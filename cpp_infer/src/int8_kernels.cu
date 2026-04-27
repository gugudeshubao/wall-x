#include "int8_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <stdexcept>

// =====================================================================
// Macro helpers
// =====================================================================
#define CUBLASLT_CHECK(expr)                                               \
    do {                                                                   \
        cublasStatus_t status = (expr);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            throw std::runtime_error(                                      \
                std::string("cublasLt error at ") + __FILE__ + ":" +       \
                std::to_string(__LINE__) + " code=" +                      \
                std::to_string((int)status));                              \
        }                                                                  \
    } while (0)


// =====================================================================
// CUDA Kernel 1: Fused per-token activation quantize
//   Input:  bf16 [M, K]
//   Output: int8 [M, K],  scale f32 [M]
//
//   One block per row.  Two-phase:
//     Phase 1 – shared-mem reduction for row absmax
//     Phase 2 – quantize with inv_scale
// =====================================================================
__global__ void fused_quantize_kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t*              __restrict__ output,
    float*               __restrict__ scales,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const __nv_bfloat16* in_row  = input  + (int64_t)row * K;
    int8_t*              out_row = output + (int64_t)row * K;

    // --- Phase 1: per-row absmax ---
    float tmax = 0.0f;
    for (int j = threadIdx.x; j < K; j += blockDim.x)
        tmax = fmaxf(tmax, fabsf(__bfloat162float(in_row[j])));

    // warp-level reduction
    for (int off = 16; off > 0; off >>= 1)
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));

    // block-level reduction (up to 8 warps for blockDim ≤ 256)
    __shared__ float warp_max[8];
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    if (lid == 0) warp_max[wid] = tmax;
    __syncthreads();

    if (wid == 0) {
        tmax = (lid < (blockDim.x >> 5)) ? warp_max[lid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1)
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
    }

    __shared__ float scale_s;
    if (threadIdx.x == 0) {
        float s = fmaxf(tmax / 127.0f, 1e-10f);
        scales[row] = s;
        scale_s = s;
    }
    __syncthreads();

    // --- Phase 2: quantize ---
    const float inv_s = 1.0f / scale_s;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        float v = __bfloat162float(in_row[j]) * inv_s;
        int   q = __float2int_rn(v);
        out_row[j] = (int8_t)max(-128, min(127, q));
    }
}


// =====================================================================
// CUDA Kernel 2: Fused dequantize
//   int32 [M,N] × act_scale [M] × weight_scale [N] + bias [N] → bf16 [M,N]
// =====================================================================
__global__ void fused_dequantize_kernel(
    const int32_t*       __restrict__ input,
    const float*         __restrict__ act_scale,
    const float*         __restrict__ weight_scale,
    const __nv_bfloat16* __restrict__ bias,   // nullptr if no bias
    __nv_bfloat16*       __restrict__ output,
    int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    const int i = idx / N;
    const int j = idx % N;

    float val = (float)input[idx] * act_scale[i] * weight_scale[j];
    if (bias) val += __bfloat162float(bias[j]);
    output[idx] = __float2bfloat16(val);
}


// =====================================================================
// CUDA Kernel 3: Fused row-scale + cast
//   f32 [M,N] × act_scale [M] + bias [N] → bf16 [M,N]
//   (used when cublasLt already applied weight_scale via alpha_vector)
// =====================================================================
__global__ void fused_row_scale_cast_kernel(
    const float*         __restrict__ input,
    const float*         __restrict__ act_scale,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16*       __restrict__ output,
    int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    const int i = idx / N;
    const int j = idx % N;

    float val = input[idx] * act_scale[i];
    if (bias) val += __bfloat162float(bias[j]);
    output[idx] = __float2bfloat16(val);
}


// =====================================================================
// C++ wrappers
// =====================================================================

namespace int8_fused {

std::tuple<torch::Tensor, torch::Tensor> quantize_activation(
    const torch::Tensor& input)
{
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16,
                "quantize_activation: expect CUDA bf16 input");
    auto flat = input.reshape({-1, input.size(-1)});
    const int M = flat.size(0), K = flat.size(1);

    auto out   = torch::empty({M, K}, flat.options().dtype(torch::kInt8));
    auto scale = torch::empty({M},    flat.options().dtype(torch::kFloat32));

    // block size: up to 256, rounded to warp boundary
    int threads = std::min(256, ((K + 31) / 32) * 32);
    threads = std::max(32, threads);

    fused_quantize_kernel<<<M, threads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat.data_ptr()),
        out.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        M, K);

    return {out, scale};
}


torch::Tensor dequantize(
    const torch::Tensor& input_i32,
    const torch::Tensor& act_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& bias)
{
    const int M = input_i32.size(0), N = input_i32.size(1);
    auto output = torch::empty({M, N},
                               input_i32.options().dtype(torch::kBFloat16));

    const int total   = M * N;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    fused_dequantize_kernel<<<blocks, threads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        input_i32.data_ptr<int32_t>(),
        act_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        bias.defined()
            ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr())
            : nullptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        M, N);

    return output;
}


torch::Tensor row_scale_cast(
    const torch::Tensor& input_f32,
    const torch::Tensor& act_scale,
    const torch::Tensor& bias)
{
    const int M = input_f32.size(0), N = input_f32.size(1);
    auto output = torch::empty({M, N},
                               input_f32.options().dtype(torch::kBFloat16));

    const int total   = M * N;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    fused_row_scale_cast_kernel<<<blocks, threads, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
        input_f32.data_ptr<float>(),
        act_scale.data_ptr<float>(),
        bias.defined()
            ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr())
            : nullptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        M, N);

    return output;
}


// =====================================================================
// Route 1:  Direct cublasLt INT8 GEMM
// =====================================================================

CublasLtInt8Gemm::CublasLtInt8Gemm() {
    CUBLASLT_CHECK(cublasLtCreate(&handle_));
    cudaMalloc(&workspace_, kWorkspaceSize);
}

CublasLtInt8Gemm::~CublasLtInt8Gemm() {
    if (workspace_) cudaFree(workspace_);
    if (handle_)    cublasLtDestroy(handle_);
}

torch::Tensor CublasLtInt8Gemm::run(
    const torch::Tensor& act_int8,        // [M, K] int8  row-major
    const torch::Tensor& weight_int8_t)   // [K, N] int8  row-major
{
    const int M = act_int8.size(0);
    const int K = act_int8.size(1);
    const int N = weight_int8_t.size(1);

    auto output = torch::empty({M, N},
                               act_int8.options().dtype(torch::kInt32));

    // ---- matmul descriptor ----
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(
        &matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    cublasOperation_t opN = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // ---- matrix layouts (col-major trick for row-major data) ----
    // We compute C = A @ B  (all row-major)
    //   ⟹ col-major:  C^T = B^T @ A^T
    //   ⟹ cublasLt "A" = weight_int8_t  as col-major [N, K], lda=N
    //     cublasLt "B" = act_int8       as col-major [K, M], ldb=K
    //     cublasLt "C" = output         as col-major [N, M], ldc=N
    //     m=N, n=M, k=K

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &layoutA, CUDA_R_8I, N, K, /*ld=*/N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &layoutB, CUDA_R_8I, K, M, /*ld=*/K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &layoutC, CUDA_R_32I, N, M, /*ld=*/N));

    // ---- algorithm heuristic ----
    cublasLtMatmulPreference_t pref;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kWorkspaceSize, sizeof(kWorkspaceSize)));

    cublasLtMatmulHeuristicResult_t heurResult;
    int returnedResults = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle_, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        pref, 1, &heurResult, &returnedResults));

    if (returnedResults == 0)
        throw std::runtime_error("cublasLt: no INT8 algorithm found");

    // ---- run ----
    int32_t alpha = 1, beta = 0;
    CUBLASLT_CHECK(cublasLtMatmul(
        handle_, matmulDesc,
        &alpha,
        weight_int8_t.data_ptr(), layoutA,   // "A" in col-major
        act_int8.data_ptr(),      layoutB,   // "B" in col-major
        &beta,
        output.data_ptr(),        layoutC,   // "C"
        output.data_ptr(),        layoutC,   // "D" = same as C
        &heurResult.algo,
        workspace_, kWorkspaceSize,
        at::cuda::getCurrentCUDAStream()));

    // cleanup
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);

    return output;
}

}  // namespace int8_fused
