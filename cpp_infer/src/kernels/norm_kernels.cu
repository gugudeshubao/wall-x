#include "kernels/norm_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdexcept>

namespace fused_norm {

namespace {

constexpr int kMaxHiddenSize = 2048;
constexpr int kThreads = 256;

__global__ void rms_norm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int rows,
    int cols,
    float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float row_vals[kMaxHiddenSize];
    __shared__ float warp_sum[8];

    const int base = row * cols;

    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
        float v = __bfloat162float(input[base + idx]);
        row_vals[idx] = v;
        local_sum += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (warp_id == 0) {
        total_sum = (lane < (blockDim.x >> 5)) ? warp_sum[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            total_sum += __shfl_xor_sync(0xffffffff, total_sum, offset);
        }
    }

    __shared__ float rrms_shared;
    if (threadIdx.x == 0) {
        rrms_shared = rsqrtf(total_sum / cols + eps);
    }
    __syncthreads();

    const float rrms = rrms_shared;
    for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
        float w = __bfloat162float(weight[idx]);
        float out = row_vals[idx] * rrms * w;
        output[base + idx] = __float2bfloat16(out);
    }
}

__global__ void fused_add_rms_norm_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int rows,
    int cols,
    float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float row_vals[kMaxHiddenSize];
    __shared__ float warp_sum[8];

    const int base = row * cols;

    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
        float x = __bfloat162float(input[base + idx]);
        float res = __bfloat162float(residual[base + idx]);
        float new_res = res + x;
        row_vals[idx] = new_res;
        residual[base + idx] = __float2bfloat16(new_res);
        local_sum += new_res * new_res;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (warp_id == 0) {
        total_sum = (lane < (blockDim.x >> 5)) ? warp_sum[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            total_sum += __shfl_xor_sync(0xffffffff, total_sum, offset);
        }
    }

    __shared__ float rrms_shared;
    if (threadIdx.x == 0) {
        rrms_shared = rsqrtf(total_sum / cols + eps);
    }
    __syncthreads();

    const float rrms = rrms_shared;
    for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
        float w = __bfloat162float(weight[idx]);
        float out = row_vals[idx] * rrms * w;
        output[base + idx] = __float2bfloat16(out);
    }
}

void check_inputs(const torch::Tensor& input,
                  const torch::Tensor& weight,
                  const char* op_name) {
    TORCH_CHECK(input.is_cuda(), op_name, ": input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), op_name, ": weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, op_name, ": input must be bf16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, op_name, ": weight must be bf16");
    TORCH_CHECK(input.is_contiguous(), op_name, ": input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), op_name, ": weight must be contiguous");
    TORCH_CHECK(input.size(-1) <= kMaxHiddenSize,
                op_name, ": hidden size must be <= ", kMaxHiddenSize);
    TORCH_CHECK(weight.numel() == input.size(-1),
                op_name, ": weight size must match hidden size");
}

}  // namespace

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       float eps) {
    check_inputs(input, weight, "rms_norm");
    int cols = input.size(-1);
    auto output = torch::empty_like(input);
    auto flat = input.reshape({-1, cols});
    auto flat_out = output.reshape({-1, cols});
    int rows = flat.size(0);

    rms_norm_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(flat_out.data_ptr()),
        rows,
        cols,
        eps);
    return output;
}

torch::Tensor fused_add_rms_norm(torch::Tensor residual,
                                 const torch::Tensor& input,
                                 const torch::Tensor& weight,
                                 float eps) {
    check_inputs(input, weight, "fused_add_rms_norm");
    TORCH_CHECK(residual.is_cuda(), "fused_add_rms_norm: residual must be CUDA");
    TORCH_CHECK(residual.dtype() == torch::kBFloat16, "fused_add_rms_norm: residual must be bf16");
    TORCH_CHECK(residual.is_contiguous(), "fused_add_rms_norm: residual must be contiguous");
    TORCH_CHECK(residual.sizes() == input.sizes(),
                "fused_add_rms_norm: residual/input shape mismatch");

    int cols = input.size(-1);
    auto output = torch::empty_like(residual);
    auto flat_input = input.reshape({-1, cols});
    auto flat_residual = residual.reshape({-1, cols});
    auto flat_out = output.reshape({-1, cols});
    int rows = flat_input.size(0);

    fused_add_rms_norm_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(flat_residual.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(flat_out.data_ptr()),
        rows,
        cols,
        eps);
    return output;
}

}  // namespace fused_norm
