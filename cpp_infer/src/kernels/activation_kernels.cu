#include "kernels/activation_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace fused_act {

namespace {

constexpr int kThreads = 256;

__global__ void fused_silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int base = row * cols;
    for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
        float gate_val = __bfloat162float(gate[base + idx]);
        float up_val = __bfloat162float(up[base + idx]);
        float silu = gate_val / (1.0f + __expf(-gate_val));
        out[base + idx] = __float2bfloat16(silu * up_val);
    }
}

void check_inputs(const torch::Tensor& gate,
                  const torch::Tensor& up,
                  const char* op_name) {
    TORCH_CHECK(gate.is_cuda(), op_name, ": gate must be CUDA");
    TORCH_CHECK(up.is_cuda(), op_name, ": up must be CUDA");
    TORCH_CHECK(gate.dtype() == torch::kBFloat16, op_name, ": gate must be bf16");
    TORCH_CHECK(up.dtype() == torch::kBFloat16, op_name, ": up must be bf16");
    TORCH_CHECK(gate.sizes() == up.sizes(), op_name, ": gate/up shape mismatch");
    TORCH_CHECK(gate.is_contiguous(), op_name, ": gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(), op_name, ": up must be contiguous");
    TORCH_CHECK(gate.dim() >= 2, op_name, ": expected at least 2D tensor");
}

}  // namespace

torch::Tensor silu_mul(const torch::Tensor& gate,
                       const torch::Tensor& up) {
    check_inputs(gate, up, "fused_silu_mul");
    auto out = torch::empty_like(gate);
    auto flat_gate = gate.reshape({-1, gate.size(-1)});
    auto flat_up = up.reshape({-1, up.size(-1)});
    auto flat_out = out.reshape({-1, out.size(-1)});
    int rows = flat_gate.size(0);
    int cols = flat_gate.size(1);

    fused_silu_mul_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(flat_up.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(flat_out.data_ptr()),
        rows,
        cols);
    return out;
}

}  // namespace fused_act
