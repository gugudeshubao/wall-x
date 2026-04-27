#pragma once

#include <torch/torch.h>

namespace cutlass_int8 {

/// CUTLASS fused INT8 GEMM with dequantization epilogue.
///
/// Computes: output[M,N] = (act_int8[M,K] @ weight_int8^T[K,N]) * act_scale[M] * weight_scale[N]
/// All scale multiplications happen in the GEMM epilogue — no INT32 writeback to GMEM.
/// Output is BF16 directly.
///
/// @param act_int8       [M, K] int8   — quantized activations
/// @param weight_int8    [N, K] int8   — quantized weights (original, NOT transposed)
/// @param act_scale      [M] f32       — per-token activation scale
/// @param weight_scale   [N] f32       — per-channel weight scale
/// @return               [M, N] bf16
torch::Tensor gemm_dequant(
    const torch::Tensor& act_int8,
    const torch::Tensor& weight_int8,
    const torch::Tensor& act_scale,
    const torch::Tensor& weight_scale);

}  // namespace cutlass_int8
