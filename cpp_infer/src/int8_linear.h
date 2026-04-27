#pragma once

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>

#include "int8_kernels.h"       // fused quant/dequant CUDA kernels
#include "cutlass_int8_gemm.h"  // CUTLASS fused INT8 GEMM + dequant epilogue

// =============================================================================
// W8A8 Dynamic Quantization: INT8 Linear Layer
//
// Weight:     per-channel symmetric INT8 (static, from offline quantization)
// Activation: per-token symmetric INT8 (dynamic, computed at runtime)
// GEMM:       CUTLASS fused INT8 GEMM with dequant epilogue
//
// Pipeline (2 CUDA kernels total):
//   1. fused_quantize_activation: bf16 → (int8, scale)    [1 kernel]
//   2. CUTLASS GEMM+dequant:     int8 × int8 → bf16      [1 kernel]
//      (INT32 accumulator stays in registers, scale + bf16 cast in epilogue)
//
// Benchmark (M=32, K=N=2048, Orin SM 8.7):
//   bf16 baseline: 55.7us | CUTLASS fused: 37.7us → 1.47x speedup
// =============================================================================

namespace int8_quant {

/// CUTLASS-fused INT8 linear: fused_quant + CUTLASS GEMM+dequant (2 kernels)
///
/// @param input          [*, K] bf16  — activation tensor
/// @param weight_int8    [N, K] int8  — quantized weight (original layout)
/// @param weight_scale   [N] f32     — per-channel weight scale
/// @param bias           [N] bf16    — optional bias
/// @return               [*, N] bf16
inline torch::Tensor linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_int8,
    const torch::Tensor& weight_scale,
    const torch::Tensor& bias = {})
{
    auto orig_sizes = input.sizes().vec();
    int64_t K = input.size(-1);
    auto flat = input.reshape({-1, K});  // [M, K]

    // 1. Fused per-token activation quantization (1 CUDA kernel)
    auto [act_int8, act_scale] = int8_fused::quantize_activation(flat);

    // 2. CUTLASS fused GEMM + dequant epilogue (1 CUDA kernel)
    //    INT32 accumulator stays in registers → scale multiply + bf16 cast in epilogue
    auto out = cutlass_int8::gemm_dequant(act_int8, weight_int8, act_scale, weight_scale);

    // 3. Optional bias add
    if (bias.defined()) {
        out = out + bias;
    }

    orig_sizes.back() = weight_int8.size(0);  // N (out_features)
    return out.reshape(orig_sizes);
}

}  // namespace int8_quant


// =============================================================================
// LinearOp: Transparent bf16/INT8 linear layer wrapper.
//
// Usage in load_weights():
//   op.load(get("proj.weight"), try_get("proj.weight_scale"), try_get("proj.bias"));
//
// Usage in forward():
//   auto out = op.forward(x);
//
// Automatically selects bf16 (torch::linear) or INT8 (CUTLASS fused)
// based on weight dtype at load time.
// =============================================================================
class LinearOp {
public:
    /// Load weight (bf16 or int8), optional scale, optional bias.
    /// Detects quantized weights by checking dtype == int8 && scale is defined.
    void load(const torch::Tensor& weight,
              const torch::Tensor& scale = {},
              const torch::Tensor& bias = {}) {
        bias_ = bias;
        if (weight.dtype() == torch::kInt8 && scale.defined()) {
            // INT8 quantized path: store weight as [N, K] for CUTLASS ColumnMajor B
            weight_int8_ = weight.contiguous();  // [N, K] — original layout
            weight_scale_ = scale;                // [N]
            use_int8_ = true;
        } else {
            // bf16 standard path
            weight_ = weight;
            use_int8_ = false;
        }
    }

    /// Forward: x @ W^T + bias
    torch::Tensor forward(const torch::Tensor& x) const {
        if (use_int8_) {
            return int8_quant::linear(x, weight_int8_, weight_scale_, bias_);
        }
        return torch::linear(x, weight_,
                              bias_.defined() ? bias_ : torch::Tensor());
    }

    /// Check if this layer uses INT8 quantization
    bool is_int8() const { return use_int8_; }

    /// Access raw bf16 weight (only valid when !is_int8())
    const torch::Tensor& weight_bf16() const { return weight_; }

private:
    // bf16 path
    torch::Tensor weight_;           // [N, K] bf16

    // INT8 path
    torch::Tensor weight_int8_;      // [N, K] int8 (original layout for CUTLASS)
    torch::Tensor weight_scale_;     // [N] f32 (per-channel scale)

    // Shared
    torch::Tensor bias_;             // [N] bf16 (optional)
    bool use_int8_ = false;
};
