#pragma once

#include <torch/torch.h>
#include <cublasLt.h>

// =============================================================================
// Fused INT8 CUDA Kernels for W8A8 Dynamic Quantization
//
// Two optimization routes vs naive PyTorch-op approach:
//   Route 1 (cublasLt): fused_quant + cublasLt_gemm + fused_dequant  (3 kernels)
//   Route 2 (_int_mm):  fused_quant + at::_int_mm  + fused_dequant  (3 kernels)
// vs Naive:             ~10 PyTorch kernels per linear layer
// =============================================================================

namespace int8_fused {

// ----- Fused CUDA kernels (shared by both routes) -----

/// Fused per-token activation quantize: bf16 [M,K] → (int8 [M,K], scale f32 [M])
/// One block per row, shared-memory reduction for absmax.
std::tuple<torch::Tensor, torch::Tensor> quantize_activation(
    const torch::Tensor& input);   // [M, K] bf16

/// Fused dequantize: int32 [M,N] × act_scale [M] × weight_scale [N] + bias → bf16 [M,N]
/// Single elementwise kernel with two scale lookups.
torch::Tensor dequantize(
    const torch::Tensor& input_i32,
    const torch::Tensor& act_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& bias = {});

/// Fused row-scale + bf16 cast: f32 [M,N] × act_scale [M] + bias → bf16 [M,N]
/// (For Route 1 if cublasLt outputs f32 with weight_scale already applied)
torch::Tensor row_scale_cast(
    const torch::Tensor& input_f32,
    const torch::Tensor& act_scale,
    const torch::Tensor& bias = {});


// ----- Route 1: Direct cublasLt INT8 GEMM -----

class CublasLtInt8Gemm {
public:
    CublasLtInt8Gemm();
    ~CublasLtInt8Gemm();

    /// C[M,N] = act_int8[M,K] @ weight_int8_t[K,N], output int32
    /// Uses cublasLt directly (bypasses at::_int_mm overhead).
    torch::Tensor run(
        const torch::Tensor& act_int8,       // [M, K] int8
        const torch::Tensor& weight_int8_t); // [K, N] int8

private:
    cublasLtHandle_t handle_ = nullptr;
    void* workspace_ = nullptr;
    static constexpr size_t kWorkspaceSize = 4 * 1024 * 1024;  // 4MB
};

}  // namespace int8_fused
