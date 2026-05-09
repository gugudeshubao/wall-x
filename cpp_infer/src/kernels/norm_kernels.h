#pragma once

#include <torch/torch.h>

namespace fused_norm {

// Standalone RMSNorm for the common hidden sizes in wall-x.
// Supports up to 2048 columns (decoder 2048, vision 1280).
torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       float eps);

// Fused residual add + RMSNorm.
// Updates residual in-place: residual += input
// Returns normalized output tensor with the same shape.
torch::Tensor fused_add_rms_norm(torch::Tensor residual,
                                 const torch::Tensor& input,
                                 const torch::Tensor& weight,
                                 float eps);

}  // namespace fused_norm
