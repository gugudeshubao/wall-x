#pragma once

#include <torch/torch.h>

namespace fused_act {

// Fused SiLU(gate) * up for bf16 CUDA tensors with matching shape.
torch::Tensor silu_mul(const torch::Tensor& gate,
                       const torch::Tensor& up);

}  // namespace fused_act
