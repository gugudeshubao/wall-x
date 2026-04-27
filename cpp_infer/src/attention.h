#pragma once
#include <torch/torch.h>
#include "int8_linear.h"
#include "kv_cache.h"
#include "triton_loader.h"
#include "utils.h"

// Forward declarations for CUDA ops from csrc/
extern void launch_multimodal_rope_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin,
    torch::Tensor q_out, torch::Tensor k_out,
    std::vector<int> mrope_section_doubled
);

class Attention {
public:
    Attention() = default;
    void init(const ModelConfig& config, int layer_idx);
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward pass:
    //   x: [batch, seq_len, hidden_size]
    //   position_ids: [3, batch, seq_len] (temporal, height, width for mrope)
    //   cos, sin: precomputed rotary embeddings
    //   is_causal: whether to use causal mask
    // Returns: [batch, seq_len, hidden_size]
    torch::Tensor forward(const torch::Tensor& x,
                          const torch::Tensor& cos,
                          const torch::Tensor& sin,
                          KVCache& cache,
                          int layer_idx,
                          bool is_causal = true);

private:
    LinearOp q_proj_;  // [num_heads * head_dim, hidden_size]
    LinearOp k_proj_;  // [num_kv_heads * head_dim, hidden_size]
    LinearOp v_proj_;  // [num_kv_heads * head_dim, hidden_size]
    LinearOp o_proj_;  // [hidden_size, num_heads * head_dim]

    int num_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int hidden_size_ = 0;
    int layer_idx_ = 0;
    std::vector<int> mrope_section_doubled_;
};
