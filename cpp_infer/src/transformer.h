#pragma once
#include <torch/torch.h>
#include "attention.h"
#include "moe.h"
#include "kv_cache.h"
#include "triton_loader.h"
#include "utils.h"

class TransformerLayer {
public:
    TransformerLayer() = default;
    void init(const ModelConfig& config, int layer_idx);
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward pass for one layer (in-place residual add for zero-alloc overhead)
    // hidden_states: [batch, seq_len, hidden_size]  -- modified in-place
    // Returns: same tensor with residual connections applied
    torch::Tensor forward(torch::Tensor hidden_states,
                          const torch::Tensor& cos,
                          const torch::Tensor& sin,
                          KVCache& cache,
                          const TokenTypeInfo& token_types,
                          TritonKernelRegistry& triton,
                          bool is_causal = true);

private:
    // RMSNorm weights
    torch::Tensor input_layernorm_weight_;
    torch::Tensor post_attention_layernorm_weight_;

    Attention attn_;
    MoEBlock moe_;

    int layer_idx_ = 0;
    float rms_norm_eps_ = 1e-6f;
    int hidden_size_ = 0;
    bool use_moe_ = true;
};
