#include "transformer.h"
#include <torch/torch.h>

static torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight, float eps) {
    // Compute in float32 for numerical stability
    auto x_f32 = x.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, /*keepdim=*/true);
    auto normed = x_f32 * torch::rsqrt(variance + eps);
    return (weight * normed).to(x.dtype());
}

void TransformerLayer::init(const ModelConfig& config, int layer_idx) {
    layer_idx_ = layer_idx;
    rms_norm_eps_ = config.rms_norm_eps;
    hidden_size_ = config.hidden_size;
    use_moe_ = config.mlp_moe;

    attn_.init(config, layer_idx);
    if (use_moe_) {
        moe_.init(config, layer_idx);
    }
}

void TransformerLayer::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) {
            throw std::runtime_error("Weight not found: " + prefix + name);
        }
        return it->second;
    };

    input_layernorm_weight_ = get("input_layernorm.weight");
    post_attention_layernorm_weight_ = get("post_attention_layernorm.weight");

    attn_.load_weights(weights, prefix);
    if (use_moe_) {
        moe_.load_weights(weights, prefix);
    }
}

torch::Tensor TransformerLayer::forward(torch::Tensor hidden_states,
                                         const torch::Tensor& cos,
                                         const torch::Tensor& sin,
                                         KVCache& cache,
                                         const TokenTypeInfo& token_types,
                                         TritonKernelRegistry& triton,
                                         bool is_causal) {
    // --- Pre-attention RMSNorm ---
    auto normed = rms_norm(hidden_states, input_layernorm_weight_, rms_norm_eps_);

    // --- Self-Attention ---
    auto attn_output = attn_.forward(normed, cos, sin, cache, layer_idx_, is_causal);

    // --- Residual add (in-place: avoids tensor allocation) ---
    hidden_states.add_(attn_output);

    // --- Post-attention RMSNorm ---
    normed = rms_norm(hidden_states, post_attention_layernorm_weight_, rms_norm_eps_);

    // --- MoE FFN ---
    torch::Tensor ffn_output;
    if (use_moe_) {
        ffn_output = moe_.forward(normed, token_types, triton);
    } else {
        ffn_output = normed;
    }

    // --- Residual add (in-place) ---
    hidden_states.add_(ffn_output);

    return hidden_states;
}
