#include "attention.h"
#include <torch/torch.h>

void Attention::init(const ModelConfig& config, int layer_idx) {
    num_heads_ = config.num_attention_heads;
    num_kv_heads_ = config.num_key_value_heads;
    head_dim_ = config.head_dim;
    hidden_size_ = config.hidden_size;
    layer_idx_ = layer_idx;

    // mrope_section doubled for the RoPE kernel
    for (int s : config.mrope_section) {
        mrope_section_doubled_.push_back(s * 2);
    }
}

void Attention::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) {
            throw std::runtime_error("Weight not found: " + prefix + name);
        }
        return it->second;
    };

    auto try_get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        return (it != weights.end()) ? it->second : torch::Tensor();
    };

    // Load projections via LinearOp (auto-detects INT8 by weight dtype)
    auto load_proj = [&](LinearOp& op, const std::string& name) {
        op.load(get(name + ".weight"),
                try_get(name + ".weight_scale"),
                try_get(name + ".bias"));
    };

    load_proj(q_proj_, "self_attn.q_proj");
    load_proj(k_proj_, "self_attn.k_proj");
    load_proj(v_proj_, "self_attn.v_proj");
    load_proj(o_proj_, "self_attn.o_proj");

    if (q_proj_.is_int8()) {
        std::cout << "    [INT8] Attention layer " << layer_idx_ << " using W8A8 quantization" << std::endl;
    }
}

torch::Tensor Attention::forward(const torch::Tensor& x,
                                  const torch::Tensor& cos,
                                  const torch::Tensor& sin,
                                  KVCache& cache,
                                  int layer_idx,
                                  bool is_causal) {
    auto batch = x.size(0);
    auto seq_len = x.size(1);

    // Q/K/V projections via LinearOp (bf16 or INT8 depending on loaded weights)
    auto q = q_proj_.forward(x);
    auto k = k_proj_.forward(x);
    auto v = v_proj_.forward(x);

    // Reshape: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
    q = q.view({batch, seq_len, num_heads_, head_dim_}).transpose(1, 2);
    k = k.view({batch, seq_len, num_kv_heads_, head_dim_}).transpose(1, 2);
    v = v.view({batch, seq_len, num_kv_heads_, head_dim_}).transpose(1, 2);

    // Apply multimodal RoPE via CUDA kernel
    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    launch_multimodal_rope_forward(q, k, cos, sin, q_out, k_out, mrope_section_doubled_);
    q = q_out;
    k = k_out;

    // Update KV cache
    cache.update(layer_idx, k, v);

    // Get full K, V from cache (includes newly written data via get_with_new)
    // current_len_ hasn't been advanced yet, so we pass new_seq as extra
    auto [cached_k, cached_v] = cache.get_with_new(layer_idx, k.size(2));

    // Expand KV heads for GQA: [batch, kv_heads, seq, dim] -> [batch, num_heads, seq, dim]
    int num_groups = num_heads_ / num_kv_heads_;
    if (num_groups > 1) {
        cached_k = cached_k.unsqueeze(2).expand({batch, num_kv_heads_, num_groups, -1, head_dim_})
                          .reshape({batch, num_heads_, -1, head_dim_});
        cached_v = cached_v.unsqueeze(2).expand({batch, num_kv_heads_, num_groups, -1, head_dim_})
                          .reshape({batch, num_heads_, -1, head_dim_});
    }

    // SDPA: scaled_dot_product_attention with is_causal=true, NO attention_mask
    // This forces cuDNN backend (0.076ms, near TRT-LLM level)
    auto attn_output = torch::scaled_dot_product_attention(
        q, cached_k, cached_v,
        /*attn_mask=*/{},           // No mask! This is the key insight from article 2
        /*dropout_p=*/0.0,
        /*is_causal=*/is_causal
    );

    // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden_size]
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch, seq_len, hidden_size_});

    // Output projection
    return o_proj_.forward(attn_output);
}
