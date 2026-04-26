#pragma once
#include <torch/torch.h>
#include <vector>

class KVCache {
public:
    KVCache() = default;

    // Pre-allocate cache for all layers
    // Shape per layer: K,V = [batch, kv_heads, max_seq, head_dim]
    void allocate(int num_layers, int batch_size, int max_seq_len,
                  int num_kv_heads, int head_dim, torch::Device device);

    // Append new K, V to cache for a given layer
    // new_k, new_v: [batch, kv_heads, new_seq, head_dim]
    void update(int layer_idx, const torch::Tensor& new_k, const torch::Tensor& new_v);

    // Get cached K, V up to current length
    // Returns [batch, kv_heads, current_len, head_dim]
    std::pair<torch::Tensor, torch::Tensor> get(int layer_idx) const;

    // Get cached K, V including newly written data (before advance)
    // extra: number of new tokens written by update() in this pass
    // Returns [batch, kv_heads, current_len + extra, head_dim]
    std::pair<torch::Tensor, torch::Tensor> get_with_new(int layer_idx, int extra) const;

    // Advance current_len by n (call after all layers processed)
    void advance(int n);

    // Get current sequence length
    int current_len() const { return current_len_; }

    // Truncate cache to given length (for ODE prefix truncation)
    void truncate(int new_len);

    // Reset cache (for new sequence)
    void reset();

    bool is_allocated() const { return !k_cache_.empty(); }

private:
    std::vector<torch::Tensor> k_cache_;  // [num_layers] each [batch, kv_heads, max_seq, head_dim]
    std::vector<torch::Tensor> v_cache_;
    int current_len_ = 0;
    int max_seq_len_ = 0;
    int batch_size_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
};
