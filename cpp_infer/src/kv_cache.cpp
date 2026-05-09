#include "kv_cache.h"

void KVCache::allocate(int num_layers, int batch_size, int max_seq_len,
                       int num_kv_heads, int head_dim, torch::Device device) {
    if ((int)k_cache_.size() == num_layers &&
        batch_size_ == batch_size &&
        max_seq_len_ == max_seq_len &&
        num_kv_heads_ == num_kv_heads &&
        head_dim_ == head_dim &&
        device_ == device) {
        current_len_ = 0;
        return;
    }

    batch_size_ = batch_size;
    max_seq_len_ = max_seq_len;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    device_ = device;
    current_len_ = 0;

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    k_cache_.clear();
    v_cache_.clear();
    k_cache_.reserve(num_layers);
    v_cache_.reserve(num_layers);

    for (int i = 0; i < num_layers; i++) {
        k_cache_.push_back(torch::zeros({batch_size, num_kv_heads, max_seq_len, head_dim}, options));
        v_cache_.push_back(torch::zeros({batch_size, num_kv_heads, max_seq_len, head_dim}, options));
    }
}

void KVCache::update(int layer_idx, const torch::Tensor& new_k, const torch::Tensor& new_v) {
    int new_seq = new_k.size(2);
    // Copy new K, V into cache at current position
    k_cache_[layer_idx].index({torch::indexing::Slice(), torch::indexing::Slice(),
                                torch::indexing::Slice(current_len_, current_len_ + new_seq)})
        .copy_(new_k);
    v_cache_[layer_idx].index({torch::indexing::Slice(), torch::indexing::Slice(),
                                torch::indexing::Slice(current_len_, current_len_ + new_seq)})
        .copy_(new_v);

    // Only update length after all layers are processed (caller manages this)
}

std::pair<torch::Tensor, torch::Tensor> KVCache::get(int layer_idx) const {
    // Return slice [batch, kv_heads, 0:current_len, head_dim]
    auto k = k_cache_[layer_idx].index({torch::indexing::Slice(), torch::indexing::Slice(),
                                          torch::indexing::Slice(0, current_len_)});
    auto v = v_cache_[layer_idx].index({torch::indexing::Slice(), torch::indexing::Slice(),
                                          torch::indexing::Slice(0, current_len_)});
    return {k, v};
}

std::pair<torch::Tensor, torch::Tensor> KVCache::get_with_new(int layer_idx, int extra) const {
    // Return slice [batch, kv_heads, 0:current_len+extra, head_dim]
    // Used during forward pass before advance() is called
    int total = current_len_ + extra;
    auto k = k_cache_[layer_idx].index({torch::indexing::Slice(), torch::indexing::Slice(),
                                          torch::indexing::Slice(0, total)});
    auto v = v_cache_[layer_idx].index({torch::indexing::Slice(), torch::indexing::Slice(),
                                          torch::indexing::Slice(0, total)});
    return {k, v};
}

void KVCache::advance(int n) {
    current_len_ += n;
    if (current_len_ > max_seq_len_) {
        current_len_ = max_seq_len_;
    }
}

void KVCache::truncate(int new_len) {
    if (new_len >= 0 && new_len <= max_seq_len_) {
        current_len_ = new_len;
    }
}

void KVCache::reset() {
    current_len_ = 0;
    // No need to zero out data; slicing handles bounds
}
