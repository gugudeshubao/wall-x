#pragma once
#include <torch/torch.h>
#include "utils.h"

// Forward declarations for CUDA ops from csrc/
extern torch::Tensor fused_rot_pos_emb_cuda(
    torch::Tensor inv_freq,
    torch::Tensor grid_thw,
    int spatial_merge_size);

extern std::tuple<torch::Tensor, torch::Tensor> get_window_index_cuda(
    torch::Tensor grid_thw,
    int spatial_merge_size,
    int vit_merger_window_size,
    int patch_size,
    int spatial_merge_unit);

// ViT MLP block (SiLU-gated: out = down_proj(silu(gate_proj(x)) * up_proj(x)))
class VisionMLP {
public:
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::Tensor gate_proj_weight_, gate_proj_bias_;
    torch::Tensor up_proj_weight_, up_proj_bias_;
    torch::Tensor down_proj_weight_, down_proj_bias_;
};

// ViT attention (SDPA variant)
class VisionAttention {
public:
    void init(int dim, int num_heads);
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x,
                          const torch::Tensor& cu_seqlens,
                          int max_seqlen,
                          const torch::Tensor& cos,
                          const torch::Tensor& sin);

private:
    torch::Tensor qkv_weight_, qkv_bias_;
    torch::Tensor proj_weight_, proj_bias_;
    int num_heads_ = 0;
    int head_dim_ = 0;
};

// Single ViT block: LayerNorm + Attention + MLP
class VisionBlock {
public:
    void init(int dim, int num_heads);
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x,
                          const torch::Tensor& cu_seqlens,
                          int max_seqlen,
                          const torch::Tensor& cos,
                          const torch::Tensor& sin);

private:
    torch::Tensor norm1_weight_;
    torch::Tensor norm2_weight_;
    VisionAttention attn_;
    VisionMLP mlp_;
    int dim_ = 0;
    float eps_ = 1e-6f;
};

// Patch merger: spatial merge + MLP projection
class PatchMerger {
public:
    void init(int dim, int context_dim, int spatial_merge_size);
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::Tensor ln_q_weight_;
    torch::Tensor mlp_0_weight_, mlp_0_bias_;  // Linear 1
    torch::Tensor mlp_2_weight_, mlp_2_bias_;  // Linear 2
    int hidden_size_ = 0;  // context_dim * spatial_merge_size^2
    float eps_ = 1e-6f;
};

// Complete Vision Encoder
class VisionEncoder {
public:
    VisionEncoder() = default;
    void init(const ModelConfig& config);
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward: pixel_values -> vision embeddings
    // pixel_values: raw pixel data for Conv3D
    // grid_thw: [num_images, 3] temporal, height, width grid dimensions
    // Returns: [total_merged_tokens, out_hidden_size]
    torch::Tensor forward(const torch::Tensor& pixel_values,
                          const torch::Tensor& grid_thw);

private:
    // Patch embedding (Conv3D)
    torch::Tensor patch_embed_weight_;  // Conv3D weight
    int patch_size_ = 14;
    int temporal_patch_size_ = 2;
    int embed_dim_ = 1280;

    // Rotary embedding inverse frequencies
    torch::Tensor rotary_inv_freq_;

    // Vision blocks
    std::vector<VisionBlock> blocks_;

    // Patch merger
    PatchMerger merger_;

    // Config
    int spatial_merge_size_ = 2;
    int spatial_merge_unit_ = 4;  // spatial_merge_size^2
    int window_size_ = 112;
    int num_heads_ = 16;
    std::vector<int> fullatt_block_indexes_;
};
