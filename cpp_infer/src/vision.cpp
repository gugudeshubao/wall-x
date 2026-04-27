#include "vision.h"
#include <torch/torch.h>
#include <cmath>

// --- Helper: RMSNorm for vision ---
static torch::Tensor vision_rms_norm(const torch::Tensor& x, const torch::Tensor& weight, float eps) {
    auto x_f32 = x.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, /*keepdim=*/true);
    auto normed = x_f32 * torch::rsqrt(variance + eps);
    return (weight * normed).to(x.dtype());
}

// --- Helper: rotate_half for vision RoPE ---
static torch::Tensor rotate_half(const torch::Tensor& x) {
    auto x1 = x.index({"...", torch::indexing::Slice(0, x.size(-1) / 2)});
    auto x2 = x.index({"...", torch::indexing::Slice(x.size(-1) / 2)});
    return torch::cat({-x2, x1}, -1);
}

// --- Helper: apply rotary pos emb for vision ---
static std::pair<torch::Tensor, torch::Tensor> apply_rotary_pos_emb_vision(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& cos, const torch::Tensor& sin) {
    auto q_f = q.to(torch::kFloat32);
    auto k_f = k.to(torch::kFloat32);
    auto cos_u = cos.unsqueeze(-2);
    auto sin_u = sin.unsqueeze(-2);
    auto q_embed = (q_f * cos_u) + (rotate_half(q_f) * sin_u);
    auto k_embed = (k_f * cos_u) + (rotate_half(k_f) * sin_u);
    return {q_embed.to(q.dtype()), k_embed.to(k.dtype())};
}

// ===================== VisionMLP =====================

void VisionMLP::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };
    auto try_get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        return (it != weights.end()) ? it->second : torch::Tensor();
    };

    auto load_proj = [&](LinearOp& op, const std::string& name) {
        op.load(get(name + ".weight"),
                try_get(name + ".weight_scale"),
                try_get(name + ".bias"));
    };

    load_proj(gate_proj_, "mlp.gate_proj");
    load_proj(up_proj_, "mlp.up_proj");
    load_proj(down_proj_, "mlp.down_proj");
}

torch::Tensor VisionMLP::forward(const torch::Tensor& x) {
    auto gate = gate_proj_.forward(x);
    auto up = up_proj_.forward(x);
    return down_proj_.forward(torch::silu(gate) * up);
}

// ===================== VisionAttention =====================

void VisionAttention::init(int dim, int num_heads) {
    num_heads_ = num_heads;
    head_dim_ = dim / num_heads;
}

void VisionAttention::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };
    auto try_get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        return (it != weights.end()) ? it->second : torch::Tensor();
    };

    qkv_.load(get("attn.qkv.weight"),
              try_get("attn.qkv.weight_scale"),
              try_get("attn.qkv.bias"));
    proj_.load(get("attn.proj.weight"),
               try_get("attn.proj.weight_scale"),
               try_get("attn.proj.bias"));
}

torch::Tensor VisionAttention::forward(const torch::Tensor& x,
                                        const torch::Tensor& cu_seqlens,
                                        int max_seqlen,
                                        const torch::Tensor& cos,
                                        const torch::Tensor& sin) {
    int seq_length = x.size(0);

    // QKV projection: [seq, dim] -> [seq, 3*dim]
    auto qkv = qkv_.forward(x);
    // Reshape: [seq, 3, num_heads, head_dim]
    qkv = qkv.reshape({seq_length, 3, num_heads_, head_dim_});
    // Permute: [3, seq, num_heads, head_dim] and unbind
    qkv = qkv.permute({1, 0, 2, 3});
    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    // Apply vision rotary position embedding
    auto [q_rot, k_rot] = apply_rotary_pos_emb_vision(q, k, cos, sin);

    // Build attention mask from cu_seqlens (block-diagonal)
    auto attn_mask = torch::zeros({1, seq_length, seq_length},
                                   torch::TensorOptions().device(q.device()).dtype(torch::kBool));
    int num_seqs = cu_seqlens.size(0) - 1;
    auto cu_acc = cu_seqlens.to(torch::kCPU).to(torch::kInt64);
    auto cu_ptr = cu_acc.data_ptr<int64_t>();
    for (int i = 0; i < num_seqs; i++) {
        int start = cu_ptr[i];
        int end = cu_ptr[i + 1];
        attn_mask.index({torch::indexing::Slice(),
                         torch::indexing::Slice(start, end),
                         torch::indexing::Slice(start, end)}) = true;
    }

    // SDPA: [num_heads, seq, head_dim] format
    q_rot = q_rot.transpose(0, 1);  // [num_heads, seq, head_dim]
    k_rot = k_rot.transpose(0, 1);
    v = v.transpose(0, 1);

    auto attn_output = torch::scaled_dot_product_attention(
        q_rot, k_rot, v, attn_mask, /*dropout_p=*/0.0);

    // Reshape back: [num_heads, seq, head_dim] -> [seq, dim]
    attn_output = attn_output.transpose(0, 1).reshape({seq_length, -1});
    return proj_.forward(attn_output);
}

// ===================== VisionBlock =====================

void VisionBlock::init(int dim, int num_heads) {
    dim_ = dim;
    attn_.init(dim, num_heads);
}

void VisionBlock::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };
    norm1_weight_ = get("norm1.weight");
    norm2_weight_ = get("norm2.weight");
    attn_.load_weights(weights, prefix);
    mlp_.load_weights(weights, prefix);
}

torch::Tensor VisionBlock::forward(const torch::Tensor& x,
                                    const torch::Tensor& cu_seqlens,
                                    int max_seqlen,
                                    const torch::Tensor& cos,
                                    const torch::Tensor& sin) {
    // Attention with residual
    auto normed = vision_rms_norm(x, norm1_weight_, eps_);
    auto attn_out = attn_.forward(normed, cu_seqlens, max_seqlen, cos, sin);
    auto h = x + attn_out;

    // MLP with residual
    normed = vision_rms_norm(h, norm2_weight_, eps_);
    auto mlp_out = mlp_.forward(normed);
    return h + mlp_out;
}

// ===================== PatchMerger =====================

void PatchMerger::init(int dim, int context_dim, int spatial_merge_size) {
    hidden_size_ = context_dim * spatial_merge_size * spatial_merge_size;
}

void PatchMerger::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };
    auto try_get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        return (it != weights.end()) ? it->second : torch::Tensor();
    };

    ln_q_weight_ = get("merger.ln_q.weight");
    mlp_0_.load(get("merger.mlp.0.weight"),
                try_get("merger.mlp.0.weight_scale"),
                try_get("merger.mlp.0.bias"));
    mlp_2_.load(get("merger.mlp.2.weight"),
                try_get("merger.mlp.2.weight_scale"),
                try_get("merger.mlp.2.bias"));
}

torch::Tensor PatchMerger::forward(const torch::Tensor& x) {
    // x: [total_tokens, dim]
    // Apply layer norm, reshape for spatial merge, then MLP
    auto normed = vision_rms_norm(x, ln_q_weight_, eps_);
    auto merged = normed.view({-1, hidden_size_});
    auto h = mlp_0_.forward(merged);
    h = torch::gelu(h, "tanh");
    return mlp_2_.forward(h);
}

// ===================== VisionEncoder =====================

void VisionEncoder::init(const ModelConfig& config) {
    patch_size_ = config.vision_patch_size;
    temporal_patch_size_ = 2;
    embed_dim_ = config.vision_hidden_size;
    spatial_merge_size_ = config.vision_spatial_merge_size;
    spatial_merge_unit_ = spatial_merge_size_ * spatial_merge_size_;
    window_size_ = config.vision_window_size;
    num_heads_ = config.vision_num_heads;
    fullatt_block_indexes_ = config.vision_fullatt_block_indexes;

    // Initialize blocks
    blocks_.resize(config.vision_depth);
    for (int i = 0; i < config.vision_depth; i++) {
        blocks_[i].init(config.vision_hidden_size, config.vision_num_heads);
    }

    // Initialize merger
    merger_.init(config.vision_out_hidden_size, config.vision_hidden_size, spatial_merge_size_);
}

void VisionEncoder::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };

    // Patch embed Conv3D weight
    patch_embed_weight_ = get("patch_embed.proj.weight");

    // Rotary embedding inv_freq (computed, not in checkpoint)
    {
        int head_dim = embed_dim_ / num_heads_;
        auto arange = torch::arange(0, head_dim / 2, torch::kFloat32);
        rotary_inv_freq_ = 1.0f / torch::pow(10000.0f, arange * (2.0f / head_dim));
        rotary_inv_freq_ = rotary_inv_freq_.to(torch::kCUDA);
    }

    // Load each block
    for (int i = 0; i < (int)blocks_.size(); i++) {
        std::string block_prefix = prefix + "blocks." + std::to_string(i) + ".";
        blocks_[i].load_weights(weights, block_prefix);
    }

    // Load merger
    merger_.load_weights(weights, prefix);
}

torch::Tensor VisionEncoder::forward(const torch::Tensor& pixel_values,
                                      const torch::Tensor& grid_thw) {
    // --- 1. Patch Embedding (Conv3D) ---
    // pixel_values: flattened patches
    // Reshape: [-1, C, T, P, P] for Conv3D
    auto patches = pixel_values.view({-1, 3, temporal_patch_size_, patch_size_, patch_size_});
    auto target_dtype = patch_embed_weight_.dtype();
    patches = patches.to(target_dtype);

    // Manual Conv3D via torch::conv3d
    auto hidden_states = torch::conv3d(patches, patch_embed_weight_,
                                        /*bias=*/{},
                                        /*stride=*/{temporal_patch_size_, patch_size_, patch_size_});
    // Flatten: [num_patches, embed_dim]
    hidden_states = hidden_states.view({-1, embed_dim_});

    // --- 2. Rotary Position Embedding ---
    auto rotary_pos_emb = fused_rot_pos_emb_cuda(rotary_inv_freq_, grid_thw, spatial_merge_size_);

    // --- 3. Window Index Generation ---
    int vit_merger_window_size = window_size_ / spatial_merge_size_ / patch_size_;
    auto [window_index, cu_window_seqlens] = get_window_index_cuda(
        grid_thw.to(torch::kInt32), spatial_merge_size_,
        vit_merger_window_size, patch_size_, spatial_merge_unit_);

    // --- 4. Reorder by window index ---
    int seq_len = hidden_states.size(0);
    hidden_states = hidden_states.reshape({seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    hidden_states = hidden_states.index_select(0, window_index);
    hidden_states = hidden_states.reshape({seq_len, -1});

    rotary_pos_emb = rotary_pos_emb.reshape({seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    rotary_pos_emb = rotary_pos_emb.index_select(0, window_index);
    rotary_pos_emb = rotary_pos_emb.reshape({seq_len, -1});

    // Position embeddings: cos and sin from rotary_pos_emb
    // rotary_pos_emb has shape [seq_len, head_dim] from fused_rot_pos_emb_cuda
    auto cos = rotary_pos_emb.cos();
    auto sin = rotary_pos_emb.sin();

    // --- 5. Full attention cumulative sequence lengths ---
    auto cu_seqlens_full = torch::repeat_interleave(
        grid_thw.index({torch::indexing::Slice(), 1}) * grid_thw.index({torch::indexing::Slice(), 2}),
        grid_thw.index({torch::indexing::Slice(), 0})
    ).cumsum(0, torch::kInt32);
    cu_seqlens_full = torch::nn::functional::pad(cu_seqlens_full, torch::nn::functional::PadFuncOptions({1, 0}));

    auto cu_full_cpu = cu_seqlens_full.to(torch::kCPU);
    int max_seqlen_full = 0;
    {
        auto acc = cu_full_cpu.accessor<int, 1>();
        for (int i = 1; i < acc.size(0); i++) {
            max_seqlen_full = std::max(max_seqlen_full, acc[i] - acc[i - 1]);
        }
    }

    // cu_window_seqlens comes from CUDA kernel as tensor
    auto cu_win_cpu = cu_window_seqlens.to(torch::kCPU);
    int max_seqlen_window = 0;
    {
        auto acc = cu_win_cpu.accessor<int, 1>();
        for (int i = 1; i < acc.size(0); i++) {
            max_seqlen_window = std::max(max_seqlen_window, acc[i] - acc[i - 1]);
        }
    }

    // --- 6. Process through vision blocks ---
    for (int i = 0; i < (int)blocks_.size(); i++) {
        bool is_fullatt = std::find(fullatt_block_indexes_.begin(),
                                     fullatt_block_indexes_.end(), i) != fullatt_block_indexes_.end();
        auto& cu = is_fullatt ? cu_seqlens_full : cu_window_seqlens;
        int max_sl = is_fullatt ? max_seqlen_full : max_seqlen_window;
        hidden_states = blocks_[i].forward(hidden_states, cu, max_sl, cos, sin);
    }

    // --- 7. Patch merger ---
    hidden_states = merger_.forward(hidden_states);

    // --- 8. Reverse window reordering ---
    auto reverse_indices = torch::argsort(window_index);
    hidden_states = hidden_states.index_select(0, reverse_indices);

    return hidden_states;
}
