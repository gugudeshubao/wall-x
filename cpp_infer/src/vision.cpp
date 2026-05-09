#include "vision.h"
#include "kernels/activation_kernels.h"
#include "kernels/norm_kernels.h"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

// --- Helper: RMSNorm for vision ---
static torch::Tensor vision_rms_norm(const torch::Tensor& x, const torch::Tensor& weight, float eps) {
    bool disable_fused_ops = std::getenv("WALLX_DISABLE_VISION_FUSED_OPS") != nullptr;
    if (!disable_fused_ops &&
        x.is_cuda() &&
        x.dtype() == torch::kBFloat16 &&
        x.is_contiguous() &&
        weight.defined() &&
        weight.is_cuda() &&
        weight.dtype() == torch::kBFloat16 &&
        weight.is_contiguous() &&
        weight.numel() == x.size(-1) &&
        x.size(-1) <= 2048) {
        return fused_norm::rms_norm(x, weight, eps);
    }
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

static VisionAttentionLayout make_attention_layout(std::vector<int64_t> offsets) {
    VisionAttentionLayout layout;
    layout.offsets = std::move(offsets);
    layout.num_seqs = static_cast<int>(layout.offsets.size()) - 1;
    layout.single_sequence = (layout.num_seqs == 1);
    layout.uniform_blocks = (layout.num_seqs > 1);

    if (layout.num_seqs <= 0) {
        return layout;
    }

    layout.block_len = layout.offsets[1] - layout.offsets[0];
    if (layout.block_len <= 0) {
        layout.uniform_blocks = false;
        return layout;
    }

    for (int i = 1; i < layout.num_seqs; i++) {
        if ((layout.offsets[i + 1] - layout.offsets[i]) != layout.block_len) {
            layout.uniform_blocks = false;
            break;
        }
    }

    return layout;
}

static VisionAttentionLayout analyze_attention_layout(const torch::Tensor& cu_seqlens) {
    auto cu_cpu = cu_seqlens.to(torch::kCPU, torch::kInt64).contiguous();
    auto* cu_ptr = cu_cpu.data_ptr<int64_t>();
    return make_attention_layout(std::vector<int64_t>(cu_ptr, cu_ptr + cu_cpu.numel()));
}

static VisionAttentionLayout build_full_attention_layout(const torch::Tensor& grid_thw) {
    auto grid_cpu = grid_thw.to(torch::kCPU, torch::kInt64).contiguous();
    auto acc = grid_cpu.accessor<int64_t, 2>();

    int64_t total_windows = 0;
    for (int i = 0; i < acc.size(0); i++) {
        total_windows += acc[i][0];
    }

    std::vector<int64_t> offsets;
    offsets.reserve(total_windows + 1);
    offsets.push_back(0);

    int64_t total = 0;
    for (int i = 0; i < acc.size(0); i++) {
        int64_t block_len = acc[i][1] * acc[i][2];
        for (int64_t t = 0; t < acc[i][0]; t++) {
            total += block_len;
            offsets.push_back(total);
        }
    }

    return make_attention_layout(std::move(offsets));
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
                try_get(name + ".bias"),
                try_get(name + ".weight_orig_shape"));
    };

    load_proj(gate_proj_, "mlp.gate_proj");
    load_proj(up_proj_, "mlp.up_proj");
    load_proj(down_proj_, "mlp.down_proj");
}

torch::Tensor VisionMLP::forward(const torch::Tensor& x) {
    auto gate = gate_proj_.forward(x);
    auto up = up_proj_.forward(x);
    if (std::getenv("WALLX_DISABLE_VISION_FUSED_OPS") != nullptr) {
        return down_proj_.forward(torch::silu(gate) * up);
    }
    auto hidden = fused_act::silu_mul(gate, up);
    return down_proj_.forward(hidden);
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
              try_get("attn.qkv.bias"),
              try_get("attn.qkv.weight_orig_shape"));
    proj_.load(get("attn.proj.weight"),
               try_get("attn.proj.weight_scale"),
               try_get("attn.proj.bias"),
               try_get("attn.proj.weight_orig_shape"));
}

torch::Tensor VisionAttention::forward(const torch::Tensor& x,
                                        const VisionAttentionLayout& layout,
                                        const torch::Tensor& cos,
                                        const torch::Tensor& sin) {
    int seq_length = x.size(0);
    bool disable_attn_fastpath = std::getenv("WALLX_DISABLE_VISION_ATTN_FASTPATH") != nullptr;

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

    // Fast path 1: single sequence -> no mask needed
    if (!disable_attn_fastpath && layout.single_sequence) {
        q_rot = q_rot.transpose(0, 1);  // [num_heads, seq, head_dim]
        k_rot = k_rot.transpose(0, 1);
        v = v.transpose(0, 1);

        auto attn_output = torch::scaled_dot_product_attention(
            q_rot, k_rot, v, /*attn_mask=*/{}, /*dropout_p=*/0.0);

        attn_output = attn_output.transpose(0, 1).reshape({seq_length, -1});
        return proj_.forward(attn_output);
    }

    // Fast path 2: equal-sized blocks -> reshape into batch dimension, no mask.
    if (!disable_attn_fastpath && layout.uniform_blocks) {
        auto q_batched = q_rot.reshape({layout.num_seqs, layout.block_len, num_heads_, head_dim_})
                            .permute({0, 2, 1, 3});
        auto k_batched = k_rot.reshape({layout.num_seqs, layout.block_len, num_heads_, head_dim_})
                            .permute({0, 2, 1, 3});
        auto v_batched = v.reshape({layout.num_seqs, layout.block_len, num_heads_, head_dim_})
                            .permute({0, 2, 1, 3});

        auto attn_output = torch::scaled_dot_product_attention(
            q_batched, k_batched, v_batched,
            /*attn_mask=*/{}, /*dropout_p=*/0.0);

        attn_output = attn_output.permute({0, 2, 1, 3}).reshape({seq_length, -1});
        return proj_.forward(attn_output);
    }

    // Fallback: build explicit block-diagonal mask.
    auto attn_mask = torch::zeros({1, seq_length, seq_length},
                                   torch::TensorOptions().device(q.device()).dtype(torch::kBool));
    for (int i = 0; i < layout.num_seqs; i++) {
        int64_t start = layout.offsets[i];
        int64_t end = layout.offsets[i + 1];
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

VisionAttentionDebug VisionAttention::forward_debug(const torch::Tensor& x,
                                                    const VisionAttentionLayout& layout,
                                                    const torch::Tensor& cos,
                                                    const torch::Tensor& sin) {
    VisionAttentionDebug dbg;
    int seq_length = x.size(0);

    auto qkv = qkv_.forward(x);
    qkv = qkv.reshape({seq_length, 3, num_heads_, head_dim_});
    qkv = qkv.permute({1, 0, 2, 3});
    dbg.q = qkv[0];
    dbg.k = qkv[1];
    dbg.v = qkv[2];

    auto qk_rot = apply_rotary_pos_emb_vision(dbg.q, dbg.k, cos, sin);
    dbg.q_rot = qk_rot.first;
    dbg.k_rot = qk_rot.second;

    torch::Tensor attn_output;
    if (layout.single_sequence) {
        auto q_rot = dbg.q_rot.transpose(0, 1);
        auto k_rot = dbg.k_rot.transpose(0, 1);
        auto v = dbg.v.transpose(0, 1);
        attn_output = torch::scaled_dot_product_attention(
            q_rot, k_rot, v, /*attn_mask=*/{}, /*dropout_p=*/0.0);
        attn_output = attn_output.transpose(0, 1).reshape({seq_length, -1});
    } else if (layout.uniform_blocks) {
        auto q_batched = dbg.q_rot.reshape({layout.num_seqs, layout.block_len, num_heads_, head_dim_})
                            .permute({0, 2, 1, 3});
        auto k_batched = dbg.k_rot.reshape({layout.num_seqs, layout.block_len, num_heads_, head_dim_})
                            .permute({0, 2, 1, 3});
        auto v_batched = dbg.v.reshape({layout.num_seqs, layout.block_len, num_heads_, head_dim_})
                            .permute({0, 2, 1, 3});
        attn_output = torch::scaled_dot_product_attention(
            q_batched, k_batched, v_batched,
            /*attn_mask=*/{}, /*dropout_p=*/0.0);
        attn_output = attn_output.permute({0, 2, 1, 3}).reshape({seq_length, -1});
    } else {
        auto attn_mask = torch::zeros({1, seq_length, seq_length},
                                       torch::TensorOptions().device(x.device()).dtype(torch::kBool));
        for (int i = 0; i < layout.num_seqs; i++) {
            int64_t start = layout.offsets[i];
            int64_t end = layout.offsets[i + 1];
            attn_mask.index({torch::indexing::Slice(),
                             torch::indexing::Slice(start, end),
                             torch::indexing::Slice(start, end)}) = true;
        }
        auto q_rot = dbg.q_rot.transpose(0, 1);
        auto k_rot = dbg.k_rot.transpose(0, 1);
        auto v = dbg.v.transpose(0, 1);
        attn_output = torch::scaled_dot_product_attention(
            q_rot, k_rot, v, attn_mask, /*dropout_p=*/0.0);
        attn_output = attn_output.transpose(0, 1).reshape({seq_length, -1});
    }

    dbg.output = proj_.forward(attn_output);
    return dbg;
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
                                    const VisionAttentionLayout& layout,
                                    const torch::Tensor& cos,
                                    const torch::Tensor& sin) {
    // Attention with residual
    auto normed = vision_rms_norm(x, norm1_weight_, eps_);
    auto attn_out = attn_.forward(normed, layout, cos, sin);
    torch::Tensor h;
    if (std::getenv("WALLX_DISABLE_VISION_FUSED_OPS") == nullptr &&
        x.is_cuda() &&
        x.dtype() == torch::kBFloat16 &&
        x.is_contiguous() &&
        attn_out.is_contiguous() &&
        norm2_weight_.defined() &&
        norm2_weight_.dtype() == torch::kBFloat16 &&
        norm2_weight_.is_contiguous() &&
        norm2_weight_.numel() == x.size(-1) &&
        x.size(-1) <= 2048) {
        h = x.clone();
        normed = fused_norm::fused_add_rms_norm(h, attn_out, norm2_weight_, eps_);
    } else {
        h = x + attn_out;
        normed = vision_rms_norm(h, norm2_weight_, eps_);
    }

    // MLP with residual
    auto mlp_out = mlp_.forward(normed);
    return h + mlp_out;
}

VisionBlockDebug VisionBlock::forward_debug(const torch::Tensor& x,
                                            const VisionAttentionLayout& layout,
                                            const torch::Tensor& cos,
                                            const torch::Tensor& sin) {
    VisionBlockDebug dbg;
    dbg.norm1_out = vision_rms_norm(x, norm1_weight_, eps_);
    auto attn_dbg = attn_.forward_debug(dbg.norm1_out, layout, cos, sin);
    dbg.q = attn_dbg.q;
    dbg.k = attn_dbg.k;
    dbg.v = attn_dbg.v;
    dbg.q_rot = attn_dbg.q_rot;
    dbg.k_rot = attn_dbg.k_rot;
    dbg.attn_out = attn_dbg.output;

    if (x.is_cuda() &&
        x.dtype() == torch::kBFloat16 &&
        x.is_contiguous() &&
        dbg.attn_out.is_contiguous() &&
        norm2_weight_.defined() &&
        norm2_weight_.dtype() == torch::kBFloat16 &&
        norm2_weight_.is_contiguous() &&
        norm2_weight_.numel() == x.size(-1) &&
        x.size(-1) <= 2048) {
        dbg.after_attn = x.clone();
        dbg.norm2_out = fused_norm::fused_add_rms_norm(
            dbg.after_attn, dbg.attn_out, norm2_weight_, eps_);
    } else {
        dbg.after_attn = x + dbg.attn_out;
        dbg.norm2_out = vision_rms_norm(dbg.after_attn, norm2_weight_, eps_);
    }

    dbg.mlp_out = mlp_.forward(dbg.norm2_out);
    dbg.output = dbg.after_attn + dbg.mlp_out;
    return dbg;
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
                try_get("merger.mlp.0.bias"),
                try_get("merger.mlp.0.weight_orig_shape"));
    mlp_2_.load(get("merger.mlp.2.weight"),
                try_get("merger.mlp.2.weight_scale"),
                try_get("merger.mlp.2.bias"),
                try_get("merger.mlp.2.weight_orig_shape"));
}

torch::Tensor PatchMerger::forward(const torch::Tensor& x) {
    // x: [total_tokens, dim]
    // Apply layer norm, reshape for spatial merge, then MLP
    auto normed = vision_rms_norm(x, ln_q_weight_, eps_);
    auto merged = normed.view({-1, hidden_size_});
    auto h = mlp_0_.forward(merged);
    h = torch::gelu(h);
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
        // Match Python vision path:
        //   Qwen2_5_VisionRotaryEmbedding(dim=head_dim // 2)
        //   inv_freq = 1 / (10000 ** (arange(0, dim, 2) / dim))
        auto arange = torch::arange(0, head_dim / 2, 2, torch::kFloat32);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> VisionEncoder::forward_debug(
    const torch::Tensor& pixel_values,
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
    auto after_reorder = hidden_states;

    rotary_pos_emb = rotary_pos_emb.reshape({seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    rotary_pos_emb = rotary_pos_emb.index_select(0, window_index);
    rotary_pos_emb = rotary_pos_emb.reshape({seq_len, -1});

    // Match Python vision path:
    //   emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    //   position_embeddings = (emb.cos(), emb.sin())
    auto emb = torch::cat({rotary_pos_emb, rotary_pos_emb}, -1);
    auto cos = emb.cos();
    auto sin = emb.sin();

    // --- 5. Attention layouts ---
    auto full_layout = build_full_attention_layout(grid_thw);
    auto window_layout = analyze_attention_layout(cu_window_seqlens);
    torch::Tensor after_block0;
    torch::Tensor after_block7;

    // --- 6. Process through vision blocks ---
    for (int i = 0; i < (int)blocks_.size(); i++) {
        bool is_fullatt = std::find(fullatt_block_indexes_.begin(),
                                     fullatt_block_indexes_.end(), i) != fullatt_block_indexes_.end();
        auto& layout = is_fullatt ? full_layout : window_layout;
        hidden_states = blocks_[i].forward(hidden_states, layout, cos, sin);
        if (i == 0) {
            after_block0 = hidden_states;
        }
        if (i == 7) {
            after_block7 = hidden_states;
        }
    }

    auto pre_merger = hidden_states;

    // --- 7. Patch merger ---
    hidden_states = merger_.forward(hidden_states);

    // --- 8. Reverse window reordering ---
    auto reverse_indices = torch::argsort(window_index);
    hidden_states = hidden_states.index_select(0, reverse_indices);

    return {after_reorder, after_block0, after_block7, pre_merger, hidden_states};
}

torch::Tensor VisionEncoder::forward(const torch::Tensor& pixel_values,
                                      const torch::Tensor& grid_thw) {
    return std::get<4>(forward_debug(pixel_values, grid_thw));
}

VisionBlockDebug VisionEncoder::run_block_debug(
    const torch::Tensor& hidden_states_reordered,
    const torch::Tensor& grid_thw,
    int block_idx) {
    TORCH_CHECK(block_idx >= 0 && block_idx < (int)blocks_.size(),
                "invalid vision block index: ", block_idx);

    auto rotary_pos_emb = fused_rot_pos_emb_cuda(rotary_inv_freq_, grid_thw, spatial_merge_size_);
    int vit_merger_window_size = window_size_ / spatial_merge_size_ / patch_size_;
    auto [window_index, cu_window_seqlens] = get_window_index_cuda(
        grid_thw.to(torch::kInt32), spatial_merge_size_,
        vit_merger_window_size, patch_size_, spatial_merge_unit_);

    int seq_len = hidden_states_reordered.size(0);
    rotary_pos_emb = rotary_pos_emb.reshape({seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    rotary_pos_emb = rotary_pos_emb.index_select(0, window_index);
    rotary_pos_emb = rotary_pos_emb.reshape({seq_len, -1});

    auto emb = torch::cat({rotary_pos_emb, rotary_pos_emb}, -1);
    auto cos = emb.cos();
    auto sin = emb.sin();

    auto full_layout = build_full_attention_layout(grid_thw);
    auto window_layout = analyze_attention_layout(cu_window_seqlens);
    bool is_fullatt = std::find(fullatt_block_indexes_.begin(),
                                fullatt_block_indexes_.end(), block_idx) != fullatt_block_indexes_.end();
    auto& layout = is_fullatt ? full_layout : window_layout;

    return blocks_[block_idx].forward_debug(hidden_states_reordered, layout, cos, sin);
}
