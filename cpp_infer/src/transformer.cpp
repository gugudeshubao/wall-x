#include "transformer.h"
#include "kernels/norm_kernels.h"
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>

static torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight, float eps) {
    // Compute in float32 for numerical stability
    auto x_f32 = x.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, /*keepdim=*/true);
    auto normed = x_f32 * torch::rsqrt(variance + eps);
    return (weight * normed).to(x.dtype());
}

static bool can_use_triton_decoder_norm(const torch::Tensor& x,
                                        const torch::Tensor& weight,
                                        int hidden_size) {
    return x.is_cuda()
        && x.dtype() == torch::kBFloat16
        && x.size(-1) == hidden_size
        && weight.defined()
        && weight.dtype() == torch::kBFloat16
        && weight.numel() == hidden_size
        && x.is_contiguous();
}

static torch::Tensor triton_rms_norm(const torch::Tensor& x,
                                     const torch::Tensor& weight,
                                     int hidden_size,
                                     TritonKernelRegistry& triton) {
    auto out = torch::empty_like(x);
    int M = x.numel() / hidden_size;
    auto stream = reinterpret_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());
    triton.rmsnorm(
        x.data_ptr(), weight.data_ptr(), out.data_ptr(), M, stream);
    return out;
}

static torch::Tensor triton_fused_add_rms_norm(torch::Tensor residual,
                                               const torch::Tensor& x,
                                               const torch::Tensor& weight,
                                               int hidden_size,
                                               TritonKernelRegistry& triton) {
    auto out = torch::empty_like(residual);
    int M = residual.numel() / hidden_size;
    auto stream = reinterpret_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());
    triton.fused_add_rmsnorm(
        const_cast<void*>(x.data_ptr()),
        residual.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        M,
        stream);
    return out;
}

static bool can_use_cuda_fused_norm(const torch::Tensor& x,
                                    const torch::Tensor& weight,
                                    int hidden_size) {
    return x.is_cuda()
        && x.dtype() == torch::kBFloat16
        && x.is_contiguous()
        && x.size(-1) == hidden_size
        && weight.defined()
        && weight.is_cuda()
        && weight.dtype() == torch::kBFloat16
        && weight.is_contiguous()
        && weight.numel() == hidden_size
        && hidden_size == 2048;
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
    bool disable_fused_norm = std::getenv("WALLX_DISABLE_DECODER_FUSED_NORM") != nullptr;
    // --- Pre-attention RMSNorm ---
    torch::Tensor normed;
    if (!disable_fused_norm &&
        can_use_cuda_fused_norm(hidden_states, input_layernorm_weight_, hidden_size_)) {
        normed = fused_norm::rms_norm(hidden_states, input_layernorm_weight_, rms_norm_eps_);
    } else if (!disable_fused_norm &&
        hidden_size_ == 2048 &&
        triton.has("rmsnorm_h2048") &&
        can_use_triton_decoder_norm(hidden_states, input_layernorm_weight_, hidden_size_)) {
        normed = triton_rms_norm(hidden_states, input_layernorm_weight_, hidden_size_, triton);
    } else {
        normed = rms_norm(hidden_states, input_layernorm_weight_, rms_norm_eps_);
    }

    // --- Self-Attention ---
    auto attn_output = attn_.forward(normed, cos, sin, cache, layer_idx_, is_causal);

    // --- Post-attention RMSNorm ---
    if (!disable_fused_norm &&
        can_use_cuda_fused_norm(hidden_states, post_attention_layernorm_weight_, hidden_size_) &&
        attn_output.is_contiguous()) {
        normed = fused_norm::fused_add_rms_norm(
            hidden_states, attn_output, post_attention_layernorm_weight_, rms_norm_eps_);
    } else if (!disable_fused_norm &&
        hidden_size_ == 2048 &&
        triton.has("fused_add_rmsnorm_h2048") &&
        can_use_triton_decoder_norm(hidden_states, post_attention_layernorm_weight_, hidden_size_) &&
        attn_output.is_contiguous()) {
        normed = triton_fused_add_rms_norm(
            hidden_states, attn_output, post_attention_layernorm_weight_, hidden_size_, triton);
    } else {
        // --- Residual add (in-place: avoids tensor allocation) ---
        hidden_states.add_(attn_output);
        normed = rms_norm(hidden_states, post_attention_layernorm_weight_, rms_norm_eps_);
    }

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
