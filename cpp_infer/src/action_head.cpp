#include "action_head.h"
#include <torch/torch.h>
#include <cmath>

// ===================== SinusoidalPosEmb =====================

torch::Tensor SinusoidalPosEmb::forward(const torch::Tensor& x) {
    int half_dim = dim_ / 2;
    float emb_factor = std::log(10000.0f) / (half_dim - 1);
    auto emb = torch::exp(
        torch::arange(half_dim, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)) * (-emb_factor)
    );
    // x: [B], emb: [half_dim] -> outer product: [B, half_dim]
    auto out = x.unsqueeze(1).to(torch::kFloat32) * emb.unsqueeze(0);
    return torch::cat({out.sin(), out.cos()}, -1);
}

// ===================== Conv1dBlock =====================

void Conv1dBlock::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };
    conv_weight_ = get("0.weight");
    conv_bias_ = get("0.bias");
    norm_weight_ = get("1.weight");
    norm_bias_ = get("1.bias");
}

torch::Tensor Conv1dBlock::forward(const torch::Tensor& x) {
    // Conv1d -> GroupNorm -> Mish
    int kernel_size = conv_weight_.size(2);
    int padding = kernel_size / 2;
    auto h = torch::conv1d(x, conv_weight_, conv_bias_, /*stride=*/1, /*padding=*/padding);
    h = torch::group_norm(h, n_groups_, norm_weight_, norm_bias_);
    return torch::mish(h);
}

// ===================== ConditionalResidualBlock1D =====================

void ConditionalResidualBlock1D::init(int in_channels, int out_channels) {
    out_channels_ = out_channels;
    need_residual_conv_ = (in_channels != out_channels);
}

void ConditionalResidualBlock1D::load_weights(const WeightMap& weights, const std::string& prefix) {
    block0_.load_weights(weights, prefix + "blocks.0.");
    block1_.load_weights(weights, prefix + "blocks.1.");

    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };

    // FiLM conditioning: cond_encoder.1 (Linear after Mish)
    cond_linear_weight_ = get("cond_encoder.1.weight");
    cond_linear_bias_ = get("cond_encoder.1.bias");

    // Residual conv (if dimensions differ)
    if (need_residual_conv_) {
        residual_conv_weight_ = get("residual_conv.weight");
        residual_conv_bias_ = get("residual_conv.bias");
    }
}

torch::Tensor ConditionalResidualBlock1D::forward(const torch::Tensor& x, const torch::Tensor& cond) {
    // x: [B, C_in, T], cond: [B, cond_dim]
    auto out = block0_.forward(x);

    // FiLM modulation: Mish(cond) -> Linear -> reshape to scale + bias
    auto embed = torch::mish(cond);
    embed = torch::linear(embed, cond_linear_weight_, cond_linear_bias_);
    // Reshape: [B, 2*out_channels] -> [B, 2, out_channels, 1]
    embed = embed.reshape({embed.size(0), 2, out_channels_, 1});
    auto scale = embed.select(1, 0);  // [B, out_channels, 1]
    auto bias = embed.select(1, 1);   // [B, out_channels, 1]
    out = scale * out + bias;

    out = block1_.forward(out);

    // Residual connection
    torch::Tensor residual;
    if (need_residual_conv_) {
        residual = torch::conv1d(x, residual_conv_weight_, residual_conv_bias_);
    } else {
        residual = x;
    }
    return out + residual;
}

// ===================== Downsample1d =====================

void Downsample1d::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };
    weight_ = get("conv.weight");
    bias_ = get("conv.bias");
}

torch::Tensor Downsample1d::forward(const torch::Tensor& x) {
    return torch::conv1d(x, weight_, bias_, /*stride=*/2, /*padding=*/1);
}

// ===================== Upsample1d =====================

void Upsample1d::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };
    weight_ = get("conv.weight");
    bias_ = get("conv.bias");
}

torch::Tensor Upsample1d::forward(const torch::Tensor& x) {
    return torch::conv_transpose1d(x, weight_, bias_, /*stride=*/2, /*padding=*/1);
}

// ===================== ConditionalUnet1D =====================
// Note: init/load_weights would be implemented based on actual model architecture.
// The wall-x model uses ActionProcessor (flow-matching), not ConditionalUnet1D for inference.
// This is included for completeness but the flow action path through the LLM is the primary one.

void ConditionalUnet1D::init(int input_dim, int global_cond_dim, int dsed, std::vector<int> down_dims) {
    dsed_ = dsed;
    sinusoidal_emb_ = SinusoidalPosEmb(dsed);
    // Architecture would be initialized here based on down_dims
}

void ConditionalUnet1D::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };

    // Diffusion step encoder
    dsenc_linear1_weight_ = get("diffusion_step_encoder.1.weight");
    dsenc_linear1_bias_ = get("diffusion_step_encoder.1.bias");
    dsenc_linear2_weight_ = get("diffusion_step_encoder.3.weight");
    dsenc_linear2_bias_ = get("diffusion_step_encoder.3.bias");

    // Down/mid/up modules would be loaded similarly
    // Omitted for brevity - the primary inference path uses ActionProcessor
}

torch::Tensor ConditionalUnet1D::forward(const torch::Tensor& sample,
                                          const torch::Tensor& timestep,
                                          const torch::Tensor& global_cond) {
    // (B, T, C) -> (B, C, T)
    auto x = sample.permute({0, 2, 1});
    int batch = x.size(0);

    // 1. Encode timestep
    auto timesteps = timestep.expand(batch);
    auto global_feature = sinusoidal_emb_.forward(timesteps);
    global_feature = torch::linear(global_feature, dsenc_linear1_weight_, dsenc_linear1_bias_);
    global_feature = torch::mish(global_feature);
    global_feature = torch::linear(global_feature, dsenc_linear2_weight_, dsenc_linear2_bias_);

    if (global_cond.defined()) {
        global_feature = torch::cat({global_feature, global_cond}, -1);
    }

    // 2. Down path with skip connections
    std::vector<torch::Tensor> h;
    for (auto& block : down_modules_) {
        x = block.resnet1.forward(x, global_feature);
        x = block.resnet2.forward(x, global_feature);
        h.push_back(x);
        if (block.has_downsample) {
            x = block.downsample.forward(x);
        }
    }

    // 3. Mid
    x = mid_module_0_.forward(x, global_feature);
    x = mid_module_1_.forward(x, global_feature);

    // 4. Up path with skip connections
    for (auto& block : up_modules_) {
        auto skip = h.back();
        h.pop_back();
        x = torch::cat({x, skip}, 1);
        x = block.resnet1.forward(x, global_feature);
        x = block.resnet2.forward(x, global_feature);
        if (block.has_upsample) {
            x = block.upsample.forward(x);
        }
    }

    // 5. Final conv
    x = final_conv_block_.forward(x);
    x = torch::conv1d(x, final_conv_weight_, final_conv_bias_);

    // (B, C, T) -> (B, T, C)
    return x.permute({0, 2, 1});
}

// ===================== ActionProcessor =====================

void ActionProcessor::init(const ModelConfig& config) {
    action_dim_ = config.action_dim;
    action_hidden_size_ = config.dim_inputs[1];  // 2048
    hidden_size_ = config.hidden_size;
    time_embed_ = SinusoidalPosEmb(action_hidden_size_);
}

void ActionProcessor::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) throw std::runtime_error("Weight not found: " + prefix + name);
        return it->second;
    };

    w1_weight_ = get("w1.weight");
    w2_weight_ = get("w2.weight");
    w2_action_weight_ = w2_weight_.slice(/*dim=*/1, /*start=*/0, /*end=*/action_hidden_size_).contiguous();
    w2_time_weight_ = w2_weight_.slice(/*dim=*/1, /*start=*/action_hidden_size_,
                                       /*end=*/action_hidden_size_ * 2).contiguous();
    w3_weight_ = get("w3.weight");
    proj_back_weight_ = get("action_proj_back.weight");

    // Load all normalizer stats (keyed by dataset name)
    for (const auto& [key, tensor] : weights) {
        if (key.find(prefix + "normalizer_action.min.") == 0) {
            auto dataset = key.substr((prefix + "normalizer_action.min.").size());
            norm_min_map_[dataset] = tensor;
        } else if (key.find(prefix + "normalizer_action.delta.") == 0) {
            auto dataset = key.substr((prefix + "normalizer_action.delta.").size());
            norm_delta_map_[dataset] = tensor;
        }
    }
    std::cout << "[ActionProcessor] Loaded normalizers for " << norm_min_map_.size() << " datasets" << std::endl;
}

torch::Tensor ActionProcessor::step(const torch::Tensor& timestep,
                                     const torch::Tensor& noisy_action,
                                     const torch::Tensor& dof_mask) {
    // Mirrors Python ActionProcessor.step()
    // noisy_action: [batch, horizon, action_dim]
    // timestep: [batch]
    // dof_mask: [batch, 1, action_dim] or [batch, horizon, action_dim]

    // Concatenate dof_mask with noisy_action: [batch, horizon, action_dim*2]
    auto action_in = noisy_action.to(w1_weight_.dtype());
    auto mask = dof_mask.to(w1_weight_.dtype());
    // 1. Sinusoidal time embedding: [batch] -> [batch, action_hidden_size]
    auto time_emb = time_embed_.forward(timestep.to(torch::kFloat32));

    if (mask.dim() == 3 && mask.size(1) == 1) {
        mask = mask.expand({-1, action_in.size(1), -1});
    }
    auto w1_input = torch::cat({action_in, mask}, -1);

    // 2. Project input through w1: [batch, horizon, action_dim*2] -> [batch, horizon, action_hidden_size]
    auto action_embed = torch::linear(w1_input, w1_weight_);

    if (!use_adarms_) {
        // Avoid repeating the same time branch projection across the full horizon.
        // Original math:
        //   w2(cat(action_embed, time_emb)) =
        //   linear(action_embed, w2_action_weight_) + linear(time_emb, w2_time_weight_)
        auto action_proj = torch::linear(action_embed, w2_action_weight_);
        auto time_proj = torch::linear(time_emb.to(action_embed.dtype()), w2_time_weight_);
        action_proj.add_(time_proj.unsqueeze(1));

        // w3(silu(action_proj)) - cast to float32 for silu then back
        auto embed = torch::linear(
            torch::silu(action_proj.to(torch::kFloat32)).to(action_proj.dtype()),
            w3_weight_);

        // Pad to full hidden_size if action_hidden_size < hidden_size
        if (action_hidden_size_ < hidden_size_) {
            int pad_size = hidden_size_ - action_hidden_size_;
            auto padding = torch::zeros({embed.size(0), embed.size(1), pad_size},
                                         embed.options());
            embed = torch::cat({embed, padding}, -1);
        }

        return embed;
    } else {
        // AdaRMS path (not used in default wall-x 3B config)
        return action_embed;
    }
}

torch::Tensor ActionProcessor::action_proj_back(const torch::Tensor& hidden) {
    // hidden: [N, action_hidden_size] -> [N, action_dim]
    // Both input and weight must have the same dtype for torch::linear
    auto h = hidden.to(torch::kFloat32);
    return torch::linear(h, proj_back_weight_.to(torch::kFloat32));
}

torch::Tensor ActionProcessor::unnormalize(const torch::Tensor& action,
                                            const std::string& dataset_name) {
    // Reverse of normalize: x = (x + 1) / 2 * delta + min
    auto min_it = norm_min_map_.find(dataset_name);
    auto delta_it = norm_delta_map_.find(dataset_name);
    if (min_it == norm_min_map_.end() || delta_it == norm_delta_map_.end()) {
        std::cerr << "[ActionProcessor] Warning: normalizer not found for dataset '"
                  << dataset_name << "', returning raw action" << std::endl;
        return action;
    }

    auto x = (action + 1.0f) / 2.0f;
    auto delta = delta_it->second.to(action.device());
    auto min_val = min_it->second.to(action.device());
    return x * delta + min_val;
}
