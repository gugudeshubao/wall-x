#pragma once
#include <torch/torch.h>
#include "utils.h"

// Sinusoidal position embedding for diffusion timestep
class SinusoidalPosEmb {
public:
    SinusoidalPosEmb(int dim) : dim_(dim) {}
    // x: [batch] float timestep -> [batch, dim]
    torch::Tensor forward(const torch::Tensor& x);
private:
    int dim_;
};

// Conv1D block: Conv1d + GroupNorm + Mish
class Conv1dBlock {
public:
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::Tensor conv_weight_, conv_bias_;
    torch::Tensor norm_weight_, norm_bias_;
    int n_groups_ = 8;
};

// Conditional residual block with FiLM modulation
class ConditionalResidualBlock1D {
public:
    void init(int in_channels, int out_channels);
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& cond);

private:
    Conv1dBlock block0_, block1_;
    // FiLM: Mish -> Linear -> Dropout -> Unflatten
    torch::Tensor cond_linear_weight_, cond_linear_bias_;
    // Residual conv: Conv1d(in, out, 1) or identity
    torch::Tensor residual_conv_weight_, residual_conv_bias_;
    int out_channels_ = 0;
    bool need_residual_conv_ = false;
};

// Downsample: Conv1d(dim, dim, 3, stride=2, padding=1)
class Downsample1d {
public:
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x);
private:
    torch::Tensor weight_, bias_;
};

// Upsample: ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
class Upsample1d {
public:
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x);
private:
    torch::Tensor weight_, bias_;
};

// ConditionalUnet1D: diffusion noise prediction / flow velocity network
class ConditionalUnet1D {
public:
    ConditionalUnet1D() = default;
    void init(int input_dim, int global_cond_dim,
              int dsed = 256, std::vector<int> down_dims = {256, 512, 1024});
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // sample: [B, T, input_dim], timestep: [B], global_cond: [B, cond_dim]
    // Returns: [B, T, input_dim]
    torch::Tensor forward(const torch::Tensor& sample,
                          const torch::Tensor& timestep,
                          const torch::Tensor& global_cond);

private:
    // Diffusion step encoder: SinPosEmb -> Linear -> Mish -> Linear
    torch::Tensor dsenc_linear1_weight_, dsenc_linear1_bias_;
    torch::Tensor dsenc_linear2_weight_, dsenc_linear2_bias_;
    int dsed_ = 256;

    // Down modules: each has [resnet1, resnet2, downsample]
    struct DownBlock {
        ConditionalResidualBlock1D resnet1, resnet2;
        Downsample1d downsample;
        bool has_downsample = true;
    };
    std::vector<DownBlock> down_modules_;

    // Mid modules: 2x ConditionalResidualBlock1D
    ConditionalResidualBlock1D mid_module_0_, mid_module_1_;

    // Up modules: each has [resnet1, resnet2, upsample]
    struct UpBlock {
        ConditionalResidualBlock1D resnet1, resnet2;
        Upsample1d upsample;
        bool has_upsample = true;
    };
    std::vector<UpBlock> up_modules_;

    // Final conv: Conv1dBlock + Conv1d
    Conv1dBlock final_conv_block_;
    torch::Tensor final_conv_weight_, final_conv_bias_;

    SinusoidalPosEmb sinusoidal_emb_{256};
};

// ActionProcessor: flow-matching action embedding + velocity prediction
class ActionProcessor {
public:
    ActionProcessor() = default;
    void init(const ModelConfig& config);
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // step: embed noisy action at given timestep for decoder input
    // timestep: [batch], noisy_action: [batch, horizon, action_dim]
    // dof_mask: [batch, 1, action_dim] binary mask for active DOFs
    // Returns: action_embed [batch, horizon, hidden_size]
    torch::Tensor step(const torch::Tensor& timestep,
                       const torch::Tensor& noisy_action,
                       const torch::Tensor& dof_mask);

    // action_proj_back: project hidden states back to action space
    // hidden: [N, action_hidden_size] -> [N, action_dim]
    torch::Tensor action_proj_back(const torch::Tensor& hidden);

    // Unnormalize predicted action
    torch::Tensor unnormalize(const torch::Tensor& action,
                              const std::string& dataset_name);

    int action_hidden_size() const { return action_hidden_size_; }
    int action_dim() const { return action_dim_; }

private:
    // Weights
    torch::Tensor w1_weight_;           // [action_dim -> action_hidden_size]
    torch::Tensor w2_weight_;           // [action_hidden_size*2 -> action_hidden_size]
    torch::Tensor w2_action_weight_;    // first half of w2 input: action branch
    torch::Tensor w2_time_weight_;      // second half of w2 input: time embedding branch
    torch::Tensor w3_weight_;           // [action_hidden_size -> action_hidden_size]
    torch::Tensor proj_back_weight_;    // [action_hidden_size -> action_dim]

    // Normalizer stats (per dataset)
    std::unordered_map<std::string, torch::Tensor> norm_min_map_;
    std::unordered_map<std::string, torch::Tensor> norm_delta_map_;

    SinusoidalPosEmb time_embed_{256};

    int action_dim_ = 20;
    int action_hidden_size_ = 2048;
    int hidden_size_ = 2048;
    bool use_adarms_ = false;
};
