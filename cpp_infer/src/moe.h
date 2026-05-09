#pragma once
#include <torch/torch.h>
#include "int8_linear.h"
#include "triton_loader.h"
#include "utils.h"

// Forward declarations for CUDA ops from csrc/
extern std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> moe_permute_topK_op(
    torch::Tensor input, torch::Tensor indices, int64_t num_out_tokens,
    std::vector<torch::Tensor> workspace, int64_t max_expanded_token_num);

extern torch::Tensor moe_recover_topK_op(
    torch::Tensor input, torch::Tensor row_id_map, torch::Tensor prob_opt,
    int64_t num_tokens, int64_t num_topK);

extern void AsymmetricDualExpertGemm(
    torch::Tensor input_expert0, torch::Tensor input_expert1,
    torch::Tensor weight_expert0, torch::Tensor weight_expert1,
    torch::Tensor output_expert0, torch::Tensor output_expert1,
    bool trans_a, bool trans_b);

// Token type information for MoE routing
struct TokenTypeInfo {
    torch::Tensor token_types;  // [batch * seq_len] int tensor: 0=text/vision, 1=action
    int num_tokens_per_expert[2] = {0, 0};
    bool grouped_by_expert = false;
};

// MoE block: 2 experts with asymmetric intermediate sizes
// Expert 0: intermediate=11008 (standard LLM)
// Expert 1: intermediate=2048 (action tokens)
class MoEBlock {
public:
    MoEBlock() = default;
    void init(const ModelConfig& config, int layer_idx);
    void load_weights(const WeightMap& weights, const std::string& prefix);

    // Forward: route tokens to experts, compute FFN, merge back
    // x: [batch, seq_len, hidden_size]
    torch::Tensor forward(const torch::Tensor& x,
                          const TokenTypeInfo& token_types,
                          TritonKernelRegistry& triton);

private:
    // Expert 0 (standard): gate_proj, up_proj, down_proj
    LinearOp gate_proj_0_, up_proj_0_, down_proj_0_;
    // Expert 1 (action): gate_proj, up_proj, down_proj
    LinearOp gate_proj_1_, up_proj_1_, down_proj_1_;

    int hidden_size_ = 0;
    int intermediate_0_ = 0;  // 11008
    int intermediate_1_ = 0;  // 2048
    int layer_idx_ = 0;
    bool use_dual_gemm_ = true;
};

// Simple MLP (non-MoE) for layers that don't use MoE
class MLP {
public:
    void init(int hidden_size, int intermediate_size);
    void load_weights(const WeightMap& weights, const std::string& prefix);
    torch::Tensor forward(const torch::Tensor& x, TritonKernelRegistry& triton);

private:
    LinearOp gate_proj_, up_proj_, down_proj_;
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
};
