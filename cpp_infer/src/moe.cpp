#include "moe.h"

void MoEBlock::init(const ModelConfig& config, int layer_idx) {
    hidden_size_ = config.hidden_size;
    intermediate_0_ = config.expert_intermediate_sizes[0];
    intermediate_1_ = config.expert_intermediate_sizes[1];
    layer_idx_ = layer_idx;
}

void MoEBlock::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) {
            throw std::runtime_error("Weight not found: " + prefix + name);
        }
        return it->second;
    };

    // Expert 0 weights
    gate_proj_0_ = get("moe.experts.0.gate_proj.weight");
    up_proj_0_ = get("moe.experts.0.up_proj.weight");
    down_proj_0_ = get("moe.experts.0.down_proj.weight");

    // Expert 1 weights
    gate_proj_1_ = get("moe.experts.1.gate_proj.weight");
    up_proj_1_ = get("moe.experts.1.up_proj.weight");
    down_proj_1_ = get("moe.experts.1.down_proj.weight");
}

torch::Tensor MoEBlock::forward(const torch::Tensor& x,
                                 const TokenTypeInfo& token_types,
                                 TritonKernelRegistry& triton) {
    auto batch = x.size(0);
    auto seq_len = x.size(1);
    auto hidden = x.size(2);

    // Flatten: [batch, seq, hidden] -> [total_tokens, hidden]
    auto flat_x = x.reshape({-1, hidden});
    int total_tokens = flat_x.size(0);

    // TokenTypeRouter: route by token_type % num_experts
    // token_type 0 -> expert 0 (text/vision)
    // token_type 1 -> expert 1 (action)
    // indices must be [total_tokens, 1] (2D, int32) for moe_permute_topK_op
    auto indices = token_types.token_types.remainder(2).to(torch::kInt32).unsqueeze(1);

    // Permute tokens by expert assignment
    std::vector<torch::Tensor> workspace;
    auto [permuted, row_id_map, ws] = moe_permute_topK_op(
        flat_x, indices, total_tokens, workspace, total_tokens);

    // Split permuted tokens by expert
    int n0 = token_types.num_tokens_per_expert[0];
    int n1 = token_types.num_tokens_per_expert[1];

    if (n0 > 0 && n1 > 0) {
        auto input_0 = permuted.slice(0, 0, n0);
        auto input_1 = permuted.slice(0, n0, n0 + n1);

        // Gate projections via dual asymmetric GEMM
        auto gate_out_0 = torch::empty({n0, intermediate_0_}, flat_x.options());
        auto gate_out_1 = torch::empty({n1, intermediate_1_}, flat_x.options());
        AsymmetricDualExpertGemm(input_0, input_1, gate_proj_0_, gate_proj_1_,
                                  gate_out_0, gate_out_1, false, true);

        // Up projections via dual asymmetric GEMM
        auto up_out_0 = torch::empty({n0, intermediate_0_}, flat_x.options());
        auto up_out_1 = torch::empty({n1, intermediate_1_}, flat_x.options());
        AsymmetricDualExpertGemm(input_0, input_1, up_proj_0_, up_proj_1_,
                                  up_out_0, up_out_1, false, true);

        // Fused SiLU * mul (in-place to avoid allocation)
        if (triton.has("fused_silu_mul_n11008")) {
            auto hidden_0 = torch::empty_like(gate_out_0);
            triton.fused_silu_mul(gate_out_0.data_ptr(), up_out_0.data_ptr(),
                                   hidden_0.data_ptr(), n0, intermediate_0_);
            gate_out_0 = hidden_0;
        } else {
            torch::silu_(gate_out_0).mul_(up_out_0);
        }

        if (triton.has("fused_silu_mul_n2048")) {
            auto hidden_1 = torch::empty_like(gate_out_1);
            triton.fused_silu_mul(gate_out_1.data_ptr(), up_out_1.data_ptr(),
                                   hidden_1.data_ptr(), n1, intermediate_1_);
            gate_out_1 = hidden_1;
        } else {
            torch::silu_(gate_out_1).mul_(up_out_1);
        }

        // Down projections via dual asymmetric GEMM
        auto out_0 = torch::empty({n0, hidden_size_}, flat_x.options());
        auto out_1 = torch::empty({n1, hidden_size_}, flat_x.options());
        AsymmetricDualExpertGemm(gate_out_0, gate_out_1, down_proj_0_, down_proj_1_,
                                  out_0, out_1, false, true);

        // Concatenate expert outputs
        auto expert_out = torch::cat({out_0, out_1}, 0);

        // Unpermute: restore original token order
        auto prob = torch::ones({total_tokens, 1}, flat_x.options());
        auto output = moe_recover_topK_op(expert_out, row_id_map, prob, total_tokens, 1);

        return output.reshape({batch, seq_len, hidden});
    } else if (n0 > 0) {
        // Only expert 0
        auto gate_out = torch::linear(permuted, gate_proj_0_);
        auto up_out = torch::linear(permuted, up_proj_0_);
        torch::silu_(gate_out).mul_(up_out);
        auto output = torch::linear(gate_out, down_proj_0_);
        auto prob = torch::ones({total_tokens, 1}, flat_x.options());
        output = moe_recover_topK_op(output, row_id_map, prob, total_tokens, 1);
        return output.reshape({batch, seq_len, hidden});
    } else {
        // Only expert 1
        auto gate_out = torch::linear(permuted, gate_proj_1_);
        auto up_out = torch::linear(permuted, up_proj_1_);
        torch::silu_(gate_out).mul_(up_out);
        auto output = torch::linear(gate_out, down_proj_1_);
        auto prob = torch::ones({total_tokens, 1}, flat_x.options());
        output = moe_recover_topK_op(output, row_id_map, prob, total_tokens, 1);
        return output.reshape({batch, seq_len, hidden});
    }
}

// --- Simple MLP ---

void MLP::init(int hidden_size, int intermediate_size) {
    hidden_size_ = hidden_size;
    intermediate_size_ = intermediate_size;
}

void MLP::load_weights(const WeightMap& weights, const std::string& prefix) {
    auto get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        if (it == weights.end()) {
            throw std::runtime_error("Weight not found: " + prefix + name);
        }
        return it->second;
    };

    gate_proj_ = get("mlp.gate_proj.weight");
    up_proj_ = get("mlp.up_proj.weight");
    down_proj_ = get("mlp.down_proj.weight");
}

torch::Tensor MLP::forward(const torch::Tensor& x, TritonKernelRegistry& triton) {
    auto gate_out = torch::linear(x, gate_proj_);
    auto up_out = torch::linear(x, up_proj_);

    // In-place SiLU * mul: reuses gate_out buffer
    torch::silu_(gate_out).mul_(up_out);

    return torch::linear(gate_out, down_proj_);
}
