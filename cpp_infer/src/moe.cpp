#include "moe.h"
#include "kernels/activation_kernels.h"
#include <cstdlib>

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

    // Expert 0 weights
    load_proj(gate_proj_0_, "moe.experts.0.gate_proj");
    load_proj(up_proj_0_, "moe.experts.0.up_proj");
    load_proj(down_proj_0_, "moe.experts.0.down_proj");

    // Expert 1 weights
    load_proj(gate_proj_1_, "moe.experts.1.gate_proj");
    load_proj(up_proj_1_, "moe.experts.1.up_proj");
    load_proj(down_proj_1_, "moe.experts.1.down_proj");

    use_dual_gemm_ =
        !gate_proj_0_.is_int8() && !up_proj_0_.is_int8() && !down_proj_0_.is_int8() &&
        !gate_proj_1_.is_int8() && !up_proj_1_.is_int8() && !down_proj_1_.is_int8();

    if (!use_dual_gemm_) {
        std::cout << "    [INT8] MoE layer " << layer_idx_
                  << " using per-expert LinearOp path" << std::endl;
    }
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

    // Split permuted tokens by expert
    int n0 = token_types.num_tokens_per_expert[0];
    int n1 = token_types.num_tokens_per_expert[1];
    bool disable_grouped_fastpath = std::getenv("WALLX_DISABLE_MOE_GROUPED_FASTPATH") != nullptr;

    // Fast path: all tokens routed to a single expert. Skip permute/recover entirely.
    if (n0 == total_tokens) {
        auto gate_out = gate_proj_0_.forward(flat_x);
        auto up_out = up_proj_0_.forward(flat_x);
        gate_out = fused_act::silu_mul(gate_out, up_out);
        auto output = down_proj_0_.forward(gate_out);
        return output.reshape({batch, seq_len, hidden});
    }

    if (n1 == total_tokens) {
        auto gate_out = gate_proj_1_.forward(flat_x);
        auto up_out = up_proj_1_.forward(flat_x);
        gate_out = fused_act::silu_mul(gate_out, up_out);
        auto output = down_proj_1_.forward(gate_out);
        return output.reshape({batch, seq_len, hidden});
    }

    if (!disable_grouped_fastpath && token_types.grouped_by_expert && n0 > 0 && n1 > 0) {
        auto input_0 = flat_x.slice(0, 0, n0);
        auto input_1 = flat_x.slice(0, n0, n0 + n1);

        torch::Tensor gate_out_0, gate_out_1, up_out_0, up_out_1;
        if (use_dual_gemm_) {
            gate_out_0 = torch::empty({n0, intermediate_0_}, flat_x.options());
            gate_out_1 = torch::empty({n1, intermediate_1_}, flat_x.options());
            AsymmetricDualExpertGemm(
                input_0, input_1,
                gate_proj_0_.weight_bf16(), gate_proj_1_.weight_bf16(),
                gate_out_0, gate_out_1, false, true);

            up_out_0 = torch::empty({n0, intermediate_0_}, flat_x.options());
            up_out_1 = torch::empty({n1, intermediate_1_}, flat_x.options());
            AsymmetricDualExpertGemm(
                input_0, input_1,
                up_proj_0_.weight_bf16(), up_proj_1_.weight_bf16(),
                up_out_0, up_out_1, false, true);
        } else {
            gate_out_0 = gate_proj_0_.forward(input_0);
            gate_out_1 = gate_proj_1_.forward(input_1);
            up_out_0 = up_proj_0_.forward(input_0);
            up_out_1 = up_proj_1_.forward(input_1);
        }

        if (gate_out_0.is_cuda() && gate_out_0.dtype() == torch::kBFloat16 &&
            gate_out_0.is_contiguous() && up_out_0.is_contiguous()) {
            gate_out_0 = fused_act::silu_mul(gate_out_0, up_out_0);
        } else if (triton.has("fused_silu_mul_n11008")) {
            auto hidden_0 = torch::empty_like(gate_out_0);
            triton.fused_silu_mul(gate_out_0.data_ptr(), up_out_0.data_ptr(),
                                   hidden_0.data_ptr(), n0, intermediate_0_);
            gate_out_0 = hidden_0;
        } else {
            torch::silu_(gate_out_0).mul_(up_out_0);
        }

        if (gate_out_1.is_cuda() && gate_out_1.dtype() == torch::kBFloat16 &&
            gate_out_1.is_contiguous() && up_out_1.is_contiguous()) {
            gate_out_1 = fused_act::silu_mul(gate_out_1, up_out_1);
        } else if (triton.has("fused_silu_mul_n2048")) {
            auto hidden_1 = torch::empty_like(gate_out_1);
            triton.fused_silu_mul(gate_out_1.data_ptr(), up_out_1.data_ptr(),
                                   hidden_1.data_ptr(), n1, intermediate_1_);
            gate_out_1 = hidden_1;
        } else {
            torch::silu_(gate_out_1).mul_(up_out_1);
        }

        torch::Tensor out_0, out_1;
        if (use_dual_gemm_) {
            out_0 = torch::empty({n0, hidden_size_}, flat_x.options());
            out_1 = torch::empty({n1, hidden_size_}, flat_x.options());
            AsymmetricDualExpertGemm(
                gate_out_0, gate_out_1,
                down_proj_0_.weight_bf16(), down_proj_1_.weight_bf16(),
                out_0, out_1, false, true);
        } else {
            out_0 = down_proj_0_.forward(gate_out_0);
            out_1 = down_proj_1_.forward(gate_out_1);
        }

        return torch::cat({out_0, out_1}, 0).reshape({batch, seq_len, hidden});
    }

    // TokenTypeRouter: route by token_type % num_experts
    // token_type 0 -> expert 0 (text/vision)
    // token_type 1 -> expert 1 (action)
    // indices must be [total_tokens, 1] (2D, int32) for moe_permute_topK_op
    auto indices = token_types.token_types.remainder(2).to(torch::kInt32).unsqueeze(1);

    // Permute tokens by expert assignment
    std::vector<torch::Tensor> workspace;
    auto [permuted, row_id_map, ws] = moe_permute_topK_op(
        flat_x, indices, total_tokens, workspace, total_tokens);

    if (n0 > 0 && n1 > 0) {
        auto input_0 = permuted.slice(0, 0, n0);
        auto input_1 = permuted.slice(0, n0, n0 + n1);

        torch::Tensor gate_out_0, gate_out_1, up_out_0, up_out_1;
        if (use_dual_gemm_) {
            // Gate projections via dual asymmetric GEMM
            gate_out_0 = torch::empty({n0, intermediate_0_}, flat_x.options());
            gate_out_1 = torch::empty({n1, intermediate_1_}, flat_x.options());
            AsymmetricDualExpertGemm(
                input_0, input_1,
                gate_proj_0_.weight_bf16(), gate_proj_1_.weight_bf16(),
                gate_out_0, gate_out_1, false, true);

            // Up projections via dual asymmetric GEMM
            up_out_0 = torch::empty({n0, intermediate_0_}, flat_x.options());
            up_out_1 = torch::empty({n1, intermediate_1_}, flat_x.options());
            AsymmetricDualExpertGemm(
                input_0, input_1,
                up_proj_0_.weight_bf16(), up_proj_1_.weight_bf16(),
                up_out_0, up_out_1, false, true);
        } else {
            gate_out_0 = gate_proj_0_.forward(input_0);
            gate_out_1 = gate_proj_1_.forward(input_1);
            up_out_0 = up_proj_0_.forward(input_0);
            up_out_1 = up_proj_1_.forward(input_1);
        }

        // Fused SiLU * mul
        if (gate_out_0.is_cuda() && gate_out_0.dtype() == torch::kBFloat16 &&
            gate_out_0.is_contiguous() && up_out_0.is_contiguous()) {
            gate_out_0 = fused_act::silu_mul(gate_out_0, up_out_0);
        } else if (triton.has("fused_silu_mul_n11008")) {
            auto hidden_0 = torch::empty_like(gate_out_0);
            triton.fused_silu_mul(gate_out_0.data_ptr(), up_out_0.data_ptr(),
                                   hidden_0.data_ptr(), n0, intermediate_0_);
            gate_out_0 = hidden_0;
        } else {
            torch::silu_(gate_out_0).mul_(up_out_0);
        }

        if (gate_out_1.is_cuda() && gate_out_1.dtype() == torch::kBFloat16 &&
            gate_out_1.is_contiguous() && up_out_1.is_contiguous()) {
            gate_out_1 = fused_act::silu_mul(gate_out_1, up_out_1);
        } else if (triton.has("fused_silu_mul_n2048")) {
            auto hidden_1 = torch::empty_like(gate_out_1);
            triton.fused_silu_mul(gate_out_1.data_ptr(), up_out_1.data_ptr(),
                                   hidden_1.data_ptr(), n1, intermediate_1_);
            gate_out_1 = hidden_1;
        } else {
            torch::silu_(gate_out_1).mul_(up_out_1);
        }

        torch::Tensor out_0, out_1;
        if (use_dual_gemm_) {
            // Down projections via dual asymmetric GEMM
            out_0 = torch::empty({n0, hidden_size_}, flat_x.options());
            out_1 = torch::empty({n1, hidden_size_}, flat_x.options());
            AsymmetricDualExpertGemm(
                gate_out_0, gate_out_1,
                down_proj_0_.weight_bf16(), down_proj_1_.weight_bf16(),
                out_0, out_1, false, true);
        } else {
            out_0 = down_proj_0_.forward(gate_out_0);
            out_1 = down_proj_1_.forward(gate_out_1);
        }

        // Concatenate expert outputs
        auto expert_out = torch::cat({out_0, out_1}, 0);

        // Unpermute: restore original token order
        auto prob = torch::ones({total_tokens, 1}, flat_x.options());
        auto output = moe_recover_topK_op(expert_out, row_id_map, prob, total_tokens, 1);

        return output.reshape({batch, seq_len, hidden});
    } else {
        TORCH_CHECK(false, "Unexpected MoE token routing state: n0=", n0, ", n1=", n1,
                    ", total=", total_tokens);
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
    auto try_get = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(prefix + name);
        return (it != weights.end()) ? it->second : torch::Tensor();
    };

    gate_proj_.load(get("mlp.gate_proj.weight"),
                    try_get("mlp.gate_proj.weight_scale"),
                    try_get("mlp.gate_proj.bias"),
                    try_get("mlp.gate_proj.weight_orig_shape"));
    up_proj_.load(get("mlp.up_proj.weight"),
                  try_get("mlp.up_proj.weight_scale"),
                  try_get("mlp.up_proj.bias"),
                  try_get("mlp.up_proj.weight_orig_shape"));
    down_proj_.load(get("mlp.down_proj.weight"),
                    try_get("mlp.down_proj.weight_scale"),
                    try_get("mlp.down_proj.bias"),
                    try_get("mlp.down_proj.weight_orig_shape"));
}

torch::Tensor MLP::forward(const torch::Tensor& x, TritonKernelRegistry& triton) {
    auto gate_out = gate_proj_.forward(x);
    auto up_out = up_proj_.forward(x);

    gate_out = fused_act::silu_mul(gate_out, up_out);

    return down_proj_.forward(gate_out);
}
