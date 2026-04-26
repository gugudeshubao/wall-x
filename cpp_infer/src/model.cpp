#include "model.h"
#include <torch/torch.h>
#include <iostream>
#include <algorithm>

// ===================== RMSNorm helper =====================

torch::Tensor WallXModel::rms_norm(const torch::Tensor& x, const torch::Tensor& weight) {
    auto x_f32 = x.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, true);
    auto normed = x_f32 * torch::rsqrt(variance + config_.rms_norm_eps);
    return (weight * normed).to(x.dtype());
}

// ===================== Rotary Embedding =====================

std::pair<torch::Tensor, torch::Tensor> WallXModel::compute_rotary_emb(
    const torch::Tensor& position_ids, int seq_len) {
    auto inv_freq = rotary_inv_freq_.to(torch::kFloat32);
    int dim = inv_freq.size(0);

    auto pos = position_ids.to(torch::kFloat32);

    auto batch = pos.size(1);

    // Multimodal RoPE: pos is [3, batch, seq] -> permute to [batch, seq, 3]
    // Compute freqs for each position dim separately and concat
    // mrope_section: [16, 24, 24] - each section uses a different pos dim
    auto pos_bsn = pos.permute({1, 2, 0});  // [batch, seq, 3]

    // Build per-section freqs
    std::vector<torch::Tensor> cos_parts, sin_parts;
    int freq_offset = 0;
    for (int section_idx = 0; section_idx < (int)config_.mrope_section.size(); section_idx++) {
        int section_size = config_.mrope_section[section_idx];  // 16, 24, 24
        // Get position for this dimension (temporal, height, width)
        auto pos_dim = pos_bsn.index({"...", section_idx});  // [batch, seq]
        // Get inv_freq slice for this section
        auto freq_slice = inv_freq.slice(0, freq_offset, freq_offset + section_size);  // [section_size]
        // Compute outer product: [batch, seq] x [section_size] -> [batch, seq, section_size]
        auto freqs = torch::einsum("bs,d->bsd", {pos_dim.to(inv_freq.device()), freq_slice});
        cos_parts.push_back(freqs.cos());
        sin_parts.push_back(freqs.sin());
        freq_offset += section_size;
    }

    auto cos = torch::cat(cos_parts, -1);  // [batch, seq, head_dim/2]
    auto sin = torch::cat(sin_parts, -1);  // [batch, seq, head_dim/2]

    // Double for the full head_dim (cos/sin for both halves of rotate_half)
    cos = torch::cat({cos, cos}, -1);  // [batch, seq, head_dim]
    sin = torch::cat({sin, sin}, -1);

    return {cos, sin};
}

// ===================== Init =====================

void WallXModel::init(const ModelConfig& config) {
    config_ = config;

    // Initialize transformer layers
    layers_.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        layers_[i].init(config, i);
    }

    // Initialize vision encoder
    vision_.init(config);

    // Initialize action processor
    action_proc_.init(config);
}

// ===================== Load Weights =====================

void WallXModel::load_weights(const std::string& checkpoint_path) {
    std::cout << "[MODEL] Loading weights from: " << checkpoint_path << std::endl;
    Timer timer;
    timer.start();

    auto weights = load_weights_from_dir(checkpoint_path, torch::kCUDA);
    std::cout << "[MODEL] Loaded " << weights.size() << " tensors in "
              << timer.elapsed_ms() << " ms" << std::endl;

    // Token embedding
    embed_tokens_ = weights.at("model.embed_tokens.weight");

    // Final norm
    final_norm_weight_ = weights.at("model.norm.weight");

    // LM head (may be tied with embed_tokens)
    auto it = weights.find("lm_head.weight");
    if (it != weights.end()) {
        lm_head_weight_ = it->second;
    } else {
        lm_head_weight_ = embed_tokens_;  // Tied weights
    }

    // Load transformer layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        layers_[i].load_weights(weights, prefix);
    }
    std::cout << "[MODEL] Loaded " << config_.num_hidden_layers << " transformer layers" << std::endl;

    // Load vision encoder
    vision_.load_weights(weights, "visual.");
    std::cout << "[MODEL] Loaded vision encoder" << std::endl;

    // Load action processor
    action_proc_.load_weights(weights, "action_preprocessor.");
    std::cout << "[MODEL] Loaded action processor" << std::endl;

    // Rotary embedding inv_freq
    auto inv_it = weights.find("model.rotary_emb.inv_freq");
    if (inv_it != weights.end()) {
        rotary_inv_freq_ = inv_it->second;
    } else {
        // Compute inv_freq from config
        int head_dim = config_.head_dim;
        auto arange = torch::arange(0, head_dim, 2, torch::kFloat32);
        rotary_inv_freq_ = 1.0f / torch::pow(config_.rope_theta, arange / head_dim);
        rotary_inv_freq_ = rotary_inv_freq_.to(torch::kCUDA);
    }

    std::cout << "[MODEL] Total weight loading time: " << timer.elapsed_ms() << " ms" << std::endl;
}

void WallXModel::load_triton_kernels(const std::string& kernels_dir) {
    std::cout << "[MODEL] Loading Triton kernels from: " << kernels_dir << std::endl;
    triton_.load_directory(kernels_dir);
}

// ===================== Transformer Forward =====================

torch::Tensor WallXModel::transformer_forward(
    const torch::Tensor& inputs_embeds,
    const torch::Tensor& position_ids,
    const TokenTypeInfo& token_types,
    bool use_cache,
    bool is_causal) {

    auto [cos, sin] = compute_rotary_emb(position_ids, inputs_embeds.size(1));

    auto hidden_states = inputs_embeds;

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        hidden_states = layers_[i].forward(
            hidden_states, cos, sin,
            kv_cache_, token_types, triton_, is_causal);
        // After first layer, hidden_states IS the residual for next layer
    }

    // Final layer norm
    hidden_states = rms_norm(hidden_states, final_norm_weight_);

    // Advance KV cache length after processing all layers
    if (use_cache) {
        int new_tokens = inputs_embeds.size(1);
        kv_cache_.advance(new_tokens);
    }

    return hidden_states;
}

torch::Tensor WallXModel::transformer_forward_postfix(
    const torch::Tensor& inputs_embeds,
    const torch::Tensor& position_ids,
    const TokenTypeInfo& token_types,
    int prefix_len) {

    auto [cos, sin] = compute_rotary_emb(position_ids, inputs_embeds.size(1));

    auto hidden_states = inputs_embeds;

    // Use cached prefix KV from previous forward pass
    // Each layer writes postfix KV at current_len_ (=prefix_len),
    // and get_with_new retrieves [0:prefix_len+postfix_len]
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        hidden_states = layers_[i].forward(
            hidden_states, cos, sin,
            kv_cache_, token_types, triton_, /*is_causal=*/false);
    }

    // Do NOT advance cache - postfix is temporary and overwritten each ODE step
    hidden_states = rms_norm(hidden_states, final_norm_weight_);
    return hidden_states;
}

// Overload 2: use pre-computed rotary embeddings (avoids recomputing cos/sin in ODE loop)
torch::Tensor WallXModel::transformer_forward_postfix(
    const torch::Tensor& inputs_embeds,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const TokenTypeInfo& token_types) {

    auto hidden_states = inputs_embeds;

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        hidden_states = layers_[i].forward(
            hidden_states, cos, sin,
            kv_cache_, token_types, triton_, /*is_causal=*/false);
    }

    hidden_states = rms_norm(hidden_states, final_norm_weight_);
    return hidden_states;
}

// ===================== Generate Flow Action =====================

WallXModel::GenerateResult WallXModel::generate_flow_action(
    const torch::Tensor& input_ids,
    const torch::Tensor& pixel_values,
    const torch::Tensor& image_grid_thw,
    const torch::Tensor& moe_token_types,
    const std::string& dataset_name,
    int num_inference_timesteps) {

    GenerateResult result;
    CudaTimer total_timer, step_timer;
    total_timer.start();

    int batch_size = input_ids.size(0);
    int seq_len = input_ids.size(1);
    auto device = input_ids.device();
    int action_dim = config_.action_dim;
    int action_horizon = config_.action_horizon;

    // --- 1. Token Embedding ---
    auto inputs_embeds = torch::embedding(embed_tokens_, input_ids);

    // --- 2. Vision Encoding ---
    step_timer.start();
    if (pixel_values.defined()) {
        auto image_embeds = vision_.forward(pixel_values, image_grid_thw);
        // Scatter image embeddings into input_ids positions
        auto image_mask = (input_ids == config_.image_token_id);
        auto mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds);
        inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds.to(inputs_embeds.dtype()));
    }
    result.vit_ms = step_timer.elapsed_ms();

    // --- 3. Position IDs (3D RoPE) ---
    auto [position_ids, rope_deltas] = get_rope_index(
        input_ids, image_grid_thw,
        /*video_grid_thw=*/torch::optional<torch::Tensor>(),
        /*second_per_grid_ts=*/torch::optional<torch::Tensor>(),
        /*attention_mask=*/torch::optional<torch::Tensor>(),
        config_.vision_spatial_merge_size,
        config_.image_token_id,
        /*video_token_id=*/config_.image_token_id + 1,  // placeholder
        config_.vision_start_token_id,
        /*tokens_per_second=*/1.0f);

    // --- 4. MoE token type info ---
    TokenTypeInfo token_types;
    token_types.token_types = moe_token_types.reshape({-1});
    for (int i = 0; i < config_.num_experts; i++) {
        token_types.num_tokens_per_expert[i] = (moe_token_types == i).sum().item<int>();
    }

    // --- 5. Initialize noisy action ---
    auto noise = torch::randn({batch_size, action_horizon, action_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto noisy_action = noise.clone();

    // DOF mask: all-ones for default (all DOFs active)
    auto dof_mask = torch::ones({batch_size, 1, action_dim},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Time schedule: linspace(0, 1, num_steps+1)
    auto times = torch::linspace(0.0f, 1.0f, num_inference_timesteps + 1, device);
    float dt = (times[1] - times[0]).item<float>();

    // First step: embed noisy action at t=0
    auto time_0 = times[0].unsqueeze(0).expand(batch_size);
    auto action_embed = action_proc_.step(time_0, noisy_action, dof_mask);
    action_embed = action_embed.reshape({-1, config_.hidden_size}).to(inputs_embeds.dtype());

    // Place action embeddings into input sequence
    auto flow_action_mask = (input_ids == config_.action_token_id);
    inputs_embeds.index_put_({flow_action_mask}, action_embed);

    // --- 6. Allocate KV cache ---
    kv_cache_.allocate(config_.num_hidden_layers, batch_size, seq_len + 64,
                        config_.num_key_value_heads, config_.head_dim, device);

    // --- 7. Prefill forward pass ---
    step_timer.start();
    auto hidden_states = transformer_forward(inputs_embeds, position_ids, token_types,
                                              /*use_cache=*/true, /*is_causal=*/true);

    // Extract action hidden states and compute first velocity
    auto action_hidden = hidden_states.index({flow_action_mask}).to(torch::kFloat32);
    auto v_0 = action_proc_.action_proj_back(
        action_hidden.index({torch::indexing::Slice(), torch::indexing::Slice(0, action_proc_.action_hidden_size())}));

    // First Euler step
    noisy_action = noisy_action + dt * v_0.reshape({batch_size, action_horizon, action_dim});
    result.prefill_ms = step_timer.elapsed_ms();

    // --- 8. Truncate KV cache to prefix (before action tokens) ---
    auto prefix_length = torch::argmax(flow_action_mask.to(torch::kFloat32), /*dim=*/1).min().item<int>();
    kv_cache_.truncate(prefix_length);

    // Postfix data
    auto postfix_position_ids = position_ids.index({torch::indexing::Slice(),
                                                      torch::indexing::Slice(),
                                                      torch::indexing::Slice(prefix_length, torch::indexing::None)});
    auto postfix_embeds = inputs_embeds.index({torch::indexing::Slice(),
                                                 torch::indexing::Slice(prefix_length, torch::indexing::None)});
    auto postfix_input_ids = input_ids.index({torch::indexing::Slice(),
                                                torch::indexing::Slice(prefix_length, torch::indexing::None)});
    auto postfix_moe_types = moe_token_types.index({torch::indexing::Slice(),
                                                       torch::indexing::Slice(prefix_length, torch::indexing::None)});

    TokenTypeInfo postfix_token_types;
    postfix_token_types.token_types = postfix_moe_types.reshape({-1});
    for (int i = 0; i < config_.num_experts; i++) {
        postfix_token_types.num_tokens_per_expert[i] = (postfix_moe_types == i).sum().item<int>();
    }

    // --- 9. ODE integration (remaining timesteps) ---
    step_timer.start();

    auto remaining_times = times.index({torch::indexing::Slice(1, torch::indexing::None)});

    // Pre-compute values that don't change across ODE steps
    auto postfix_action_mask = (postfix_input_ids == config_.action_token_id);
    auto [postfix_cos, postfix_sin] = compute_rotary_emb(postfix_position_ids, postfix_embeds.size(1));
    auto ode_embeds_buf = postfix_embeds.clone();  // pre-allocated buffer for embedding scatter

    auto velocity_fn = [&](float t, const torch::Tensor& state) -> torch::Tensor {
        // Create timestep tensor
        auto timestep = torch::full({batch_size}, t, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        // Embed noisy action at current timestep
        auto embed = action_proc_.step(timestep, state, dof_mask);
        embed = embed.reshape({-1, config_.hidden_size}).to(postfix_embeds.dtype());

        // Replace action embeddings in postfix (reuse pre-allocated buffer)
        ode_embeds_buf.copy_(postfix_embeds);
        ode_embeds_buf.index_put_({postfix_action_mask}, embed);

        // Forward through transformer (using cached prefix KV + pre-computed rotary)
        auto hs = transformer_forward_postfix(ode_embeds_buf, postfix_cos, postfix_sin,
                                               postfix_token_types);

        // Extract velocity from action hidden states
        auto act_hs = hs.index({postfix_action_mask}).to(torch::kFloat32);
        auto v_pred = action_proc_.action_proj_back(
            act_hs.index({torch::indexing::Slice(), torch::indexing::Slice(0, action_proc_.action_hidden_size())}));

        return v_pred.reshape({batch_size, action_horizon, action_dim});
    };

    auto final_action = EulerODESolver::solve(velocity_fn, noisy_action, remaining_times);
    result.ode_ms = step_timer.elapsed_ms();

    // --- 10. Unnormalize ---
    result.predict_action = action_proc_.unnormalize(final_action, dataset_name);
    result.total_ms = total_timer.elapsed_ms();

    return result;
}
