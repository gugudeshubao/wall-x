#include "model.h"
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

// ===================== RMSNorm helper =====================

torch::Tensor WallXModel::rms_norm(const torch::Tensor& x, const torch::Tensor& weight) {
    auto x_f32 = x.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, true);
    auto normed = x_f32 * torch::rsqrt(variance + config_.rms_norm_eps);
    return (weight * normed).to(x.dtype());
}

bool WallXModel::maybe_replay_flow_action_graph(
    const torch::Tensor& noisy_action,
    const torch::Tensor& postfix_embeds,
    const torch::Tensor& postfix_position_ids,
    const torch::Tensor& postfix_input_ids,
    const torch::Tensor& postfix_moe_types,
    const torch::Tensor& postfix_cos,
    const torch::Tensor& postfix_sin,
    const torch::Tensor& dof_mask,
    const torch::Tensor& remaining_times,
    const TokenTypeInfo& postfix_token_types,
    int batch_size,
    int action_horizon,
    int action_dim,
    const std::string& dataset_name,
    torch::Tensor& final_action,
    double& ode_ms) {

    if (std::getenv("WALLX_DISABLE_FLOW_GRAPH") != nullptr) {
        return false;
    }

    // Current fast path is specialized for the fixed-shape benchmark / deployment case.
    if (!noisy_action.is_cuda() ||
        batch_size != 1 ||
        config_.hidden_size != 2048 ||
        postfix_embeds.size(1) != action_horizon ||
        remaining_times.dim() != 1 ||
        remaining_times.size(0) <= 0) {
        return false;
    }

    if (!(postfix_input_ids == config_.action_token_id).all().item<bool>()) {
        return false;
    }

    const int seq_len = postfix_embeds.size(1);
    const int prefix_length = kv_cache_.current_len();
    const int num_steps = remaining_times.size(0);

    auto needs_rebuild = [&]() {
        if (!flow_action_graph_ || !flow_action_graph_->initialized) {
            return true;
        }
        return flow_action_graph_->batch_size != batch_size ||
               flow_action_graph_->seq_len != seq_len ||
               flow_action_graph_->action_horizon != action_horizon ||
               flow_action_graph_->action_dim != action_dim ||
               flow_action_graph_->num_steps != num_steps ||
               flow_action_graph_->prefix_length != prefix_length;
    };

    if (needs_rebuild()) {
        flow_action_graph_ = std::make_unique<FlowActionGraphCache>();
        auto& cache = *flow_action_graph_;
        cache.batch_size = batch_size;
        cache.seq_len = seq_len;
        cache.action_horizon = action_horizon;
        cache.action_dim = action_dim;
        cache.num_steps = num_steps;
        cache.prefix_length = prefix_length;
        cache.state = noisy_action.clone();
        cache.dof_mask = dof_mask.clone();
        cache.postfix_embeds = postfix_embeds.clone();
        cache.postfix_cos = postfix_cos.clone();
        cache.postfix_sin = postfix_sin.clone();
        cache.ode_embeds_buf = postfix_embeds.clone();
        cache.postfix_token_types.token_types = postfix_token_types.token_types.clone();
        for (int i = 0; i < config_.num_experts; i++) {
            cache.postfix_token_types.num_tokens_per_expert[i] =
                postfix_token_types.num_tokens_per_expert[i];
        }
        cache.postfix_token_types.grouped_by_expert = postfix_token_types.grouped_by_expert;

        auto times_cpu = remaining_times.to(torch::kCPU, torch::kFloat32).contiguous();
        auto* t_ptr = times_cpu.data_ptr<float>();
        cache.timestep_tensors.reserve(num_steps);
        cache.dts.reserve(num_steps);
        for (int i = 0; i < num_steps; i++) {
            cache.timestep_tensors.push_back(
                torch::full({batch_size}, t_ptr[i],
                            torch::TensorOptions().dtype(torch::kFloat32).device(noisy_action.device())));
            float dt = (i < num_steps - 1) ? (t_ptr[i + 1] - t_ptr[i]) : (1.0f - t_ptr[i]);
            cache.dts.push_back(dt);
        }

        // Ensure tensors prepared on the caller's stream are visible before
        // the side-stream warmup/capture touches them.
        cudaDeviceSynchronize();

        // Warm up allocations on a side stream before capture.
        auto capture_stream = at::cuda::getStreamFromPool(false, noisy_action.device().index());
        cache.capture_stream = std::make_unique<c10::cuda::CUDAStream>(capture_stream);
        {
            c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
            auto state = cache.state.clone();
            for (int i = 0; i < num_steps; i++) {
                auto embed = action_proc_.step(cache.timestep_tensors[i], state, cache.dof_mask);
                embed = embed.to(cache.postfix_embeds.dtype());
                cache.ode_embeds_buf.copy_(embed);
                auto hs = transformer_forward_postfix(
                    cache.ode_embeds_buf, cache.postfix_cos, cache.postfix_sin,
                    cache.postfix_token_types);
                auto act_hs = hs.reshape({-1, config_.hidden_size}).to(torch::kFloat32);
                auto v_pred = action_proc_.action_proj_back(
                    act_hs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(0, action_proc_.action_hidden_size())}));
                state.add_(v_pred.reshape({batch_size, action_horizon, action_dim}), cache.dts[i]);
            }
            cudaStreamSynchronize(capture_stream.stream());
        }

        {
            c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
            cache.graph.capture_begin(at::cuda::graph_pool_handle(), cudaStreamCaptureModeGlobal);
            for (int i = 0; i < num_steps; i++) {
                auto embed = action_proc_.step(cache.timestep_tensors[i], cache.state, cache.dof_mask);
                embed = embed.to(cache.postfix_embeds.dtype());
                cache.ode_embeds_buf.copy_(embed);
                auto hs = transformer_forward_postfix(
                    cache.ode_embeds_buf, cache.postfix_cos, cache.postfix_sin,
                    cache.postfix_token_types);
                auto act_hs = hs.reshape({-1, config_.hidden_size}).to(torch::kFloat32);
                auto v_pred = action_proc_.action_proj_back(
                    act_hs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(0, action_proc_.action_hidden_size())}));
                cache.state.add_(v_pred.reshape({batch_size, action_horizon, action_dim}), cache.dts[i]);
            }
            cache.graph.capture_end();
        }

        cache.initialized = true;
    }

    auto& cache = *flow_action_graph_;
    {
        c10::cuda::CUDAStreamGuard stream_guard(*cache.capture_stream);
        cache.state.copy_(noisy_action);

        CudaTimer graph_timer;
        graph_timer.start(cache.capture_stream->stream());
        cache.graph.replay();
        ode_ms = graph_timer.elapsed_ms(cache.capture_stream->stream());
    }
    final_action = cache.state;
    return true;
}

bool WallXModel::maybe_replay_vqa_decode_graph(
    const torch::Tensor& next_token_from_prefill,
    int64_t current_pos_start,
    int max_new_tokens,
    torch::Tensor& generated_gpu,
    double& decode_ms) {

    // Disabled for now: decode graph capture is less stable than Flow Action's
    // fixed-shape ODE loop because it mixes graph replay with per-step KV growth.
    // Keep the implementation scaffold for future work, but fall back to the
    // stable GPU-only decode loop today.
    return false;

    if (!next_token_from_prefill.is_cuda() ||
        next_token_from_prefill.dtype() != torch::kLong ||
        next_token_from_prefill.numel() != 1 ||
        max_new_tokens <= 1 ||
        kv_cache_.current_len() <= 0) {
        return false;
    }

    const int batch_size = 1;
    const int seq_len = kv_cache_.current_len();

    auto needs_rebuild = [&]() {
        if (!vqa_decode_graph_ || !vqa_decode_graph_->initialized) {
            return true;
        }
        return vqa_decode_graph_->batch_size != batch_size ||
               vqa_decode_graph_->seq_len != seq_len ||
               vqa_decode_graph_->max_new_tokens != max_new_tokens ||
               vqa_decode_graph_->current_pos_start != current_pos_start;
    };

    if (needs_rebuild()) {
        vqa_decode_graph_ = std::make_unique<VQADecodeGraphCache>();
        auto& cache = *vqa_decode_graph_;
        cache.batch_size = batch_size;
        cache.seq_len = seq_len;
        cache.max_new_tokens = max_new_tokens;
        cache.current_pos_start = static_cast<int>(current_pos_start);
        cache.token_id = torch::zeros({1, 1},
            torch::TensorOptions().dtype(torch::kLong).device(next_token_from_prefill.device()));
        cache.decode_pos = torch::zeros({3, 1, 1},
            torch::TensorOptions().dtype(torch::kLong).device(next_token_from_prefill.device()));
        cache.next_token = torch::zeros({1, 1},
            torch::TensorOptions().dtype(torch::kLong).device(next_token_from_prefill.device()));
        auto single_type = torch::zeros({1},
            torch::TensorOptions().dtype(torch::kLong).device(next_token_from_prefill.device()));
        cache.decode_token_types.token_types = single_type;
        cache.decode_token_types.num_tokens_per_expert[0] = 1;
        for (int i = 1; i < config_.num_experts; i++) {
            cache.decode_token_types.num_tokens_per_expert[i] = 0;
        }
        cache.decode_token_types.grouped_by_expert = true;

        auto capture_stream = at::cuda::getStreamFromPool(false, next_token_from_prefill.device().index());
        cache.capture_stream = std::make_unique<c10::cuda::CUDAStream>(capture_stream);
        cache.graphs.reserve(max_new_tokens - 1);

        {
            c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
            kv_cache_.truncate(seq_len);
            for (int step = 0; step < max_new_tokens - 1; step++) {
                cache.token_id.zero_();
                cache.decode_pos.fill_(current_pos_start + step);
                auto token_embed = torch::embedding(embed_tokens_, cache.token_id);
                auto decode_hidden = transformer_forward(
                    token_embed,
                    cache.decode_pos,
                    cache.decode_token_types,
                    /*use_cache=*/true,
                    /*is_causal=*/false);
                auto step_logits = torch::matmul(decode_hidden.index({0, 0}), lm_head_weight_.t());
                cache.next_token.copy_(step_logits.argmax().reshape({1, 1}));
            }
            cudaStreamSynchronize(capture_stream.stream());
            kv_cache_.truncate(seq_len);
        }

        for (int step = 0; step < max_new_tokens - 1; step++) {
            auto graph = std::make_unique<at::cuda::CUDAGraph>();
            {
                c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
                cache.token_id.zero_();
                cache.decode_pos.fill_(current_pos_start + step);
                graph->capture_begin(at::cuda::graph_pool_handle(), cudaStreamCaptureModeGlobal);
                auto token_embed = torch::embedding(embed_tokens_, cache.token_id);
                auto decode_hidden = transformer_forward(
                    token_embed,
                    cache.decode_pos,
                    cache.decode_token_types,
                    /*use_cache=*/true,
                    /*is_causal=*/false);
                auto step_logits = torch::matmul(decode_hidden.index({0, 0}), lm_head_weight_.t());
                cache.next_token.copy_(step_logits.argmax().reshape({1, 1}));
                graph->capture_end();
            }
            cache.graphs.push_back(std::move(graph));
        }

        kv_cache_.truncate(seq_len);
        cache.initialized = true;
    }

    auto& cache = *vqa_decode_graph_;
    {
        c10::cuda::CUDAStreamGuard stream_guard(*cache.capture_stream);
        kv_cache_.truncate(seq_len);
        cache.token_id.copy_(next_token_from_prefill);
        generated_gpu.index_put_({0}, next_token_from_prefill.reshape({1}));

        CudaTimer timer;
        timer.start(cache.capture_stream->stream());
        for (int step = 0; step < max_new_tokens - 1; step++) {
            cache.decode_pos.fill_(current_pos_start + step);
            cache.graphs[step]->replay();
            generated_gpu.index_put_({step + 1}, cache.next_token.reshape({1}));
            cache.token_id.copy_(cache.next_token);
        }
        cudaStreamSynchronize(cache.capture_stream->stream());
        decode_ms = timer.elapsed_ms(cache.capture_stream->stream());
    }
    return true;
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

    // Convert to bfloat16 to match Q/K dtype (CUDA kernel dispatches on Q's dtype)
    cos = cos.to(torch::kBFloat16);
    sin = sin.to(torch::kBFloat16);

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

torch::Tensor WallXModel::encode_image(
    const torch::Tensor& pixel_values,
    const torch::Tensor& image_grid_thw) {
    if (!pixel_values.defined()) {
        return torch::Tensor();
    }
    return vision_.forward(pixel_values, image_grid_thw);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> WallXModel::encode_image_debug(
    const torch::Tensor& pixel_values,
    const torch::Tensor& image_grid_thw) {
    if (!pixel_values.defined()) {
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
    return vision_.forward_debug(pixel_values, image_grid_thw);
}

VisionBlockDebug WallXModel::encode_image_block_debug(
    const torch::Tensor& hidden_states_reordered,
    const torch::Tensor& image_grid_thw,
    int block_idx) {
    return vision_.run_block_debug(hidden_states_reordered, image_grid_thw, block_idx);
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
    token_types.grouped_by_expert = true;

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
    postfix_token_types.grouped_by_expert = true;

    // --- 9. ODE integration (remaining timesteps) ---
    step_timer.start();

    auto remaining_times = times.index({torch::indexing::Slice(1, torch::indexing::None)});

    // Pre-compute values that don't change across ODE steps
    auto [postfix_cos, postfix_sin] = compute_rotary_emb(postfix_position_ids, postfix_embeds.size(1));

    torch::Tensor final_action;
    double graph_ode_ms = 0.0;
    if (maybe_replay_flow_action_graph(
            noisy_action,
            postfix_embeds,
            postfix_position_ids,
            postfix_input_ids,
            postfix_moe_types,
            postfix_cos,
            postfix_sin,
            dof_mask,
            remaining_times,
            postfix_token_types,
            batch_size,
            action_horizon,
            action_dim,
            dataset_name,
            final_action,
            graph_ode_ms)) {
        result.ode_ms = static_cast<float>(graph_ode_ms);
    } else {
        auto postfix_action_mask = (postfix_input_ids == config_.action_token_id);
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

        final_action = EulerODESolver::solve(velocity_fn, noisy_action, remaining_times);
        result.ode_ms = step_timer.elapsed_ms();
    }

    // --- 10. Unnormalize ---
    result.predict_action = action_proc_.unnormalize(final_action, dataset_name);
    result.total_ms = total_timer.elapsed_ms();

    return result;
}

// ===================== Generate Text (VQA) =====================

WallXModel::TextGenerateResult WallXModel::generate_text(
    const torch::Tensor& input_ids,
    const torch::Tensor& pixel_values,
    const torch::Tensor& image_grid_thw,
    const torch::Tensor& image_embeds_override,
    int max_new_tokens,
    bool greedy) {

    TextGenerateResult result;
    CudaTimer total_timer, step_timer;
    total_timer.start();

    int batch_size = input_ids.size(0);
    int seq_len = input_ids.size(1);
    auto device = input_ids.device();

    // --- 1. Token Embedding ---
    auto inputs_embeds = torch::embedding(embed_tokens_, input_ids);

    // --- 2. Vision Encoding ---
    step_timer.start();
    if (image_embeds_override.defined() || pixel_values.defined()) {
        auto image_embeds = image_embeds_override.defined()
            ? image_embeds_override
            : vision_.forward(pixel_values, image_grid_thw);
        auto image_mask = (input_ids == config_.image_token_id);
        auto mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds);
        inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds.to(inputs_embeds.dtype()));
    }
    result.vit_ms = step_timer.elapsed_ms();

    // --- 3. Position IDs (3D RoPE) ---
    torch::Tensor position_ids;
    if (image_grid_thw.defined()) {
        auto pos_and_delta = get_rope_index(
            input_ids, image_grid_thw,
            /*video_grid_thw=*/torch::optional<torch::Tensor>(),
            /*second_per_grid_ts=*/torch::optional<torch::Tensor>(),
            /*attention_mask=*/torch::optional<torch::Tensor>(),
            config_.vision_spatial_merge_size,
            config_.image_token_id,
            /*video_token_id=*/config_.image_token_id + 1,
            config_.vision_start_token_id,
            /*tokens_per_second=*/1.0f);
        position_ids = std::get<0>(pos_and_delta);
    } else {
        // Text-only debug/inference path: all three mRoPE axes share the same 1D positions.
        position_ids = torch::arange(
            seq_len,
            torch::TensorOptions().dtype(torch::kLong).device(device)
        ).view({1, 1, seq_len}).expand({3, batch_size, seq_len});
    }

    // --- 4. MoE token types: all type 0 (text/vision), no action tokens ---
    TokenTypeInfo token_types;
    auto moe_types = torch::zeros({1, seq_len}, torch::TensorOptions().dtype(torch::kLong).device(device));
    token_types.token_types = moe_types.reshape({-1});
    token_types.num_tokens_per_expert[0] = seq_len;
    for (int i = 1; i < config_.num_experts; i++) {
        token_types.num_tokens_per_expert[i] = 0;
    }
    token_types.grouped_by_expert = true;

    // --- 5. Allocate KV cache ---
    kv_cache_.allocate(config_.num_hidden_layers, batch_size, seq_len + max_new_tokens,
                        config_.num_key_value_heads, config_.head_dim, device);

    // --- 6. Prefill forward ---
    step_timer.start();
    auto hidden_states = transformer_forward(inputs_embeds, position_ids, token_types,
                                              /*use_cache=*/true, /*is_causal=*/true);

    // Apply lm_head to get logits for the last position
    auto last_hidden = hidden_states.index({0, -1});  // [hidden_size]
    auto logits = torch::matmul(last_hidden, lm_head_weight_.t());  // [vocab_size]
    auto next_token_tensor = logits.argmax().reshape({1, 1});
    result.prefill_ms = step_timer.elapsed_ms();

    // --- 7. Decode loop ---
    step_timer.start();

    // Track generated tokens on GPU to avoid per-step GPU->CPU sync.
    auto generated_gpu = torch::empty({max_new_tokens},
        torch::TensorOptions().dtype(torch::kLong).device(device));
    generated_gpu.index_put_({0}, next_token_tensor.reshape({1}));

    // Compute next position: max position from prefill + 1
    // For text tokens in multimodal RoPE, all 3 dims have the same position
    int64_t current_pos = position_ids.max().item<int64_t>() + 1;

    // Decode token type info (single text token per step)
    TokenTypeInfo decode_token_types;
    auto single_type = torch::zeros({1}, torch::TensorOptions().dtype(torch::kLong).device(device));
    decode_token_types.token_types = single_type;
    decode_token_types.num_tokens_per_expert[0] = 1;
    for (int i = 1; i < config_.num_experts; i++) {
        decode_token_types.num_tokens_per_expert[i] = 0;
    }
    decode_token_types.grouped_by_expert = true;

    double decode_graph_ms = 0.0;
    if (maybe_replay_vqa_decode_graph(
            next_token_tensor,
            current_pos,
            max_new_tokens,
            generated_gpu,
            decode_graph_ms)) {
        result.decode_ms = static_cast<float>(decode_graph_ms);
    } else {
        auto token_id = torch::empty({1, 1}, torch::TensorOptions().dtype(torch::kLong).device(device));
        auto decode_pos = torch::empty({3, 1, 1}, torch::TensorOptions().dtype(torch::kLong).device(device));

        for (int step = 1; step < max_new_tokens; step++) {
            token_id.copy_(next_token_tensor);
            auto token_embed = torch::embedding(embed_tokens_, token_id);

            // Position IDs for decode: [3, 1, 1], all dims = current_pos
            decode_pos.fill_(current_pos);

            // Forward through transformer (use_cache=true advances KV cache)
            auto decode_hidden = transformer_forward(token_embed, decode_pos, decode_token_types,
                                                       /*use_cache=*/true, /*is_causal=*/false);

            // Apply lm_head
            auto step_logits = torch::matmul(decode_hidden.index({0, 0}), lm_head_weight_.t());
            next_token_tensor = step_logits.argmax().reshape({1, 1});

            generated_gpu.index_put_({step}, next_token_tensor.reshape({1}));
            current_pos++;
        }

        result.decode_ms = step_timer.elapsed_ms();
    }

    // --- 8. Package results ---
    auto generated_cpu = generated_gpu.to(torch::kCPU);
    auto generated_acc = generated_cpu.accessor<int64_t, 1>();
    result.num_tokens = max_new_tokens;
    for (int i = 0; i < max_new_tokens; i++) {
        if (generated_acc[i] == config_.eos_token_id) {
            result.num_tokens = i + 1;
            break;
        }
    }
    result.generated_ids = generated_cpu.index({torch::indexing::Slice(0, result.num_tokens)});
    result.total_ms = total_timer.elapsed_ms();

    // Reset KV cache for next inference
    kv_cache_.reset();

    return result;
}

torch::Tensor WallXModel::dump_text_logits_teacher_forced(
    const torch::Tensor& input_ids,
    const torch::Tensor& pixel_values,
    const torch::Tensor& image_grid_thw,
    const torch::Tensor& teacher_token_ids) {

    int batch_size = input_ids.size(0);
    int seq_len = input_ids.size(1);
    auto device = input_ids.device();
    int num_steps = teacher_token_ids.numel();

    TORCH_CHECK(batch_size == 1, "Teacher-forced logits dump currently expects batch_size=1");
    TORCH_CHECK(teacher_token_ids.dim() == 1, "teacher_token_ids must be 1D");
    bool recompute_full = std::getenv("WALLX_DEBUG_VQA_RECOMPUTE") != nullptr;

    torch::Tensor image_embeds;
    if (pixel_values.defined()) {
        image_embeds = vision_.forward(pixel_values, image_grid_thw);
    }

    auto build_inputs_embeds = [&](const torch::Tensor& cur_input_ids) {
        auto cur_inputs_embeds = torch::embedding(embed_tokens_, cur_input_ids);
        if (image_embeds.defined()) {
            auto image_mask = (cur_input_ids == config_.image_token_id);
            auto mask_expanded = image_mask.unsqueeze(-1).expand_as(cur_inputs_embeds);
            cur_inputs_embeds = cur_inputs_embeds.masked_scatter(
                mask_expanded, image_embeds.to(cur_inputs_embeds.dtype()));
        }
        return cur_inputs_embeds;
    };

    auto inputs_embeds = build_inputs_embeds(input_ids);

    auto [position_ids, rope_deltas] = get_rope_index(
        input_ids, image_grid_thw,
        /*video_grid_thw=*/torch::optional<torch::Tensor>(),
        /*second_per_grid_ts=*/torch::optional<torch::Tensor>(),
        /*attention_mask=*/torch::optional<torch::Tensor>(),
        config_.vision_spatial_merge_size,
        config_.image_token_id,
        /*video_token_id=*/config_.image_token_id + 1,
        config_.vision_start_token_id,
        /*tokens_per_second=*/1.0f);

    TokenTypeInfo token_types;
    auto moe_types = torch::zeros({1, seq_len}, torch::TensorOptions().dtype(torch::kLong).device(device));
    token_types.token_types = moe_types.reshape({-1});
    token_types.num_tokens_per_expert[0] = seq_len;
    for (int i = 1; i < config_.num_experts; i++) {
        token_types.num_tokens_per_expert[i] = 0;
    }
    token_types.grouped_by_expert = true;

    kv_cache_.allocate(config_.num_hidden_layers, batch_size, seq_len + num_steps,
                        config_.num_key_value_heads, config_.head_dim, device);

    std::vector<torch::Tensor> logits_cpu;
    logits_cpu.reserve(num_steps);

    auto hidden_states = transformer_forward(inputs_embeds, position_ids, token_types,
                                             /*use_cache=*/true, /*is_causal=*/true);
    auto last_hidden = hidden_states.index({0, -1});
    auto logits = torch::matmul(last_hidden, lm_head_weight_.t());
    logits_cpu.push_back(logits.to(torch::kFloat32).cpu());

    if (num_steps <= 1) {
        kv_cache_.reset();
        return torch::stack(logits_cpu, 0);
    }

    int64_t current_pos = position_ids.max().item<int64_t>() + 1;
    TokenTypeInfo decode_token_types;
    auto single_type = torch::zeros({1}, torch::TensorOptions().dtype(torch::kLong).device(device));
    decode_token_types.token_types = single_type;
    decode_token_types.num_tokens_per_expert[0] = 1;
    for (int i = 1; i < config_.num_experts; i++) {
        decode_token_types.num_tokens_per_expert[i] = 0;
    }
    decode_token_types.grouped_by_expert = true;

    auto teacher_gpu = teacher_token_ids.to(device, torch::kLong).contiguous();
    auto token_id = torch::empty({1, 1}, torch::TensorOptions().dtype(torch::kLong).device(device));
    auto decode_pos = torch::empty({3, 1, 1}, torch::TensorOptions().dtype(torch::kLong).device(device));

    for (int step = 1; step < num_steps; step++) {
        torch::Tensor step_logits;
        if (recompute_full) {
            kv_cache_.reset();
            auto forced_prefix = teacher_gpu.index({torch::indexing::Slice(0, step)}).reshape({1, step});
            auto cur_input_ids = torch::cat({input_ids, forced_prefix}, /*dim=*/1);
            auto cur_inputs_embeds = build_inputs_embeds(cur_input_ids);
            auto [cur_position_ids, cur_rope_deltas] = get_rope_index(
                cur_input_ids, image_grid_thw,
                /*video_grid_thw=*/torch::optional<torch::Tensor>(),
                /*second_per_grid_ts=*/torch::optional<torch::Tensor>(),
                /*attention_mask=*/torch::optional<torch::Tensor>(),
                config_.vision_spatial_merge_size,
                config_.image_token_id,
                /*video_token_id=*/config_.image_token_id + 1,
                config_.vision_start_token_id,
                /*tokens_per_second=*/1.0f);
            TokenTypeInfo cur_token_types;
            auto cur_moe_types = torch::zeros({1, cur_input_ids.size(1)},
                torch::TensorOptions().dtype(torch::kLong).device(device));
            cur_token_types.token_types = cur_moe_types.reshape({-1});
            cur_token_types.num_tokens_per_expert[0] = cur_input_ids.size(1);
            for (int i = 1; i < config_.num_experts; i++) {
                cur_token_types.num_tokens_per_expert[i] = 0;
            }
            cur_token_types.grouped_by_expert = true;
            auto cur_hidden_states = transformer_forward(
                cur_inputs_embeds, cur_position_ids, cur_token_types,
                /*use_cache=*/false, /*is_causal=*/true);
            step_logits = torch::matmul(cur_hidden_states.index({0, -1}), lm_head_weight_.t());
        } else {
            token_id.copy_(teacher_gpu.index({step - 1}).reshape({1, 1}));
            auto token_embed = torch::embedding(embed_tokens_, token_id);
            decode_pos.fill_(current_pos);
            auto decode_hidden = transformer_forward(token_embed, decode_pos, decode_token_types,
                                                     /*use_cache=*/true, /*is_causal=*/false);
            step_logits = torch::matmul(decode_hidden.index({0, 0}), lm_head_weight_.t());
            current_pos++;
        }
        logits_cpu.push_back(step_logits.to(torch::kFloat32).cpu());
    }

    kv_cache_.reset();
    return torch::stack(logits_cpu, 0);
}
