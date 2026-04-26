#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>

#include "utils.h"
#include "transformer.h"
#include "vision.h"
#include "action_head.h"
#include "ode_solver.h"
#include "kv_cache.h"
#include "triton_loader.h"
#include "weight_loader.h"

// Forward declaration for RoPE index CUDA kernel (from csrc/rope_index.cu)
extern std::tuple<torch::Tensor, torch::Tensor> get_rope_index(
    const torch::optional<torch::Tensor>& input_ids,
    const torch::optional<torch::Tensor>& image_grid_thw,
    const torch::optional<torch::Tensor>& video_grid_thw,
    const torch::optional<torch::Tensor>& second_per_grid_ts,
    const torch::optional<torch::Tensor>& attention_mask,
    int spatial_merge_size,
    int image_token_id,
    int video_token_id,
    int vision_start_token_id,
    float tokens_per_second);

// Full wall-x model: ViT + LLM (MoE) + Action Head
class WallXModel {
public:
    WallXModel() = default;

    // Initialize model structure from config
    void init(const ModelConfig& config);

    // Load weights from safetensors checkpoint directory
    void load_weights(const std::string& checkpoint_path);

    // Load Triton cubin kernels
    void load_triton_kernels(const std::string& kernels_dir);

    // Generate flow action (main inference entry point)
    // Returns: predicted action [batch, horizon, action_dim]
    struct GenerateResult {
        torch::Tensor predict_action;
        float total_ms = 0;
        float vit_ms = 0;
        float prefill_ms = 0;
        float ode_ms = 0;
    };

    GenerateResult generate_flow_action(
        const torch::Tensor& input_ids,        // [batch, seq_len]
        const torch::Tensor& pixel_values,      // raw pixel data
        const torch::Tensor& image_grid_thw,    // [num_images, 3]
        const torch::Tensor& moe_token_types,   // [batch, seq_len] 0=text/vision, 1=action
        const std::string& dataset_name,
        int num_inference_timesteps = 5);

private:
    ModelConfig config_;

    // Token embedding
    torch::Tensor embed_tokens_;  // [vocab_size, hidden_size]

    // Transformer layers
    std::vector<TransformerLayer> layers_;

    // Final layer norm
    torch::Tensor final_norm_weight_;

    // LM head (for text generation, not used in action inference)
    torch::Tensor lm_head_weight_;

    // Vision encoder
    VisionEncoder vision_;

    // Action processor
    ActionProcessor action_proc_;

    // KV cache
    KVCache kv_cache_;

    // Triton kernels
    TritonKernelRegistry triton_;

    // Rotary embedding (LLM)
    torch::Tensor rotary_inv_freq_;
    int max_position_embeddings_ = 32768;

    // RMSNorm helper
    torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight);

    // Compute rotary embeddings (cos, sin) from position_ids
    std::pair<torch::Tensor, torch::Tensor> compute_rotary_emb(
        const torch::Tensor& position_ids, int seq_len);

    // Forward through all transformer layers
    torch::Tensor transformer_forward(
        const torch::Tensor& inputs_embeds,
        const torch::Tensor& position_ids,
        const TokenTypeInfo& token_types,
        bool use_cache,
        bool is_causal = true);

    // Forward through transformer layers for postfix (with cached prefix KV)
    // Overload 1: compute rotary embeddings internally
    torch::Tensor transformer_forward_postfix(
        const torch::Tensor& inputs_embeds,
        const torch::Tensor& position_ids,
        const TokenTypeInfo& token_types,
        int prefix_len);

    // Overload 2: use pre-computed rotary embeddings (for ODE loop caching)
    torch::Tensor transformer_forward_postfix(
        const torch::Tensor& inputs_embeds,
        const torch::Tensor& cos,
        const torch::Tensor& sin,
        const TokenTypeInfo& token_types);
};
