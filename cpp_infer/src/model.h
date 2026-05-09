#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
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

    // Debug helper: run the vision encoder only.
    torch::Tensor encode_image(
        const torch::Tensor& pixel_values,
        const torch::Tensor& image_grid_thw);

    // Debug helper: returns {after_reorder_hidden, block0_hidden, block7_hidden, pre_merger_hidden, final_image_embeds}.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> encode_image_debug(
        const torch::Tensor& pixel_values,
        const torch::Tensor& image_grid_thw);

    VisionBlockDebug encode_image_block_debug(
        const torch::Tensor& hidden_states_reordered,
        const torch::Tensor& image_grid_thw,
        int block_idx);

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

    // Generate text (VQA inference entry point)
    // Returns: generated token IDs and timing
    struct TextGenerateResult {
        torch::Tensor generated_ids;  // [num_tokens] generated token IDs
        int num_tokens = 0;           // actual tokens generated (may < max if EOS hit)
        float total_ms = 0;
        float vit_ms = 0;
        float prefill_ms = 0;
        float decode_ms = 0;
    };

    TextGenerateResult generate_text(
        const torch::Tensor& input_ids,        // [batch, seq_len]
        const torch::Tensor& pixel_values,      // raw pixel data
        const torch::Tensor& image_grid_thw,    // [num_images, 3]
        const torch::Tensor& image_embeds_override = {},
        int max_new_tokens = 64,
        bool greedy = true);

    // Debug helper: run VQA under teacher forcing and dump per-step logits.
    // teacher_token_ids: [num_steps] generated token IDs from a reference run.
    // Returns logits tensor on CPU: [num_steps, vocab_size] float32.
    torch::Tensor dump_text_logits_teacher_forced(
        const torch::Tensor& input_ids,
        const torch::Tensor& pixel_values,
        const torch::Tensor& image_grid_thw,
        const torch::Tensor& teacher_token_ids);

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

    struct FlowActionGraphCache {
        bool initialized = false;
        int batch_size = 0;
        int seq_len = 0;
        int action_horizon = 0;
        int action_dim = 0;
        int num_steps = 0;
        int prefix_length = 0;
        std::vector<float> dts;
        std::vector<torch::Tensor> timestep_tensors;

        torch::Tensor state;
        torch::Tensor dof_mask;
        torch::Tensor postfix_embeds;
        torch::Tensor postfix_cos;
        torch::Tensor postfix_sin;
        torch::Tensor postfix_action_mask;
        torch::Tensor ode_embeds_buf;
        TokenTypeInfo postfix_token_types;

        at::cuda::CUDAGraph graph;
        std::unique_ptr<c10::cuda::CUDAStream> capture_stream;
    };

    std::unique_ptr<FlowActionGraphCache> flow_action_graph_;

    struct VQADecodeGraphCache {
        bool initialized = false;
        int batch_size = 0;
        int seq_len = 0;
        int max_new_tokens = 0;
        int current_pos_start = 0;
        torch::Tensor token_id;
        torch::Tensor decode_pos;
        torch::Tensor next_token;
        TokenTypeInfo decode_token_types;
        std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs;
        std::unique_ptr<c10::cuda::CUDAStream> capture_stream;
    };

    std::unique_ptr<VQADecodeGraphCache> vqa_decode_graph_;

    // Compute rotary embeddings (cos, sin) from position_ids
    std::pair<torch::Tensor, torch::Tensor> compute_rotary_emb(
        const torch::Tensor& position_ids, int seq_len);

    bool maybe_replay_flow_action_graph(
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
        double& ode_ms);

    bool maybe_replay_vqa_decode_graph(
        const torch::Tensor& next_token_from_prefill,
        int64_t current_pos_start,
        int max_new_tokens,
        torch::Tensor& generated_gpu,
        double& decode_ms);

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
