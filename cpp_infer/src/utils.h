#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

// Model configuration matching wall-x 3B (wall-oss-flow)
struct ModelConfig {
    int hidden_size = 2048;
    int num_hidden_layers = 36;
    int num_attention_heads = 16;
    int num_key_value_heads = 2;
    int head_dim = 128;  // hidden_size / num_attention_heads
    int vocab_size = 151936;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    std::string hidden_act = "silu";
    bool tie_word_embeddings = true;

    // MoE config
    int num_experts = 2;
    int expert_intermediate_sizes[2] = {11008, 2048};
    bool attention_moe = false;
    bool mlp_moe = true;

    // Vision config
    int vision_depth = 32;
    int vision_hidden_size = 1280;
    int vision_num_heads = 16;
    int vision_intermediate_size = 3420;
    int vision_patch_size = 14;
    int vision_spatial_merge_size = 2;
    int vision_window_size = 112;
    int vision_out_hidden_size = 2048;
    std::vector<int> vision_fullatt_block_indexes = {7, 15, 23, 31};

    // RoPE config
    std::vector<int> mrope_section = {16, 24, 24};

    // Action config
    int action_dim = 20;
    int action_horizon = 32;
    int num_inference_timesteps = 5;
    std::vector<int> dim_inputs = {2048, 2048};

    // Special token IDs
    int bos_token_id = 151643;
    int eos_token_id = 151645;
    int vision_start_token_id = 151652;
    int vision_end_token_id = 151653;
    int image_token_id = 151655;
    int action_token_id = 151666;
    int propri_token_id = 151665;
};

// Timer utility for profiling
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    void print(const std::string& label) const {
        std::cout << "[TIMER] " << label << ": " << elapsed_ms() << " ms" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// CUDA event-based timer for GPU operations
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&end_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(end_);
    }
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }
    float elapsed_ms(cudaStream_t stream = 0) {
        cudaEventRecord(end_, stream);
        cudaEventSynchronize(end_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, end_);
        return ms;
    }
private:
    cudaEvent_t start_, end_;
};

// Weight name mapping: HuggingFace -> internal
using WeightMap = std::unordered_map<std::string, torch::Tensor>;

inline void check_tensor(const torch::Tensor& t, const std::string& name,
                         const std::vector<int64_t>& expected_shape = {}) {
    if (!t.defined()) {
        throw std::runtime_error("Tensor not defined: " + name);
    }
    if (!expected_shape.empty()) {
        auto shape = t.sizes();
        if (shape.size() != expected_shape.size()) {
            throw std::runtime_error(name + ": expected " +
                std::to_string(expected_shape.size()) + " dims, got " +
                std::to_string(shape.size()));
        }
    }
}
