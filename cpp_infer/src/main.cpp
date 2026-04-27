#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>

#include "model.h"
#include "utils.h"

// Simple CLI argument parser
struct Args {
    std::string model_path;
    std::string kernels_dir = "";
    std::string dataset_name = "x2_normal";
    std::string mode = "action";  // "action" or "vqa"
    int num_timesteps = 5;
    int max_new_tokens = 64;
    int warmup_runs = 2;
    int benchmark_runs = 10;
    bool benchmark_mode = false;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if ((arg == "--kernels" || arg == "-k") && i + 1 < argc) {
            args.kernels_dir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            args.dataset_name = argv[++i];
        } else if (arg == "--timesteps" && i + 1 < argc) {
            args.num_timesteps = std::stoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (arg == "--max_new_tokens" && i + 1 < argc) {
            args.max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup_runs = std::stoi(argv[++i]);
        } else if (arg == "--benchmark" && i + 1 < argc) {
            args.benchmark_runs = std::stoi(argv[++i]);
            args.benchmark_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "wall-x C++ Inference Engine\n"
                      << "Usage: wallx_infer [options]\n"
                      << "  --model, -m PATH      Path to model checkpoint directory\n"
                      << "  --kernels, -k PATH    Path to Triton cubin kernels directory\n"
                      << "  --mode MODE           Inference mode: action or vqa (default: action)\n"
                      << "  --dataset NAME        Dataset name for normalizer (default: x2_normal)\n"
                      << "  --timesteps N         Number of ODE timesteps (default: 5)\n"
                      << "  --max_new_tokens N    Max tokens to generate in VQA mode (default: 64)\n"
                      << "  --warmup N            Number of warmup runs (default: 2)\n"
                      << "  --benchmark N         Run N benchmark iterations\n"
                      << "  --help, -h            Show this help\n";
            exit(0);
        }
    }
    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        exit(1);
    }
    return args;
}

// Create dummy inputs for testing/benchmarking
struct DummyInputs {
    torch::Tensor input_ids;
    torch::Tensor pixel_values;
    torch::Tensor image_grid_thw;
    torch::Tensor moe_token_types;
};

DummyInputs create_dummy_inputs(const ModelConfig& config, torch::Device device) {
    DummyInputs inputs;

    // Simulate a typical inference scenario:
    // ~200 text tokens + ~256 image tokens + ~32 action tokens
    int text_len = 200;
    int image_tokens = 256;  // 16x16 patches after merge
    int action_tokens = config.action_horizon;  // 32
    int total_len = text_len + image_tokens + action_tokens;

    // Input IDs: text tokens + image tokens + action tokens
    auto opts_long = torch::TensorOptions().dtype(torch::kLong).device(device);
    auto opts_float = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    inputs.input_ids = torch::randint(0, config.vocab_size, {1, total_len}, opts_long);

    // Set image token region
    inputs.input_ids.index({0, torch::indexing::Slice(text_len, text_len + image_tokens)}) = config.image_token_id;

    // Set action token region
    inputs.input_ids.index({0, torch::indexing::Slice(text_len + image_tokens, torch::indexing::None)}) = config.action_token_id;

    // Pixel values: simulate one image (e.g., 224x224 -> patches)
    int num_patches = image_tokens * config.vision_spatial_merge_size * config.vision_spatial_merge_size;
    inputs.pixel_values = torch::randn({num_patches, 3 * 2 * config.vision_patch_size * config.vision_patch_size},
                                        opts_float);

    // Image grid: [1, 3] -> (temporal=1, height=32, width=32) for 1024 patches -> 256 after merge
    inputs.image_grid_thw = torch::tensor({{1, 32, 32}}, opts_long.device(device));

    // MoE token types: 0 for text/vision, 1 for action
    inputs.moe_token_types = torch::zeros({1, total_len}, opts_long);
    inputs.moe_token_types.index({0, torch::indexing::Slice(text_len + image_tokens, torch::indexing::None)}) = 1;

    return inputs;
}

// Create VQA dummy inputs (no action tokens)
DummyInputs create_vqa_dummy_inputs(const ModelConfig& config, torch::Device device) {
    DummyInputs inputs;

    int text_len = 200;
    int image_tokens = 256;
    int total_len = text_len + image_tokens;

    auto opts_long = torch::TensorOptions().dtype(torch::kLong).device(device);
    auto opts_float = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    inputs.input_ids = torch::randint(0, config.vocab_size, {1, total_len}, opts_long);

    // Set image token region
    inputs.input_ids.index({0, torch::indexing::Slice(text_len, text_len + image_tokens)}) = config.image_token_id;

    // Pixel values (same as Flow Action)
    int num_patches = image_tokens * config.vision_spatial_merge_size * config.vision_spatial_merge_size;
    inputs.pixel_values = torch::randn({num_patches, 3 * 2 * config.vision_patch_size * config.vision_patch_size},
                                        opts_float);

    inputs.image_grid_thw = torch::tensor({{1, 32, 32}}, opts_long.device(device));

    // MoE token types: all 0 (no action tokens)
    inputs.moe_token_types = torch::zeros({1, total_len}, opts_long);

    return inputs;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    // Set up CUDA
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available!" << std::endl;
        return 1;
    }

    torch::Device device(torch::kCUDA, 0);
    std::cout << "========================================" << std::endl;
    std::cout << " wall-x C++ Inference Engine" << std::endl;
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << " Device: " << prop.name << std::endl;
        std::cout << " SM: " << prop.major << "." << prop.minor << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // Initialize model
    ModelConfig config;
    WallXModel model;
    model.init(config);

    // Load weights
    model.load_weights(args.model_path);

    // Load Triton kernels (if available)
    if (!args.kernels_dir.empty()) {
        model.load_triton_kernels(args.kernels_dir);
    }
#ifdef KERNELS_DIR
    else {
        model.load_triton_kernels(KERNELS_DIR);
    }
#endif

    // Create dummy inputs
    std::cout << "\n[MODE] " << args.mode << std::endl;

    if (args.mode == "vqa") {
        // ==================== VQA Mode ====================
        auto inputs = create_vqa_dummy_inputs(config, device);
        std::cout << "[INPUT] Sequence length: " << inputs.input_ids.size(1) << std::endl;
        std::cout << "[INPUT] Max new tokens: " << args.max_new_tokens << std::endl;

        // Warmup
        std::cout << "\n--- Warmup (" << args.warmup_runs << " runs) ---" << std::endl;
        for (int i = 0; i < args.warmup_runs; i++) {
            torch::NoGradGuard no_grad;
            auto result = model.generate_text(
                inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw,
                args.max_new_tokens);
            std::cout << "  Run " << (i + 1) << ": " << result.total_ms << " ms, "
                      << result.num_tokens << " tokens" << std::endl;
        }

        if (args.benchmark_mode) {
            std::cout << "\n--- Benchmark (" << args.benchmark_runs << " runs) ---" << std::endl;
            float total_ms = 0, vit_ms = 0, prefill_ms = 0, decode_ms = 0;
            int total_tokens = 0;

            for (int i = 0; i < args.benchmark_runs; i++) {
                torch::NoGradGuard no_grad;
                auto result = model.generate_text(
                    inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw,
                    args.max_new_tokens);

                total_ms += result.total_ms;
                vit_ms += result.vit_ms;
                prefill_ms += result.prefill_ms;
                decode_ms += result.decode_ms;
                total_tokens += result.num_tokens;

                float tok_s = result.num_tokens / (result.total_ms / 1000.0f);
                std::cout << "  Run " << (i + 1) << ": total=" << result.total_ms
                          << "ms, tokens=" << result.num_tokens
                          << ", tok/s=" << tok_s << std::endl;
            }

            int n = args.benchmark_runs;
            float avg_tokens = (float)total_tokens / n;
            float avg_tok_s = avg_tokens / ((total_ms / n) / 1000.0f);
            std::cout << "\n========== VQA BENCHMARK RESULTS ==========" << std::endl;
            std::cout << "  Average total:   " << (total_ms / n) << " ms" << std::endl;
            std::cout << "  Average ViT:     " << (vit_ms / n) << " ms" << std::endl;
            std::cout << "  Average prefill: " << (prefill_ms / n) << " ms" << std::endl;
            std::cout << "  Average decode:  " << (decode_ms / n) << " ms" << std::endl;
            std::cout << "  Avg tokens:      " << avg_tokens << std::endl;
            std::cout << "  Avg tok/s:       " << avg_tok_s << std::endl;
            std::cout << "============================================" << std::endl;
        } else {
            // Single inference run
            std::cout << "\n--- VQA Inference ---" << std::endl;
            torch::NoGradGuard no_grad;
            auto result = model.generate_text(
                inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw,
                args.max_new_tokens);

            std::cout << "Generated " << result.num_tokens << " tokens" << std::endl;
            std::cout << "Token IDs: " << result.generated_ids << std::endl;

            float tok_s = result.num_tokens / (result.total_ms / 1000.0f);
            std::cout << "\n--- Timing ---" << std::endl;
            std::cout << "  ViT encoding:  " << result.vit_ms << " ms" << std::endl;
            std::cout << "  Prefill:       " << result.prefill_ms << " ms" << std::endl;
            std::cout << "  Decode:        " << result.decode_ms << " ms" << std::endl;
            std::cout << "  Total:         " << result.total_ms << " ms" << std::endl;
            std::cout << "  Throughput:    " << tok_s << " tok/s" << std::endl;
        }
    } else {
        // ==================== Flow Action Mode ====================
        auto inputs = create_dummy_inputs(config, device);
        std::cout << "[INPUT] Sequence length: " << inputs.input_ids.size(1) << std::endl;
        std::cout << "[INPUT] Action tokens: " << config.action_horizon << std::endl;
        std::cout << "[INPUT] ODE timesteps: " << args.num_timesteps << std::endl;

        // Warmup
        std::cout << "\n--- Warmup (" << args.warmup_runs << " runs) ---" << std::endl;
        for (int i = 0; i < args.warmup_runs; i++) {
            torch::NoGradGuard no_grad;
            auto result = model.generate_flow_action(
                inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw,
                inputs.moe_token_types, args.dataset_name, args.num_timesteps);
            std::cout << "  Run " << (i + 1) << ": " << result.total_ms << " ms" << std::endl;
        }

        // Benchmark
        if (args.benchmark_mode) {
            std::cout << "\n--- Benchmark (" << args.benchmark_runs << " runs) ---" << std::endl;
            float total_ms = 0, vit_ms = 0, prefill_ms = 0, ode_ms = 0;

            for (int i = 0; i < args.benchmark_runs; i++) {
                torch::NoGradGuard no_grad;
                auto result = model.generate_flow_action(
                    inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw,
                    inputs.moe_token_types, args.dataset_name, args.num_timesteps);

                total_ms += result.total_ms;
                vit_ms += result.vit_ms;
                prefill_ms += result.prefill_ms;
                ode_ms += result.ode_ms;

                std::cout << "  Run " << (i + 1) << ": total=" << result.total_ms
                          << "ms, vit=" << result.vit_ms
                          << "ms, prefill=" << result.prefill_ms
                          << "ms, ode=" << result.ode_ms << "ms" << std::endl;
            }

            int n = args.benchmark_runs;
            std::cout << "\n========== ACTION BENCHMARK RESULTS ==========" << std::endl;
            std::cout << "  Average total:   " << (total_ms / n) << " ms" << std::endl;
            std::cout << "  Average ViT:     " << (vit_ms / n) << " ms" << std::endl;
            std::cout << "  Average prefill: " << (prefill_ms / n) << " ms" << std::endl;
            std::cout << "  Average ODE:     " << (ode_ms / n) << " ms" << std::endl;
            std::cout << "  Throughput:      " << (1000.0f * n / total_ms) << " infer/s" << std::endl;
            std::cout << "===============================================" << std::endl;
        } else {
            // Single inference run
            std::cout << "\n--- Inference ---" << std::endl;
            torch::NoGradGuard no_grad;
            auto result = model.generate_flow_action(
                inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw,
                inputs.moe_token_types, args.dataset_name, args.num_timesteps);

            std::cout << "Predicted action shape: " << result.predict_action.sizes() << std::endl;
            std::cout << "Action (first 5 dims): " << result.predict_action[0][0].slice(0, 0, 5) << std::endl;

            std::cout << "\n--- Timing ---" << std::endl;
            std::cout << "  ViT encoding:  " << result.vit_ms << " ms" << std::endl;
            std::cout << "  Prefill:       " << result.prefill_ms << " ms" << std::endl;
            std::cout << "  ODE loop:      " << result.ode_ms << " ms" << std::endl;
            std::cout << "  Total:         " << result.total_ms << " ms" << std::endl;
        }
    }

    return 0;
}
