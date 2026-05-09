#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <vector>
#include <filesystem>

#include "model.h"
#include "utils.h"
#include "weight_loader.h"

// Simple CLI argument parser
struct Args {
    std::string model_path;
    std::string kernels_dir = "";
    std::string dataset_name = "x2_normal";
    std::string mode = "action";  // "action" or "vqa"
    std::string input_dir = "";   // directory with pre-saved .pt tensors (for accuracy validation)
    std::string teacher_tokens_path = "";
    std::string dump_logits_path = "";
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
        } else if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
            args.input_dir = argv[++i];
        } else if (arg == "--teacher_tokens" && i + 1 < argc) {
            args.teacher_tokens_path = argv[++i];
        } else if (arg == "--dump_logits" && i + 1 < argc) {
            args.dump_logits_path = argv[++i];
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
                      << "  --input, -i PATH      Load pre-saved tensors from directory (for accuracy validation)\n"
                      << "  --teacher_tokens PATH Teacher-forced token tensor (.pt) for VQA debug\n"
                      << "  --dump_logits PATH    Save teacher-forced VQA logits tensor (.pt)\n"
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
    torch::Tensor image_embeds;
    torch::Tensor vision_after_reorder;
    torch::Tensor vision_block0;
    torch::Tensor vision_block7;
    torch::Tensor vision_block7_input;
    torch::Tensor vision_block7_q;
    torch::Tensor vision_block7_k;
    torch::Tensor vision_block7_v;
    torch::Tensor vision_block7_q_rot;
    torch::Tensor vision_block7_k_rot;
    torch::Tensor vision_block7_norm1;
    torch::Tensor vision_block7_attn_out;
    torch::Tensor vision_block7_after_attn;
    torch::Tensor vision_block7_norm2;
    torch::Tensor vision_block7_mlp_out;
    torch::Tensor vision_block7_output;
    torch::Tensor vision_pre_merger;
    torch::Tensor vision_block15_input;
    torch::Tensor vision_block15_q;
    torch::Tensor vision_block15_k;
    torch::Tensor vision_block15_v;
    torch::Tensor vision_block15_q_rot;
    torch::Tensor vision_block15_k_rot;
    torch::Tensor vision_block15_norm1;
    torch::Tensor vision_block15_attn_out;
    torch::Tensor vision_block15_after_attn;
    torch::Tensor vision_block15_norm2;
    torch::Tensor vision_block15_mlp_out;
    torch::Tensor vision_block15_output;
    torch::Tensor vision_block23_input;
    torch::Tensor vision_block23_q;
    torch::Tensor vision_block23_k;
    torch::Tensor vision_block23_v;
    torch::Tensor vision_block23_q_rot;
    torch::Tensor vision_block23_k_rot;
    torch::Tensor vision_block23_norm1;
    torch::Tensor vision_block23_attn_out;
    torch::Tensor vision_block23_after_attn;
    torch::Tensor vision_block23_norm2;
    torch::Tensor vision_block23_mlp_out;
    torch::Tensor vision_block23_output;
    torch::Tensor moe_token_types;
};

torch::Tensor load_token_ids_from_text(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: Failed to open teacher token file: " << path << std::endl;
        exit(1);
    }

    std::vector<int64_t> token_ids;
    int64_t token = 0;
    while (ifs >> token) {
        token_ids.push_back(token);
    }

    if (token_ids.empty()) {
        std::cerr << "ERROR: No token IDs found in " << path << std::endl;
        exit(1);
    }

    return torch::tensor(token_ids, torch::TensorOptions().dtype(torch::kLong));
}

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

// Load pre-saved tensors from a directory (exported by export_vqa_inputs.py)
DummyInputs load_inputs_from_dir(const std::string& input_dir, torch::Device device) {
    DummyInputs inputs;
    std::string sf_path = input_dir + "/inputs.safetensors";
    std::cout << "[INPUT] Loading tensors from " << sf_path << std::endl;

    auto weights = load_safetensors(sf_path, device);

    auto find_tensor = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(name);
        if (it == weights.end()) {
            std::cerr << "ERROR: Tensor '" << name << "' not found in " << sf_path << std::endl;
            exit(1);
        }
        return it->second;
    };
    auto find_optional_tensor = [&](const std::string& name) -> torch::Tensor {
        auto it = weights.find(name);
        return (it != weights.end()) ? it->second : torch::Tensor();
    };

    inputs.input_ids = find_tensor("input_ids");
    inputs.pixel_values = find_optional_tensor("pixel_values");
    inputs.image_grid_thw = find_optional_tensor("image_grid_thw");
    inputs.image_embeds = find_optional_tensor("image_embeds");
    inputs.vision_after_reorder = find_optional_tensor("vision_after_reorder");
    inputs.vision_block0 = find_optional_tensor("vision_block0");
    inputs.vision_block7 = find_optional_tensor("vision_block7");
    inputs.vision_block7_input = find_optional_tensor("vision_block7_input");
    inputs.vision_block7_q = find_optional_tensor("vision_block7_q");
    inputs.vision_block7_k = find_optional_tensor("vision_block7_k");
    inputs.vision_block7_v = find_optional_tensor("vision_block7_v");
    inputs.vision_block7_q_rot = find_optional_tensor("vision_block7_q_rot");
    inputs.vision_block7_k_rot = find_optional_tensor("vision_block7_k_rot");
    inputs.vision_block7_norm1 = find_optional_tensor("vision_block7_norm1");
    inputs.vision_block7_attn_out = find_optional_tensor("vision_block7_attn_out");
    inputs.vision_block7_after_attn = find_optional_tensor("vision_block7_after_attn");
    inputs.vision_block7_norm2 = find_optional_tensor("vision_block7_norm2");
    inputs.vision_block7_mlp_out = find_optional_tensor("vision_block7_mlp_out");
    inputs.vision_block7_output = find_optional_tensor("vision_block7_output");
    inputs.vision_pre_merger = find_optional_tensor("vision_pre_merger");
    inputs.vision_block15_input = find_optional_tensor("vision_block15_input");
    inputs.vision_block15_q = find_optional_tensor("vision_block15_q");
    inputs.vision_block15_k = find_optional_tensor("vision_block15_k");
    inputs.vision_block15_v = find_optional_tensor("vision_block15_v");
    inputs.vision_block15_q_rot = find_optional_tensor("vision_block15_q_rot");
    inputs.vision_block15_k_rot = find_optional_tensor("vision_block15_k_rot");
    inputs.vision_block15_norm1 = find_optional_tensor("vision_block15_norm1");
    inputs.vision_block15_attn_out = find_optional_tensor("vision_block15_attn_out");
    inputs.vision_block15_after_attn = find_optional_tensor("vision_block15_after_attn");
    inputs.vision_block15_norm2 = find_optional_tensor("vision_block15_norm2");
    inputs.vision_block15_mlp_out = find_optional_tensor("vision_block15_mlp_out");
    inputs.vision_block15_output = find_optional_tensor("vision_block15_output");
    inputs.vision_block23_input = find_optional_tensor("vision_block23_input");
    inputs.vision_block23_q = find_optional_tensor("vision_block23_q");
    inputs.vision_block23_k = find_optional_tensor("vision_block23_k");
    inputs.vision_block23_v = find_optional_tensor("vision_block23_v");
    inputs.vision_block23_q_rot = find_optional_tensor("vision_block23_q_rot");
    inputs.vision_block23_k_rot = find_optional_tensor("vision_block23_k_rot");
    inputs.vision_block23_norm1 = find_optional_tensor("vision_block23_norm1");
    inputs.vision_block23_attn_out = find_optional_tensor("vision_block23_attn_out");
    inputs.vision_block23_after_attn = find_optional_tensor("vision_block23_after_attn");
    inputs.vision_block23_norm2 = find_optional_tensor("vision_block23_norm2");
    inputs.vision_block23_mlp_out = find_optional_tensor("vision_block23_mlp_out");
    inputs.vision_block23_output = find_optional_tensor("vision_block23_output");

    // MoE token types: all 0 for VQA
    inputs.moe_token_types = torch::zeros_like(inputs.input_ids);

    std::cout << "[INPUT] input_ids:      " << inputs.input_ids.sizes() << std::endl;
    if (inputs.pixel_values.defined()) {
        std::cout << "[INPUT] pixel_values:   " << inputs.pixel_values.sizes() << std::endl;
    } else {
        std::cout << "[INPUT] pixel_values:   <none>" << std::endl;
    }
    if (inputs.image_grid_thw.defined()) {
        std::cout << "[INPUT] image_grid_thw: " << inputs.image_grid_thw.sizes() << std::endl;
    } else {
        std::cout << "[INPUT] image_grid_thw: <none>" << std::endl;
    }
    if (inputs.image_embeds.defined()) {
        std::cout << "[INPUT] image_embeds:   " << inputs.image_embeds.sizes() << std::endl;
    }
    if (inputs.vision_after_reorder.defined()) {
        std::cout << "[INPUT] vision_after_reorder: " << inputs.vision_after_reorder.sizes() << std::endl;
    }
    if (inputs.vision_block0.defined()) {
        std::cout << "[INPUT] vision_block0: " << inputs.vision_block0.sizes() << std::endl;
    }
    if (inputs.vision_block7.defined()) {
        std::cout << "[INPUT] vision_block7: " << inputs.vision_block7.sizes() << std::endl;
    }
    if (inputs.vision_block7_q.defined()) {
        std::cout << "[INPUT] vision_block7_q: " << inputs.vision_block7_q.sizes() << std::endl;
    }
    if (inputs.vision_block7_k.defined()) {
        std::cout << "[INPUT] vision_block7_k: " << inputs.vision_block7_k.sizes() << std::endl;
    }
    if (inputs.vision_block7_v.defined()) {
        std::cout << "[INPUT] vision_block7_v: " << inputs.vision_block7_v.sizes() << std::endl;
    }
    if (inputs.vision_block7_q_rot.defined()) {
        std::cout << "[INPUT] vision_block7_q_rot: " << inputs.vision_block7_q_rot.sizes() << std::endl;
    }
    if (inputs.vision_block7_k_rot.defined()) {
        std::cout << "[INPUT] vision_block7_k_rot: " << inputs.vision_block7_k_rot.sizes() << std::endl;
    }
    if (inputs.vision_pre_merger.defined()) {
        std::cout << "[INPUT] vision_pre_merger: " << inputs.vision_pre_merger.sizes() << std::endl;
    }
    if (inputs.vision_block15_input.defined()) {
        std::cout << "[INPUT] vision_block15_input: " << inputs.vision_block15_input.sizes() << std::endl;
    }
    if (inputs.vision_block23_input.defined()) {
        std::cout << "[INPUT] vision_block23_input: " << inputs.vision_block23_input.sizes() << std::endl;
    }

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
        DummyInputs inputs;
        if (!args.input_dir.empty()) {
            inputs = load_inputs_from_dir(args.input_dir, device);
        } else {
            inputs = create_vqa_dummy_inputs(config, device);
        }
        std::cout << "[INPUT] Sequence length: " << inputs.input_ids.size(1) << std::endl;
        std::cout << "[INPUT] Max new tokens: " << args.max_new_tokens << std::endl;

        if (inputs.pixel_values.defined() &&
            inputs.image_grid_thw.defined() &&
            inputs.image_embeds.defined()) {
            torch::NoGradGuard no_grad;
            auto debug_tuple = model.encode_image_debug(inputs.pixel_values, inputs.image_grid_thw);
            auto cpp_after_reorder = std::get<0>(debug_tuple);
            auto cpp_block0 = std::get<1>(debug_tuple);
            auto cpp_block7 = std::get<2>(debug_tuple);
            auto cpp_pre_merger = std::get<3>(debug_tuple);
            auto cpp_image_embeds = std::get<4>(debug_tuple);
            auto cpp_f = cpp_image_embeds.flatten().to(torch::kFloat32);
            auto ref_f = inputs.image_embeds.flatten().to(torch::kFloat32);
            auto dot = (cpp_f * ref_f).sum().item<float>();
            auto norm_cpp = cpp_f.norm().item<float>();
            auto norm_ref = ref_f.norm().item<float>();
            auto cosine = dot / (norm_cpp * norm_ref + 1e-10f);
            auto diff = (cpp_f - ref_f).abs();
            auto max_abs = diff.max().item<float>();
            auto mean_abs = diff.mean().item<float>();
            std::cout << "[VISION CMP] cpp vs python image_embeds:"
                      << " cosine=" << cosine
                      << " max_abs=" << max_abs
                      << " mean_abs=" << mean_abs << std::endl;

            if (inputs.vision_after_reorder.defined() && cpp_after_reorder.defined()) {
                auto cpp_ar = cpp_after_reorder.flatten().to(torch::kFloat32);
                auto ref_ar = inputs.vision_after_reorder.flatten().to(torch::kFloat32);
                auto dot_ar = (cpp_ar * ref_ar).sum().item<float>();
                auto norm_cpp_ar = cpp_ar.norm().item<float>();
                auto norm_ref_ar = ref_ar.norm().item<float>();
                auto cosine_ar = dot_ar / (norm_cpp_ar * norm_ref_ar + 1e-10f);
                auto diff_ar = (cpp_ar - ref_ar).abs();
                auto max_abs_ar = diff_ar.max().item<float>();
                auto mean_abs_ar = diff_ar.mean().item<float>();
                std::cout << "[VISION CMP] cpp vs python after_reorder:"
                          << " cosine=" << cosine_ar
                          << " max_abs=" << max_abs_ar
                          << " mean_abs=" << mean_abs_ar << std::endl;
            }

            if (inputs.vision_block0.defined() && cpp_block0.defined()) {
                auto cpp_b0 = cpp_block0.flatten().to(torch::kFloat32);
                auto ref_b0 = inputs.vision_block0.flatten().to(torch::kFloat32);
                auto dot_b0 = (cpp_b0 * ref_b0).sum().item<float>();
                auto norm_cpp_b0 = cpp_b0.norm().item<float>();
                auto norm_ref_b0 = ref_b0.norm().item<float>();
                auto cosine_b0 = dot_b0 / (norm_cpp_b0 * norm_ref_b0 + 1e-10f);
                auto diff_b0 = (cpp_b0 - ref_b0).abs();
                auto max_abs_b0 = diff_b0.max().item<float>();
                auto mean_abs_b0 = diff_b0.mean().item<float>();
                std::cout << "[VISION CMP] cpp vs python block0:"
                          << " cosine=" << cosine_b0
                          << " max_abs=" << max_abs_b0
                          << " mean_abs=" << mean_abs_b0 << std::endl;
            }

            if (inputs.vision_block7.defined() && cpp_block7.defined()) {
                auto cpp_b7 = cpp_block7.flatten().to(torch::kFloat32);
                auto ref_b7 = inputs.vision_block7.flatten().to(torch::kFloat32);
                auto dot_b7 = (cpp_b7 * ref_b7).sum().item<float>();
                auto norm_cpp_b7 = cpp_b7.norm().item<float>();
                auto norm_ref_b7 = ref_b7.norm().item<float>();
                auto cosine_b7 = dot_b7 / (norm_cpp_b7 * norm_ref_b7 + 1e-10f);
                auto diff_b7 = (cpp_b7 - ref_b7).abs();
                auto max_abs_b7 = diff_b7.max().item<float>();
                auto mean_abs_b7 = diff_b7.mean().item<float>();
                std::cout << "[VISION CMP] cpp vs python block7:"
                          << " cosine=" << cosine_b7
                          << " max_abs=" << max_abs_b7
                          << " mean_abs=" << mean_abs_b7 << std::endl;
            }

            if (inputs.vision_block7_input.defined()) {
                std::cout << "[VISION DBG] block7_input present, q=" << inputs.vision_block7_q.defined()
                          << " k=" << inputs.vision_block7_k.defined()
                          << " v=" << inputs.vision_block7_v.defined()
                          << " q_rot=" << inputs.vision_block7_q_rot.defined()
                          << " k_rot=" << inputs.vision_block7_k_rot.defined()
                          << std::endl;
                auto dbg = model.encode_image_block_debug(inputs.vision_block7_input, inputs.image_grid_thw, 7);
                auto cmp = [&](const char* label, const torch::Tensor& cpp_t, const torch::Tensor& ref_t) {
                    auto cpp_f = cpp_t.flatten().to(torch::kFloat32);
                    auto ref_f = ref_t.flatten().to(torch::kFloat32);
                    auto dot = (cpp_f * ref_f).sum().item<float>();
                    auto norm_cpp = cpp_f.norm().item<float>();
                    auto norm_ref = ref_f.norm().item<float>();
                    auto cosine = dot / (norm_cpp * norm_ref + 1e-10f);
                    auto diff = (cpp_f - ref_f).abs();
                    std::cout << "[VISION CMP] cpp vs python " << label
                              << ": cosine=" << cosine
                              << " max_abs=" << diff.max().item<float>()
                              << " mean_abs=" << diff.mean().item<float>()
                              << std::endl;
                };
                if (inputs.vision_block7_q.defined()) cmp("block7_q", dbg.q, inputs.vision_block7_q);
                if (inputs.vision_block7_k.defined()) cmp("block7_k", dbg.k, inputs.vision_block7_k);
                if (inputs.vision_block7_v.defined()) cmp("block7_v", dbg.v, inputs.vision_block7_v);
                if (inputs.vision_block7_q_rot.defined()) cmp("block7_q_rot", dbg.q_rot, inputs.vision_block7_q_rot);
                if (inputs.vision_block7_k_rot.defined()) cmp("block7_k_rot", dbg.k_rot, inputs.vision_block7_k_rot);
                if (inputs.vision_block7_norm1.defined()) cmp("block7_norm1", dbg.norm1_out, inputs.vision_block7_norm1);
                if (inputs.vision_block7_attn_out.defined()) cmp("block7_attn_out", dbg.attn_out, inputs.vision_block7_attn_out);
                if (inputs.vision_block7_after_attn.defined()) cmp("block7_after_attn", dbg.after_attn, inputs.vision_block7_after_attn);
                if (inputs.vision_block7_norm2.defined()) cmp("block7_norm2", dbg.norm2_out, inputs.vision_block7_norm2);
                if (inputs.vision_block7_mlp_out.defined()) cmp("block7_mlp_out", dbg.mlp_out, inputs.vision_block7_mlp_out);
                if (inputs.vision_block7_output.defined()) cmp("block7_output_dbg", dbg.output, inputs.vision_block7_output);
            }

            if (inputs.vision_pre_merger.defined() && cpp_pre_merger.defined()) {
                auto cpp_pm = cpp_pre_merger.flatten().to(torch::kFloat32);
                auto ref_pm = inputs.vision_pre_merger.flatten().to(torch::kFloat32);
                auto dot_pm = (cpp_pm * ref_pm).sum().item<float>();
                auto norm_cpp_pm = cpp_pm.norm().item<float>();
                auto norm_ref_pm = ref_pm.norm().item<float>();
                auto cosine_pm = dot_pm / (norm_cpp_pm * norm_ref_pm + 1e-10f);
                auto diff_pm = (cpp_pm - ref_pm).abs();
                auto max_abs_pm = diff_pm.max().item<float>();
                auto mean_abs_pm = diff_pm.mean().item<float>();
                std::cout << "[VISION CMP] cpp vs python pre_merger:"
                          << " cosine=" << cosine_pm
                          << " max_abs=" << max_abs_pm
                          << " mean_abs=" << mean_abs_pm << std::endl;
            }

            if (inputs.vision_block15_input.defined()) {
                std::cout << "[VISION DBG] block15_input present, q=" << inputs.vision_block15_q.defined()
                          << " k=" << inputs.vision_block15_k.defined()
                          << " v=" << inputs.vision_block15_v.defined()
                          << " q_rot=" << inputs.vision_block15_q_rot.defined()
                          << " k_rot=" << inputs.vision_block15_k_rot.defined()
                          << std::endl;
                auto dbg15 = model.encode_image_block_debug(inputs.vision_block15_input, inputs.image_grid_thw, 15);
                auto cmp15 = [&](const char* label, const torch::Tensor& cpp_t, const torch::Tensor& ref_t) {
                    auto cpp_f = cpp_t.flatten().to(torch::kFloat32);
                    auto ref_f = ref_t.flatten().to(torch::kFloat32);
                    auto dot = (cpp_f * ref_f).sum().item<float>();
                    auto norm_cpp = cpp_f.norm().item<float>();
                    auto norm_ref = ref_f.norm().item<float>();
                    auto cosine = dot / (norm_cpp * norm_ref + 1e-10f);
                    auto diff = (cpp_f - ref_f).abs();
                    std::cout << "[VISION CMP] cpp vs python " << label
                              << ": cosine=" << cosine
                              << " max_abs=" << diff.max().item<float>()
                              << " mean_abs=" << diff.mean().item<float>()
                              << std::endl;
                };
                if (inputs.vision_block15_q.defined()) cmp15("block15_q", dbg15.q, inputs.vision_block15_q);
                if (inputs.vision_block15_k.defined()) cmp15("block15_k", dbg15.k, inputs.vision_block15_k);
                if (inputs.vision_block15_v.defined()) cmp15("block15_v", dbg15.v, inputs.vision_block15_v);
                if (inputs.vision_block15_q_rot.defined()) cmp15("block15_q_rot", dbg15.q_rot, inputs.vision_block15_q_rot);
                if (inputs.vision_block15_k_rot.defined()) cmp15("block15_k_rot", dbg15.k_rot, inputs.vision_block15_k_rot);
                if (inputs.vision_block15_norm1.defined()) cmp15("block15_norm1", dbg15.norm1_out, inputs.vision_block15_norm1);
                if (inputs.vision_block15_attn_out.defined()) cmp15("block15_attn_out", dbg15.attn_out, inputs.vision_block15_attn_out);
                if (inputs.vision_block15_after_attn.defined()) cmp15("block15_after_attn", dbg15.after_attn, inputs.vision_block15_after_attn);
                if (inputs.vision_block15_norm2.defined()) cmp15("block15_norm2", dbg15.norm2_out, inputs.vision_block15_norm2);
                if (inputs.vision_block15_mlp_out.defined()) cmp15("block15_mlp_out", dbg15.mlp_out, inputs.vision_block15_mlp_out);
                if (inputs.vision_block15_output.defined()) cmp15("block15_output_dbg", dbg15.output, inputs.vision_block15_output);
            }

            if (inputs.vision_block23_input.defined()) {
                std::cout << "[VISION DBG] block23_input present, q=" << inputs.vision_block23_q.defined()
                          << " k=" << inputs.vision_block23_k.defined()
                          << " v=" << inputs.vision_block23_v.defined()
                          << " q_rot=" << inputs.vision_block23_q_rot.defined()
                          << " k_rot=" << inputs.vision_block23_k_rot.defined()
                          << std::endl;
                auto dbg23 = model.encode_image_block_debug(inputs.vision_block23_input, inputs.image_grid_thw, 23);
                auto cmp23 = [&](const char* label, const torch::Tensor& cpp_t, const torch::Tensor& ref_t) {
                    auto cpp_f = cpp_t.flatten().to(torch::kFloat32);
                    auto ref_f = ref_t.flatten().to(torch::kFloat32);
                    auto dot = (cpp_f * ref_f).sum().item<float>();
                    auto norm_cpp = cpp_f.norm().item<float>();
                    auto norm_ref = ref_f.norm().item<float>();
                    auto cosine = dot / (norm_cpp * norm_ref + 1e-10f);
                    auto diff = (cpp_f - ref_f).abs();
                    std::cout << "[VISION CMP] cpp vs python " << label
                              << ": cosine=" << cosine
                              << " max_abs=" << diff.max().item<float>()
                              << " mean_abs=" << diff.mean().item<float>()
                              << std::endl;
                };
                if (inputs.vision_block23_q.defined()) cmp23("block23_q", dbg23.q, inputs.vision_block23_q);
                if (inputs.vision_block23_k.defined()) cmp23("block23_k", dbg23.k, inputs.vision_block23_k);
                if (inputs.vision_block23_v.defined()) cmp23("block23_v", dbg23.v, inputs.vision_block23_v);
                if (inputs.vision_block23_q_rot.defined()) cmp23("block23_q_rot", dbg23.q_rot, inputs.vision_block23_q_rot);
                if (inputs.vision_block23_k_rot.defined()) cmp23("block23_k_rot", dbg23.k_rot, inputs.vision_block23_k_rot);
                if (inputs.vision_block23_norm1.defined()) cmp23("block23_norm1", dbg23.norm1_out, inputs.vision_block23_norm1);
                if (inputs.vision_block23_attn_out.defined()) cmp23("block23_attn_out", dbg23.attn_out, inputs.vision_block23_attn_out);
                if (inputs.vision_block23_after_attn.defined()) cmp23("block23_after_attn", dbg23.after_attn, inputs.vision_block23_after_attn);
                if (inputs.vision_block23_norm2.defined()) cmp23("block23_norm2", dbg23.norm2_out, inputs.vision_block23_norm2);
                if (inputs.vision_block23_mlp_out.defined()) cmp23("block23_mlp_out", dbg23.mlp_out, inputs.vision_block23_mlp_out);
                if (inputs.vision_block23_output.defined()) cmp23("block23_output_dbg", dbg23.output, inputs.vision_block23_output);
            }
        }

        if (!args.dump_logits_path.empty()) {
            if (args.teacher_tokens_path.empty()) {
                std::cerr << "ERROR: --dump_logits requires --teacher_tokens" << std::endl;
                return 1;
            }

            auto teacher_tokens = load_token_ids_from_text(args.teacher_tokens_path);

            std::cout << "\n--- Teacher-Forced Logits Dump ---" << std::endl;
            std::cout << "Teacher tokens: " << teacher_tokens.sizes() << std::endl;

            torch::NoGradGuard no_grad;
            auto logits = model.dump_text_logits_teacher_forced(
                inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw,
                teacher_tokens);
            if (std::filesystem::path(args.dump_logits_path).extension() == ".txt") {
                std::ofstream ofs(args.dump_logits_path);
                auto logits_acc = logits.accessor<float, 2>();
                auto teacher_acc = teacher_tokens.accessor<int64_t, 1>();
                for (int step = 0; step < logits.size(0); step++) {
                    int64_t teacher_id = teacher_acc[step];
                    int64_t top_id = 0;
                    float top_logit = logits_acc[step][0];
                    int64_t rank = 1;
                    for (int64_t vid = 1; vid < logits.size(1); vid++) {
                        float v = logits_acc[step][vid];
                        if (v > top_logit) {
                            top_logit = v;
                            top_id = vid;
                        }
                        if (v > logits_acc[step][teacher_id]) {
                            rank++;
                        }
                    }
                    ofs << step
                        << " teacher=" << teacher_id
                        << " top=" << top_id
                        << " teacher_rank=" << rank
                        << " teacher_logit=" << logits_acc[step][teacher_id]
                        << " top_logit=" << top_logit
                        << "\n";
                }
                ofs.close();
            } else {
                torch::save(logits, args.dump_logits_path);
            }
            std::cout << "Saved logits debug output to " << args.dump_logits_path
                      << " with shape " << logits.sizes() << std::endl;
            return 0;
        }

        // Warmup
        std::cout << "\n--- Warmup (" << args.warmup_runs << " runs) ---" << std::endl;
        for (int i = 0; i < args.warmup_runs; i++) {
            torch::NoGradGuard no_grad;
            auto result = model.generate_text(
                inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw,
                inputs.image_embeds,
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
                    inputs.image_embeds,
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
                inputs.image_embeds,
                args.max_new_tokens);

            std::cout << "Generated " << result.num_tokens << " tokens" << std::endl;
            std::cout << "Token IDs: " << result.generated_ids << std::endl;

            // Save generated token IDs for comparison with Python baseline
            if (!args.input_dir.empty()) {
                std::string out_path = args.input_dir + "/cpp_output_tokens.txt";
                std::ofstream ofs(out_path);
                auto ids = result.generated_ids.cpu().to(torch::kLong);
                for (int i = 0; i < ids.size(0); i++) {
                    if (i > 0) ofs << " ";
                    ofs << ids[i].item<int64_t>();
                }
                ofs << std::endl;
                std::cout << "Saved output tokens to " << out_path << std::endl;
            }

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
