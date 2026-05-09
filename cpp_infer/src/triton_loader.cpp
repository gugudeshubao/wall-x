#include "triton_loader.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

static std::string read_text_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

static std::string extract_json_string(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    pos = json.find("\"", pos);
    if (pos == std::string::npos) return "";
    pos++;
    auto end = json.find("\"", pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

static int extract_json_int(const std::string& json, const std::string& key, int default_value = 0) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_value;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_value;
    pos = json.find_first_of("-0123456789", pos);
    if (pos == std::string::npos) return default_value;
    auto end = json.find_first_not_of("0123456789", pos + 1);
    try {
        return std::stoi(json.substr(pos, end - pos));
    } catch (...) {
        return default_value;
    }
}

// --- TritonKernel ---

TritonKernel::~TritonKernel() {
    if (module_) {
        cuModuleUnload(module_);
    }
}

void TritonKernel::load(const std::string& cubin_path, const std::string& kernel_name) {
    name_ = kernel_name;

    // Initialize CUDA driver API (idempotent)
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuInit failed: " + std::to_string(res));
    }

    // Load cubin module
    res = cuModuleLoad(&module_, cubin_path.c_str());
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuModuleLoad failed for " + cubin_path +
                                 ": error " + std::to_string(res));
    }

    // Get function handle
    res = cuModuleGetFunction(&function_, module_, kernel_name.c_str());
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuModuleGetFunction failed for " + kernel_name +
                                 ": error " + std::to_string(res));
    }

    std::cout << "[TritonKernel] Loaded " << kernel_name << " from " << cubin_path << std::endl;
}

void TritonKernel::launch(void** args, unsigned int grid_x, unsigned int grid_y,
                          unsigned int grid_z, unsigned int block_x,
                          unsigned int shared_mem, CUstream stream) {
    assert(function_ && "Kernel not loaded!");
    CUresult res = cuLaunchKernel(
        function_,
        grid_x, grid_y, grid_z,     // grid
        block_x, 1, 1,              // block
        shared_mem,                   // shared memory bytes
        stream,                       // stream
        args,                         // kernel args
        nullptr                       // extra
    );
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed for " + name_ +
                                 ": error " + std::to_string(res));
    }
}

void TritonKernel::launch_1d(void** args, unsigned int num_programs,
                             unsigned int block_x, unsigned int shared_mem,
                             CUstream stream) {
    launch(args, num_programs, 1, 1, block_x, shared_mem, stream);
}

// --- TritonKernelRegistry ---

void TritonKernelRegistry::load_directory(const std::string& dir_path) {
    if (!fs::exists(dir_path)) {
        std::cerr << "[TritonKernelRegistry] Warning: kernel directory not found: "
                  << dir_path << std::endl;
        return;
    }

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".cubin") {
            std::string name = entry.path().stem().string();
            std::string cubin_path = entry.path().string();

            // Try to find kernel function name from JSON metadata
            std::string meta_path = entry.path().parent_path().string() + "/" + name + ".json";
            std::string kernel_func_name = name;  // default: same as file name
            unsigned int block_x = 256;
            unsigned int shared_mem = 0;

            // For Triton-compiled kernels, the function name inside the cubin
            // often differs from the file stem. Prefer metadata when available.
            if (fs::exists(meta_path)) {
                auto meta_json = read_text_file(meta_path);
                auto meta_kernel_name = extract_json_string(meta_json, "kernel_name");
                if (!meta_kernel_name.empty()) {
                    kernel_func_name = meta_kernel_name;
                }
                int num_warps = extract_json_int(meta_json, "num_warps", 0);
                if (num_warps > 0) {
                    block_x = static_cast<unsigned int>(num_warps * 32);
                }
                shared_mem = static_cast<unsigned int>(extract_json_int(meta_json, "shared_mem", 0));
            }

            try {
                TritonKernel kernel;
                kernel.load(cubin_path, kernel_func_name);
                kernel.set_launch_config(block_x, shared_mem);
                kernels_[name] = std::move(kernel);
            } catch (const std::exception& e) {
                std::cerr << "[TritonKernelRegistry] Failed to load " << name
                          << ": " << e.what() << std::endl;
            }
        }
    }
    std::cout << "[TritonKernelRegistry] Loaded " << kernels_.size()
              << " kernels from " << dir_path << std::endl;
}

TritonKernel& TritonKernelRegistry::get(const std::string& name) {
    auto it = kernels_.find(name);
    if (it == kernels_.end()) {
        throw std::runtime_error("Kernel not found: " + name);
    }
    return it->second;
}

bool TritonKernelRegistry::has(const std::string& name) const {
    return kernels_.count(name) > 0;
}

void TritonKernelRegistry::fused_add_rmsnorm(void* x, void* residual, void* weight,
                                              void* out, int M, CUstream stream) {
    auto& kernel = get("fused_add_rmsnorm_h2048");
    // Args must match kernel signature: x_ptr, residual_ptr, weight_ptr, out_ptr, M
    void* args[] = {&x, &residual, &weight, &out, &M};
    // Grid: one program per row, block: num_warps * 32
    kernel.launch_1d(args, static_cast<unsigned int>(M), kernel.block_x(), kernel.shared_mem(), stream);
}

void TritonKernelRegistry::rmsnorm(void* x, void* weight, void* out,
                                    int M, CUstream stream) {
    auto& kernel = get("rmsnorm_h2048");
    void* args[] = {&x, &weight, &out, &M};
    kernel.launch_1d(args, static_cast<unsigned int>(M), kernel.block_x(), kernel.shared_mem(), stream);
}

void TritonKernelRegistry::fused_silu_mul(void* gate, void* up, void* out,
                                           int M, int N, CUstream stream) {
    // Select kernel variant based on N
    std::string kernel_name;
    if (N == 11008) {
        kernel_name = "fused_silu_mul_n11008";
    } else if (N == 2048) {
        kernel_name = "fused_silu_mul_n2048";
    } else {
        throw std::runtime_error("No fused_silu_mul kernel for N=" + std::to_string(N));
    }

    auto& kernel = get(kernel_name);
    void* args[] = {&gate, &up, &out, &M, &N};
    kernel.launch_1d(args, static_cast<unsigned int>(M), kernel.block_x(), kernel.shared_mem(), stream);
}
