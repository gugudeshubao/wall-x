#include "triton_loader.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

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

            // For Triton-compiled kernels, the function name inside the cubin
            // may differ. We use the file stem as default.
            try {
                TritonKernel kernel;
                kernel.load(cubin_path, kernel_func_name);
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
    kernel.launch_1d(args, static_cast<unsigned int>(M), 8 * 32, 0, stream);
}

void TritonKernelRegistry::rmsnorm(void* x, void* weight, void* out,
                                    int M, CUstream stream) {
    auto& kernel = get("rmsnorm_h2048");
    void* args[] = {&x, &weight, &out, &M};
    kernel.launch_1d(args, static_cast<unsigned int>(M), 8 * 32, 0, stream);
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
    unsigned int block_x = (N == 11008) ? 8 * 32 : 4 * 32;
    kernel.launch_1d(args, static_cast<unsigned int>(M), block_x, 0, stream);
}
