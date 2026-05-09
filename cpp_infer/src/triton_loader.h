#pragma once

#include <cuda.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

// Loads Triton-compiled cubin files and launches kernels via CUDA Driver API
class TritonKernel {
public:
    TritonKernel() = default;
    ~TritonKernel();

    // Load cubin file and extract kernel function
    void load(const std::string& cubin_path, const std::string& kernel_name);

    // Launch kernel with given arguments
    // grid: {gridX, gridY, gridZ}
    // block: {blockX, blockY, blockZ}  (typically num_warps * 32, 1, 1)
    void launch(void** args, unsigned int grid_x, unsigned int grid_y,
                unsigned int grid_z, unsigned int block_x,
                unsigned int shared_mem = 0, CUstream stream = nullptr);

    // Convenience: launch 1D grid
    void launch_1d(void** args, unsigned int num_programs,
                   unsigned int block_x = 256,
                   unsigned int shared_mem = 0, CUstream stream = nullptr);

    bool is_loaded() const { return function_ != nullptr; }
    const std::string& name() const { return name_; }
    void set_launch_config(unsigned int block_x, unsigned int shared_mem) {
        block_x_ = block_x;
        shared_mem_ = shared_mem;
    }
    unsigned int block_x() const { return block_x_; }
    unsigned int shared_mem() const { return shared_mem_; }

private:
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;
    std::string name_;
    unsigned int block_x_ = 256;
    unsigned int shared_mem_ = 0;
};

// Registry: manages all Triton kernels for the inference engine
class TritonKernelRegistry {
public:
    // Load all cubin files from a directory
    void load_directory(const std::string& dir_path);

    // Get a loaded kernel by name
    TritonKernel& get(const std::string& name);
    bool has(const std::string& name) const;

    // Launch helpers for specific wall-x kernels

    // fused_add_rmsnorm: residual += x, out = rmsnorm(residual, weight)
    // Args: x_ptr, residual_ptr, weight_ptr, out_ptr, M
    void fused_add_rmsnorm(void* x, void* residual, void* weight, void* out,
                           int M, CUstream stream = nullptr);

    // rmsnorm: out = rmsnorm(x, weight)
    // Args: x_ptr, weight_ptr, out_ptr, M
    void rmsnorm(void* x, void* weight, void* out,
                 int M, CUstream stream = nullptr);

    // fused_silu_mul: out = silu(gate) * up
    // Args: gate_ptr, up_ptr, out_ptr, M, N
    void fused_silu_mul(void* gate, void* up, void* out,
                        int M, int N, CUstream stream = nullptr);

private:
    std::unordered_map<std::string, TritonKernel> kernels_;
};
