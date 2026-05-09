// =====================================================================
// bench_int8.cpp — INT8 Linear Layer Micro-benchmark
//
// Compares 4 implementations for W8A8 dynamic quantization:
//   1. bf16 baseline  — torch::linear (bf16 GEMM)
//   2. Naive INT8     — PyTorch ops quant + at::_int_mm + PyTorch ops dequant
//   3. Route 2 (fused + _int_mm): fused_quant + at::_int_mm + fused_dequant
//   4. Route 1 (fused + cublasLt): fused_quant + cublasLt   + fused_dequant
//
// Usage:  ./bench_int8          (default: M=32 K=2048 N=2048)
//         ./bench_int8 32 2048 2048
// =====================================================================

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>

#include "int8_linear.h"    // naive int8_quant::linear
#include "kernels/int8_kernels.h"   // fused kernels + CublasLtInt8Gemm
#include "kernels/cutlass_int8_gemm.h"  // CUTLASS fused GEMM + dequant epilogue

// ---- helpers ----
static void cuda_sync() { cudaDeviceSynchronize(); }

struct BenchResult {
    std::string name;
    double us;       // microseconds per call
};

template <typename Fn>
double bench(Fn&& fn, int warmup = 50, int iters = 200) {
    for (int i = 0; i < warmup; ++i) fn();
    cuda_sync();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    cuda_sync();
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return us / iters;
}

static int round_up_to_8(int x) {
    return ((x + 7) / 8) * 8;
}

// ---- naive INT8 linear (reproduces current int8_quant::linear) ----
static torch::Tensor naive_int8_linear(
    const torch::Tensor& input,            // [M, K] bf16
    const torch::Tensor& weight_int8_t,    // [K, N] int8
    const torch::Tensor& weight_scale)     // [N] f32
{
    int64_t K = input.size(-1);
    auto flat = input.reshape({-1, K});

    // 1. dynamic per-token quant (multiple PyTorch ops)
    auto act_f32   = flat.to(torch::kFloat32);
    auto act_absmax = act_f32.abs().amax(/*dim=*/1, /*keepdim=*/true);
    auto act_scale  = (act_absmax / 127.0f).clamp_min(1e-10f);
    auto act_int8   = (act_f32 / act_scale).round().clamp(-128, 127)
                          .to(torch::kInt8);

    // 2. INT8 GEMM
    auto out_i32 = at::_int_mm(act_int8, weight_int8_t);

    // 3. dequantize (multiple PyTorch ops)
    auto out_f32 = out_i32.to(torch::kFloat32) * act_scale
                   * weight_scale.unsqueeze(0);
    return out_f32.to(torch::kBFloat16);
}


// ---- Route 2: fused quant + _int_mm + fused dequant ----
static torch::Tensor route2_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_int8_t,
    const torch::Tensor& weight_scale)
{
    auto [act_int8, act_scale] = int8_fused::quantize_activation(input);
    auto out_i32 = at::_int_mm(act_int8, weight_int8_t);
    return int8_fused::dequantize(out_i32, act_scale, weight_scale);
}


// ---- Route 1: fused quant + cublasLt + fused dequant ----
static torch::Tensor route1_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_int8_t,
    const torch::Tensor& weight_scale,
    int8_fused::CublasLtInt8Gemm& gemm)
{
    auto [act_int8, act_scale] = int8_fused::quantize_activation(input);
    auto out_i32 = gemm.run(act_int8, weight_int8_t);
    return int8_fused::dequantize(out_i32, act_scale, weight_scale);
}

// ---- Route 3: fused quant + CUTLASS fused GEMM+dequant (2 kernels total) ----
static torch::Tensor route3_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_int8,    // [N, K] NOT transposed
    const torch::Tensor& weight_scale)
{
    auto [act_int8, act_scale] = int8_fused::quantize_activation(input);
    return cutlass_int8::gemm_dequant(act_int8, weight_int8, act_scale, weight_scale);
}

// ---- Route 3b: CUTLASS fused GEMM+dequant + separate bias add ----
static torch::Tensor route3_bias_manual(
    const torch::Tensor& input,
    const torch::Tensor& weight_int8,    // [N, K] NOT transposed
    const torch::Tensor& weight_scale,
    const torch::Tensor& bias)
{
    auto [act_int8, act_scale] = int8_fused::quantize_activation(input);
    auto out = cutlass_int8::gemm_dequant(act_int8, weight_int8, act_scale, weight_scale);
    return out + bias;
}

// ---- Route 3c: CUTLASS fused GEMM+dequant+bias (all in epilogue) ----
static torch::Tensor route3_bias_fused(
    const torch::Tensor& input,
    const torch::Tensor& weight_int8,    // [N, K] NOT transposed
    const torch::Tensor& weight_scale,
    const torch::Tensor& bias)
{
    auto [act_int8, act_scale] = int8_fused::quantize_activation(input);
    return cutlass_int8::gemm_dequant(act_int8, weight_int8, act_scale, weight_scale, bias);
}

// ---- Padded-K old path: explicit zeros+copy before quantization ----
static torch::Tensor route3_bias_fused_old_pad(
    const torch::Tensor& input,
    const torch::Tensor& weight_int8_padded,   // [N_pad, K_pad]
    const torch::Tensor& weight_scale_padded,  // [N_pad]
    const torch::Tensor& bias_padded,          // [N_pad]
    int64_t orig_k,
    int64_t orig_n)
{
    auto flat = input.reshape({-1, input.size(-1)});
    int64_t padded_k = weight_int8_padded.size(1);
    auto padded = torch::zeros({flat.size(0), padded_k}, flat.options());
    padded.slice(1, 0, orig_k).copy_(flat);

    auto [act_int8, act_scale] = int8_fused::quantize_activation(padded);
    auto out = cutlass_int8::gemm_dequant(
        act_int8, weight_int8_padded, act_scale, weight_scale_padded, bias_padded);
    return out.slice(1, 0, orig_n).reshape({input.size(0), orig_n});
}

// ---- Padded-K new path: quantization kernel zero-fills tail directly ----
static torch::Tensor route3_bias_fused_new_pad(
    const torch::Tensor& input,
    const torch::Tensor& weight_int8_padded,   // [N_pad, K_pad]
    const torch::Tensor& weight_scale_padded,  // [N_pad]
    const torch::Tensor& bias_padded,          // [N_pad]
    int64_t orig_k,
    int64_t orig_n)
{
    return int8_quant::linear(
        input, weight_int8_padded, weight_scale_padded, bias_padded, orig_k, orig_n);
}


// =====================================================================
int main(int argc, char* argv[]) {

    int M = 32, K = 2048, N = 2048;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }

    std::cout << "============================================\n"
              << "  INT8 Linear Micro-benchmark\n"
              << "  M=" << M << "  K=" << K << "  N=" << N << "\n"
              << "============================================\n\n";

    auto opts_bf16 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);

    // ---- prepare data ----
    auto act_bf16   = torch::randn({M, K}, opts_bf16);
    auto weight_bf16 = torch::randn({N, K}, opts_bf16);
    auto bias_bf16 = torch::randn({N}, opts_bf16);

    int N_pad = round_up_to_8(N);
    int K_pad = round_up_to_8(K);
    auto weight_bf16_padded = torch::zeros({N_pad, K_pad}, opts_bf16);
    weight_bf16_padded.slice(0, 0, N).slice(1, 0, K).copy_(weight_bf16);
    auto bias_bf16_padded = torch::zeros({N_pad}, opts_bf16);
    bias_bf16_padded.slice(0, 0, N).copy_(bias_bf16);

    // quantize weight (per-channel symmetric, offline)
    auto w_f32  = weight_bf16.to(torch::kFloat32);
    auto w_absmax = w_f32.abs().amax(/*dim=*/1);
    auto w_scale  = (w_absmax / 127.0f).clamp_min(1e-10f);
    auto w_int8   = (w_f32 / w_scale.unsqueeze(1)).round()
                        .clamp(-128, 127).to(torch::kInt8);
    auto w_int8_t = w_int8.t().contiguous();  // [K, N] for _int_mm

    auto w_pad_f32 = weight_bf16_padded.to(torch::kFloat32);
    auto w_pad_absmax = w_pad_f32.abs().amax(/*dim=*/1);
    auto w_scale_pad = (w_pad_absmax / 127.0f).clamp_min(1e-10f);
    auto w_int8_pad = (w_pad_f32 / w_scale_pad.unsqueeze(1)).round()
                          .clamp(-128, 127).to(torch::kInt8);

    bool int_mm_compatible = (M >= 24) && (M % 8 == 0) && (K % 8 == 0);
    bool cutlass_native_compatible = (K % 8 == 0) && (N % 8 == 0);

    // ---- cublasLt handle (reusable) ----
    int8_fused::CublasLtInt8Gemm cublaslt_gemm;

    // ---- correctness check ----
    {
        auto ref = torch::linear(act_bf16, weight_bf16);
        auto ref_bias = torch::linear(act_bf16, weight_bf16, bias_bf16);
        auto r_route3_old_pad = route3_bias_fused_old_pad(
            act_bf16, w_int8_pad, w_scale_pad, bias_bf16_padded, K, N);
        auto r_route3_new_pad = route3_bias_fused_new_pad(
            act_bf16, w_int8_pad, w_scale_pad, bias_bf16_padded, K, N);

        auto cos = [](const torch::Tensor& a, const torch::Tensor& b) {
            auto af = a.flatten().to(torch::kFloat32);
            auto bf = b.flatten().to(torch::kFloat32);
            return (af * bf).sum().item<float>() /
                   (af.norm().item<float>() * bf.norm().item<float>() + 1e-10f);
        };

        std::cout << "Correctness (cosine sim vs bf16 ref):\n"
                  << std::fixed << std::setprecision(6);
        if (int_mm_compatible) {
            auto r_naive  = naive_int8_linear(act_bf16, w_int8_t, w_scale);
            auto r_route2 = route2_linear(act_bf16, w_int8_t, w_scale);
            auto r_route1 = route1_linear(act_bf16, w_int8_t, w_scale, cublaslt_gemm);
            std::cout << "  Naive  : " << cos(r_naive, ref) << "\n"
                      << "  Route2 : " << cos(r_route2, ref) << "\n"
                      << "  Route1 : " << cos(r_route1, ref) << "\n";
        } else {
            std::cout << "  Naive  : skipped (requires K % 8 == 0 and padded _int_mm path)\n"
                      << "  Route2 : skipped (requires K % 8 == 0 and padded _int_mm path)\n"
                      << "  Route1 : skipped (uses _int_mm-compatible synthetic reference path)\n";
        }
        if (cutlass_native_compatible) {
            auto r_route3 = route3_linear(act_bf16, w_int8, w_scale);
            auto r_route3_bias_manual = route3_bias_manual(act_bf16, w_int8, w_scale, bias_bf16);
            auto r_route3_bias_fused = route3_bias_fused(act_bf16, w_int8, w_scale, bias_bf16);
            std::cout
                      << "  Route3 : " << cos(r_route3, ref) << "\n"
                      << "  R3+bias(manual): " << cos(r_route3_bias_manual, ref_bias) << "\n"
                      << "  R3+bias(fused) : " << cos(r_route3_bias_fused, ref_bias) << "\n"
                      << "  fused≈manual   : " << cos(r_route3_bias_fused, r_route3_bias_manual) << "\n";
        } else {
            std::cout
                      << "  Route3 : skipped (native CUTLASS path requires K,N % 8 == 0)\n"
                      << "  R3+bias(manual): skipped (native CUTLASS path requires K,N % 8 == 0)\n"
                      << "  R3+bias(fused) : skipped (native CUTLASS path requires K,N % 8 == 0)\n"
                      << "  fused≈manual   : skipped (native CUTLASS path requires K,N % 8 == 0)\n";
        }
        std::cout
                  << "  old-pad≈ref    : " << cos(r_route3_old_pad, ref_bias) << "\n"
                  << "  new-pad≈ref    : " << cos(r_route3_new_pad, ref_bias) << "\n"
                  << "  new≈old-pad    : " << cos(r_route3_new_pad, r_route3_old_pad) << "\n\n";
    }

    // ---- benchmark ----
    std::vector<BenchResult> results;

    // 1. bf16 baseline
    results.push_back({"bf16 baseline",
        bench([&]{ torch::linear(act_bf16, weight_bf16); })});

    if (int_mm_compatible) {
        // 2. Naive INT8
        results.push_back({"Naive INT8 (~10 kernels)",
            bench([&]{ naive_int8_linear(act_bf16, w_int8_t, w_scale); })});

        // 3. Route 2: fused + _int_mm
        results.push_back({"Route2: fused + _int_mm",
            bench([&]{ route2_linear(act_bf16, w_int8_t, w_scale); })});

        // 4. Route 1: fused + cublasLt
        results.push_back({"Route1: fused + cublasLt",
            bench([&]{ route1_linear(act_bf16, w_int8_t, w_scale, cublaslt_gemm); })});
    }

    if (cutlass_native_compatible) {
        // 5. Route 3: fused_quant + CUTLASS fused GEMM+dequant
        results.push_back({"Route3: CUTLASS fused (2 kern)",
            bench([&]{ route3_linear(act_bf16, w_int8, w_scale); })});

        // 6. Route 3 + separate bias add
        results.push_back({"Route3+bias: manual add",
            bench([&]{ route3_bias_manual(act_bf16, w_int8, w_scale, bias_bf16); })});

        // 7. Route 3 + fused bias in epilogue
        results.push_back({"Route3+bias: fused epi",
            bench([&]{ route3_bias_fused(act_bf16, w_int8, w_scale, bias_bf16); })});
    }

    if (K_pad != K || N_pad != N) {
        results.push_back({"Route3 padded: old zeros+copy",
            bench([&]{ route3_bias_fused_old_pad(
                act_bf16, w_int8_pad, w_scale_pad, bias_bf16_padded, K, N); })});
        results.push_back({"Route3 padded: quant+zerofill",
            bench([&]{ route3_bias_fused_new_pad(
                act_bf16, w_int8_pad, w_scale_pad, bias_bf16_padded, K, N); })});
    }

    // ---- also bench individual kernels ----
    // Fused quant
    results.push_back({"  [kernel] fused_quant",
        bench([&]{ int8_fused::quantize_activation(act_bf16); })});

    // _int_mm only
    auto [tmp_act, tmp_scale] = int8_fused::quantize_activation(act_bf16);
    if (int_mm_compatible) {
        results.push_back({"  [kernel] at::_int_mm",
            bench([&]{ at::_int_mm(tmp_act, w_int8_t); })});

        // cublasLt only
        results.push_back({"  [kernel] cublasLt GEMM",
            bench([&]{ cublaslt_gemm.run(tmp_act, w_int8_t); })});
    }

    if (cutlass_native_compatible) {
        // CUTLASS fused GEMM+dequant only (no quant)
        results.push_back({"  [kernel] CUTLASS GEMM+deq",
            bench([&]{ cutlass_int8::gemm_dequant(tmp_act, w_int8, tmp_scale, w_scale); })});

        // CUTLASS fused GEMM+dequant+bias only (no quant)
        results.push_back({"  [kernel] CUTLASS GEMM+deq+bias",
            bench([&]{ cutlass_int8::gemm_dequant(tmp_act, w_int8, tmp_scale, w_scale, bias_bf16); })});
    }

    // Fused dequant
    if (int_mm_compatible) {
        auto tmp_i32 = at::_int_mm(tmp_act, w_int8_t);
        results.push_back({"  [kernel] fused_dequant",
            bench([&]{ int8_fused::dequantize(tmp_i32, tmp_scale, w_scale); })});
    }

    // ---- print table ----
    std::cout << "--------------------------------------------\n"
              << std::left << std::setw(32) << "Method"
              << std::right << std::setw(10) << "us/call"
              << std::setw(10) << "vs bf16" << "\n"
              << "--------------------------------------------\n";

    double bf16_us = results[0].us;
    for (auto& r : results) {
        std::cout << std::left << std::setw(32) << r.name
                  << std::right << std::setw(8) << std::fixed
                  << std::setprecision(1) << r.us << "us";
        if (r.name.find("[kernel]") == std::string::npos) {
            std::cout << std::setw(8) << std::setprecision(2)
                      << (r.us / bf16_us) << "x";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------------------------------\n";

    return 0;
}
