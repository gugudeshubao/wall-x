"""
Benchmark Triton Fused Kernels vs PyTorch Baseline on Jetson Orin

Tests with actual wall-x 3B model dimensions:
  hidden_size=2048, intermediate_size=11008/2048
  Measures latency and memory bandwidth utilization.

Usage:
  source /data/wy/wall-x/venv/bin/activate
  python benchmark_kernels.py
"""

import torch
import time
import sys

from fused_add_rmsnorm import fused_add_rmsnorm, triton_rmsnorm
from fused_silu_mul import fused_silu_mul


def benchmark_fn(fn, warmup=50, iters=200):
    """Benchmark a function, return median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # us
    times.sort()
    return times[len(times) // 2]  # median


def benchmark_fused_add_rmsnorm(M, N, eps=1e-6):
    """Compare fused_add_rmsnorm vs separate add + rmsnorm."""
    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.ones(N, device=device, dtype=dtype)

    # --- PyTorch baseline ---
    def pytorch_baseline():
        residual_ref = torch.randn(M, N, device=device, dtype=dtype)
        r = residual_ref + x
        rms = torch.sqrt(r.float().pow(2).mean(-1, keepdim=True) + eps)
        return (r.float() / rms * weight.float()).to(dtype)

    # Pre-allocate for fair comparison
    residual_pt = torch.randn(M, N, device=device, dtype=dtype)
    def pt_fn():
        residual_pt.copy_(residual_pt)  # simulate in-place
        r = residual_pt + x
        rms = torch.sqrt(r.float().pow(2).mean(-1, keepdim=True) + eps)
        return (r.float() / rms * weight.float()).to(dtype)

    residual_tri = torch.randn(M, N, device=device, dtype=dtype)
    def tri_fn():
        return fused_add_rmsnorm(x, residual_tri, weight, eps)

    pt_us = benchmark_fn(pt_fn)
    tri_us = benchmark_fn(tri_fn)

    # Bandwidth: read 2*M*N (x + residual) + write 2*M*N (residual + out) + read N (weight)
    bytes_moved = (2 * M * N + 2 * M * N) * 2 + N * 2  # bf16 = 2 bytes
    bw_tri = bytes_moved / (tri_us * 1e-6) / 1e9  # GB/s

    return pt_us, tri_us, bw_tri


def benchmark_fused_silu_mul(M, N):
    """Compare fused_silu_mul vs separate silu + mul."""
    dtype = torch.bfloat16
    device = "cuda"

    gate = torch.randn(M, N, device=device, dtype=dtype)
    up = torch.randn(M, N, device=device, dtype=dtype)

    def pt_fn():
        return torch.nn.functional.silu(gate) * up

    def tri_fn():
        return fused_silu_mul(gate, up)

    pt_us = benchmark_fn(pt_fn)
    tri_us = benchmark_fn(tri_fn)

    # Bandwidth: read 2*M*N (gate + up) + write M*N (out)
    bytes_moved = (2 * M * N + M * N) * 2  # bf16
    bw_tri = bytes_moved / (tri_us * 1e-6) / 1e9

    return pt_us, tri_us, bw_tri


def main():
    print("=" * 80)
    print("Triton Fused Kernel Benchmark - Jetson Orin (SM 8.7)")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 80)
    print()

    # Orin theoretical peak: 204.8 GB/s (LPDDR5, unified memory)
    PEAK_BW = 204.8

    # --- fused_add_rmsnorm ---
    print("--- fused_add_rmsnorm (residual add + RMSNorm) ---")
    print(f"{'M':>6} {'N':>6} {'PyTorch(us)':>12} {'Triton(us)':>12} {'Speedup':>8} {'BW(GB/s)':>10} {'BW%':>6}")
    print("-" * 70)

    for M, desc in [(1, "decode"), (32, "ODE"), (128, "prefill"), (512, "long_seq")]:
        N = 2048
        pt_us, tri_us, bw = benchmark_fused_add_rmsnorm(M, N)
        speedup = pt_us / tri_us
        bw_pct = bw / PEAK_BW * 100
        print(f"{M:>6} {N:>6} {pt_us:>12.1f} {tri_us:>12.1f} {speedup:>7.2f}x {bw:>9.1f} {bw_pct:>5.1f}%")

    print()

    # --- fused_silu_mul ---
    print("--- fused_silu_mul (SiLU activation * up projection) ---")
    print(f"{'M':>6} {'N':>6} {'PyTorch(us)':>12} {'Triton(us)':>12} {'Speedup':>8} {'BW(GB/s)':>10} {'BW%':>6}")
    print("-" * 70)

    for M, desc in [(1, "decode"), (32, "ODE"), (128, "prefill")]:
        for N, expert in [(11008, "exp0"), (2048, "exp1")]:
            pt_us, tri_us, bw = benchmark_fused_silu_mul(M, N)
            speedup = pt_us / tri_us
            bw_pct = bw / PEAK_BW * 100
            label = f"{desc}/{expert}"
            print(f"{M:>6} {N:>6} {pt_us:>12.1f} {tri_us:>12.1f} {speedup:>7.2f}x {bw:>9.1f} {bw_pct:>5.1f}%  [{label}]")

    print()
    print("Peak memory bandwidth: {:.1f} GB/s (LPDDR5)".format(PEAK_BW))


if __name__ == "__main__":
    main()
