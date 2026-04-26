#!/usr/bin/env python3
"""
Benchmark: 4 Flash Attention implementations on Jetson AGX Orin (SM 8.7)

1. Triton FA    - flash_attn.flash_attn_triton.flash_attn_func
2. FA2 CUDA     - flash_attn.flash_attn_func (C++ CUDA, v2.8.3)
3. cuDNN SDPA   - torch.nn.functional.scaled_dot_product_attention
4. TRT-LLM FA   - tensorrt_llm (if available)

Tests wall-x-like configurations:
  - Prefill:  batch=1, seq=420, nheads=16, head_dim=128, causal=True
  - Postfix:  batch=1, seq_q=32, seq_kv=420, nheads=16, head_dim=128 (cross-attn)
  - Decode:   batch=1, seq_q=1, seq_kv=420, nheads=16, head_dim=128
"""

import sys
import time
import math
import torch
import torch.nn.functional as F

# ============================================================
# Config
# ============================================================
WARMUP = 5
REPEATS = 50
DTYPE = torch.bfloat16
DEVICE = "cuda"

CONFIGS = [
    # (name, batch, seq_q, seq_kv, nheads_q, nheads_kv, head_dim, causal)
    ("Prefill-420",   1, 420, 420, 16, 4, 128, True),
    ("Prefill-488",   1, 488, 488, 16, 4, 128, True),
    ("Postfix-32",    1, 32,  420, 16, 4, 128, False),
    ("Decode-1",      1, 1,   420, 16, 4, 128, False),
]

# ============================================================
# Helper: CUDA timer
# ============================================================
def bench_fn(fn, warmup=WARMUP, repeats=REPEATS):
    """Benchmark a function using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Trim top/bottom 10%
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    avg = sum(trimmed) / len(trimmed)
    return avg, min(times), max(times)


def expand_kv_for_gqa(k, v, nheads_q, nheads_kv):
    """Expand KV for GQA: [b, s, nkv, d] -> [b, s, nq, d]"""
    if nheads_q == nheads_kv:
        return k, v
    ratio = nheads_q // nheads_kv
    b, s, _, d = k.shape
    k = k.unsqueeze(3).expand(b, s, nheads_kv, ratio, d).reshape(b, s, nheads_q, d)
    v = v.unsqueeze(3).expand(b, s, nheads_kv, ratio, d).reshape(b, s, nheads_q, d)
    return k, v


# ============================================================
# Backend 1: Triton FA (from flash_attn)
# ============================================================
def load_triton_fa():
    try:
        import triton
        from flash_attn.flash_attn_triton import flash_attn_func as triton_fa_func
        print(f"[OK] Triton FA loaded (triton {triton.__version__})")
        return triton_fa_func
    except Exception as e:
        print(f"[SKIP] Triton FA: {e}")
        return None


# ============================================================
# Backend 2: FA2 CUDA (flash_attn C++)
# ============================================================
def load_fa2_cuda():
    try:
        from flash_attn import flash_attn_func as fa2_func
        # Quick test
        q = torch.randn(1, 4, 2, 64, dtype=DTYPE, device=DEVICE)
        k = torch.randn(1, 4, 2, 64, dtype=DTYPE, device=DEVICE)
        v = torch.randn(1, 4, 2, 64, dtype=DTYPE, device=DEVICE)
        fa2_func(q, k, v, causal=True)
        import flash_attn
        ver = getattr(flash_attn, '__version__', 'unknown')
        print(f"[OK] FA2 CUDA loaded (v{ver})")
        return fa2_func
    except Exception as e:
        print(f"[SKIP] FA2 CUDA: {e}")
        return None


# ============================================================
# Backend 3: cuDNN SDPA (PyTorch built-in)
# ============================================================
def load_cudnn_sdpa():
    """Always available in PyTorch 2.x"""
    print(f"[OK] cuDNN SDPA loaded (PyTorch {torch.__version__})")
    return "sdpa"


# ============================================================
# Backend 4: TRT-LLM FA
# ============================================================
def load_trtllm_fa():
    try:
        import tensorrt_llm
        # Try to find attention function
        if hasattr(tensorrt_llm, 'functional') and hasattr(tensorrt_llm.functional, 'attention'):
            print(f"[OK] TRT-LLM FA loaded (v{tensorrt_llm.__version__})")
            return tensorrt_llm.functional.attention
        else:
            # TRT-LLM's attention is deeply embedded in the runtime
            # Try alternative: use the raw FMHA plugin
            print(f"[SKIP] TRT-LLM FA: attention function not directly callable")
            print(f"  (TRT-LLM v{tensorrt_llm.__version__} found, but FA is embedded in engine runtime)")
            return None
    except ImportError:
        print(f"[SKIP] TRT-LLM: not installed")
        return None
    except Exception as e:
        print(f"[SKIP] TRT-LLM FA: {e}")
        return None


# ============================================================
# Run benchmarks
# ============================================================
def run_benchmark(config_name, batch, seq_q, seq_kv, nheads_q, nheads_kv, head_dim, causal,
                  triton_fa, fa2_cuda, trtllm_fa):
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  batch={batch}, seq_q={seq_q}, seq_kv={seq_kv}, "
          f"nheads_q={nheads_q}, nheads_kv={nheads_kv}, head_dim={head_dim}, causal={causal}")
    print(f"{'='*60}")

    results = {}

    # --- Generate inputs ---
    # For Triton FA and FA2: shape (batch, seqlen, nheads, headdim)
    q_bshd = torch.randn(batch, seq_q, nheads_q, head_dim, dtype=DTYPE, device=DEVICE)
    k_bshd = torch.randn(batch, seq_kv, nheads_kv, head_dim, dtype=DTYPE, device=DEVICE)
    v_bshd = torch.randn(batch, seq_kv, nheads_kv, head_dim, dtype=DTYPE, device=DEVICE)

    # For SDPA: shape (batch, nheads, seqlen, headdim) + expand GQA
    k_exp, v_exp = expand_kv_for_gqa(k_bshd, v_bshd, nheads_q, nheads_kv)
    q_sdpa = q_bshd.transpose(1, 2)  # (b, nh, sq, hd)
    k_sdpa = k_exp.transpose(1, 2)   # (b, nh, skv, hd)
    v_sdpa = v_exp.transpose(1, 2)   # (b, nh, skv, hd)

    # For FA2/Triton with GQA: expand KV to match Q heads
    k_fa_exp, v_fa_exp = expand_kv_for_gqa(k_bshd, v_bshd, nheads_q, nheads_kv)

    # --- 1. Triton FA ---
    if triton_fa is not None:
        try:
            # Triton FA = FlashAttnFunc.apply, only positional args allowed
            # signature: (q, k, v, bias, causal, softmax_scale)
            # Triton FA doesn't natively support GQA, must expand
            fn = lambda: triton_fa(q_bshd, k_fa_exp, v_fa_exp, None, causal, None)
            avg, mn, mx = bench_fn(fn)
            results["Triton FA"] = avg
            print(f"  Triton FA:   {avg:.3f} ms  (min={mn:.3f}, max={mx:.3f})")
        except Exception as e:
            print(f"  Triton FA:   FAILED - {e}")

    # --- 2. FA2 CUDA ---
    if fa2_cuda is not None:
        try:
            # FA2 supports GQA natively (different nheads for q and kv)
            if causal and seq_q == seq_kv:
                fn = lambda: fa2_cuda(q_bshd, k_bshd, v_bshd, causal=True)
            else:
                # For cross-attention (seq_q != seq_kv), causal may not apply
                fn = lambda: fa2_cuda(q_bshd, k_bshd, v_bshd, causal=False)
            avg, mn, mx = bench_fn(fn)
            results["FA2 CUDA"] = avg
            print(f"  FA2 CUDA:    {avg:.3f} ms  (min={mn:.3f}, max={mx:.3f})")
        except Exception as e:
            print(f"  FA2 CUDA:    FAILED - {e}")

    # --- 3. cuDNN SDPA (isolated) ---
    try:
        use_causal = causal and (seq_q == seq_kv)
        # Properly isolate cuDNN backend: disable all others
        def cudnn_attn():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(True)
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None, dropout_p=0.0,
                is_causal=use_causal
            )
            # Restore defaults
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            return out
        avg, mn, mx = bench_fn(cudnn_attn)
        results["cuDNN SDPA"] = avg
        print(f"  cuDNN SDPA:  {avg:.3f} ms  (min={mn:.3f}, max={mx:.3f})")
    except Exception as e:
        print(f"  cuDNN SDPA:  FAILED - {e}")
        # Restore defaults on failure
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)

    # --- 4. mem_efficient SDPA (isolated) ---
    try:
        use_causal = causal and (seq_q == seq_kv)
        def mem_efficient_attn():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None, dropout_p=0.0,
                is_causal=use_causal
            )
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(True)
            return out
        avg, mn, mx = bench_fn(mem_efficient_attn)
        results["MemEff SDPA"] = avg
        print(f"  MemEff SDPA: {avg:.3f} ms  (min={mn:.3f}, max={mx:.3f})")
    except Exception as e:
        print(f"  MemEff SDPA: FAILED - {e}")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)

    # --- 5. PyTorch Math (truly isolated, no cuDNN) ---
    try:
        def math_attn():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(False)
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None, dropout_p=0.0,
                is_causal=causal and (seq_q == seq_kv)
            )
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(True)
            return out
        avg, mn, mx = bench_fn(math_attn)
        results["Math (ref)"] = avg
        print(f"  Math (ref):  {avg:.3f} ms  (min={mn:.3f}, max={mx:.3f})")
    except Exception as e:
        print(f"  Math (ref):  FAILED - {e}")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)

    return results


def main():
    print("=" * 60)
    print(" Flash Attention Benchmark: 4 Implementations on Orin")
    print(f" Device: {torch.cuda.get_device_name()}")
    print(f" PyTorch: {torch.__version__}")
    print(f" CUDA: {torch.version.cuda}")
    print(f" Dtype: {DTYPE}")
    print(f" Warmup: {WARMUP}, Repeats: {REPEATS}")
    print("=" * 60)

    # Load backends
    print("\n--- Loading backends ---")
    triton_fa = load_triton_fa()
    fa2_cuda = load_fa2_cuda()
    sdpa = load_cudnn_sdpa()
    trtllm_fa = load_trtllm_fa()

    # Run all configs
    all_results = {}
    for cfg in CONFIGS:
        name = cfg[0]
        results = run_benchmark(name, *cfg[1:], triton_fa, fa2_cuda, trtllm_fa)
        all_results[name] = results

    # Summary table
    print("\n\n" + "=" * 80)
    print(" SUMMARY TABLE (ms, lower is better)")
    print("=" * 80)

    # Collect all backends that have at least one result
    all_backends = set()
    for r in all_results.values():
        all_backends.update(r.keys())
    backends = sorted(all_backends)

    # Header
    header = f"{'Config':<16}" + "".join(f"{'│ ' + b:<16}" for b in backends)
    print(header)
    print("─" * len(header))

    for cfg_name, results in all_results.items():
        row = f"{cfg_name:<16}"
        for b in backends:
            if b in results:
                row += f"│ {results[b]:>8.3f} ms   "
            else:
                row += f"│ {'N/A':>8}      "
        print(row)

    # Speedup vs Math reference
    print("\n" + "=" * 80)
    print(" SPEEDUP vs Math (reference)")
    print("=" * 80)
    for cfg_name, results in all_results.items():
        if "Math (ref)" not in results:
            continue
        ref = results["Math (ref)"]
        row = f"{cfg_name:<16}"
        for b in backends:
            if b in results and b != "Math (ref)":
                speedup = ref / results[b]
                row += f"│ {b}: {speedup:.2f}x  "
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
