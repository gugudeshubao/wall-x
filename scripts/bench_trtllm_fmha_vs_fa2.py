#!/usr/bin/env python3
"""
Benchmark: TRT-LLM FMHA (SM 8.7 precompiled cubin) vs open-source FA2.

Tests both prefill (context) and decode (generation) attention latency
with wall-x model dimensions:
  - num_heads=16, num_kv_heads=2 (GQA 8:1)
  - head_dim=128, hidden_size=2048
  - prefill seq_len=420, decode seq_len=1 (kv_len=420)

Usage:
  python bench_trtllm_fmha_vs_fa2.py
"""

import time
import sys
import torch
import numpy as np

# ──────────────────────────────────────────────────────────
# Config matching wall-x model
# ──────────────────────────────────────────────────────────
BATCH = 1
NUM_HEADS = 16       # query heads
NUM_KV_HEADS = 2     # KV heads (GQA 8:1)
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM   # 2048
PREFILL_SEQ = 420    # wall-x VQA typical
WARMUP = 5
ITERS = 50
DTYPE_STR = "bfloat16"
DTYPE_TORCH = torch.bfloat16

# ──────────────────────────────────────────────────────────
# 1. Open-source FA2 benchmark
# ──────────────────────────────────────────────────────────
def bench_fa2_prefill():
    from flash_attn import flash_attn_func
    q = torch.randn(BATCH, PREFILL_SEQ, NUM_HEADS, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')
    k = torch.randn(BATCH, PREFILL_SEQ, NUM_KV_HEADS, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')
    v = torch.randn(BATCH, PREFILL_SEQ, NUM_KV_HEADS, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')

    # warmup
    for _ in range(WARMUP):
        _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()

    # time
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return times


def bench_fa2_decode():
    """Decode: Q has seq_len=1, KV has seq_len=420 (splitkv path)."""
    from flash_attn import flash_attn_func
    q = torch.randn(BATCH, 1, NUM_HEADS, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')
    k = torch.randn(BATCH, PREFILL_SEQ, NUM_KV_HEADS, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')
    v = torch.randn(BATCH, PREFILL_SEQ, NUM_KV_HEADS, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')

    for _ in range(WARMUP):
        _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return times


# ──────────────────────────────────────────────────────────
# 2. TRT-LLM FMHA benchmark (builds TRT engine with FMHA plugin)
# ──────────────────────────────────────────────────────────
def build_trtllm_prefill_engine():
    """Build a TRT engine with a single FMHA layer (context/prefill)."""
    import tensorrt_llm
    from tensorrt_llm import Tensor, str_dtype_to_trt
    from tensorrt_llm.functional import gpt_attention
    from tensorrt_llm.models.generation_mixin import GenerationMixin
    from tensorrt_llm.plugin.plugin import ContextFMHAType

    tensorrt_llm.logger.set_level('error')

    max_batch_size = BATCH
    max_input_len = PREFILL_SEQ
    max_seq_len = PREFILL_SEQ
    num_layers = 1
    kv_dtype = str_dtype_to_trt(DTYPE_STR)

    # QKV packed: (batch, seq, (num_heads + 2*num_kv_heads) * head_dim)
    qkv_dim = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    qkv_shape = (max_batch_size, max_input_len, qkv_dim)

    builder = tensorrt_llm.Builder()
    builder_config = builder.create_builder_config(
        name="fmha_bench",
        precision=DTYPE_STR,
    )
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    net.plugin_config.gpt_attention_plugin = DTYPE_STR
    net.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    net.plugin_config.remove_input_padding = False

    with tensorrt_llm.net_guard(net):
        inputs = GenerationMixin().prepare_attention_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=1,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_DIM,
            num_layers=num_layers,
            kv_dtype=kv_dtype,
            remove_input_padding=False,
            use_gpt_attention_plugin=True,
            paged_kv_cache=False,
        )

        qkv = Tensor(name='qkv',
                      dtype=str_dtype_to_trt(DTYPE_STR),
                      shape=[-1, -1, qkv_dim],
                      dim_range=dict(
                          batch_size=[(1, max_batch_size, max_batch_size)],
                          tokens=[(1, max_input_len, max_input_len)],
                          hidden_size=[qkv_dim],
                      ))

        sequence_length = inputs['sequence_length']
        host_past_key_value_lengths = inputs['host_past_key_value_lengths']
        host_max_attention_window_sizes = inputs['host_max_attention_window_sizes']
        host_sink_token_length = inputs['host_sink_token_length']
        context_lengths = inputs['context_lengths']
        host_request_types = inputs['host_request_types']
        past_key_value = inputs['past_key_value']
        if past_key_value:
            past_key_value = past_key_value[0]
        cache_indirection = inputs['cache_indirection']
        host_runtime_perf_knobs = inputs['host_runtime_perf_knobs']

        outputs = gpt_attention(
            qkv=qkv,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            host_past_key_value_lengths=host_past_key_value_lengths,
            host_max_attention_window_sizes=host_max_attention_window_sizes,
            host_sink_token_length=host_sink_token_length,
            context_lengths=context_lengths,
            cache_indirection=cache_indirection,
            host_request_types=host_request_types,
            layer_idx=0,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            hidden_size_per_head=HEAD_DIM,
            q_scaling=1.0,
            rotary_embedding_dim=0,
            max_context_length=max_input_len,
            host_context_lengths=inputs.get('host_context_lengths'),
            use_cache=False,
            host_runtime_perf_knobs=host_runtime_perf_knobs,
        )

        net._mark_output(outputs[0], 'output',
                         dtype=str_dtype_to_trt(DTYPE_STR))

    engine = builder.build_engine(net, builder_config)
    return engine


def bench_trtllm_prefill():
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    engine = build_trtllm_prefill_engine()
    if engine is None:
        print("ERROR: Failed to build TRT-LLM engine")
        return None

    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)

    qkv_dim = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    qkv = torch.randn(BATCH, PREFILL_SEQ, qkv_dim,
                       dtype=DTYPE_TORCH, device='cuda') * 1e-3
    context_lengths = torch.full([BATCH], PREFILL_SEQ,
                                 dtype=torch.int32, device='cuda')
    host_request_types = torch.zeros([BATCH], dtype=torch.int32, device='cpu')
    host_max_attention_window_sizes = torch.tensor(
        [PREFILL_SEQ], dtype=torch.int32, device='cpu')
    host_sink_token_length = torch.tensor([0], dtype=torch.int32, device='cpu')
    host_runtime_perf_knobs = torch.tensor(
        [-1] * 16, dtype=torch.int64, device='cpu')

    output = torch.zeros(BATCH, PREFILL_SEQ, NUM_HEADS * HEAD_DIM,
                         dtype=DTYPE_TORCH, device='cuda')

    inputs = {
        'qkv': qkv,
        'context_lengths': context_lengths,
        'host_request_types': host_request_types,
        'host_max_attention_window_sizes': host_max_attention_window_sizes,
        'host_sink_token_length': host_sink_token_length,
        'host_runtime_perf_knobs': host_runtime_perf_knobs,
    }
    outputs = {'output': output}

    inputs_info = [
        tensorrt_llm.runtime.TensorInfo(
            name, torch_dtype_to_trt(tensor.dtype), tensor.shape)
        for name, tensor in inputs.items()
    ]
    session.infer_shapes(inputs_info)
    stream = torch.cuda.current_stream()

    # warmup
    for _ in range(WARMUP):
        session.run(inputs=inputs, outputs=outputs,
                    stream=stream.cuda_stream)
        torch.cuda.synchronize()

    # time
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session.run(inputs=inputs, outputs=outputs,
                    stream=stream.cuda_stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return times


# ──────────────────────────────────────────────────────────
# 3. PyTorch SDPA benchmark (baseline)
# ──────────────────────────────────────────────────────────
def bench_sdpa_prefill():
    q = torch.randn(BATCH, NUM_HEADS, PREFILL_SEQ, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')
    k = torch.randn(BATCH, NUM_KV_HEADS, PREFILL_SEQ, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')
    v = torch.randn(BATCH, NUM_KV_HEADS, PREFILL_SEQ, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')

    # expand KV for GQA
    k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)

    for _ in range(WARMUP):
        _ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)
        torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return times


def bench_sdpa_decode():
    q = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')
    k = torch.randn(BATCH, NUM_KV_HEADS, PREFILL_SEQ, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')
    v = torch.randn(BATCH, NUM_KV_HEADS, PREFILL_SEQ, HEAD_DIM,
                     dtype=DTYPE_TORCH, device='cuda')

    k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)

    for _ in range(WARMUP):
        _ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False)  # decode: no causal mask needed
        torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return times


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def report(name, times):
    if times is None:
        print(f"  {name:30s}  FAILED")
        return
    arr = np.array(times)
    print(f"  {name:30s}  {arr.mean():.3f} ms  "
          f"(std {arr.std():.3f}, min {arr.min():.3f}, max {arr.max():.3f})")


if __name__ == "__main__":
    print(f"Config: batch={BATCH}, num_heads={NUM_HEADS}, "
          f"num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, "
          f"seq_len={PREFILL_SEQ}, dtype={DTYPE_STR}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    # --- Prefill (context) ---
    print("=" * 60)
    print(f"PREFILL: Q/K/V seq_len = {PREFILL_SEQ} (causal)")
    print("=" * 60)

    print("\n[1] SDPA (PyTorch cuDNN):")
    sdpa_p = bench_sdpa_prefill()
    report("SDPA prefill", sdpa_p)

    print("\n[2] Open-source FA2 v2.8.3:")
    fa2_p = bench_fa2_prefill()
    report("FA2 prefill", fa2_p)

    print("\n[3] TRT-LLM FMHA (SM 8.7 cubin):")
    try:
        trt_p = bench_trtllm_prefill()
        report("TRT-LLM FMHA prefill", trt_p)
    except Exception as e:
        print(f"  TRT-LLM FMHA prefill FAILED: {e}")
        trt_p = None

    # --- Decode ---
    print()
    print("=" * 60)
    print(f"DECODE: Q seq_len=1, KV seq_len={PREFILL_SEQ}")
    print("=" * 60)

    print("\n[1] SDPA (PyTorch cuDNN):")
    sdpa_d = bench_sdpa_decode()
    report("SDPA decode", sdpa_d)

    print("\n[2] Open-source FA2 v2.8.3:")
    fa2_d = bench_fa2_decode()
    report("FA2 decode", fa2_d)

    # TRT-LLM decode requires KV cache engine - skip for now
    print("\n[3] TRT-LLM FMHA decode: (requires KV cache engine, skipped)")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if sdpa_p and fa2_p:
        print(f"  Prefill: FA2 vs SDPA = {np.mean(fa2_p)/np.mean(sdpa_p):.2f}x")
    if trt_p and fa2_p:
        print(f"  Prefill: TRT-LLM vs FA2 = {np.mean(trt_p)/np.mean(fa2_p):.2f}x")
    if trt_p and sdpa_p:
        print(f"  Prefill: TRT-LLM vs SDPA = {np.mean(trt_p)/np.mean(sdpa_p):.2f}x")
    if sdpa_d and fa2_d:
        print(f"  Decode:  FA2 vs SDPA = {np.mean(fa2_d)/np.mean(sdpa_d):.2f}x")
