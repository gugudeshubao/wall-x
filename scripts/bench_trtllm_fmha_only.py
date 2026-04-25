#!/usr/bin/env python3
"""
Standalone TRT-LLM FMHA benchmark.
Runs with system Python (where tensorrt_llm is installed).
"""
import time
import sys
from collections import OrderedDict
import torch
import numpy as np

BATCH = 1
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
PREFILL_SEQ = 420
WARMUP = 5
ITERS = 50
DTYPE_STR = "bfloat16"
DTYPE_TORCH = torch.bfloat16

def build_prefill_engine():
    import tensorrt_llm
    from tensorrt_llm import Tensor, str_dtype_to_trt
    from tensorrt_llm.functional import gpt_attention
    from tensorrt_llm.models.generation_mixin import GenerationMixin
    from tensorrt_llm.plugin.plugin import ContextFMHAType

    tensorrt_llm.logger.set_level('error')

    kv_dtype = str_dtype_to_trt(DTYPE_STR)
    qkv_dim = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM

    builder = tensorrt_llm.Builder()
    builder_config = builder.create_builder_config(
        name="fmha_bench", precision=DTYPE_STR)
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    net.plugin_config.gpt_attention_plugin = DTYPE_STR
    net.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    net.plugin_config.remove_input_padding = False

    with tensorrt_llm.net_guard(net):
        inputs = GenerationMixin().prepare_attention_inputs(
            max_batch_size=BATCH,
            max_beam_width=1,
            max_input_len=PREFILL_SEQ,
            max_seq_len=PREFILL_SEQ,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_DIM,
            num_layers=1,
            kv_dtype=kv_dtype,
            remove_input_padding=False,
            use_gpt_attention_plugin=True,
            paged_kv_cache=False,
        )

        qkv = Tensor(name='qkv',
                      dtype=str_dtype_to_trt(DTYPE_STR),
                      shape=[-1, -1, qkv_dim],
                      dim_range=OrderedDict(
                          batch_size=[(1, BATCH, BATCH)],
                          tokens=[(1, PREFILL_SEQ, PREFILL_SEQ)],
                          hidden_size=[qkv_dim],
                      ))

        outputs = gpt_attention(
            qkv=qkv,
            past_key_value=inputs['past_key_value'][0] if inputs['past_key_value'] else None,
            sequence_length=inputs['sequence_length'],
            host_past_key_value_lengths=inputs['host_past_key_value_lengths'],
            host_max_attention_window_sizes=inputs['host_max_attention_window_sizes'],
            host_sink_token_length=inputs['host_sink_token_length'],
            context_lengths=inputs['context_lengths'],
            cache_indirection=inputs['cache_indirection'],
            host_request_types=inputs['host_request_types'],
            layer_idx=0,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            hidden_size_per_head=HEAD_DIM,
            q_scaling=1.0,
            rotary_embedding_dim=0,
            max_context_length=PREFILL_SEQ,
            host_context_lengths=inputs.get('host_context_lengths'),
            use_cache=False,
            host_runtime_perf_knobs=inputs['host_runtime_perf_knobs'],
        )

        net._mark_output(outputs[0], 'output',
                         dtype=str_dtype_to_trt(DTYPE_STR))

    return builder.build_engine(net, builder_config)


def main():
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    print(f"TRT-LLM version: {tensorrt_llm.__version__}")
    print(f"Config: batch={BATCH}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, "
          f"head_dim={HEAD_DIM}, seq={PREFILL_SEQ}, dtype={DTYPE_STR}")
    print()

    print("Building TRT engine with FMHA plugin...")
    t0 = time.time()
    engine = build_prefill_engine()
    print(f"Engine built in {time.time()-t0:.1f}s")

    if engine is None:
        print("ERROR: engine build failed")
        sys.exit(1)

    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)

    qkv_dim = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    qkv = torch.randn(BATCH, PREFILL_SEQ, qkv_dim,
                       dtype=DTYPE_TORCH, device='cuda') * 1e-3
    context_lengths = torch.full([BATCH], PREFILL_SEQ,
                                 dtype=torch.int32, device='cuda')
    host_request_types = torch.zeros([BATCH], dtype=torch.int32, device='cpu')
    host_max_attn = torch.tensor([PREFILL_SEQ], dtype=torch.int32, device='cpu')
    host_sink = torch.tensor([0], dtype=torch.int32, device='cpu')
    host_perf = torch.tensor([-1]*16, dtype=torch.int64, device='cpu')

    output = torch.zeros(BATCH, PREFILL_SEQ, NUM_HEADS * HEAD_DIM,
                         dtype=DTYPE_TORCH, device='cuda')

    inputs = {
        'qkv': qkv,
        'context_lengths': context_lengths,
        'host_request_types': host_request_types,
        'host_max_attention_window_sizes': host_max_attn,
        'host_sink_token_length': host_sink,
        'host_runtime_perf_knobs': host_perf,
    }
    outputs_dict = {'output': output}

    inputs_info = [
        tensorrt_llm.runtime.TensorInfo(
            name, torch_dtype_to_trt(t.dtype), t.shape)
        for name, t in inputs.items()
    ]
    session.infer_shapes(inputs_info)
    stream = torch.cuda.current_stream()

    # warmup
    print("Warming up...")
    for _ in range(WARMUP):
        session.run(inputs=inputs, outputs=outputs_dict,
                    stream=stream.cuda_stream)
        torch.cuda.synchronize()

    # benchmark
    print(f"Running {ITERS} iterations...")
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session.run(inputs=inputs, outputs=outputs_dict,
                    stream=stream.cuda_stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print(f"\nTRT-LLM FMHA prefill:")
    print(f"  Mean: {arr.mean():.3f} ms")
    print(f"  Std:  {arr.std():.3f} ms")
    print(f"  Min:  {arr.min():.3f} ms")
    print(f"  Max:  {arr.max():.3f} ms")

    # Compare with previous results
    print(f"\nFor reference (from venv run):")
    print(f"  SDPA prefill:  0.127 ms")
    print(f"  FA2 prefill:   0.306 ms")
    print(f"  TRT-LLM FMHA:  {arr.mean():.3f} ms")
    print(f"  TRT-LLM vs FA2:  {arr.mean()/0.306:.2f}x")
    print(f"  TRT-LLM vs SDPA: {arr.mean()/0.127:.2f}x")


if __name__ == "__main__":
    main()
