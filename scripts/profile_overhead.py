#!/usr/bin/env python3
"""Profile time breakdown: preprocessing, prefill, decode overhead."""
import os, sys, time, glob, traceback
import torch
import numpy as np


def load_model(model_path, device="cuda"):
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
    from safetensors.torch import load_file

    config_path = os.path.join(model_path, "config.json")
    model_config = Qwen2_5_VLMoEForAction.config_class.from_pretrained(config_path)
    model_config._attn_implementation = "flash_attention_2"

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = Qwen2_5_VLMoEForAction(model_config, processor=processor)
    model.resize_token_embeddings(len(processor.tokenizer))

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    state_dict = {}
    for f in safetensor_files:
        sd = load_file(f, device="cpu")
        state_dict.update(sd)
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model, processor


def main():
    model_path = "/data/wy/models/wall-oss-flow"
    image_path = "/data/wy/wall-x/test_images/fruits_on_table.png"
    question = "Describe what you see in this image."
    max_new_tokens = 64

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        model, processor = load_model(model_path)
    except Exception as e:
        print(f"Load failed: {e}")
        traceback.print_exc()
        return

    from PIL import Image

    # Warmup
    print("\nWarmup...")
    pil_img = Image.open(image_path).convert("RGB").resize((448, 448))
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)
    torch.cuda.synchronize()
    print("Warmup done.")

    results = {}

    # 1. Image load + resize
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        pil_img = Image.open(image_path).convert("RGB").resize((448, 448))
    t1 = time.perf_counter()
    results["image_load_resize_ms"] = (t1 - t0) / 10 * 1000

    # 2. Processor (tokenize + vision encode)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    results["tokenize_process_ms"] = (t1 - t0) / 10 * 1000

    input_len = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_len}")

    # 3. generate() with different token counts to isolate prefill vs decode
    token_counts = [1, 4, 16, 64]
    gen_times = {}
    for nt in token_counts:
        times = []
        for _ in range(3):
            pil_img = Image.open(image_path).convert("RGB").resize((448, 448))
            messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": question}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to("cuda")

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=nt, do_sample=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            actual_tokens = generated.shape[1] - input_len
            times.append(((t1 - t0) * 1000, actual_tokens))

        avg_ms = np.mean([t[0] for t in times])
        avg_tok = np.mean([t[1] for t in times])
        gen_times[nt] = (avg_ms, avg_tok)
        print(f"  generate(max={nt:>3}): {avg_ms:>8.1f} ms, actual tokens: {avg_tok:.0f}")

    # 4. Kernel launch overhead
    a = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10000):
        a = a + 1
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    results["kernel_launch_us"] = (t1 - t0) / 10000 * 1e6

    # 5. Argmax overhead (vocab=152064)
    logits = torch.randn(1, 1, 152064, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        _ = logits.argmax(dim=-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    results["argmax_vocab_us"] = (t1 - t0) / 200 * 1e6

    # 6. Compute decode cost per token via linear regression
    # generate(N) = prefill + N * decode_per_token + overhead
    if len(gen_times) >= 2:
        xs = np.array([gen_times[k][1] for k in token_counts if gen_times[k][1] > 0])
        ys = np.array([gen_times[k][0] for k in token_counts if gen_times[k][1] > 0])
        if len(xs) >= 2:
            # Linear fit: y = a*x + b, where a = decode_per_token, b = prefill + overhead
            coeffs = np.polyfit(xs, ys, 1)
            decode_per_token = coeffs[0]
            prefill_plus_overhead = coeffs[1]
            results["decode_per_token_ms"] = decode_per_token
            results["prefill_overhead_ms"] = prefill_plus_overhead

    # Print results
    print("\n" + "=" * 65)
    print("  Overhead Breakdown")
    print("=" * 65)
    print(f"  Image load+resize:       {results['image_load_resize_ms']:>8.1f} ms")
    print(f"  Tokenize+process:        {results['tokenize_process_ms']:>8.1f} ms")
    print(f"  Kernel launch overhead:  {results['kernel_launch_us']:>8.1f} us/launch")
    print(f"  Argmax (vocab=152K):     {results['argmax_vocab_us']:>8.1f} us/call")

    print(f"\n  generate() scaling (greedy, do_sample=False):")
    for nt in token_counts:
        ms, tok = gen_times[nt]
        tok_s = tok / (ms / 1000) if ms > 0 else 0
        print(f"    max_tokens={nt:>3}: {ms:>8.1f} ms  ({tok:.0f} tokens, {tok_s:.1f} tok/s)")

    if "decode_per_token_ms" in results:
        dpt = results["decode_per_token_ms"]
        pfo = results["prefill_overhead_ms"]
        print(f"\n  Linear regression: generate(N) = {pfo:.1f} + {dpt:.1f} * N")
        print(f"  Prefill + overhead:      {pfo:>8.1f} ms")
        print(f"  Decode per token:        {dpt:>8.1f} ms")

        # Budget for 64 tokens
        total_64 = pfo + dpt * 64
        print(f"\n  === Budget for 64 tokens ===")
        print(f"  Prefill + overhead:      {pfo:>8.1f} ms  ({pfo/total_64*100:.1f}%)")
        print(f"  Decode (64 * {dpt:.1f}):     {dpt*64:>8.1f} ms  ({dpt*64/total_64*100:.1f}%)")
        print(f"  Total estimated:         {total_64:>8.1f} ms")

        # Compare with nsys kernel time
        nsys_kernel_ms = 2101.0  # from previous profiling
        print(f"\n  === GPU Utilization ===")
        print(f"  GPU kernel time (nsys):  {nsys_kernel_ms:>8.1f} ms")
        print(f"  Wall clock (64 tok):     {gen_times[64][0]:>8.1f} ms")
        gpu_util = nsys_kernel_ms / gen_times[64][0] * 100
        overhead_ms = gen_times[64][0] - nsys_kernel_ms
        print(f"  GPU utilization:         {gpu_util:>7.1f}%")
        print(f"  Non-kernel overhead:     {overhead_ms:>8.1f} ms ({100-gpu_util:.1f}%)")

        # Per-decode-step breakdown
        nsys_per_decode_kernel = (nsys_kernel_ms - 200) / 64  # rough prefill ~200ms
        decode_overhead_per_step = dpt - nsys_per_decode_kernel
        print(f"\n  === Per Decode Step ===")
        print(f"  Wall clock per step:     {dpt:>8.1f} ms")
        print(f"  GPU kernel per step:     {nsys_per_decode_kernel:>8.1f} ms (est.)")
        print(f"  Overhead per step:       {decode_overhead_per_step:>8.1f} ms ({decode_overhead_per_step/dpt*100:.1f}%)")


if __name__ == "__main__":
    main()
