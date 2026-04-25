#!/usr/bin/env python3
"""Lightweight VQA profiling script for nsys/ncu capture.

Usage with nsys:
    nsys profile -o vqa_nsys python profile_vqa.py --model_path /path/to/model --image /path/to/image.png

Usage with ncu (single kernel pass, very slow):
    ncu --set full --target-processes all -o vqa_ncu python profile_vqa.py --model_path /path/to/model --image /path/to/image.png --ncu_mode

The --ncu_mode flag skips warmup and only runs generate once with max_new_tokens=16
to keep ncu capture time manageable.
"""
import os, sys, time, argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image", required=True, help="Path to a single test image")
    parser.add_argument("--question", default="Describe what you see in this image.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--ncu_mode", action="store_true",
                        help="NCU mode: no warmup, max_new_tokens=16, single run")
    parser.add_argument("--attn", default="flash_attention_2",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention implementation (default: flash_attention_2)")
    args = parser.parse_args()

    if args.ncu_mode:
        args.max_new_tokens = 16  # keep ncu capture small

    # ---- load model ----
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
    from PIL import Image
    from safetensors.torch import load_file
    import glob

    print(f"Loading model from {args.model_path} (attn={args.attn}) ...")
    t0 = time.time()

    # Load config and set attention implementation
    config_path = os.path.join(args.model_path, "config.json")
    model_config = Qwen2_5_VLMoEForAction.config_class.from_pretrained(config_path)
    model_config._attn_implementation = args.attn

    # Load processor and construct model
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    model = Qwen2_5_VLMoEForAction(model_config, processor=processor)
    model.resize_token_embeddings(len(processor.tokenizer))

    # Load weights
    safetensor_files = glob.glob(os.path.join(args.model_path, "*.safetensors"))
    state_dict = {}
    for f in safetensor_files:
        sd = load_file(f, device="cpu")
        state_dict.update(sd)
    model.load_state_dict(state_dict, strict=False)

    model.eval().to("cuda", dtype=torch.bfloat16)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s, peak GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"Attention: {model_config._attn_implementation}")

    # ---- prepare input ----
    pil_img = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image} ({pil_img.size[0]}x{pil_img.size[1]})")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[pil_img],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    input_len = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_len}, max_new_tokens: {args.max_new_tokens}")

    # ---- warmup (skip in ncu mode) ----
    if not args.ncu_mode:
        print("Warmup run ...")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        torch.cuda.synchronize()
        print("Warmup done.")

    # ---- profiled run ----
    print(f"\n{'='*60}")
    print(f"Starting profiled inference ({'NCU mode' if args.ncu_mode else 'normal'}) ...")

    # CUDA range marker for nsys
    if hasattr(torch.cuda, 'nvtx'):
        torch.cuda.nvtx.range_push("vqa_generate")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    if hasattr(torch.cuda, 'nvtx'):
        torch.cuda.nvtx.range_pop()

    elapsed_ms = (t1 - t0) * 1000
    out_tokens = generated.shape[1] - input_len
    tok_per_s = out_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

    # Decode output
    output_ids = generated[0][input_len:]
    output_text = processor.decode(output_ids, skip_special_tokens=True)

    print(f"{'='*60}")
    print(f"Latency:     {elapsed_ms:.1f} ms")
    print(f"Tokens:      {out_tokens}")
    print(f"Tokens/sec:  {tok_per_s:.1f}")
    print(f"Peak GPU:    {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"Answer:      {output_text[:200]}")
    print(f"{'='*60}")
    print("Done. Profile data captured.")

if __name__ == "__main__":
    main()
