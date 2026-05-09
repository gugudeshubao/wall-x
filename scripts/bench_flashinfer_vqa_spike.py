#!/usr/bin/env python3
"""Benchmark a Python-side FlashInfer spike on wall-x VQA.

This script does not modify the model source tree. It monkey patches
`Qwen2_5_VLSdpaAttention.forward()` at runtime and replaces the text-decoder
attention with FlashInfer when the shape/semantics match the common VQA path:

- batch size = 1
- attention_mask is either None or all-ones
- q_len > 1: prefill -> FlashInfer prefill
- q_len == 1: decode -> FlashInfer decode

Everything else falls back to the original SDPA implementation.
"""

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def install_flashinfer_patch(flashinfer_root: str, patch_mode: str):
    if flashinfer_root and flashinfer_root not in sys.path:
        sys.path.insert(0, flashinfer_root)

    import flashinfer  # noqa: F401
    from wall_x.model.qwen2_5_based import modeling_qwen2_5_vl as qwen_vl

    original_forward = qwen_vl.Qwen2_5_VLSdpaAttention.forward

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
    ):
        import flashinfer

        if output_attentions:
            return original_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        if bsz != 1 or hidden_states.device.type != "cuda":
            return original_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                return original_forward(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            if torch.any(attention_mask != 1).item():
                return original_forward(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = qwen_vl.apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            if use_cache:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                past_key_states, past_value_states = past_key_value[self.layer_idx]
                key_states = torch.cat([past_key_states, key_states], dim=-2)
                value_states = torch.cat([past_value_states, value_states], dim=-2)

        use_flashinfer = (
            (patch_mode == "flashinfer-both")
            or (patch_mode == "flashinfer-prefill" and q_len > 1)
            or (patch_mode == "flashinfer-decode" and q_len == 1)
        )
        if not use_flashinfer:
            return original_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        # FlashInfer natively supports GQA, so do not repeat KV heads.
        key_fi = key_states[0].permute(1, 0, 2).contiguous()
        value_fi = value_states[0].permute(1, 0, 2).contiguous()

        if q_len == 1:
            query_fi = query_states[0, :, 0, :].contiguous()
            attn_output = flashinfer.decode.single_decode_with_kv_cache(
                query_fi,
                key_fi,
                value_fi,
            )
            attn_output = attn_output.view(1, 1, self.hidden_size)
        else:
            query_fi = query_states[0].permute(1, 0, 2).contiguous()
            attn_output = flashinfer.prefill.single_prefill_with_kv_cache(
                query_fi,
                key_fi,
                value_fi,
                causal=True,
                backend="auto",
            )
            attn_output = attn_output.view(1, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    qwen_vl.Qwen2_5_VLSdpaAttention.forward = patched_forward
    return original_forward, qwen_vl.Qwen2_5_VLSdpaAttention


def uninstall_flashinfer_patch(original_forward, cls):
    cls.forward = original_forward


def load_model(model_path, mode, flashinfer_root=None, device="cuda"):
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )
    from safetensors.torch import load_file

    original_forward = None
    patched_cls = None
    if mode != "sdpa":
        original_forward, patched_cls = install_flashinfer_patch(flashinfer_root, mode)

    t0 = time.time()
    config_path = os.path.join(model_path, "config.json")
    model_config = Qwen2_5_VLMoEForAction.config_class.from_pretrained(config_path)
    model_config._attn_implementation = "sdpa"

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = Qwen2_5_VLMoEForAction(model_config, processor=processor)
    model.resize_token_embeddings(len(processor.tokenizer))

    state_dict = {}
    for f in glob.glob(os.path.join(model_path, "*.safetensors")):
        state_dict.update(load_file(f, device="cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device, dtype=torch.bfloat16)

    load_time = time.time() - t0
    return model, processor, load_time, original_forward, patched_cls


def unload_model(model, processor, original_forward, patched_cls):
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if original_forward is not None and patched_cls is not None:
        uninstall_flashinfer_patch(original_forward, patched_cls)
    time.sleep(1)


def prepare_input(processor, image_path, question, device="cuda"):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)


def benchmark(model, inputs, max_new_tokens, warmup, runs):
    input_len = inputs["input_ids"].shape[1]

    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()

    latencies = []
    outputs = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        outputs.append(generated[0][input_len:].tolist())

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "avg_tok_s": len(outputs[0]) / (float(np.mean(latencies)) / 1000.0),
        "tokens": outputs[0],
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def teacher_forced_decode(model, inputs, teacher_tokens: torch.Tensor):
    """Run generation under teacher forcing and dump step-wise ranking stats."""
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    attention_mask = torch.ones_like(input_ids)
    moe_token_types = torch.zeros_like(input_ids)
    base_seq_len = input_ids.shape[1]
    base_cache_position = torch.arange(base_seq_len, device=input_ids.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            moe_token_types=moe_token_types,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
            cache_position=base_cache_position,
        )

    past_key_values = outputs.past_key_values
    hidden = outputs.hidden_states[-1][:, -1, :]
    logits = model.lm_head(hidden)

    rows = []
    for step in range(teacher_tokens.numel()):
        teacher_id = int(teacher_tokens[step].item())
        step_logits = logits[0].float().cpu()
        top_id = int(step_logits.argmax().item())
        teacher_logit = float(step_logits[teacher_id].item())
        top_logit = float(step_logits[top_id].item())
        teacher_rank = int((step_logits > step_logits[teacher_id]).sum().item() + 1)
        rows.append((step, teacher_id, top_id, teacher_rank, teacher_logit, top_logit))

        if step == teacher_tokens.numel() - 1:
            break

        next_token = teacher_tokens[step].view(1, 1).to(input_ids.device)
        current_total_len = base_seq_len + step + 1
        next_mask = torch.ones(
            (1, current_total_len), device=input_ids.device, dtype=input_ids.dtype
        )
        next_cache_position = torch.tensor(
            [base_seq_len + step], device=input_ids.device, dtype=torch.long
        )
        with torch.no_grad():
            decode_outputs = model(
                input_ids=next_token,
                attention_mask=next_mask,
                pixel_values=None,
                image_grid_thw=None,
                moe_token_types=torch.zeros_like(next_token),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
                cache_position=next_cache_position,
            )
        past_key_values = decode_outputs.past_key_values
        hidden = decode_outputs.hidden_states[-1][:, -1, :]
        logits = model.lm_head(hidden)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashInfer VQA spike on wall-x")
    parser.add_argument("--model_path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--image", default="/data/wy/wall-x/test_images/fruits_on_table.png")
    parser.add_argument("--question", default="Describe what you see in this image.")
    parser.add_argument("--flashinfer_root", default="/tmp/flashinfer_orin_nodeps")
    parser.add_argument(
        "--mode",
        choices=[
            "sdpa",
            "flashinfer-prefill",
            "flashinfer-decode",
            "flashinfer-both",
            "both",
        ],
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--teacher_tokens", default=None)
    args = parser.parse_args()

    modes = ["sdpa", "flashinfer-both"] if args.mode == "both" else [args.mode]
    all_stats = {}

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for mode in modes:
        print(f"\n{'='*72}")
        print(f"Mode: {mode}")
        print(f"{'='*72}")
        model, processor, load_time, original_forward, patched_cls = load_model(
            args.model_path, mode, args.flashinfer_root
        )
        inputs = prepare_input(processor, args.image, args.question)
        print(f"Load time: {load_time:.1f}s, input_len={inputs['input_ids'].shape[1]}")
        if args.teacher_tokens:
            teacher_tokens = torch.load(args.teacher_tokens, map_location="cpu").to(torch.long)
            rows = teacher_forced_decode(model, inputs, teacher_tokens.to(inputs["input_ids"].device))
            for step, teacher_id, top_id, teacher_rank, teacher_logit, top_logit in rows:
                print(
                    f"step={step} teacher={teacher_id} top={top_id} "
                    f"teacher_rank={teacher_rank} teacher_logit={teacher_logit:.4f} top_logit={top_logit:.4f}"
                )
            all_stats[mode] = {"teacher_rows": rows}
            unload_model(model, processor, original_forward, patched_cls)
            continue
        stats = benchmark(model, inputs, args.max_new_tokens, args.warmup, args.runs)
        print(f"Mean latency: {stats['mean_ms']:.1f} ms (std {stats['std_ms']:.1f})")
        print(f"Min/Max: {stats['min_ms']:.1f} / {stats['max_ms']:.1f} ms")
        print(f"Throughput: {stats['avg_tok_s']:.2f} tok/s")
        print(f"Peak GPU: {stats['peak_gpu_gb']:.2f} GB")
        print(f"Tokens: {stats['tokens']}")
        all_stats[mode] = stats
        unload_model(model, processor, original_forward, patched_cls)

    if len(all_stats) == 2:
        base = all_stats["sdpa"]
        fi = all_stats["flashinfer-both"]
        print(f"\n{'#'*72}")
        print("FlashInfer Spike vs SDPA")
        print(f"{'#'*72}")
        print(f"SDPA mean:       {base['mean_ms']:.1f} ms")
        print(f"FlashInfer mean: {fi['mean_ms']:.1f} ms")
        print(f"Speedup:         {base['mean_ms'] / fi['mean_ms']:.2f}x")
        print(f"SDPA tok/s:      {base['avg_tok_s']:.2f}")
        print(f"FlashInfer tok/s:{fi['avg_tok_s']:.2f}")
        print(f"Token match:     {base['tokens'] == fi['tokens']}")


if __name__ == "__main__":
    main()
