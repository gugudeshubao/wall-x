#!/usr/bin/env python3
"""
Export step-by-step VQA decode trace from the original wall-x model.

Purpose:
  - capture per-step decode logits and chosen token ids
  - diagnose where TRT VQA-specialized runner diverges
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file


def build_vqa_inputs(model, image, question: str):
    processor = model.processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    return text, inputs


@torch.no_grad()
def run_decode_trace(model, inputs, max_new_tokens: int):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)
    moe_token_types = torch.zeros_like(input_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        moe_token_types=moe_token_types,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    past_key_values = outputs.past_key_values
    logits_gpu = outputs.logits[:, -1, :].float()
    next_token = logits_gpu.argmax(dim=-1)

    traces = []
    traces.append(
        {
            "step": 0,
            "token_id": int(next_token.item()),
            "logits": logits_gpu.cpu(),
        }
    )

    current_pos = int(outputs.logits.shape[1])  # next absolute position

    for step in range(1, max_new_tokens):
        decode_input_ids = next_token.view(1, 1)
        decode_attention_mask = torch.ones_like(decode_input_ids)
        decode_pos = torch.full((3, 1, 1), current_pos, dtype=torch.long, device=decode_input_ids.device)
        decode_moe_types = torch.zeros_like(decode_input_ids)
        cache_position = torch.tensor([current_pos], dtype=torch.long, device=decode_input_ids.device)

        outputs = model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            position_ids=decode_pos,
            cache_position=cache_position,
            past_key_values=past_key_values,
            moe_token_types=decode_moe_types,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        past_key_values = outputs.past_key_values
        logits_gpu = outputs.logits[:, -1, :].float()
        next_token = logits_gpu.argmax(dim=-1)
        traces.append(
            {
                "step": step,
                "token_id": int(next_token.item()),
                "logits": logits_gpu.cpu(),
            }
        )
        current_pos += 1

    return traces


def main():
    parser = argparse.ArgumentParser(description="Export original wall-x step-by-step VQA decode trace")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question", default="Describe what you see in this image.")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    print(f"[LOAD] model from {args.model_path}")
    t0 = time.perf_counter()
    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cuda").bfloat16()
    print(f"[LOAD] done in {(time.perf_counter() - t0):.1f}s")

    image = Image.open(args.image).convert("RGB")
    prompt_text, inputs = build_vqa_inputs(model, image, args.question)
    traces = run_decode_trace(model, inputs, args.max_new_tokens)

    tensors = {}
    manifest = {
        "model_path": args.model_path,
        "image": str(args.image),
        "question": args.question,
        "prompt_text": prompt_text,
        "steps": [],
    }
    for tr in traces:
        step = tr["step"]
        tensors[f"step_logits_{step}"] = tr["logits"]
        manifest["steps"].append(
            {"step": step, "token_id": tr["token_id"]}
        )

    save_file(tensors, str(out_dir / "decode_trace.safetensors"))
    with open(out_dir / "decode_trace_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {out_dir / 'decode_trace.safetensors'}")
    print(f"[SAVE] {out_dir / 'decode_trace_manifest.json'}")
    print("[TOKENS]", [s["token_id"] for s in manifest["steps"]])


if __name__ == "__main__":
    main()
