#!/usr/bin/env python3
"""
Export VQA-specialized decoder reference tensors for TensorRT Phase A.

This script does NOT build TensorRT engines. It prepares the reference inputs
and outputs needed by the Phase A decoder-only route:

  - inputs_embeds after token embedding + image embedding scatter
  - position_ids / rope_deltas
  - moe_token_types (all zeros for VQA-specialized path)
  - prefill last-token logits
  - greedy generated tokens for a short decode run

The goal is to lock down the Python-side reference interface before building
the TensorRT decoder engine.
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
def build_inputs_embeds_and_positions(model, inputs):
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)
    attention_mask = inputs.get("attention_mask", None)

    inputs_embeds = model.model.embed_tokens(input_ids)

    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    position_ids, rope_deltas = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=attention_mask,
    )

    moe_token_types = torch.zeros_like(input_ids)
    return inputs_embeds, position_ids, rope_deltas, moe_token_types, attention_mask


@torch.no_grad()
def run_prefill_reference(model, inputs, moe_token_types):
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        pixel_values=inputs.get("pixel_values", None),
        image_grid_thw=inputs.get("image_grid_thw", None),
        moe_token_types=moe_token_types,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    logits = outputs.logits
    last_logits = logits[:, -1, :].contiguous()
    return outputs, last_logits


@torch.no_grad()
def run_first_decode_step(
    model,
    base_input_ids,
    base_attention_mask,
    next_token_id,
    past_key_values,
    current_pos,
):
    device = next_token_id.device
    full_input_ids = torch.cat([base_input_ids, next_token_id.view(1, 1)], dim=1)
    full_attention_mask = torch.cat(
        [base_attention_mask, torch.ones((base_attention_mask.shape[0], 1), dtype=base_attention_mask.dtype, device=device)],
        dim=1,
    )
    cache_position = torch.tensor([current_pos], dtype=torch.long, device=device)
    full_moe_types = torch.zeros_like(full_input_ids)

    model_inputs = model.prepare_inputs_for_generation(
        input_ids=full_input_ids,
        attention_mask=full_attention_mask,
        past_key_values=past_key_values,
        moe_token_types=full_moe_types,
        cache_position=cache_position,
        use_cache=True,
    )
    decode_pos = torch.full(
        (3, 1, 1),
        current_pos,
        dtype=torch.long,
        device=device,
    )
    model_inputs["position_ids"] = decode_pos
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    logits = outputs.logits[:, -1, :].contiguous()
    decode_input_ids = model_inputs["input_ids"]
    decode_attention_mask = model_inputs["attention_mask"]
    decode_moe_types = model_inputs["moe_token_types"]
    token_embed = model.model.embed_tokens(decode_input_ids).contiguous()
    return (
        decode_input_ids,
        decode_attention_mask,
        token_embed,
        decode_pos,
        decode_moe_types,
        outputs,
        logits,
    )


def dump_past_kv_tensors(past_key_values):
    dumped = {}
    if hasattr(past_key_values, "key_cache"):
        for i, (k, v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
            dumped[f"past_key_{i}"] = k.detach().cpu()
            dumped[f"past_value_{i}"] = v.detach().cpu()
    elif hasattr(past_key_values, "layers"):
        for i, layer in enumerate(past_key_values.layers):
            dumped[f"past_key_{i}"] = layer.keys.detach().cpu()
            dumped[f"past_value_{i}"] = layer.values.detach().cpu()
    else:
        raise RuntimeError("Unsupported past_key_values structure")
    return dumped


def prepare_mrope_ready(cos: torch.Tensor, sin: torch.Tensor, mrope_section):
    mrope_section = mrope_section * 2
    cos_m = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(1)
    sin_m = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(1)
    return cos_m, sin_m


@torch.no_grad()
def run_greedy_reference(model, inputs, moe_token_types, max_new_tokens: int):
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        pixel_values=inputs.get("pixel_values", None),
        image_grid_thw=inputs.get("image_grid_thw", None),
        moe_token_types=moe_token_types,
        max_new_tokens=max_new_tokens,
        eos_token_id=[model.processor.tokenizer.eos_token_id],
        use_cache=True,
        pad_token_id=model.processor.tokenizer.pad_token_id,
        do_sample=False,
    )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:].contiguous()
    answer = model.processor.decode(gen_ids, skip_special_tokens=True)
    return gen_ids, answer


def main():
    parser = argparse.ArgumentParser(
        description="Export VQA-specialized decoder reference tensors"
    )
    parser.add_argument("--model-path", required=True, help="Path to wall-x model")
    parser.add_argument("--image", required=True, help="Path to a test image")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--question",
        default="Describe what you see in this image.",
        help="VQA question",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Greedy reference decode length",
    )
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

    inputs_embeds, position_ids, rope_deltas, moe_token_types, attention_mask = (
        build_inputs_embeds_and_positions(model, inputs)
    )

    print(f"[SHAPE] input_ids      {list(inputs['input_ids'].shape)}")
    print(f"[SHAPE] pixel_values   {list(inputs['pixel_values'].shape)}")
    print(f"[SHAPE] inputs_embeds  {list(inputs_embeds.shape)}")
    print(f"[SHAPE] position_ids   {list(position_ids.shape)}")

    prefill_outputs, last_logits = run_prefill_reference(model, inputs, moe_token_types)
    prefill_kv_tensors = dump_past_kv_tensors(prefill_outputs.past_key_values)
    gen_ids, answer = run_greedy_reference(
        model, inputs, moe_token_types, args.max_new_tokens
    )
    next_token_id = last_logits.argmax(dim=-1)
    current_pos = int(position_ids.max().item() + 1)
    (
        decode_input_ids,
        decode_attention_mask,
        decode_inputs_embeds,
        decode_position_ids,
        decode_moe_types,
        decode_outputs,
        decode_last_logits,
    ) = run_first_decode_step(
        model,
        inputs["input_ids"],
        attention_mask,
        next_token_id,
        prefill_outputs.past_key_values,
        current_pos,
    )
    decode_cos, decode_sin = model.model.rotary_emb(
        decode_inputs_embeds, decode_position_ids
    )
    decode_cos_m, decode_sin_m = prepare_mrope_ready(
        decode_cos, decode_sin, model.config.rope_scaling["mrope_section"]
    )
    present_kv_tensors = dump_past_kv_tensors(decode_outputs.past_key_values)

    tensors = {
        "input_ids": inputs["input_ids"].cpu(),
        "attention_mask": attention_mask.cpu()
        if attention_mask is not None
        else torch.empty(0, dtype=torch.int64),
        "pixel_values": inputs["pixel_values"].cpu(),
        "image_grid_thw": inputs["image_grid_thw"].cpu(),
        "inputs_embeds": inputs_embeds.cpu(),
        "position_ids": position_ids.cpu(),
        "rope_deltas": rope_deltas.cpu(),
        "moe_token_types": moe_token_types.cpu(),
        "prefill_last_logits": last_logits.float().cpu(),
        "greedy_token_ids": gen_ids.cpu(),
        "decode_input_ids": decode_input_ids.cpu(),
        "decode_attention_mask": decode_attention_mask.cpu(),
        "decode_inputs_embeds": decode_inputs_embeds.cpu(),
        "decode_position_ids": decode_position_ids.cpu(),
        "decode_moe_token_types": decode_moe_types.cpu(),
        "decode_cos_mrope": decode_cos_m.cpu(),
        "decode_sin_mrope": decode_sin_m.cpu(),
        "decode_last_logits": decode_last_logits.float().cpu(),
    }
    tensors.update({f"prefill_{k}": v for k, v in prefill_kv_tensors.items()})
    tensors.update({f"present_{k}": v for k, v in present_kv_tensors.items()})
    save_file(tensors, str(out_dir / "decoder_reference.safetensors"))

    manifest = {
        "model_path": args.model_path,
        "image": str(args.image),
        "question": args.question,
        "prompt_text": prompt_text,
        "max_new_tokens": args.max_new_tokens,
        "input_seq_len": int(inputs["input_ids"].shape[1]),
        "generated_token_ids": gen_ids.tolist(),
        "generated_text": answer,
        "first_decode_input_id": int(next_token_id.item()),
        "first_decode_position": current_pos,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {out_dir / 'decoder_reference.safetensors'}")
    print(f"[SAVE] {out_dir / 'manifest.json'}")
    print(f"[TEXT] {answer}")


if __name__ == "__main__":
    main()
