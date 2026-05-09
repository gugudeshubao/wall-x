#!/usr/bin/env python3
"""
Export INT8 VQA inputs plus vision-debug tensors for one image.

This script patches the bf16 model with INT8 weights, runs the Python INT8
model on a single image, and saves:
  - inputs.safetensors
  - baseline_tokens.pt
  - vision intermediate tensors (after_reorder / block0 / block7 / ...).

Usage:
    python scripts/export_vqa_inputs_int8_debug.py \
        --model-path /data/wy/models/wall-oss-flow \
        --int8-path /data/wy/models/wall-oss-flow-int8-pad-moe \
        --image-dir /data/wy/wall-x/test_images \
        --image-name robot_gripper \
        --output-dir /data/wy/wall-x/vqa_test_inputs_int8_debug
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from wall_x.fusions import ops
try:
    from scripts.bench_int8_vqa import load_int8_weights, patch_model_int8
except ModuleNotFoundError:
    from bench_int8_vqa import load_int8_weights, patch_model_int8


def export_vision_debug(model, pixel_values, image_grid_thw):
    """Run the INT8 vision encoder and export intermediate tensors."""
    vision = model.visual

    with torch.no_grad():
        hidden_states = vision.patch_embed(pixel_values)
        rotary_pos_emb = ops.rot_pos_emb(
            vision.rotary_pos_emb.inv_freq, image_grid_thw, vision.spatial_merge_size
        )
        window_index, cu_window_seqlens = ops.get_window_index(
            grid_thw=image_grid_thw,
            window_size=vision.window_size,
            spatial_merge_size=vision.spatial_merge_size,
            patch_size=vision.patch_size,
            spatial_merge_unit=vision.spatial_merge_unit,
        )

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // vision.spatial_merge_unit, vision.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)
        after_reorder = hidden_states

        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // vision.spatial_merge_unit, vision.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen_full = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        max_seqlen_window = (
            cu_window_seqlens[1:] - cu_window_seqlens[:-1]
        ).max().item()

        after_block0 = None
        after_block7 = None
        after_block15 = None
        after_block23 = None
        block7_input = None
        block7_q = block7_k = block7_v = None
        block7_q_rot = block7_k_rot = None
        block7_norm1 = block7_attn_out = block7_after_attn = None
        block7_norm2 = block7_mlp_out = block7_output = None
        block15_input = None
        block15_q = block15_k = block15_v = None
        block15_q_rot = block15_k_rot = None
        block15_norm1 = block15_attn_out = block15_after_attn = None
        block15_norm2 = block15_mlp_out = block15_output = None
        block23_input = None
        block23_q = block23_k = block23_v = None
        block23_q_rot = block23_k_rot = None
        block23_norm1 = block23_attn_out = block23_after_attn = None
        block23_norm2 = block23_mlp_out = block23_output = None

        for layer_num, blk in enumerate(vision.blocks):
            if layer_num in vision.fullatt_block_indexes:
                cu_now = cu_seqlens
                max_now = max_seqlen_full
            else:
                cu_now = cu_window_seqlens
                max_now = max_seqlen_window

            if layer_num in (7, 15, 23):
                is_block7 = layer_num == 7
                is_block15 = layer_num == 15
                cur_input = hidden_states
                cur_norm1 = blk.norm1(hidden_states)[0]
                seq_length = cur_norm1.shape[0]
                q, k, v = (
                    blk.attn.qkv(cur_norm1)
                    .reshape(seq_length, 3, blk.attn.num_heads, -1)
                    .permute(1, 0, 2, 3)
                    .unbind(0)
                )
                if is_block7:
                    block7_input = cur_input
                    block7_norm1 = cur_norm1
                    block7_q, block7_k, block7_v = q, k, v
                elif is_block15:
                    block15_input = cur_input
                    block15_norm1 = cur_norm1
                    block15_q, block15_k, block15_v = q, k, v
                else:
                    block23_input = cur_input
                    block23_norm1 = cur_norm1
                    block23_q, block23_k, block23_v = q, k, v
                # Vision attention in this model path always uses the 2D vision RoPE helper.
                q_rot, k_rot = q.float(), k.float()
                cos, sin = position_embeddings
                cos = cos.unsqueeze(-2)
                sin = sin.unsqueeze(-2)
                q_rot = (q_rot * cos) + (torch.cat([-q_rot[..., q_rot.shape[-1] // 2 :], q_rot[..., : q_rot.shape[-1] // 2]], dim=-1) * sin)
                k_rot = (k_rot * cos) + (torch.cat([-k_rot[..., k_rot.shape[-1] // 2 :], k_rot[..., : k_rot.shape[-1] // 2]], dim=-1) * sin)
                if is_block7:
                    block7_q_rot = q_rot.to(q.dtype)
                    block7_k_rot = k_rot.to(k.dtype)
                elif is_block15:
                    block15_q_rot = q_rot.to(q.dtype)
                    block15_k_rot = k_rot.to(k.dtype)
                else:
                    block23_q_rot = q_rot.to(q.dtype)
                    block23_k_rot = k_rot.to(k.dtype)
                cur_attn_out = blk.attn(
                    cur_norm1,
                    cu_seqlens=cu_now,
                    max_seqlen=max_now,
                    position_embeddings=position_embeddings,
                )
                cur_after_attn = hidden_states + cur_attn_out
                cur_norm2 = blk.norm2(cur_after_attn)[0]
                cur_mlp_out = blk.mlp(cur_norm2)
                cur_output = cur_after_attn + cur_mlp_out
                if is_block7:
                    block7_attn_out = cur_attn_out
                    block7_after_attn = cur_after_attn
                    block7_norm2 = cur_norm2
                    block7_mlp_out = cur_mlp_out
                    block7_output = cur_output
                elif is_block15:
                    block15_attn_out = cur_attn_out
                    block15_after_attn = cur_after_attn
                    block15_norm2 = cur_norm2
                    block15_mlp_out = cur_mlp_out
                    block15_output = cur_output
                else:
                    block23_attn_out = cur_attn_out
                    block23_after_attn = cur_after_attn
                    block23_norm2 = cur_norm2
                    block23_mlp_out = cur_mlp_out
                    block23_output = cur_output
                hidden_states = cur_output
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_now,
                    max_seqlen=max_now,
                    position_embeddings=position_embeddings,
                )

            if layer_num == 0:
                after_block0 = hidden_states
            if layer_num == 7:
                after_block7 = hidden_states
            if layer_num == 15:
                after_block15 = hidden_states
            if layer_num == 23:
                after_block23 = hidden_states

        pre_merger = hidden_states
        image_embeds = vision.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        image_embeds = image_embeds[reverse_indices, :]

    return {
        "vision_after_reorder": after_reorder.cpu(),
        "vision_block0": after_block0.cpu(),
        "vision_block7": after_block7.cpu(),
        "vision_block7_input": block7_input.cpu(),
        "vision_block7_q": block7_q.cpu(),
        "vision_block7_k": block7_k.cpu(),
        "vision_block7_v": block7_v.cpu(),
        "vision_block7_q_rot": block7_q_rot.cpu(),
        "vision_block7_k_rot": block7_k_rot.cpu(),
        "vision_block7_norm1": block7_norm1.cpu(),
        "vision_block7_attn_out": block7_attn_out.cpu(),
        "vision_block7_after_attn": block7_after_attn.cpu(),
        "vision_block7_norm2": block7_norm2.cpu(),
        "vision_block7_mlp_out": block7_mlp_out.cpu(),
        "vision_block7_output": block7_output.cpu(),
        "vision_block15": after_block15.cpu(),
        "vision_block15_input": block15_input.cpu(),
        "vision_block15_q": block15_q.cpu(),
        "vision_block15_k": block15_k.cpu(),
        "vision_block15_v": block15_v.cpu(),
        "vision_block15_q_rot": block15_q_rot.cpu(),
        "vision_block15_k_rot": block15_k_rot.cpu(),
        "vision_block15_norm1": block15_norm1.cpu(),
        "vision_block15_attn_out": block15_attn_out.cpu(),
        "vision_block15_after_attn": block15_after_attn.cpu(),
        "vision_block15_norm2": block15_norm2.cpu(),
        "vision_block15_mlp_out": block15_mlp_out.cpu(),
        "vision_block15_output": block15_output.cpu(),
        "vision_block23": after_block23.cpu(),
        "vision_block23_input": block23_input.cpu(),
        "vision_block23_q": block23_q.cpu(),
        "vision_block23_k": block23_k.cpu(),
        "vision_block23_v": block23_v.cpu(),
        "vision_block23_q_rot": block23_q_rot.cpu(),
        "vision_block23_k_rot": block23_k_rot.cpu(),
        "vision_block23_norm1": block23_norm1.cpu(),
        "vision_block23_attn_out": block23_attn_out.cpu(),
        "vision_block23_after_attn": block23_after_attn.cpu(),
        "vision_block23_norm2": block23_norm2.cpu(),
        "vision_block23_mlp_out": block23_mlp_out.cpu(),
        "vision_block23_output": block23_output.cpu(),
        "vision_pre_merger": pre_merger.cpu(),
        "image_embeds": image_embeds.cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Export INT8 VQA debug inputs")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--int8-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question", default="Describe what you see in this image.")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

    print(f"Loading bf16 model from {args.model_path} ...")
    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cuda").bfloat16()
    processor = model.processor

    print(f"Patching with INT8 weights from {args.int8_path} ...")
    int8_weights = load_int8_weights(args.int8_path, device="cuda")
    patch_model_int8(model, int8_weights)
    del int8_weights

    img_path = Path(args.image_dir) / f"{args.image_name}.png"
    if not img_path.exists():
        for ext in [".jpg", ".jpeg", ".webp", ".bmp"]:
            cand = Path(args.image_dir) / f"{args.image_name}{ext}"
            if cand.exists():
                img_path = cand
                break
    if not img_path.exists():
        raise FileNotFoundError(f"image not found for {args.image_name} in {args.image_dir}")

    image = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": args.question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")

    save_dir = out_dir / args.image_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_tensors = {
        "input_ids": inputs["input_ids"].cpu(),
        "pixel_values": inputs["pixel_values"].cpu(),
        "image_grid_thw": inputs["image_grid_thw"].cpu(),
    }
    save_tensors.update(export_vision_debug(model, inputs["pixel_values"], inputs["image_grid_thw"]))
    save_file(save_tensors, str(save_dir / "inputs.safetensors"))

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    (save_dir / "python_int8_tokens.txt").write_text(" ".join(map(str, gen_ids.tolist())))
    print("saved", save_dir)


if __name__ == "__main__":
    main()
