#!/usr/bin/env python3
"""
Export preprocessed VQA inputs for C++ inference engine validation.

Preprocesses real test images through Qwen2.5-VL processor, saves the tensor
inputs (input_ids, pixel_values, image_grid_thw) as .pt files that libtorch
can load. Also runs Python inference to produce baseline answers for comparison.

Usage (on Orin):
    python scripts/export_vqa_inputs.py \
        --model-path /data/wy/models/wall-oss-flow \
        --image-dir /data/wy/wall-x/test_images \
        --output-dir /data/wy/wall-x/vqa_test_inputs \
        --max-new-tokens 20
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file
from wall_x.fusions import ops


def main():
    parser = argparse.ArgumentParser(
        description="Export preprocessed VQA inputs for C++ engine validation")
    parser.add_argument("--model-path", required=True,
                        help="Path to model directory")
    parser.add_argument("--image-dir", required=True,
                        help="Directory containing test images")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for tensor files")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Max tokens for Python baseline (default: 20)")
    parser.add_argument("--question", default="Describe what you see in this image.",
                        help="VQA question to ask")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    image_dir = Path(args.image_dir)
    exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts
    ])
    if not image_paths:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} test images")

    # Load model and processor
    print(f"\nLoading model from {args.model_path} ...")
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cuda").bfloat16()
    processor = model.processor
    print(f"Model loaded. Peak GPU: "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    def export_vision_debug(image, pixel_values, image_grid_thw):
        """Run Python vision encoder and export intermediate tensors for comparison."""
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
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)
            after_reorder = hidden_states

            rotary_pos_emb = rotary_pos_emb.reshape(
                seq_len // vision.spatial_merge_unit, vision.spatial_merge_unit, -1
            )
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

            cu_seqlens = torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
            max_seqlen_full = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            max_seqlen_window = (
                (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max().item()
            )

            after_block0 = None
            after_block7 = None
            for layer_num, blk in enumerate(vision.blocks):
                if layer_num in vision.fullatt_block_indexes:
                    cu_now = cu_seqlens
                    max_now = max_seqlen_full
                else:
                    cu_now = cu_window_seqlens
                    max_now = max_seqlen_window
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
            pre_merger = hidden_states
            image_embeds = vision.merger(hidden_states)
            reverse_indices = torch.argsort(window_index)
            image_embeds = image_embeds[reverse_indices, :]

        return {
            "vision_after_reorder": after_reorder.cpu(),
            "vision_block0": after_block0.cpu(),
            "vision_block7": after_block7.cpu(),
            "vision_pre_merger": pre_merger.cpu(),
            "image_embeds": image_embeds.cpu(),
        }

    results = []

    for img_path in image_paths:
        img_name = img_path.stem  # e.g., "fruits_on_table"
        img_dir = output_dir / img_name
        img_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"{'='*60}")

        image = Image.open(img_path).convert("RGB")

        # Build chat message (same format as VQA scripts)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.question},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[image],
            padding=True, return_tensors="pt",
        ).to("cuda")

        # Extract and save tensors for C++ engine
        input_ids = inputs['input_ids']          # [1, seq_len] long
        pixel_values = inputs['pixel_values']    # [num_patches, feat_dim] bf16
        image_grid_thw = inputs['image_grid_thw']  # [1, 3] long

        print(f"  input_ids:      {list(input_ids.shape)} {input_ids.dtype}")
        print(f"  pixel_values:   {list(pixel_values.shape)} {pixel_values.dtype}")
        print(f"  image_grid_thw: {list(image_grid_thw.shape)} {image_grid_thw.dtype}")

        # Save tensors as safetensors (libtorch compatible via weight_loader)
        save_tensors = {
            "input_ids": input_ids.cpu(),
            "pixel_values": pixel_values.cpu(),
            "image_grid_thw": image_grid_thw.cpu(),
        }

        # Export Python vision debug tensors for step-by-step compare.
        vision_debug = export_vision_debug(image, pixel_values, image_grid_thw)
        save_tensors.update(vision_debug)

        save_file(save_tensors, str(img_dir / "inputs.safetensors"))
        print(f"  Saved tensors to {img_dir}/inputs.safetensors")

        # Run Python inference as baseline
        print(f"  Running Python inference (max_new_tokens={args.max_new_tokens})...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        gen_ids = output_ids[0][input_ids.shape[1]:]
        answer = processor.decode(gen_ids, skip_special_tokens=True)
        n_tok = len(gen_ids)

        # Save baseline token IDs
        torch.save(gen_ids.cpu(), str(img_dir / "baseline_tokens.pt"))

        print(f"  Answer: {answer[:80]}{'...' if len(answer)>80 else ''}")
        print(f"  Tokens: {n_tok}, Latency: {latency*1000:.0f}ms")
        print(f"  Token IDs: {gen_ids.tolist()}")

        results.append({
            "image": img_path.name,
            "question": args.question,
            "answer": answer,
            "token_ids": gen_ids.tolist(),
            "num_tokens": n_tok,
            "latency_ms": round(latency * 1000, 1),
            "input_seq_len": input_ids.shape[1],
            "num_patches": pixel_values.shape[0],
        })

    # Save manifest
    manifest = {
        "model_path": args.model_path,
        "question": args.question,
        "max_new_tokens": args.max_new_tokens,
        "images": results,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Total {len(results)} images exported to {output_dir}")


if __name__ == "__main__":
    main()
