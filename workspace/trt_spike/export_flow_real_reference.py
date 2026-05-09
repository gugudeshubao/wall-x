#!/usr/bin/env python3
"""
Export a Flow Action reference from the wall-x Python model using a REAL image.

Produces the same safetensors format as export_flow_dummy_reference.py so that
all downstream ODE runners (TRT, Edge-LLM) can be tested with a real-image
prefix KV cache instead of random tensors.

Usage (on Orin):
    source /data/wy/wall-x/venv/bin/activate
    export LD_LIBRARY_PATH=/data/wy/wall-x/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
    python workspace/trt_spike/export_flow_real_reference.py \
        --model-path /data/wy/models/wall-oss-flow \
        --image /data/wy/wall-x/test_images/fruits_on_table.png \
        --prompt "Pick up the red object on the table." \
        --output-dir /data/wy/wall-x/workspace/trt_spike/tmp/flow_real_1
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

# Reuse helpers from the dummy export script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from export_flow_dummy_reference import (
    load_model,
    export_reference,
    prepare_mrope_ready,
    build_explicit_causal_mask,
    bool_mask_to_additive,
    get_cache_tensors,
    trim_cache_prefix,
)


def create_real_inputs(model, processor, image_path: str, prompt: str, device="cuda"):
    """Build wall-x Flow Action inputs from a real image + text prompt."""
    image = Image.open(image_path).convert("RGB")

    # Build the VLA message in the same format as wall-x serving
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
                {"type": "text", "text": "\nProprioception: <|propri|>"},
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
    ).to(device)

    config = model.config
    action_horizon = getattr(config, "action_horizon", 32)
    action_dim = getattr(config, "action_dim", 20)
    agent_pos_mask_dim = getattr(config, "agent_pos_dim", 20)

    # Append action tokens to input_ids
    action_token_id = model.action_token_id_set.get(
        "action_token_id",
        getattr(config, "action_token_id", None),
    )
    if action_token_id is None:
        raise ValueError("Cannot find action_token_id in model config")

    action_ids = torch.full(
        (1, action_horizon), action_token_id, dtype=torch.long, device=device
    )
    input_ids = torch.cat([inputs["input_ids"], action_ids], dim=1)
    attention_mask = torch.cat(
        [inputs["attention_mask"],
         torch.ones(1, action_horizon, dtype=torch.long, device=device)],
        dim=1,
    )

    total_len = input_ids.shape[1]
    prefix_len = total_len - action_horizon  # everything before action tokens

    # moe_token_types: 0 for prefix, 1 for action postfix
    moe_token_types = torch.zeros(1, total_len, dtype=torch.long, device=device)
    moe_token_types[0, prefix_len:] = 1

    dof_mask = torch.ones(1, action_horizon, action_dim, dtype=torch.float32, device=device)
    agent_pos_mask = torch.ones(1, agent_pos_mask_dim, dtype=torch.float32, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "moe_token_types": moe_token_types,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "num_inference_timesteps": 5,
        "dataset_names": "x2_normal",
        "dof_mask": dof_mask,
        "agent_pos_mask": agent_pos_mask,
        "unnorm": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export Flow Action reference from a real image"
    )
    parser.add_argument("--model-path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument(
        "--image",
        default="/data/wy/wall-x/test_images/fruits_on_table.png",
        help="Path to real test image",
    )
    parser.add_argument(
        "--prompt",
        default="Pick up the red object on the table.",
        help="Task instruction for the robot",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-impl", default="sdpa")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] model from {args.model_path}")
    t0 = time.perf_counter()
    model, processor = load_model(args.model_path, attn_impl=args.attn_impl)
    print(f"[LOAD] done in {(time.perf_counter() - t0):.1f}s")

    print(f"[INPUT] image={args.image}")
    print(f"[INPUT] prompt={args.prompt!r}")
    inputs = create_real_inputs(
        model, processor, args.image, args.prompt, device="cuda"
    )
    print(f"[INPUT] total tokens = {inputs['input_ids'].shape[1]}")
    print(f"[INPUT] action_horizon = {inputs['action_horizon']}")

    refs, manifest = export_reference(model, inputs, args.seed)

    out_file = out_dir / "flow_real_reference.safetensors"
    manifest_file = out_dir / "flow_real_manifest.json"
    manifest["image"] = args.image
    manifest["prompt"] = args.prompt

    save_file(refs, str(out_file))
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVE] {out_file}")
    print(f"[SAVE] {manifest_file}")
    print(f"[PREFIX] {manifest['prefix_length']} tokens")
    print(f"[POSTFIX] {manifest['postfix_length']} tokens")
    print(f"[TIMINGS] {json.dumps(manifest['timings_ms'], indent=2)}")


if __name__ == "__main__":
    main()
