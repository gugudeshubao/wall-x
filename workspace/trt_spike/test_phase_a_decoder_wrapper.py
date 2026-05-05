#!/usr/bin/env python3
"""
Smoke test for Phase A VQA-specialized decoder wrapper.

Loads decoder reference tensors exported by export_vqa_decoder_reference.py,
runs the expert0-only decoder wrapper, and compares prefill last-token logits
against the Python reference.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from phase_a_decoder_wrapper import VQASpecializedDecoder


def main():
    parser = argparse.ArgumentParser(description="Test Phase A decoder wrapper")
    parser.add_argument("--model-path", required=True, help="Path to wall-x model")
    parser.add_argument(
        "--ref-dir", required=True, help="Directory containing decoder_reference.safetensors"
    )
    args = parser.parse_args()

    ref_path = Path(args.ref_dir) / "decoder_reference.safetensors"
    manifest_path = Path(args.ref_dir) / "manifest.json"
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    print(f"[LOAD] model from {args.model_path}")
    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cuda").bfloat16()

    wrapper = VQASpecializedDecoder(model).eval().to("cuda").bfloat16()

    tensors = load_file(str(ref_path), device="cuda")
    inputs_embeds = tensors["inputs_embeds"]
    position_ids = tensors["position_ids"]
    attention_mask = tensors["attention_mask"]
    ref_last_logits = tensors["prefill_last_logits"]

    if attention_mask.numel() == 0:
        attention_mask = None

    with torch.no_grad():
        logits = wrapper(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        last_logits = logits[:, -1, :].float()

    diff = (last_logits - ref_last_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        last_logits.flatten(), ref_last_logits.flatten(), dim=0
    ).item()

    print(f"[COMPARE] last-logits cosine:  {cosine:.8f}")
    print(f"[COMPARE] last-logits mean abs: {mean_abs:.8e}")
    print(f"[COMPARE] last-logits max abs:  {max_abs:.8e}")

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"[REF] greedy text: {manifest.get('generated_text', '')}")


if __name__ == "__main__":
    main()
