#!/usr/bin/env python3
"""
Export Phase A VQA-specialized decoder wrapper to ONNX.

This targets the prefill-only path first:

  inputs:
    - inputs_embeds [1, S, 2048]
    - position_ids  [3, 1, S]
    - attention_mask [1, S]

  output:
    - logits [1, S, vocab_size]

The goal is to test whether the expert0-only decoder wrapper is exportable
through standard ONNX before falling back to direct TensorRT API construction.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from phase_a_decoder_wrapper import VQASpecializedDecoder


def main():
    parser = argparse.ArgumentParser(description="Export Phase A decoder ONNX")
    parser.add_argument("--model-path", required=True, help="Path to wall-x model")
    parser.add_argument("--ref-dir", required=True, help="Directory with decoder_reference.safetensors")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    args = parser.parse_args()

    ref_path = Path(args.ref_dir) / "decoder_reference.safetensors"
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    print(f"[LOAD] model from {args.model_path}")
    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cuda").bfloat16()

    wrapper = VQASpecializedDecoder(model, force_manual_attention=True).eval().to("cuda").bfloat16()

    tensors = load_file(str(ref_path), device="cuda")
    inputs_embeds = tensors["inputs_embeds"]
    position_ids = tensors["position_ids"]
    attention_mask = tensors["attention_mask"]
    if attention_mask.numel() == 0:
        attention_mask = torch.ones(
            inputs_embeds.shape[0], inputs_embeds.shape[1],
            dtype=torch.long, device=inputs_embeds.device
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[EXPORT] starting torch.onnx.export ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (inputs_embeds, position_ids, attention_mask),
            str(output_path),
            input_names=["inputs_embeds", "position_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()
