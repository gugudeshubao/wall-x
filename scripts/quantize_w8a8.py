#!/usr/bin/env python3
"""
W8A8 INT8 Quantization for wall-x model.

Converts bf16 linear weights to per-channel symmetric INT8.
Keeps Action Head, MoE experts, embeddings, and norms in bf16.

Usage:
    python quantize_w8a8.py --model-dir /path/to/bf16/model --output-dir /path/to/int8/model
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def should_quantize(key: str) -> bool:
    """Determine if a weight key should be quantized to INT8.

    Quantize: Attention projections (q/k/v/o_proj), Vision linear layers, PatchMerger MLP.
    Skip: Action Head (ODE error), MoE experts (custom CUDA kernel), embeddings, norms.
    """
    if not key.endswith(".weight"):
        return False

    # --- SKIP rules ---
    # Action Head: ODE integration error accumulation
    if "action_preprocessor" in key:
        return False
    # Token embeddings and LM head
    if "embed_tokens" in key or "lm_head" in key:
        return False
    # Normalization layers (RMSNorm, LayerNorm)
    if "layernorm" in key or "norm" in key:
        return False
    # MoE expert weights (AsymmetricDualExpertGemm custom CUDA kernel)
    if "moe.experts" in key:
        return False
    # Conv3D patch embedding
    if "patch_embed" in key:
        return False
    # Rotary embedding inv_freq
    if "rotary_emb" in key:
        return False
    # ConditionalUnet1D / Conv1d blocks (if present, not primary path)
    if "diffusion_step_encoder" in key or "cond_encoder" in key:
        return False
    if "blocks.0.0." in key or "blocks.1.0." in key:  # Conv1dBlock internals
        return False
    if "residual_conv" in key:
        return False
    if "final_conv" in key:
        return False
    if "downsample" in key or "upsample" in key:
        return False

    # --- QUANTIZE rules ---
    # Transformer self-attention projections
    if "self_attn" in key and any(p in key for p in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return True
    # Vision attention (qkv combined, output proj)
    if "attn.qkv" in key or "attn.proj" in key:
        return True
    # Vision MLP and transformer MLP (non-MoE layers)
    if "mlp.gate_proj" in key or "mlp.up_proj" in key or "mlp.down_proj" in key:
        return True
    # Patch merger MLP
    if "merger.mlp" in key:
        return True

    return False


def quantize_per_channel(weight: torch.Tensor):
    """Per-channel symmetric INT8 quantization.

    Args:
        weight: [out_features, in_features] bf16/f16 tensor
    Returns:
        (weight_int8, scale) where:
            weight_int8: [out_features, in_features] int8
            scale: [out_features] float32  (per output-channel scale)
    """
    w = weight.float()
    # Per output-channel absmax
    scale = w.abs().amax(dim=1) / 127.0
    scale = scale.clamp(min=1e-10)  # avoid div-by-zero
    # Quantize: round + clip to [-128, 127]
    w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.to(torch.float32)


def main():
    parser = argparse.ArgumentParser(description="W8A8 INT8 Weight Quantization for wall-x")
    parser.add_argument("--model-dir", required=True, help="Path to bf16 safetensors model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for INT8 quantized model")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all safetensors files
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: No safetensors files found in {model_dir}")
        sys.exit(1)

    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(st_files)} safetensors file(s)")

    quantized_count = 0
    kept_count = 0
    total_bf16_bytes = 0
    total_int8_bytes = 0

    for sf in st_files:
        print(f"\n{'='*60}")
        print(f"Processing: {sf.name}")
        print(f"{'='*60}")
        output_tensors = OrderedDict()

        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if should_quantize(key):
                    # torch._int_mm requires both N (out_features, dim=0) and
                    # K (in_features, dim=1) to be multiples of 8
                    if tensor.dim() == 2 and (tensor.shape[0] % 8 != 0 or tensor.shape[1] % 8 != 0):
                        output_tensors[key] = tensor
                        kept_count += 1
                        print(f"  [SKIP] {key}: shape {list(tensor.shape)} not aligned to 8")
                        continue

                    # Quantize this weight
                    w_int8, scale = quantize_per_channel(tensor)
                    # Store int8 weight with SAME key (dtype identifies it)
                    output_tensors[key] = w_int8
                    # Store per-channel scale with _scale suffix
                    scale_key = key.replace(".weight", ".weight_scale")
                    output_tensors[scale_key] = scale

                    quantized_count += 1
                    orig_bytes = tensor.numel() * tensor.element_size()
                    int8_bytes = w_int8.numel() * 1 + scale.numel() * 4
                    total_bf16_bytes += orig_bytes
                    total_int8_bytes += int8_bytes

                    print(f"  [Q] {key}: {list(tensor.shape)} "
                          f"bf16({orig_bytes/1e6:.1f}MB) -> int8({int8_bytes/1e6:.1f}MB) "
                          f"({int8_bytes/orig_bytes*100:.0f}%)")
                else:
                    output_tensors[key] = tensor
                    kept_count += 1

        out_path = output_dir / sf.name
        save_file(output_tensors, str(out_path))
        print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB)")

    # Handle index file for multi-shard models
    for jf in model_dir.glob("*.json"):
        if "index" in jf.name:
            with open(jf) as f:
                index = json.load(f)
            if "weight_map" in index:
                new_map = OrderedDict()
                for key, shard in index["weight_map"].items():
                    new_map[key] = shard
                    if should_quantize(key):
                        scale_key = key.replace(".weight", ".weight_scale")
                        new_map[scale_key] = shard
                index["weight_map"] = new_map
            with open(output_dir / jf.name, 'w') as f:
                json.dump(index, f, indent=2)
            print(f"\nUpdated index: {jf.name}")
        else:
            shutil.copy2(str(jf), str(output_dir / jf.name))
            print(f"Copied config: {jf.name}")

    # Copy tokenizer files
    for pattern in ["tokenizer*", "*.model", "*.tiktoken", "special_tokens_map*"]:
        for tf in model_dir.glob(pattern):
            shutil.copy2(str(tf), str(output_dir / tf.name))

    # Summary
    print(f"\n{'='*60}")
    print(f"Quantization Summary")
    print(f"{'='*60}")
    print(f"Quantized layers:  {quantized_count}")
    print(f"Kept (bf16):       {kept_count}")
    if total_bf16_bytes > 0:
        print(f"Weight size:       {total_bf16_bytes/1e9:.2f}GB (bf16) -> {total_int8_bytes/1e9:.2f}GB (int8)")
        print(f"Compression:       {total_int8_bytes/total_bf16_bytes*100:.1f}%")
    print(f"Output:            {output_dir}")


if __name__ == "__main__":
    main()
