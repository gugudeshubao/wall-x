#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from tokenizers import Tokenizer

from llm_loader import AutoModel


def move_known_tensors_to_device(model, device):
    for module in model.modules():
        for attr in ("bias", "qweight", "qzeros", "scales", "weight", "weight_scale", "pre_quant_scale"):
            value = getattr(module, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(module, attr, value.to(device))


def build_text_only_mrope_cache(model, batch_size: int, max_positions: int, device):
    rotary_dim = int(model.config.head_dim * getattr(model.config, "partial_rotary_factor", 1.0))
    theta = float(getattr(model.config, "rope_theta", 1000000.0))
    half_dim = rotary_dim // 2
    idx = torch.arange(0, half_dim, dtype=torch.float32, device=device)
    denom = torch.pow(torch.tensor(theta, dtype=torch.float32, device=device), 2.0 * idx / rotary_dim)
    pos = torch.arange(max_positions, dtype=torch.float32, device=device).unsqueeze(1)
    inv_freq = pos / denom.unsqueeze(0)
    cos = torch.cos(inv_freq)
    sin = torch.sin(inv_freq)
    rope = torch.cat([cos, sin], dim=1).unsqueeze(0).repeat(batch_size, 1, 1).contiguous()
    return rope


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(aa, bb, dim=0).item())


def stats(t: torch.Tensor):
    x = t.reshape(-1).float().cpu()
    return {
        "shape": list(t.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "abs_mean": float(x.abs().mean().item()),
    }


def compare(a: torch.Tensor, b: torch.Tensor):
    diff = (a - b).float()
    return {
        "cosine": cosine(a, b),
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "a": stats(a),
        "b": stats(b),
    }


def run(model, tokenizer, text, device):
    model.eval()
    move_known_tensors_to_device(model, device)
    input_ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64, device=device)
    hidden = model.model.embed_tokens(input_ids)
    batch = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    past = tuple(
        torch.zeros(batch, 2, model.config.num_key_value_heads, 1, model.config.head_dim, dtype=torch.float16, device=device)
        for _ in range(model.config.num_hidden_layers)
    )
    rope = build_text_only_mrope_cache(model, batch, 4096, device)
    context_lengths = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    kvcache_start_index = torch.zeros((batch,), dtype=torch.int32, device=device)
    with torch.no_grad():
        attn_out, _ = model.model.layers[0].self_attn(hidden, past[0], rope, context_lengths, kvcache_start_index)
        layer_out, _ = model.model.layers[0](hidden, past[0], rope, context_lengths, kvcache_start_index)
    return attn_out, layer_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--awq", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", default="What is 2+2?")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    fp16 = AutoModel.from_pretrained(args.fp16)
    awq = AutoModel.from_pretrained(args.awq)
    fp16.to(device)
    awq.to(device)

    fp16_attn, fp16_layer = run(fp16, tokenizer, args.text, device)
    awq_attn, awq_layer = run(awq, tokenizer, args.text, device)

    out = {
        "text": args.text,
        "attn": compare(fp16_attn, awq_attn),
        "layer": compare(fp16_layer, awq_layer),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
