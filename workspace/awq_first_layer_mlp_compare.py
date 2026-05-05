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
    layer = model.model.layers[0]
    with torch.no_grad():
        input_ln = layer.input_layernorm(hidden)
        post_attn_ln = layer.post_attention_layernorm(hidden)
        mlp_out = layer.mlp(post_attn_ln)
    return input_ln, post_attn_ln, mlp_out


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

    fp16_input_ln, fp16_post_ln, fp16_mlp = run(fp16, tokenizer, args.text, device)
    awq_input_ln, awq_post_ln, awq_mlp = run(awq, tokenizer, args.text, device)

    out = {
        "text": args.text,
        "input_layernorm": compare(fp16_input_ln, awq_input_ln),
        "post_attention_layernorm": compare(fp16_post_ln, awq_post_ln),
        "mlp_out": compare(fp16_mlp, awq_mlp),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
