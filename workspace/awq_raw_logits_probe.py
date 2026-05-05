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


def run_model(model, tokenizer, text, device):
    model.eval()
    move_known_tensors_to_device(model, device)
    input_ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64, device=device)
    inputs_embeds = model.model.embed_tokens(input_ids)
    batch = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    past = tuple(
        torch.zeros(batch, 2, model.config.num_key_value_heads, 1, model.config.head_dim, dtype=torch.float16, device=device)
        for _ in range(model.config.num_hidden_layers)
    )
    rope = build_text_only_mrope_cache(model, batch, 4096, device)
    context_lengths = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    kvcache_start_index = torch.zeros((batch,), dtype=torch.int32, device=device)
    last_token_ids = torch.tensor([[seq_len - 1]], dtype=torch.int64, device=device)
    with torch.no_grad():
        logits, _ = model(inputs_embeds, past, rope, context_lengths, kvcache_start_index, last_token_ids)
        scores = logits[0, 0]
        topk = torch.topk(scores, 10)
        return {
            "top_ids": topk.indices.detach().cpu().tolist(),
            "top_scores": topk.values.detach().cpu().tolist(),
            "argmax": int(topk.indices[0].item()),
            "argmax_score": float(topk.values[0].item()),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", default="What is 2+2?")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    result = run_model(model, tokenizer, args.text, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
