#!/usr/bin/env python3
"""
Build and optionally run a TRT-LLM single-layer block for Phase A.

Scope:
  - layer0 only
  - expert0-only dense MLP
  - manual attention in TRT graph
  - prefill-only

Inputs expected from exported references:
  - hidden_in        [1, S, 2048]
  - cos_mrope        [1, 1, S, 128]
  - sin_mrope        [1, 1, S, 128]
  - causal_mask_4d   [1, 1, S, S]

Output:
  - layer0_out       [1, S, 2048]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def rotate_half_trt(x, F):
    head_dim = x.shape[-1]
    x1, x2 = F.split(x, [head_dim // 2, head_dim // 2], dim=3)
    neg_x2 = F.mul(x2, -1.0)
    return F.concat([neg_x2, x1], dim=3)


def build_engine(weights, refs, precision="bfloat16"):
    import tensorrt_llm
    from tensorrt_llm import Builder, Tensor, str_dtype_to_trt
    import tensorrt_llm.functional as F
    from tensorrt_llm.layers import RmsNorm, Linear, RowLinear

    hidden_in = refs["hidden_in"]
    cos_mrope = refs["cos_mrope"]
    sin_mrope = refs["sin_mrope"]
    causal_mask = refs["causal_mask_4d"]

    B, S, H = hidden_in.shape
    HEADS = 16
    KV_HEADS = 2
    HEAD_DIM = 128
    KV_GROUPS = HEADS // KV_HEADS
    INTER = weights["layer0.expert0.gate_proj.weight"].shape[0]
    q_dim = HEADS * HEAD_DIM
    k_dim = KV_HEADS * HEAD_DIM
    v_dim = KV_HEADS * HEAD_DIM

    builder = Builder()
    builder_config = builder.create_builder_config(name="phase_a_layer0", precision=precision)
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        x = Tensor(
            name="hidden_in",
            dtype=str_dtype_to_trt(precision),
            shape=[B, S, H],
        )
        cos = Tensor(
            name="cos_mrope",
            dtype=str_dtype_to_trt(precision),
            shape=[B, 1, S, HEAD_DIM],
        )
        sin = Tensor(
            name="sin_mrope",
            dtype=str_dtype_to_trt(precision),
            shape=[B, 1, S, HEAD_DIM],
        )
        mask = Tensor(
            name="causal_mask_4d",
            dtype=str_dtype_to_trt(precision),
            shape=[B, 1, S, S],
        )

        ln1 = RmsNorm(H, eps=1e-5, dtype=precision)
        ln1.weight.value = weights["layer0.input_layernorm.weight"]

        qkv = Linear(H, q_dim + k_dim + v_dim, bias=True, dtype=precision, gather_output=True, is_qkv=True)
        qkv.weight.value = weights["layer0.qkv.weight"]
        qkv.bias.value = weights["layer0.qkv.bias"]

        o_proj = RowLinear(q_dim, H, bias=False, dtype=precision)
        o_proj.weight.value = weights["layer0.o_proj.weight"]

        ln2 = RmsNorm(H, eps=1e-5, dtype=precision)
        ln2.weight.value = weights["layer0.post_attention_layernorm.weight"]

        gate_proj = Linear(H, INTER, bias=False, dtype=precision, gather_output=True)
        gate_proj.weight.value = weights["layer0.expert0.gate_proj.weight"]
        up_proj = Linear(H, INTER, bias=False, dtype=precision, gather_output=True)
        up_proj.weight.value = weights["layer0.expert0.up_proj.weight"]
        down_proj = RowLinear(INTER, H, bias=False, dtype=precision)
        down_proj.weight.value = weights["layer0.expert0.down_proj.weight"]

        # pre-attn norm
        h1 = ln1(x)

        # qkv projections
        qkv_out = qkv(h1)
        q, k, v = F.split(qkv_out, [q_dim, k_dim, v_dim], dim=2)

        q = F.view(q, [B, S, HEADS, HEAD_DIM])
        q = F.permute(q, [0, 2, 1, 3])
        k = F.view(k, [B, S, KV_HEADS, HEAD_DIM])
        k = F.permute(k, [0, 2, 1, 3])
        v = F.view(v, [B, S, KV_HEADS, HEAD_DIM])
        v = F.permute(v, [0, 2, 1, 3])

        # multimodal rope (manual, using precomputed cos/sin)
        q_rot = F.add(F.mul(q, cos), F.mul(rotate_half_trt(q, F), sin))
        k_rot = F.add(F.mul(k, cos), F.mul(rotate_half_trt(k, F), sin))

        # repeat kv heads
        k_rep = F.repeat_interleave(k_rot, KV_GROUPS, dim=1)
        v_rep = F.repeat_interleave(v, KV_GROUPS, dim=1)

        # manual attention
        scores = F.matmul(q_rot, k_rep, transb=True)
        scores = F.mul(scores, 1.0 / np.sqrt(HEAD_DIM))
        scores = F.add(scores, mask)
        probs = F.softmax(scores, dim=-1)
        attn = F.matmul(probs, v_rep)

        attn = F.permute(attn, [0, 2, 1, 3])
        attn = F.view(attn, [B, S, q_dim])
        attn_out = o_proj(attn)
        x2 = F.add(x, attn_out)

        # post-attn norm + expert0-only mlp
        h2 = ln2(x2)
        gate = gate_proj(h2)
        up = up_proj(h2)
        hidden = F.mul(F.silu(gate), up)
        mlp_out = down_proj(hidden)
        out = F.add(x2, mlp_out)

        out.mark_output("layer0_out", str_dtype_to_trt(precision))

    engine = builder.build_engine(net, builder_config)
    return engine


def run_engine(engine, refs, precision="bfloat16"):
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[precision]
    hidden_in = refs["hidden_in"].to("cuda").to(torch_dtype)
    cos_mrope = refs["cos_mrope"].to("cuda").to(torch_dtype)
    sin_mrope = refs["sin_mrope"].to("cuda").to(torch_dtype)
    causal_mask = refs["causal_mask_4d"].to("cuda").to(torch_dtype)

    output = torch.zeros_like(hidden_in)
    inputs = {
        "hidden_in": hidden_in,
        "cos_mrope": cos_mrope,
        "sin_mrope": sin_mrope,
        "causal_mask_4d": causal_mask,
    }
    outputs = {"layer0_out": output}

    infos = [
        tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape)
        for k, v in inputs.items()
    ]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
    torch.cuda.synchronize()
    return outputs["layer0_out"], ok


def main():
    parser = argparse.ArgumentParser(description="Build/run Phase A TRT-LLM layer0 block")
    parser.add_argument("--weights", required=True, help="layer0_weights.safetensors")
    parser.add_argument("--refs", required=True, help="layer0_ref.safetensors")
    parser.add_argument("--engine", default="", help="optional output engine path")
    parser.add_argument(
        "--precision",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Build/run precision",
    )
    args = parser.parse_args()

    weights = load_file(args.weights, device="cpu")
    refs = load_file(args.refs, device="cpu")

    print(f"[BUILD] TRT-LLM layer0 block ({args.precision}) ...")
    engine = build_engine(weights, refs, precision=args.precision)
    if engine is None:
        raise RuntimeError("engine build failed")

    if args.engine:
        Path(args.engine).write_bytes(engine)
        print(f"[SAVE] {args.engine}")

    print("[RUN] TRT-LLM layer0 block ...")
    out, ok = run_engine(engine, refs, precision=args.precision)
    if not ok:
        raise RuntimeError("session.run returned False")

    ref_out = refs["layer0_out"].to(out.dtype)
    diff = (out.cpu() - ref_out.cpu()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten().float().cpu(), ref_out.flatten().float().cpu(), dim=0
    ).item()

    print(f"[COMPARE] layer0_out cosine:   {cosine:.8f}")
    print(f"[COMPARE] layer0_out mean abs: {mean_abs:.8e}")
    print(f"[COMPARE] layer0_out max abs:  {max_abs:.8e}")
    print(f"[DEBUG] output has nan:       {torch.isnan(out).any().item()}")
    if not torch.isnan(out).all():
        finite = out[~torch.isnan(out)]
        print(f"[DEBUG] output min/max:       {finite.min().item():.8e} / {finite.max().item():.8e}")


if __name__ == "__main__":
    main()
