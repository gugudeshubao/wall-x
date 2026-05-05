#!/usr/bin/env python3
"""
Build and run a multi-layer Phase A prefill-only TRT-LLM decoder.

This extends the validated layer0 block to N layers:

  - expert0-only dense MLP
  - manual multimodal rope
  - manual attention
  - prefill-only

For accuracy, it computes a torch reference inline from the same checkpoint
weights and reference tensors, then compares TRT output to that reference.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from safetensors import safe_open
from safetensors.torch import load_file


def rotate_half_trt(x, F):
    head_dim = x.shape[-1]
    x1, x2 = F.split(x, [head_dim // 2, head_dim // 2], dim=3)
    neg_x2 = F.mul(x2, -1.0)
    return F.concat([neg_x2, x1], dim=3)


def rotate_half_torch(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class CheckpointReader:
    def __init__(self, path: str):
        self.path = path
        self.f = safe_open(path, framework="pt", device="cpu")
        self._keys = set(self.f.keys())

    def get(self, key: str):
        return self.f.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self._keys


def build_engine(
    ckpt: CheckpointReader,
    refs,
    num_layers: int,
    precision="bfloat16",
    include_head=False,
    output_kv=False,
):
    import tensorrt_llm
    from tensorrt_llm import Builder, Tensor, str_dtype_to_trt
    import tensorrt_llm.functional as F
    from tensorrt_llm.layers import RmsNorm, Linear, RowLinear

    hidden_in = refs["hidden_in"]
    B, S, H = hidden_in.shape
    HEADS = 16
    KV_HEADS = 2
    HEAD_DIM = 128
    KV_GROUPS = HEADS // KV_HEADS
    q_dim = HEADS * HEAD_DIM
    k_dim = KV_HEADS * HEAD_DIM
    v_dim = KV_HEADS * HEAD_DIM

    builder = Builder()
    builder_config = builder.create_builder_config(name=f"phase_a_prefill_{num_layers}", precision=precision)
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        x = Tensor(name="hidden_in", dtype=str_dtype_to_trt(precision), shape=[B, S, H])
        cos = Tensor(name="cos_mrope", dtype=str_dtype_to_trt(precision), shape=list(refs["cos_mrope"].shape))
        sin = Tensor(name="sin_mrope", dtype=str_dtype_to_trt(precision), shape=list(refs["sin_mrope"].shape))
        mask = Tensor(name="causal_mask_4d", dtype=str_dtype_to_trt(precision), shape=list(refs["causal_mask_4d"].shape))

        h = x
        present_keys = []
        present_vals = []
        for i in range(num_layers):
            prefix = f"model.layers.{i}."
            ln1 = RmsNorm(H, eps=1e-5, dtype=precision)
            ln1.weight.value = ckpt.get(prefix + "input_layernorm.weight")

            q_w = ckpt.get(prefix + "self_attn.q_proj.weight")
            k_w = ckpt.get(prefix + "self_attn.k_proj.weight")
            v_w = ckpt.get(prefix + "self_attn.v_proj.weight")
            q_b = ckpt.get(prefix + "self_attn.q_proj.bias")
            k_b = ckpt.get(prefix + "self_attn.k_proj.bias")
            v_b = ckpt.get(prefix + "self_attn.v_proj.bias")
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()

            qkv = Linear(H, q_dim + k_dim + v_dim, bias=True, dtype=precision, gather_output=True, is_qkv=True)
            qkv.weight.value = qkv_w
            qkv.bias.value = qkv_b

            o_proj = RowLinear(q_dim, H, bias=False, dtype=precision)
            o_proj.weight.value = ckpt.get(prefix + "self_attn.o_proj.weight")

            ln2 = RmsNorm(H, eps=1e-5, dtype=precision)
            ln2.weight.value = ckpt.get(prefix + "post_attention_layernorm.weight")

            gate_w = ckpt.get(prefix + "moe.experts.0.gate_proj.weight")
            up_w = ckpt.get(prefix + "moe.experts.0.up_proj.weight")
            down_w = ckpt.get(prefix + "moe.experts.0.down_proj.weight")
            inter = gate_w.shape[0]

            gate_proj = Linear(H, inter, bias=False, dtype=precision, gather_output=True)
            gate_proj.weight.value = gate_w
            up_proj = Linear(H, inter, bias=False, dtype=precision, gather_output=True)
            up_proj.weight.value = up_w
            down_proj = RowLinear(inter, H, bias=False, dtype=precision)
            down_proj.weight.value = down_w

            residual = h
            h1 = ln1(h)

            qkv_out = qkv(h1)
            q, k, v = F.split(qkv_out, [q_dim, k_dim, v_dim], dim=2)
            q = F.permute(F.view(q, [B, S, HEADS, HEAD_DIM]), [0, 2, 1, 3])
            k = F.permute(F.view(k, [B, S, KV_HEADS, HEAD_DIM]), [0, 2, 1, 3])
            v = F.permute(F.view(v, [B, S, KV_HEADS, HEAD_DIM]), [0, 2, 1, 3])

            q_rot = F.add(F.mul(q, cos), F.mul(rotate_half_trt(q, F), sin))
            k_rot = F.add(F.mul(k, cos), F.mul(rotate_half_trt(k, F), sin))
            if output_kv:
                present_keys.append(k_rot)
                present_vals.append(v)

            k_rep = F.repeat_interleave(k_rot, KV_GROUPS, dim=1)
            v_rep = F.repeat_interleave(v, KV_GROUPS, dim=1)

            scores = F.matmul(q_rot, k_rep, transb=True)
            scores = F.mul(scores, 1.0 / np.sqrt(HEAD_DIM))
            scores = F.add(scores, mask)
            probs = F.softmax(scores, dim=-1)
            attn = F.matmul(probs, v_rep)

            attn = F.permute(attn, [0, 2, 1, 3])
            attn = F.view(attn, [B, S, q_dim])
            attn_out = o_proj(attn)
            x2 = F.add(residual, attn_out)

            residual2 = x2
            h2 = ln2(x2)
            gate = gate_proj(h2)
            up = up_proj(h2)
            hidden = F.mul(F.silu(gate), up)
            mlp_out = down_proj(hidden)
            h = F.add(residual2, mlp_out)

        if include_head:
            final_norm = RmsNorm(H, eps=1e-5, dtype=precision)
            final_norm.weight.value = ckpt.get("model.norm.weight")
            h = final_norm(h)
            last_h = F.slice(h, starts=[0, S - 1, 0], sizes=[B, 1, H])
            last_h = F.view(last_h, [B, H])
            lm_head_weight = ckpt.get("lm_head.weight") if ckpt.has("lm_head.weight") else ckpt.get("model.embed_tokens.weight")
            lm_head = Linear(H, lm_head_weight.shape[0], bias=False, dtype=precision, gather_output=True)
            lm_head.weight.value = lm_head_weight
            logits = lm_head(last_h)
            logits.mark_output("last_logits", str_dtype_to_trt(precision))
        else:
            h.mark_output("hidden_out", str_dtype_to_trt(precision))

        if output_kv:
            for i in range(num_layers):
                present_keys[i].mark_output(
                    f"present_past_key_{i}", str_dtype_to_trt(precision)
                )
                present_vals[i].mark_output(
                    f"present_past_value_{i}", str_dtype_to_trt(precision)
                )

    return builder.build_engine(net, builder_config)


def torch_reference(ckpt: CheckpointReader, refs, num_layers: int, include_head=False):
    hidden = refs["hidden_in"].to("cuda").float()
    cos = refs["cos_mrope"].to("cuda").float()
    sin = refs["sin_mrope"].to("cuda").float()
    mask = refs["causal_mask_4d"].to("cuda").float()
    B, S, H = hidden.shape
    HEADS = 16
    KV_HEADS = 2
    HEAD_DIM = 128
    KV_GROUPS = HEADS // KV_HEADS

    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        residual = hidden

        w = ckpt.get(prefix + "input_layernorm.weight").to("cuda").float()
        var = hidden.pow(2).mean(-1, keepdim=True)
        h1 = hidden * torch.rsqrt(var + 1e-5)
        h1 = h1 * w

        q_w = ckpt.get(prefix + "self_attn.q_proj.weight").to("cuda").float()
        k_w = ckpt.get(prefix + "self_attn.k_proj.weight").to("cuda").float()
        v_w = ckpt.get(prefix + "self_attn.v_proj.weight").to("cuda").float()
        q_b = ckpt.get(prefix + "self_attn.q_proj.bias").to("cuda").float()
        k_b = ckpt.get(prefix + "self_attn.k_proj.bias").to("cuda").float()
        v_b = ckpt.get(prefix + "self_attn.v_proj.bias").to("cuda").float()
        o_w = ckpt.get(prefix + "self_attn.o_proj.weight").to("cuda").float()

        q = torch.matmul(h1, q_w.t()) + q_b
        k = torch.matmul(h1, k_w.t()) + k_b
        v = torch.matmul(h1, v_w.t()) + v_b
        q = q.view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, S, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, S, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = (q * cos) + (rotate_half_torch(q) * sin)
        k = (k * cos) + (rotate_half_torch(k) * sin)
        k = k.repeat_interleave(KV_GROUPS, dim=1)
        v = v.repeat_interleave(KV_GROUPS, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(HEAD_DIM)
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)
        attn_out = torch.matmul(attn, o_w.t())
        x2 = residual + attn_out

        residual2 = x2
        w2 = ckpt.get(prefix + "post_attention_layernorm.weight").to("cuda").float()
        var2 = x2.pow(2).mean(-1, keepdim=True)
        h2 = x2 * torch.rsqrt(var2 + 1e-5)
        h2 = h2 * w2

        gate_w = ckpt.get(prefix + "moe.experts.0.gate_proj.weight").to("cuda").float()
        up_w = ckpt.get(prefix + "moe.experts.0.up_proj.weight").to("cuda").float()
        down_w = ckpt.get(prefix + "moe.experts.0.down_proj.weight").to("cuda").float()
        gate = torch.matmul(h2, gate_w.t())
        up = torch.matmul(h2, up_w.t())
        hidden_mlp = torch_f.silu(gate) * up
        mlp_out = torch.matmul(hidden_mlp, down_w.t())
        hidden = residual2 + mlp_out

    if include_head:
        w_final = ckpt.get("model.norm.weight").to("cuda").float()
        var = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(var + 1e-5)
        hidden = hidden * w_final
        last_h = hidden[:, -1, :]
        lm_w = (ckpt.get("lm_head.weight") if ckpt.has("lm_head.weight") else ckpt.get("model.embed_tokens.weight")).to("cuda").float()
        logits = torch.matmul(last_h, lm_w.t())
        return logits.cpu()

    return hidden.cpu()


def run_engine(
    engine,
    refs,
    precision="bfloat16",
    include_head=False,
    output_kv=False,
    num_layers=0,
    warmup=3,
    iters=10,
):
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

    inputs = {
        "hidden_in": hidden_in,
        "cos_mrope": cos_mrope,
        "sin_mrope": sin_mrope,
        "causal_mask_4d": causal_mask,
    }
    if include_head:
        output = torch.zeros((hidden_in.shape[0], 153715), dtype=torch_dtype, device="cuda")
        outputs = {"last_logits": output}
    else:
        output = torch.zeros_like(hidden_in)
        outputs = {"hidden_out": output}
    if output_kv:
        kv_heads = 2
        head_dim = 128
        seq_len = hidden_in.shape[1]
        for i in range(num_layers):
            outputs[f"present_past_key_{i}"] = torch.zeros(
                (1, kv_heads, seq_len, head_dim), dtype=torch_dtype, device="cuda"
            )
            outputs[f"present_past_value_{i}"] = torch.zeros(
                (1, kv_heads, seq_len, head_dim), dtype=torch_dtype, device="cuda"
            )

    infos = [
        tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape)
        for k, v in inputs.items()
    ]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()

    # warmup
    for _ in range(warmup):
        ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    primary = outputs["last_logits"] if include_head else outputs["hidden_out"]
    return primary, outputs, ok, times


def main():
    parser = argparse.ArgumentParser(description="Build/run multi-layer Phase A TRT-LLM prefill decoder")
    parser.add_argument("--checkpoint", required=True, help="Original model.safetensors path")
    parser.add_argument("--refs", required=True, help="layer0_ref.safetensors path")
    parser.add_argument("--engine", default="", help="optional output engine path")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--include-head", action="store_true", help="Apply final norm + lm_head and compare logits")
    parser.add_argument("--output-kv", action="store_true", help="Also output per-layer KV caches from prefill")
    parser.add_argument(
        "--decoder-ref",
        default="",
        help="Optional decoder_reference.safetensors for comparing original prefill_last_logits",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    refs = load_file(args.refs, device="cpu")
    ckpt = CheckpointReader(args.checkpoint)

    print(f"[BUILD] TRT-LLM prefill decoder layers={args.num_layers} precision={args.precision} include_head={args.include_head}")
    t0 = time.perf_counter()
    engine = build_engine(
        ckpt,
        refs,
        num_layers=args.num_layers,
        precision=args.precision,
        include_head=args.include_head,
        output_kv=args.output_kv,
    )
    build_ms = (time.perf_counter() - t0) * 1000
    if engine is None:
        raise RuntimeError("engine build failed")
    print(f"[BUILD] done in {build_ms:.1f} ms")

    if args.engine:
        Path(args.engine).write_bytes(engine)
        print(f"[SAVE] {args.engine}")

    print("[RUN] TRT-LLM prefill decoder ...")
    out, outputs, ok, times = run_engine(
        engine,
        refs,
        precision=args.precision,
        include_head=args.include_head,
        output_kv=args.output_kv,
        num_layers=args.num_layers,
        warmup=args.warmup,
        iters=args.iters,
    )
    if not ok:
        raise RuntimeError("session.run returned False")

    print("[REF] torch reference ...")
    ref = torch_reference(ckpt, refs, args.num_layers, include_head=args.include_head)
    diff = (out.cpu().float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten().float().cpu(), ref.flatten().float().cpu(), dim=0
    ).item()

    print(f"[COMPARE] cosine:   {cosine:.8f}")
    print(f"[COMPARE] mean abs: {mean_abs:.8e}")
    print(f"[COMPARE] max abs:  {max_abs:.8e}")
    print(f"[DEBUG] output has nan: {torch.isnan(out).any().item()}")
    arr = np.array(times)
    print(f"[BENCH] mean: {arr.mean():.3f} ms | std: {arr.std():.3f} ms | min: {arr.min():.3f} ms | max: {arr.max():.3f} ms")

    if args.include_head and args.decoder_ref:
        dec_refs = load_file(args.decoder_ref, device="cpu")
        if "prefill_last_logits" in dec_refs:
            orig = dec_refs["prefill_last_logits"].float()
            out_cpu = out.cpu().float()
            diff2 = (out_cpu - orig).abs()
            cosine2 = torch.nn.functional.cosine_similarity(
                out_cpu.flatten(), orig.flatten(), dim=0
            ).item()
            print("[COMPARE_ORIG] vs original wall-x prefill_last_logits")
            print(f"[COMPARE_ORIG] cosine:   {cosine2:.8f}")
            print(f"[COMPARE_ORIG] mean abs: {diff2.mean().item():.8e}")
            print(f"[COMPARE_ORIG] max abs:  {diff2.max().item():.8e}")


if __name__ == "__main__":
    main()
