#!/usr/bin/env python3
"""
Build and run a single-step Phase A decode TRT-LLM decoder.

Scope:
  - VQA-specialized
  - expert0-only dense MLP
  - manual multimodal rope
  - single decode step (q_len = 1)
  - past KV cache provided as explicit tensor inputs
"""

import argparse
import time

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
        self.f = safe_open(path, framework="pt", device="cpu")
        self._keys = set(self.f.keys())

    def get(self, key: str):
        return self.f.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self._keys


def build_engine(ckpt: CheckpointReader, refs, num_layers: int, precision="bfloat16", past_len: int | None = None):
    import tensorrt_llm
    from tensorrt_llm import Builder, Tensor, str_dtype_to_trt
    import tensorrt_llm.functional as F
    from tensorrt_llm.layers import RmsNorm, Linear, RowLinear

    hidden_in = refs["decode_inputs_embeds"]
    B, S, H = hidden_in.shape  # [1, 1, 2048]
    assert S == 1
    HEADS = 16
    KV_HEADS = 2
    HEAD_DIM = 128
    KV_GROUPS = HEADS // KV_HEADS
    q_dim = HEADS * HEAD_DIM
    k_dim = KV_HEADS * HEAD_DIM
    v_dim = KV_HEADS * HEAD_DIM
    PAST = past_len if past_len is not None else refs["prefill_past_key_0"].shape[2]

    builder = Builder()
    builder_config = builder.create_builder_config(name=f"phase_a_decode_{num_layers}", precision=precision)
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        h = Tensor(name="decode_inputs_embeds", dtype=str_dtype_to_trt(precision), shape=[B, S, H])
        cos = Tensor(name="decode_cos_mrope", dtype=str_dtype_to_trt(precision), shape=list(refs["decode_cos_mrope"].shape))
        sin = Tensor(name="decode_sin_mrope", dtype=str_dtype_to_trt(precision), shape=list(refs["decode_sin_mrope"].shape))

        past_keys = []
        past_vals = []
        for i in range(num_layers):
            pk = Tensor(name=f"prefill_past_key_{i}", dtype=str_dtype_to_trt(precision), shape=[B, KV_HEADS, PAST, HEAD_DIM])
            pv = Tensor(name=f"prefill_past_value_{i}", dtype=str_dtype_to_trt(precision), shape=[B, KV_HEADS, PAST, HEAD_DIM])
            past_keys.append(pk)
            past_vals.append(pv)

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

            # append new kv to past
            full_k = F.concat([past_keys[i], k_rot], dim=2)
            full_v = F.concat([past_vals[i], v], dim=2)
            present_keys.append(full_k)
            present_vals.append(full_v)

            k_rep = F.repeat_interleave(full_k, KV_GROUPS, dim=1)
            v_rep = F.repeat_interleave(full_v, KV_GROUPS, dim=1)

            scores = F.matmul(q_rot, k_rep, transb=True)
            scores = F.mul(scores, 1.0 / np.sqrt(HEAD_DIM))
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

        final_norm = RmsNorm(H, eps=1e-5, dtype=precision)
        final_norm.weight.value = ckpt.get("model.norm.weight")
        h = final_norm(h)
        last_h = F.view(h, [B, H])
        lm_head_weight = ckpt.get("lm_head.weight") if ckpt.has("lm_head.weight") else ckpt.get("model.embed_tokens.weight")
        lm_head = Linear(H, lm_head_weight.shape[0], bias=False, dtype=precision, gather_output=True)
        lm_head.weight.value = lm_head_weight
        logits = lm_head(last_h)
        logits.mark_output("decode_last_logits", str_dtype_to_trt(precision))
        for i in range(num_layers):
            present_keys[i].mark_output(f"present_past_key_{i}", str_dtype_to_trt(precision))
            present_vals[i].mark_output(f"present_past_value_{i}", str_dtype_to_trt(precision))

    return builder.build_engine(net, builder_config)


def torch_reference(ckpt: CheckpointReader, refs, num_layers: int):
    hidden = refs["decode_inputs_embeds"].to("cuda").float()
    cos = refs["decode_cos_mrope"].to("cuda").float()
    sin = refs["decode_sin_mrope"].to("cuda").float()
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

        past_k = refs[f"prefill_past_key_{i}"].to("cuda").float()
        past_v = refs[f"prefill_past_value_{i}"].to("cuda").float()
        full_k = torch.cat([past_k, k], dim=2)
        full_v = torch.cat([past_v, v], dim=2)

        k_rep = full_k.repeat_interleave(KV_GROUPS, dim=1)
        v_rep = full_v.repeat_interleave(KV_GROUPS, dim=1)
        scores = torch.matmul(q, k_rep.transpose(-1, -2)) / np.sqrt(HEAD_DIM)
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v_rep)
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

    w_final = ckpt.get("model.norm.weight").to("cuda").float()
    var = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(var + 1e-5)
    hidden = hidden * w_final
    last_h = hidden[:, -1, :]
    lm_w = (ckpt.get("lm_head.weight") if ckpt.has("lm_head.weight") else ckpt.get("model.embed_tokens.weight")).to("cuda").float()
    logits = torch.matmul(last_h, lm_w.t())
    return logits.cpu()


def run_engine(engine, refs, num_layers: int, precision="bfloat16", warmup=3, iters=10):
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]

    inputs = {
        "decode_inputs_embeds": refs["decode_inputs_embeds"].to("cuda").to(torch_dtype),
        "decode_cos_mrope": refs["decode_cos_mrope"].to("cuda").to(torch_dtype),
        "decode_sin_mrope": refs["decode_sin_mrope"].to("cuda").to(torch_dtype),
    }
    for i in range(num_layers):
        inputs[f"prefill_past_key_{i}"] = refs[f"prefill_past_key_{i}"].to("cuda").to(torch_dtype)
        inputs[f"prefill_past_value_{i}"] = refs[f"prefill_past_value_{i}"].to("cuda").to(torch_dtype)

    outputs = {
        "decode_last_logits": torch.zeros((1, 153715), dtype=torch_dtype, device="cuda")
    }
    for i in range(num_layers):
        in_k = inputs[f"prefill_past_key_{i}"]
        in_v = inputs[f"prefill_past_value_{i}"]
        B, Hk, T, D = in_k.shape
        outputs[f"present_past_key_{i}"] = torch.zeros(
            (B, Hk, T + 1, D), dtype=in_k.dtype, device=in_k.device
        )
        outputs[f"present_past_value_{i}"] = torch.zeros(
            (B, Hk, T + 1, D), dtype=in_v.dtype, device=in_v.device
        )

    infos = [tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape) for k, v in inputs.items()]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()
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
    return outputs, ok, times


def main():
    parser = argparse.ArgumentParser(description="Build/run single-step decode TRT-LLM decoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder-ref", required=True)
    parser.add_argument("--engine", default="")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--past-len", type=int, default=None, help="Optional static past KV length override")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    refs = load_file(args.decoder_ref, device="cpu")
    ckpt = CheckpointReader(args.checkpoint)

    print(f"[BUILD] TRT-LLM decode decoder layers={args.num_layers} precision={args.precision}")
    t0 = time.perf_counter()
    engine = build_engine(
        ckpt,
        refs,
        num_layers=args.num_layers,
        precision=args.precision,
        past_len=args.past_len,
    )
    build_ms = (time.perf_counter() - t0) * 1000
    if engine is None:
        raise RuntimeError("engine build failed")
    print(f"[BUILD] done in {build_ms:.1f} ms")
    if args.engine:
        from pathlib import Path
        Path(args.engine).write_bytes(engine)
        print(f"[SAVE] {args.engine}")

    print("[RUN] TRT-LLM decode decoder ...")
    outputs, ok, times = run_engine(engine, refs, args.num_layers, precision=args.precision, warmup=args.warmup, iters=args.iters)
    if not ok:
        raise RuntimeError("session.run returned False")

    out = outputs["decode_last_logits"]
    print("[REF] torch reference ...")
    ref = torch_reference(ckpt, refs, args.num_layers)
    diff = (out.cpu().float() - ref.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(out.flatten().float().cpu(), ref.flatten().float().cpu(), dim=0).item()
    print(f"[COMPARE] cosine:   {cosine:.8f}")
    print(f"[COMPARE] mean abs: {diff.mean().item():.8e}")
    print(f"[COMPARE] max abs:  {diff.max().item():.8e}")
    arr = np.array(times)
    print(f"[BENCH] mean: {arr.mean():.3f} ms | std: {arr.std():.3f} ms | min: {arr.min():.3f} ms | max: {arr.max():.3f} ms")

    orig = refs["decode_last_logits"].float()
    diff2 = (out.cpu().float() - orig).abs()
    cosine2 = torch.nn.functional.cosine_similarity(out.flatten().float().cpu(), orig.flatten().float().cpu(), dim=0).item()
    print("[COMPARE_ORIG] vs original wall-x decode_last_logits")
    print(f"[COMPARE_ORIG] cosine:   {cosine2:.8f}")
    print(f"[COMPARE_ORIG] mean abs: {diff2.mean().item():.8e}")
    print(f"[COMPARE_ORIG] max abs:  {diff2.max().item():.8e}")


if __name__ == "__main__":
    main()
