#!/usr/bin/env python3
"""
Build and run a dynamic-past-len Phase A decode TRT-LLM decoder.

Goal:
  - replace multiple `decode_past{N}.engine` variants with one dynamic decode engine
  - keep current VQA-specialized expert0-only path
  - validate on multiple past_len points (e.g. 420 / 423)
"""

import argparse
import time
from collections import OrderedDict

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


def build_engine(
    ckpt: CheckpointReader,
    refs,
    num_layers: int,
    precision="bfloat16",
    past_min: int = 1,
    past_opt: int = 420,
    past_max: int = 512,
):
    import tensorrt_llm
    from tensorrt_llm import Builder, Tensor, str_dtype_to_trt
    import tensorrt_llm.functional as F
    from tensorrt_llm.layers import Linear, RmsNorm, RowLinear

    hidden_in = refs["decode_inputs_embeds"]
    B, S, H = hidden_in.shape
    assert B == 1 and S == 1
    HEADS = 16
    KV_HEADS = 2
    HEAD_DIM = 128
    KV_GROUPS = HEADS // KV_HEADS
    q_dim = HEADS * HEAD_DIM
    k_dim = KV_HEADS * HEAD_DIM
    v_dim = KV_HEADS * HEAD_DIM

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=f"phase_a_decode_dynamic_{num_layers}",
        precision=precision,
        force_num_profiles=1,
    )
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        h = Tensor(
            name="decode_inputs_embeds",
            dtype=str_dtype_to_trt(precision),
            shape=[1, 1, H],
            dim_range=OrderedDict(
                batch_size=[(1, 1, 1)],
                q_len=[(1, 1, 1)],
                hidden_size=[H],
            ),
        )
        cos = Tensor(
            name="decode_cos_mrope",
            dtype=str_dtype_to_trt(precision),
            shape=[1, 1, 1, HEAD_DIM],
            dim_range=OrderedDict(
                batch_size=[(1, 1, 1)],
                num_heads=[(1, 1, 1)],
                q_len=[(1, 1, 1)],
                head_dim=[HEAD_DIM],
            ),
        )
        sin = Tensor(
            name="decode_sin_mrope",
            dtype=str_dtype_to_trt(precision),
            shape=[1, 1, 1, HEAD_DIM],
            dim_range=OrderedDict(
                batch_size=[(1, 1, 1)],
                num_heads=[(1, 1, 1)],
                q_len=[(1, 1, 1)],
                head_dim=[HEAD_DIM],
            ),
        )

        past_keys = []
        past_vals = []
        past_shape = [1, KV_HEADS, -1, HEAD_DIM]
        past_range = OrderedDict(
            batch_size=[(1, 1, 1)],
            kv_heads=[KV_HEADS],
            past_len=[(past_min, past_opt, past_max)],
            head_dim=[HEAD_DIM],
        )
        for i in range(num_layers):
            pk = Tensor(
                name=f"prefill_past_key_{i}",
                dtype=str_dtype_to_trt(precision),
                shape=past_shape,
                dim_range=past_range,
            )
            pv = Tensor(
                name=f"prefill_past_value_{i}",
                dtype=str_dtype_to_trt(precision),
                shape=past_shape,
                dim_range=past_range,
            )
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


def run_engine(engine, refs, num_layers: int, precision="bfloat16", warmup=1, iters=3, past_prefix="prefill_past"):
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
        inputs[f"prefill_past_key_{i}"] = refs[f"{past_prefix}_key_{i}"].to("cuda").to(torch_dtype)
        inputs[f"prefill_past_value_{i}"] = refs[f"{past_prefix}_value_{i}"].to("cuda").to(torch_dtype)

    outputs = {"decode_last_logits": torch.zeros((1, 153715), dtype=torch_dtype, device="cuda")}
    T = refs[f"{past_prefix}_key_0"].shape[2]
    for i in range(num_layers):
        outputs[f"present_past_key_{i}"] = torch.zeros((1, 2, T + 1, 128), dtype=torch_dtype, device="cuda")
        outputs[f"present_past_value_{i}"] = torch.zeros((1, 2, T + 1, 128), dtype=torch_dtype, device="cuda")

    infos = [tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape) for k, v in inputs.items()]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()
    for _ in range(warmup):
        ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        if not ok:
            raise RuntimeError("warmup failed")
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        if not ok:
            raise RuntimeError("run failed")
        times.append((time.perf_counter() - t0) * 1000)
    return outputs, np.array(times)


def main():
    parser = argparse.ArgumentParser(description="Build/run dynamic Phase A decode TRT-LLM decoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder-ref", required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--past-min", type=int, default=1)
    parser.add_argument("--past-opt", type=int, default=420)
    parser.add_argument("--past-max", type=int, default=512)
    parser.add_argument("--engine", default="")
    args = parser.parse_args()

    refs = load_file(args.decoder_ref, device="cpu")
    ckpt = CheckpointReader(args.checkpoint)
    print(f"[BUILD] dynamic decode engine layers={args.num_layers} precision={args.precision} past=[{args.past_min},{args.past_opt},{args.past_max}]")
    t0 = time.perf_counter()
    engine = build_engine(
        ckpt, refs, num_layers=args.num_layers, precision=args.precision,
        past_min=args.past_min, past_opt=args.past_opt, past_max=args.past_max,
    )
    print(f"[BUILD] done in {(time.perf_counter()-t0)*1000:.1f} ms")
    if args.engine:
        from pathlib import Path
        Path(args.engine).write_bytes(engine)
        print(f"[SAVE] {args.engine}")

    tests = [("prefill_past", refs["prefill_past_key_0"].shape[2])]
    if "present_past_key_0" in refs:
        tests.append(("present_past", refs["present_past_key_0"].shape[2]))
    for past_prefix, test_len in tests:
        print(f"[RUN] test source={past_prefix} past_len={test_len}")
        outputs, times = run_engine(
            engine, refs, args.num_layers, precision=args.precision, past_prefix=past_prefix
        )
        print(f"[BENCH] mean={times.mean():.3f} ms std={times.std():.3f} ms")
        print(f"[OUT] present_k0={tuple(outputs['present_past_key_0'].shape)}")


if __name__ == "__main__":
    main()
