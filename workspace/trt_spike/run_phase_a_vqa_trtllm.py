#!/usr/bin/env python3
"""
Run the first full VQA-specialized TRT-LLM path.

This runner stitches together:
  - 36-layer prefill decoder engine
  - 36-layer single-step decode engine

Current scope:
  - VQA-specialized
  - expert0-only
  - vision / image scatter stays outside TRT for now
  - token embedding and next-step rotary cos/sin are host-side
  - decoder主体走 TRT
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import build_phase_a_prefill_trtllm as prefill_mod
import build_phase_a_decode_trtllm as decode_mod
import build_phase_a_decode_dynamic_trtllm as decode_dynamic_mod


def make_session(engine_bytes):
    import tensorrt_llm
    return tensorrt_llm.runtime.Session.from_serialized_engine(engine_bytes)


def maybe_load_engine(path: Path):
    if path.exists():
        return path.read_bytes()
    return None


def load_engine_source(path_or_bytes):
    if isinstance(path_or_bytes, Path):
        return path_or_bytes.read_bytes()
    return path_or_bytes


def trim_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def run_prefill_session(session, refs, num_layers: int, precision="bfloat16"):
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

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
    outputs = {
        "last_logits": torch.zeros((1, 153715), dtype=torch_dtype, device="cuda")
    }
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
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return outputs, ok, ms


def run_decode_session(session, inputs, num_layers: int, precision="bfloat16"):
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    infos = [
        tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape)
        for k, v in inputs.items()
    ]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()

    outputs = {
        "decode_last_logits": torch.zeros((1, 153715), dtype=inputs["decode_inputs_embeds"].dtype, device="cuda")
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

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return outputs, ok, ms


class CheckpointReader:
    def __init__(self, path: str):
        self.f = safe_open(path, framework="pt", device="cpu")

    def get(self, key: str):
        return self.f.get_tensor(key)


def compute_decode_cos_sin(config, inv_freq, pos: int, device, dtype):
    position_ids = torch.full((3, 1, 1), pos, dtype=torch.long, device=device)
    inv_freq_expanded = (
        inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    )
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    mrope_section = config["rope_scaling"]["mrope_section"] * 2
    cos_m = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(1)
    sin_m = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(1)
    return position_ids, cos_m, sin_m


def main():
    parser = argparse.ArgumentParser(description="Run first full VQA-specialized TRT-LLM path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder-ref", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save-engines-dir", default="")
    parser.add_argument("--save-step-logits", action="store_true", help="Save TRT per-step logits to JSON-friendly output")
    parser.add_argument("--build-only", action="store_true", help="Build engines and exit without running")
    parser.add_argument("--run-only", action="store_true", help="Run only from cached engines; fail if cache is missing")
    parser.add_argument("--dynamic-decode", action="store_true", help="Use one dynamic decode engine instead of per-past-len engines")
    args = parser.parse_args()

    if args.build_only and args.run_only:
        raise ValueError("--build-only and --run-only are mutually exclusive")

    refs = load_file(args.decoder_ref, device="cpu")
    with open(args.config_json) as f:
        config = json.load(f)
    ckpt = CheckpointReader(args.checkpoint)
    inv_freq = ckpt.get("model.rotary_emb.inv_freq") if "model.rotary_emb.inv_freq" in set(ckpt.f.keys()) else None
    if inv_freq is None:
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        arange = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / torch.pow(torch.tensor(config["rope_theta"], dtype=torch.float32), arange / head_dim)

    save_dir = Path(args.save_engines_dir) if args.save_engines_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    prefill_engine = None
    decode_engines = {}
    decode_dynamic_engine = None
    initial_past_len = refs["prefill_past_key_0"].shape[2]

    if save_dir:
        prefill_engine = maybe_load_engine(save_dir / "prefill.engine")
    if prefill_engine is None:
        if args.run_only:
            raise FileNotFoundError("prefill.engine not found in save-engines-dir")
        print("[BUILD] prefill engine ...")
        prefill_engine = prefill_mod.build_engine(
            prefill_mod.CheckpointReader(args.checkpoint),
            load_file("/data/wy/wall-x/workspace/trt_spike/tmp/layer0_ref.safetensors", device="cpu"),
            num_layers=args.num_layers,
            precision=args.precision,
            include_head=True,
            output_kv=True,
        )
        if save_dir:
            (save_dir / "prefill.engine").write_bytes(prefill_engine)
            prefill_engine = save_dir / "prefill.engine"
    else:
        print("[LOAD] prefill engine from cache")
        if save_dir:
            prefill_engine = save_dir / "prefill.engine"

    if args.dynamic_decode:
        if save_dir:
            decode_dynamic_engine = maybe_load_engine(save_dir / "decode_dynamic.engine")
        if decode_dynamic_engine is None:
            if args.run_only:
                raise FileNotFoundError("decode_dynamic.engine not found in save-engines-dir")
            print("[BUILD] dynamic decode engine ...")
            decode_dynamic_engine = decode_dynamic_mod.build_engine(
                decode_dynamic_mod.CheckpointReader(args.checkpoint),
                refs,
                num_layers=args.num_layers,
                precision=args.precision,
                past_min=1,
                past_opt=initial_past_len,
                past_max=max(initial_past_len + args.max_new_tokens + 8, 512),
            )
            if save_dir:
                (save_dir / "decode_dynamic.engine").write_bytes(decode_dynamic_engine)
                decode_dynamic_engine = save_dir / "decode_dynamic.engine"
        else:
            print("[LOAD] dynamic decode engine from cache")
            if save_dir:
                decode_dynamic_engine = save_dir / "decode_dynamic.engine"
    else:
        print("[BUILD] decode engines ...")
        for step_idx in range(1, args.max_new_tokens):
            past_len = initial_past_len + (step_idx - 1)
            cached = None
            if save_dir:
                cached = maybe_load_engine(save_dir / f"decode_past{past_len}.engine")
            if cached is not None:
                decode_engines[past_len] = save_dir / f"decode_past{past_len}.engine" if save_dir else cached
                print(f"[LOAD] decode engine past_len={past_len} from cache")
            else:
                if args.run_only:
                    raise FileNotFoundError(
                        f"decode_past{past_len}.engine not found in save-engines-dir"
                    )
                engine_bytes = decode_mod.build_engine(
                    decode_mod.CheckpointReader(args.checkpoint),
                    refs,
                    num_layers=args.num_layers,
                    precision=args.precision,
                    past_len=past_len,
                )
                if save_dir:
                    path = save_dir / f"decode_past{past_len}.engine"
                    path.write_bytes(engine_bytes)
                    decode_engines[past_len] = path
                    del engine_bytes
                else:
                    decode_engines[past_len] = engine_bytes

    if args.build_only:
        print("[DONE] build-only mode complete")
        return

    prefill_engine_src = load_engine_source(prefill_engine)
    prefill_session = make_session(prefill_engine_src)
    del prefill_engine_src
    trim_memory()

    print("[RUN] prefill ...")
    prefill_refs = load_file("/data/wy/wall-x/workspace/trt_spike/tmp/layer0_ref.safetensors", device="cpu")
    prefill_outputs, ok, prefill_ms = run_prefill_session(
        prefill_session,
        prefill_refs,
        args.num_layers,
        precision=args.precision,
    )
    del prefill_refs
    if not ok:
        raise RuntimeError("prefill session.run failed")

    del prefill_session
    trim_memory()

    generated = []
    step_logits = []
    prefill_logits = prefill_outputs.pop("last_logits")
    next_token = prefill_logits.argmax(dim=-1)
    generated.append(int(next_token.item()))
    if args.save_step_logits:
        step_logits.append(prefill_logits.detach().float().cpu().tolist())
    del prefill_logits

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.precision]
    embed_weight = (ckpt.get("model.embed_tokens.weight")).to("cuda").to(torch_dtype)
    current_pos = int(refs["decode_position_ids"][0, 0, 0].item())
    decode_times = []
    past = {}
    for i in range(args.num_layers):
        past[f"prefill_past_key_{i}"] = prefill_outputs[f"present_past_key_{i}"]
        past[f"prefill_past_value_{i}"] = prefill_outputs[f"present_past_value_{i}"]
    del prefill_outputs
    trim_memory()

    for step in range(1, args.max_new_tokens):
        token_id = next_token.view(1, 1)
        token_embed = torch.nn.functional.embedding(token_id, embed_weight)
        pos_ids, cos_m, sin_m = compute_decode_cos_sin(
            config, inv_freq.to("cuda"), current_pos, device="cuda", dtype=torch_dtype
        )
        inputs = {
            "decode_inputs_embeds": token_embed,
            "decode_cos_mrope": cos_m,
            "decode_sin_mrope": sin_m,
        }
        inputs.update(past)
        current_past_len = next(iter(past.values())).shape[2]
        if args.dynamic_decode:
            if step == 1:
                decode_engine_src = load_engine_source(decode_dynamic_engine)
                decode_session = make_session(decode_engine_src)
                del decode_engine_src
                trim_memory()
        else:
            decode_engine_src = load_engine_source(decode_engines[current_past_len])
            decode_session = make_session(decode_engine_src)
            del decode_engine_src
            trim_memory()
        outputs, ok, ms = run_decode_session(
            decode_session, inputs, args.num_layers, precision=args.precision
        )
        if not ok:
            raise RuntimeError(f"decode step {step} failed")
        decode_times.append(ms)
        decode_logits = outputs.pop("decode_last_logits")
        if args.save_step_logits:
            step_logits.append(decode_logits.detach().float().cpu().tolist())
        next_token = decode_logits.argmax(dim=-1)
        generated.append(int(next_token.item()))
        current_pos += 1
        del decode_logits
        if not args.dynamic_decode:
            del decode_session
        del inputs, token_embed, pos_ids, cos_m, sin_m
        old_past = past
        past = {}
        for i in range(args.num_layers):
            past[f"prefill_past_key_{i}"] = outputs[f"present_past_key_{i}"]
            past[f"prefill_past_value_{i}"] = outputs[f"present_past_value_{i}"]
        del old_past
        del outputs
        trim_memory()

    if args.dynamic_decode:
        del decode_session
        trim_memory()

    ref_tokens = refs["greedy_token_ids"].tolist()
    match = generated == ref_tokens[: len(generated)]
    print(f"[RESULT] prefill: {prefill_ms:.3f} ms")
    print(f"[RESULT] decode mean: {np.mean(decode_times):.3f} ms")
    print(f"[RESULT] total: {prefill_ms + sum(decode_times):.3f} ms")
    print(f"[RESULT] generated: {generated}")
    print(f"[RESULT] reference: {ref_tokens}")
    print(f"[RESULT] token_match_prefix: {match}")

    if args.save_step_logits and save_dir:
        payload = {
            "generated": generated,
            "reference_prefix": ref_tokens[: len(generated)],
            "prefill_ms": prefill_ms,
            "decode_times_ms": decode_times,
            "step_logits": step_logits,
        }
        with open(save_dir / "trt_step_logits.json", "w") as f:
            json.dump(payload, f)
        print(f"[SAVE] {save_dir / 'trt_step_logits.json'}")

    del past, embed_weight, refs, ckpt, decode_engines, decode_dynamic_engine, next_token
    trim_memory()


if __name__ == "__main__":
    main()
