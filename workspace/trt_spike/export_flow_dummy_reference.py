#!/usr/bin/env python3
"""
Export a reproducible dummy Flow Action reference from the original Python wall-x model.

Purpose:
  - keep the first Flow TRT spike aligned with the existing C++ benchmark conditions
  - export prefix/postfix tensors around generate_flow_action()
  - provide exact reference tensors for:
      1. prefetch decoder pass
      2. prefix KV trim
      3. first postfix ODE step
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def prepare_mrope_ready(cos: torch.Tensor, sin: torch.Tensor, mrope_section):
    mrope_section = mrope_section * 2
    cos_m = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(1)
    sin_m = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(1)
    return cos_m, sin_m


def build_explicit_causal_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    neg_large = -1e4
    mask = torch.full((batch_size, 1, seq_len, seq_len), neg_large, dtype=dtype, device=device)
    tri = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    mask = torch.where(tri.unsqueeze(0).unsqueeze(0), torch.zeros_like(mask), mask)
    return mask


def bool_mask_to_additive(mask: torch.Tensor, dtype: torch.dtype):
    neg_large = -1e4
    zeros = torch.zeros_like(mask, dtype=dtype)
    neg = torch.full_like(mask, neg_large, dtype=dtype)
    return torch.where(mask, zeros, neg)


def load_model(model_path, attn_impl="sdpa", device="cuda"):
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

    config_path = os.path.join(model_path, "config.json")
    model_config = Qwen2_5_VLMoEForAction.config_class.from_pretrained(config_path)
    model_config._attn_implementation = attn_impl

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = Qwen2_5_VLMoEForAction(model_config, processor=processor)
    model.resize_token_embeddings(len(processor.tokenizer))

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    state_dict = {}
    for f in safetensor_files:
        sd = load_file(f, device="cpu")
        state_dict.update(sd)
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    if hasattr(model, "action_preprocessor") and hasattr(model.action_preprocessor, "action_proj_back"):
        model.action_preprocessor.action_proj_back = model.action_preprocessor.action_proj_back.to(torch.float32)
    processor.tokenizer.padding_side = "left"
    return model, processor


def create_dummy_inputs(model, device="cuda"):
    config = model.config
    text_len = 200
    image_tokens = 256
    action_horizon = 32
    total_len = text_len + image_tokens + action_horizon

    opts_long = dict(dtype=torch.long, device=device)
    input_ids = torch.randint(100, 1000, (1, total_len), **opts_long)
    input_ids[0, text_len : text_len + image_tokens] = config.image_token_id
    input_ids[0, text_len + image_tokens :] = model.action_token_id_set["action_token_id"]

    attention_mask = torch.ones(1, total_len, **opts_long)

    merge_size = config.vision_config.spatial_merge_size
    num_patches = image_tokens * merge_size * merge_size
    patch_size = config.vision_config.patch_size
    in_channels = 3
    temporal_patch = config.vision_config.temporal_patch_size
    pixel_dim = in_channels * temporal_patch * patch_size * patch_size
    pixel_values = torch.randn(num_patches, pixel_dim, dtype=torch.bfloat16, device=device)
    image_grid_thw = torch.tensor([[1, 32, 32]], **opts_long)

    moe_token_types = torch.zeros(1, total_len, **opts_long)
    moe_token_types[0, text_len + image_tokens :] = 1

    action_dim = getattr(config, "action_dim", 20)
    dof_mask = torch.ones(1, action_horizon, action_dim, dtype=torch.float32, device=device)

    agent_pos_mask_dim = getattr(config, "agent_pos_dim", 20)
    agent_pos_mask = torch.ones(1, agent_pos_mask_dim, dtype=torch.float32, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "moe_token_types": moe_token_types,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "num_inference_timesteps": 5,
        "dataset_names": "x2_normal",
        "dof_mask": dof_mask,
        "agent_pos_mask": agent_pos_mask,
        "unnorm": False,
    }


def get_cache_tensors(cache):
    keys = {}
    values = {}
    if hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            keys[i] = cache.key_cache[i]
            values[i] = cache.value_cache[i]
    else:
        for i in range(len(cache.layers)):
            keys[i] = cache.layers[i].keys
            values[i] = cache.layers[i].values
    return keys, values


def trim_cache_prefix(cache, prefix_length: int):
    if hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :prefix_length, :].contiguous()
            cache.value_cache[i] = cache.value_cache[i][:, :, :prefix_length, :].contiguous()
    else:
        for i in range(len(cache.layers)):
            cache.layers[i].keys = cache.layers[i].keys[:, :, :prefix_length, :].contiguous()
            cache.layers[i].values = cache.layers[i].values[:, :, :prefix_length, :].contiguous()


@torch.no_grad()
def export_reference(model, inputs: dict, seed: int):
    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    moe_token_types = inputs["moe_token_types"]
    action_horizon = inputs["action_horizon"]
    action_dim = inputs["action_dim"]
    dof_mask = inputs["dof_mask"]
    dataset_names = inputs["dataset_names"]

    refs = {}
    timings = {}

    t0 = time.perf_counter()
    inputs_embeds = model.model.embed_tokens(input_ids)
    pixel_values = pixel_values.type(model.visual.dtype)
    image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
    mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))
    timings["embed_processing_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    position_ids, rope_deltas = model.get_rope_index(
        input_ids,
        image_grid_thw,
        None,
        None,
        attention_mask,
    )
    group_size = torch.zeros(model.config.num_experts, dtype=torch.long, device="cpu")
    for i in range(model.config.num_experts):
        group_size[i] = (moe_token_types == i).sum()
    start_indices = torch.cumsum(group_size, dim=0) - group_size
    end_indices = torch.cumsum(group_size, dim=0)
    cache_position = torch.arange(inputs_embeds.shape[1], device=device)
    prefetch_causal_mask = model.model._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, None, output_attentions=False, moe_token_types=moe_token_types
    )
    if prefetch_causal_mask is None:
        prefetch_causal_mask = build_explicit_causal_mask(
            inputs_embeds.shape[0], inputs_embeds.shape[1], inputs_embeds.dtype, device
        )
    prefetch_cos, prefetch_sin = model.model.rotary_emb(inputs_embeds, position_ids)
    prefetch_cos_m, prefetch_sin_m = prepare_mrope_ready(
        prefetch_cos, prefetch_sin, model.config.rope_scaling["mrope_section"]
    )
    timings["position_encoding_ms"] = (time.perf_counter() - t0) * 1000

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        size=(1, action_horizon, action_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    noisy_action = noise.clone()

    if inputs["num_inference_timesteps"] not in model.times_cache:
        model.times_cache[inputs["num_inference_timesteps"]] = torch.linspace(
            0.0,
            1.0,
            inputs["num_inference_timesteps"] + 1,
            device=device,
            dtype=torch.float32,
        )
    times = model.times_cache[inputs["num_inference_timesteps"]]
    dt = times[1] - times[0]
    time_0 = times[0].unsqueeze(0).repeat(noisy_action.shape[0])

    t0 = time.perf_counter()
    action_embed_0, adarms_cond_0 = model.action_preprocessor.step(
        timestep=time_0, noisy_action=noisy_action, dof_mask=dof_mask
    )
    flow_action_mask = input_ids == model.action_token_id_set["action_token_id"]
    inputs_embeds_t0 = inputs_embeds.clone()
    inputs_embeds_t0[flow_action_mask] = action_embed_0.reshape(-1, inputs_embeds.shape[-1]).to(inputs_embeds.dtype)
    timings["action_initialization_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    prefetch_output = model.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds_t0,
        moe_token_types=moe_token_types,
        start_indices=start_indices,
        end_indices=end_indices,
        positional_masks=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        adarms_conds=[None, adarms_cond_0],
    )
    timings["prefetch_forward_ms"] = (time.perf_counter() - t0) * 1000

    hidden_states = prefetch_output.last_hidden_state
    prefix_kv_cache = prefetch_output.past_key_values
    action_hidden_states_0 = hidden_states[flow_action_mask].to(torch.float32)
    action_pred_0 = model.action_preprocessor.action_proj_back(
        action_hidden_states_0[:, : model.action_preprocessor.action_hidden_size]
    )
    if getattr(model.config, "use_x_pred", False):
        v_0 = action_pred_0 - noise.reshape(-1, noise.shape[-1])
    else:
        v_0 = action_pred_0
    noisy_action_after_prefetch = noisy_action + dt * v_0.reshape(1, action_horizon, action_dim)

    t0 = time.perf_counter()
    has_true = flow_action_mask.any(dim=1)
    prefix_length = torch.argmax(flow_action_mask.float(), dim=1, keepdim=True)
    prefix_length[~has_true] = flow_action_mask.shape[1]
    prefix_length = int(prefix_length[0].item())
    trim_cache_prefix(prefix_kv_cache, prefix_length)

    postfix_position_ids = position_ids[:, :, prefix_length:]
    postfix_inputs_embeds = inputs_embeds_t0[:, prefix_length:, :]
    postfix_attention_mask = attention_mask[:, prefix_length:]
    postfix_moe_token_types = moe_token_types[:, prefix_length:]
    postfix_input_ids = input_ids[:, prefix_length:]

    group_size = torch.zeros(model.config.num_experts, dtype=torch.long, device="cpu")
    for i in range(model.config.num_experts):
        group_size[i] = (postfix_moe_token_types == i).sum()
    postfix_start_indices = torch.cumsum(group_size, dim=0) - group_size
    postfix_end_indices = torch.cumsum(group_size, dim=0)

    pad_token_id = model.processor.tokenizer.pad_token_id
    padding_mask = input_ids == pad_token_id
    postfix_length = input_ids.shape[-1] - prefix_length
    postfix_attention_mask_3d = torch.ones(
        (1, postfix_length, prefix_length + postfix_length),
        dtype=torch.bool,
        device=device,
    )
    postfix_attention_mask_3d[:, :, prefix_length:] = torch.tril(
        torch.ones((postfix_length, postfix_length), dtype=torch.bool, device=device)
    )
    postfix_padding_mask = padding_mask[:, prefix_length:]
    full_padding_mask = padding_mask
    for batch_idx in range(padding_mask.shape[0]):
        postfix_attention_mask_3d[batch_idx, postfix_padding_mask[batch_idx], :] = False
        postfix_attention_mask_3d[batch_idx, :, full_padding_mask[batch_idx]] = False
    postfix_attention_mask_additive_4d = bool_mask_to_additive(
        postfix_attention_mask_3d.unsqueeze(1), postfix_inputs_embeds.dtype
    )
    postfix_cos, postfix_sin = model.model.rotary_emb(postfix_inputs_embeds, postfix_position_ids)
    postfix_cos_m, postfix_sin_m = prepare_mrope_ready(
        postfix_cos, postfix_sin, model.config.rope_scaling["mrope_section"]
    )
    timings["cache_preprocessing_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    timestep_1 = times[1].unsqueeze(0).repeat(noisy_action.shape[0])
    action_embed_1, adarms_cond_1 = model.action_preprocessor.step(
        timestep=timestep_1, noisy_action=noisy_action_after_prefetch, dof_mask=dof_mask
    )
    postfix_action_mask = postfix_input_ids == model.action_token_id_set["action_token_id"]
    temp_inputs_embeds = postfix_inputs_embeds.clone()
    temp_inputs_embeds[postfix_action_mask] = action_embed_1.reshape(-1, postfix_inputs_embeds.shape[-1]).to(temp_inputs_embeds.dtype)
    postfix_output = model.model(
        input_ids=None,
        attention_mask=postfix_attention_mask_3d,
        position_ids=postfix_position_ids,
        past_key_values=prefix_kv_cache,
        inputs_embeds=temp_inputs_embeds,
        moe_token_types=postfix_moe_token_types,
        start_indices=postfix_start_indices,
        end_indices=postfix_end_indices,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        adarms_conds=[None, adarms_cond_1],
    )
    timings["first_postfix_step_ms"] = (time.perf_counter() - t0) * 1000

    postfix_hidden_states = postfix_output.last_hidden_state
    action_hidden_states_1 = postfix_hidden_states[postfix_action_mask].to(torch.float32)
    action_pred_1 = model.action_preprocessor.action_proj_back(
        action_hidden_states_1[:, : model.action_preprocessor.action_hidden_size]
    )
    if getattr(model.config, "use_x_pred", False):
        v_1 = action_pred_1 - noise.reshape(-1, noise.shape[-1])
    else:
        v_1 = action_pred_1

    torch.manual_seed(seed)
    full_out = model.generate_flow_action(**inputs)
    predict_action = full_out["predict_action"]

    refs["input_ids"] = input_ids.cpu()
    refs["attention_mask"] = attention_mask.cpu()
    refs["pixel_values"] = pixel_values.cpu()
    refs["image_grid_thw"] = image_grid_thw.cpu()
    refs["moe_token_types"] = moe_token_types.cpu()
    refs["inputs_embeds_after_scatter"] = inputs_embeds.cpu()
    refs["inputs_embeds_t0"] = inputs_embeds_t0.cpu()
    refs["position_ids"] = position_ids.cpu()
    refs["rope_deltas"] = rope_deltas.cpu()
    refs["start_indices"] = start_indices
    refs["end_indices"] = end_indices
    refs["prefetch_causal_mask_4d"] = prefetch_causal_mask.cpu()
    refs["prefetch_cos"] = prefetch_cos.cpu()
    refs["prefetch_sin"] = prefetch_sin.cpu()
    refs["prefetch_cos_mrope"] = prefetch_cos_m.cpu()
    refs["prefetch_sin_mrope"] = prefetch_sin_m.cpu()
    refs["noise"] = noise.cpu()
    refs["times"] = times.cpu()
    refs["dof_mask"] = dof_mask.cpu()
    refs["flow_action_mask"] = flow_action_mask.cpu()
    refs["prefix_length"] = torch.tensor([prefix_length], dtype=torch.long)
    refs["action_embed_t0"] = action_embed_0.cpu()
    if adarms_cond_0 is not None:
        refs["adarms_cond_t0"] = adarms_cond_0.cpu()
    refs["prefetch_hidden_states"] = hidden_states.cpu()
    refs["action_pred_t0"] = action_pred_0.cpu()
    refs["v_t0"] = v_0.reshape(1, action_horizon, action_dim).cpu()
    refs["noisy_action_after_prefetch"] = noisy_action_after_prefetch.cpu()

    prefix_keys, prefix_values = get_cache_tensors(prefix_kv_cache)
    for i in range(len(prefix_keys)):
        refs[f"prefix_past_key_{i}"] = prefix_keys[i].cpu()
        refs[f"prefix_past_value_{i}"] = prefix_values[i].cpu()

    refs["postfix_position_ids"] = postfix_position_ids.cpu()
    refs["postfix_inputs_embeds"] = postfix_inputs_embeds.cpu()
    refs["postfix_inputs_embeds_t1"] = temp_inputs_embeds.cpu()
    refs["postfix_attention_mask_3d"] = postfix_attention_mask_3d.cpu()
    refs["postfix_attention_mask_additive_4d"] = postfix_attention_mask_additive_4d.cpu()
    refs["postfix_cos"] = postfix_cos.cpu()
    refs["postfix_sin"] = postfix_sin.cpu()
    refs["postfix_cos_mrope"] = postfix_cos_m.cpu()
    refs["postfix_sin_mrope"] = postfix_sin_m.cpu()
    refs["postfix_moe_token_types"] = postfix_moe_token_types.cpu()
    refs["postfix_input_ids"] = postfix_input_ids.cpu()
    refs["postfix_start_indices"] = postfix_start_indices
    refs["postfix_end_indices"] = postfix_end_indices
    refs["postfix_action_mask"] = postfix_action_mask.cpu()
    refs["action_embed_t1"] = action_embed_1.cpu()
    if adarms_cond_1 is not None:
        refs["adarms_cond_t1"] = adarms_cond_1.cpu()
    refs["postfix_hidden_states_t1"] = postfix_hidden_states.cpu()
    refs["action_pred_t1"] = action_pred_1.cpu()
    refs["v_t1"] = v_1.reshape(1, action_horizon, action_dim).cpu()
    refs["predict_action"] = predict_action.cpu()

    manifest = {
        "seed": seed,
        "attn_impl": model.config._attn_implementation,
        "seq_len": int(input_ids.shape[1]),
        "action_horizon": int(action_horizon),
        "action_dim": int(action_dim),
        "num_inference_timesteps": int(inputs["num_inference_timesteps"]),
        "prefix_length": int(prefix_length),
        "postfix_length": int(postfix_input_ids.shape[1]),
        "timings_ms": timings,
    }
    return refs, manifest


def main():
    parser = argparse.ArgumentParser(description="Export dummy Flow Action reference for TRT spike")
    parser.add_argument("--model-path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-impl", default="sdpa")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] model from {args.model_path}")
    t0 = time.perf_counter()
    model, _ = load_model(args.model_path, attn_impl=args.attn_impl)
    print(f"[LOAD] done in {(time.perf_counter() - t0):.1f}s")

    inputs = create_dummy_inputs(model)
    refs, manifest = export_reference(model, inputs, args.seed)

    save_file(refs, str(out_dir / "flow_dummy_reference.safetensors"))
    with open(out_dir / "flow_dummy_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {out_dir / 'flow_dummy_reference.safetensors'}")
    print(f"[SAVE] {out_dir / 'flow_dummy_manifest.json'}")
    print(f"[PREFIX] {manifest['prefix_length']}")
    print(f"[POSTFIX] {manifest['postfix_length']}")


if __name__ == "__main__":
    main()
