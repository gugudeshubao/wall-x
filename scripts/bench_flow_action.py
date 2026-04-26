#!/usr/bin/env python3
"""Benchmark: Python Flow Action inference on Orin.
Matches C++ engine conditions: dummy inputs, seq=488, action_horizon=32, ODE steps=5.
"""
import os, sys, time, glob, traceback
import torch
import numpy as np


def load_model(model_path, attn_impl="flash_attention_2", device="cuda"):
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
    from safetensors.torch import load_file

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
    # Fix dtype mismatch: action_proj_back input is float32 but weight is bfloat16
    if hasattr(model, 'action_preprocessor') and hasattr(model.action_preprocessor, 'action_proj_back'):
        model.action_preprocessor.action_proj_back = model.action_preprocessor.action_proj_back.to(torch.float32)
    # FA2 requires padding_side='left'
    processor.tokenizer.padding_side = 'left'
    return model, processor


def create_dummy_inputs(model, processor, device="cuda"):
    """Create dummy inputs matching C++ engine: 200 text + 256 image + 32 action = 488 tokens."""
    config = model.config
    text_len = 200
    image_tokens = 256
    action_horizon = 32
    total_len = text_len + image_tokens + action_horizon

    opts_long = dict(dtype=torch.long, device=device)

    # input_ids
    input_ids = torch.randint(100, 1000, (1, total_len), **opts_long)
    input_ids[0, text_len:text_len + image_tokens] = config.image_token_id
    input_ids[0, text_len + image_tokens:] = model.action_token_id_set["action_token_id"]

    # attention_mask
    attention_mask = torch.ones(1, total_len, **opts_long)

    # pixel_values: patches for 256 image tokens (merge_size=2 -> 1024 patches)
    merge_size = config.vision_config.spatial_merge_size
    num_patches = image_tokens * merge_size * merge_size
    patch_size = config.vision_config.patch_size
    in_channels = 3
    temporal_patch = config.vision_config.temporal_patch_size
    pixel_dim = in_channels * temporal_patch * patch_size * patch_size
    pixel_values = torch.randn(num_patches, pixel_dim, dtype=torch.bfloat16, device=device)

    # image_grid_thw
    image_grid_thw = torch.tensor([[1, 32, 32]], **opts_long)

    # moe_token_types
    moe_token_types = torch.zeros(1, total_len, **opts_long)
    moe_token_types[0, text_len + image_tokens:] = 1

    # dof_mask (action_dim) - must match noisy_action shape: (batch, action_horizon, action_dim)
    action_dim = getattr(config, "action_dim", 20)
    dof_mask = torch.ones(1, action_horizon, action_dim, dtype=torch.float32, device=device)

    # agent_pos_mask
    agent_pos_mask_dim = getattr(config, "agent_pos_dim", 20)
    agent_pos_mask = torch.ones(1, agent_pos_mask_dim, dtype=torch.float32, device=device)

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        moe_token_types=moe_token_types,
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_inference_timesteps=5,
        dataset_names="x2_normal",
        dof_mask=dof_mask,
        agent_pos_mask=agent_pos_mask,
        unnorm=False,
    )


def main():
    model_path = "/data/wy/models/wall-oss-flow"
    warmup_runs = 2
    benchmark_runs = 10

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Use SDPA: FA2 is incompatible with the 3D attention mask in ODE steps.
    # C++ engine also uses SDPA (cuDNN fused path), so this is a fair comparison.
    attn_impl = "sdpa"
    try:
        import flash_attn
        print(f"Flash Attention available: {flash_attn.__version__} (using SDPA for Flow Action)")
    except ImportError:
        print("Flash Attention not available")
    print(f"Attention backend: {attn_impl}")

    print(f"\nLoading model ({attn_impl})...")
    t0 = time.time()
    model, processor = load_model(model_path, attn_impl=attn_impl)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    inputs = create_dummy_inputs(model, processor)
    print(f"Sequence length: {inputs['input_ids'].shape[1]}")
    print(f"Action horizon: {inputs['action_horizon']}")
    print(f"ODE timesteps: {inputs['num_inference_timesteps']}")

    # Warmup
    print(f"\n--- Warmup ({warmup_runs} runs) ---")
    for i in range(warmup_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate_flow_action(**inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timing = out.get("timing_results", {})
        print(f"  Run {i+1}: {(t1-t0)*1000:.1f}ms")

    # Benchmark
    print(f"\n--- Benchmark ({benchmark_runs} runs) ---")
    all_times = []
    all_details = []

    for i in range(benchmark_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate_flow_action(**inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_ms = (t1 - t0) * 1000
        timing = out.get("timing_results", {})
        embed_ms = timing.get("embed_processing", 0) * 1000
        prefetch_ms = timing.get("prefetch_forward", 0) * 1000
        ode_ms = timing.get("ode_integration", 0) * 1000
        other_ms = total_ms - embed_ms - prefetch_ms - ode_ms

        all_times.append(total_ms)
        all_details.append((embed_ms, prefetch_ms, ode_ms, other_ms))

        print(f"  Run {i+1}: total={total_ms:.1f}ms, "
              f"embed+vit={embed_ms:.1f}ms, prefill={prefetch_ms:.1f}ms, "
              f"ode={ode_ms:.1f}ms, other={other_ms:.1f}ms")

    # Results
    avg_total = np.mean(all_times)
    std_total = np.std(all_times)
    avg_details = np.mean(all_details, axis=0)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n========== PYTHON FLOW ACTION BENCHMARK ==========")
    print(f"  Attention:       {attn_impl}")
    print(f"  Average total:   {avg_total:.1f} ms (std {std_total:.1f})")
    print(f"  Avg embed+ViT:   {avg_details[0]:.1f} ms")
    print(f"  Avg prefill:     {avg_details[1]:.1f} ms")
    print(f"  Avg ODE:         {avg_details[2]:.1f} ms")
    print(f"  Avg other:       {avg_details[3]:.1f} ms")
    print(f"  Throughput:      {1000/avg_total:.2f} infer/s")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")
    print(f"===================================================")


if __name__ == "__main__":
    main()
