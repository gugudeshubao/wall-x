#!/usr/bin/env python3
"""Run wall-x or Edge-LLM VQA through one CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
EDGE_WALLX_VQA_DIR = REPO_ROOT / "workspace" / "edge_llm_wallx_vqa"
if str(EDGE_WALLX_VQA_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_WALLX_VQA_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_edge_backend(config_path: str):
    from vqa_wrapper_edge_llm import EdgeLLMVQAWrapper

    return EdgeLLMVQAWrapper(backend_config_path=config_path)


def build_minimal_wallx_train_config(model_path: str) -> dict:
    config_json = Path(model_path) / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"Could not find {config_json}")
    model_cfg = json.loads(config_json.read_text())
    return {
        "processor_path": model_path,
        "data": {
            "use_state_string_representation": False,
            "action_horizon_flow": model_cfg.get("action_horizon_flow", 32),
        },
        "dof_config": model_cfg.get("dof_config", {}),
        "agent_pos_config": model_cfg.get("agent_pos_config", {}),
    }


def parse_edge_router_config(path: str) -> dict[str, str]:
    data = json.loads(Path(path).read_text())
    return {
        "edge_backend_config_qwen25": data["edge_backend_config_qwen25"],
        "edge_backend_config_qwen3": data["edge_backend_config_qwen3"],
    }


def load_wallx_backend(model_path: str, train_config_path: str | None):
    import yaml
    from vqa_inference import VQAWrapper

    train_config = None
    if train_config_path and os.path.exists(train_config_path):
        with open(train_config_path, "r") as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        train_config = build_minimal_wallx_train_config(model_path)
    return VQAWrapper(model_path=model_path, train_config=train_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wall-x or Edge-LLM VQA from one CLI")
    parser.add_argument("--backend", choices=["edge", "wallx", "edge_router"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-json", default="")

    # Edge backend
    parser.add_argument("--edge-backend-config", default="")
    parser.add_argument("--edge-router-config", default="")

    # wall-x backend
    parser.add_argument("--wallx-model-path", default="")
    parser.add_argument("--wallx-train-config", default="")

    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")

    if args.backend == "edge":
        if not args.edge_backend_config:
            raise SystemExit("--edge-backend-config is required for backend=edge")
        backend = load_edge_backend(args.edge_backend_config)
        result = backend.generate_with_metadata(image, args.question)
    elif args.backend == "edge_router":
        if not args.edge_router_config:
            raise SystemExit("--edge-router-config is required for backend=edge_router")
        from wall_x.serving.vqa_backend import build_vqa_backend

        router_cfg = parse_edge_router_config(args.edge_router_config)
        backend = build_vqa_backend(
            backend="edge_router",
            edge_backend_config_qwen25=router_cfg["edge_backend_config_qwen25"],
            edge_backend_config_qwen3=router_cfg["edge_backend_config_qwen3"],
        )
        result = backend.generate_with_metadata(image, args.question)
    else:
        if not args.wallx_model_path:
            raise SystemExit("--wallx-model-path is required for backend=wallx")
        backend = load_wallx_backend(args.wallx_model_path, args.wallx_train_config or None)
        answer = backend.generate(image, args.question, max_new_tokens=20)
        result = {
            "backend": "wallx",
            "image": args.image,
            "question": args.question,
            "output_text": answer,
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
