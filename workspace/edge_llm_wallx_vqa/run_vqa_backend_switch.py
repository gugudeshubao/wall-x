#!/usr/bin/env python3
"""Run either wall-x or Edge-LLM VQA backend from one CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def load_edge_backend(config_path: str):
    from vqa_wrapper_edge_llm import EdgeLLMVQAWrapper

    return EdgeLLMVQAWrapper(backend_config_path=config_path)


def load_wallx_backend(model_path: str, train_config_path: str | None):
    import yaml
    from scripts.vqa_inference import VQAWrapper

    train_config = None
    if train_config_path:
        with open(train_config_path, "r") as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
    return VQAWrapper(model_path=model_path, train_config=train_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a VQA backend through a single CLI")
    parser.add_argument("--backend", choices=["edge", "wallx"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-json", default="")

    # Edge backend
    parser.add_argument("--edge-backend-config", default="")

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
