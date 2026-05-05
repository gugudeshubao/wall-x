#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EDGE_WALLX_VQA_DIR = REPO_ROOT / "workspace" / "edge_llm_wallx_vqa"
if str(EDGE_WALLX_VQA_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_WALLX_VQA_DIR))

from wall_x.serving.vqa_backend import build_vqa_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wall-x or Edge-LLM VQA through one CLI.")
    parser.add_argument("--backend", choices=["wallx", "edge", "edge_router"], default="wallx")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--train-config", default="")
    parser.add_argument("--edge-backend-config", default="")
    parser.add_argument("--edge-backend-config-qwen25", default="")
    parser.add_argument("--edge-backend-config-qwen3", default="")
    parser.add_argument("--edge-router-config", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    edge_backend_config_qwen25 = args.edge_backend_config_qwen25 or None
    edge_backend_config_qwen3 = args.edge_backend_config_qwen3 or None
    if args.edge_router_config:
        router_cfg = json.loads(Path(args.edge_router_config).read_text())
        edge_backend_config_qwen25 = router_cfg.get("edge_backend_config_qwen25") or edge_backend_config_qwen25
        edge_backend_config_qwen3 = router_cfg.get("edge_backend_config_qwen3") or edge_backend_config_qwen3

    backend = build_vqa_backend(
        backend=args.backend,
        model_path=args.model_path or None,
        train_config_path=args.train_config or None,
        edge_backend_config=args.edge_backend_config or None,
        edge_backend_config_qwen25=edge_backend_config_qwen25,
        edge_backend_config_qwen3=edge_backend_config_qwen3,
    )
    image = Image.open(args.image).convert("RGB")

    if hasattr(backend, "generate_with_metadata"):
        result = backend.generate_with_metadata(
            image, args.question, max_new_tokens=args.max_new_tokens
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        answer = backend.generate(image, args.question, max_new_tokens=args.max_new_tokens)
        print(answer)


if __name__ == "__main__":
    main()
