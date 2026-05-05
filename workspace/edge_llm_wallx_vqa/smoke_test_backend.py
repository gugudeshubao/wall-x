#!/usr/bin/env python3
"""Smoke test the Edge-LLM VQA backend against one wall-x case."""

import argparse
import json
from pathlib import Path

from PIL import Image

from edge_llm_vqa_backend import EdgeLLMVQABackend, EdgeLLMVQABackendConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the Edge-LLM VQA backend")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--visual-engine-dir", required=True)
    parser.add_argument("--plugin-path", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--compat-mode", default="wallx_vqa")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    backend = EdgeLLMVQABackend(
        EdgeLLMVQABackendConfig(
            build_dir=args.build_dir,
            engine_dir=args.engine_dir,
            multimodal_engine_dir=args.visual_engine_dir,
            plugin_path=args.plugin_path,
            work_root=args.work_root,
            compat_mode=args.compat_mode,
            max_new_tokens=args.max_new_tokens,
        )
    )

    image = Image.open(args.image).convert("RGB")
    result = backend.generate_with_metadata(image, args.question)
    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
