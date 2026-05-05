#!/usr/bin/env python3
"""Compare wall-x and Edge-LLM VQA backends through one CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
from difflib import SequenceMatcher
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/data/wy/wall-x/venv/bin/python"
DEFAULT_SWITCH_SCRIPT = REPO_ROOT / "scripts" / "vqa_backend_switch.py"


def run_backend(python_bin: str, backend: str, extra_args: list[str], output_json: Path) -> dict:
    cmd = [python_bin, str(DEFAULT_SWITCH_SCRIPT), "--backend", backend, *extra_args, "--output-json", str(output_json)]
    subprocess.run(cmd, check=True)
    return json.loads(output_json.read_text())


def similarity(a: str, b: str) -> float:
    a_norm = " ".join(a.split())
    b_norm = " ".join(b.split())
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare wall-x and Edge-LLM backends")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--edge-backend-config", required=True)
    parser.add_argument("--wallx-model-path", required=True)
    parser.add_argument("--wallx-train-config", default="")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    tmp_dir = REPO_ROOT / "workspace" / "edge_llm_wallx_vqa" / "compare_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    edge_out = tmp_dir / "edge.json"
    wallx_out = tmp_dir / "wallx.json"

    edge = run_backend(
        args.python_bin,
        "edge",
        [
            "--edge-backend-config",
            args.edge_backend_config,
            "--image",
            args.image,
            "--question",
            args.question,
        ],
        edge_out,
    )
    wallx = run_backend(
        args.python_bin,
        "wallx",
        [
            "--wallx-model-path",
            args.wallx_model_path,
            *(
                ["--wallx-train-config", args.wallx_train_config]
                if args.wallx_train_config
                else []
            ),
            "--image",
            args.image,
            "--question",
            args.question,
        ],
        wallx_out,
    )

    result = {
        "image": args.image,
        "question": args.question,
        "edge": {
            "latency_ms": edge.get("latency_ms"),
            "output_text": edge.get("output_text", ""),
        },
        "wallx": {
            "output_text": wallx.get("output_text", ""),
        },
        "similarity": similarity(wallx.get("output_text", ""), edge.get("output_text", "")),
        "edge_request": edge.get("raw_response", {}).get("responses", [{}])[0].get("formatted_complete_request"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
