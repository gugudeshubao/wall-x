#!/usr/bin/env python3
"""Launch a websocket VQA server backed by wall-x or TensorRT-Edge-LLM."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from wall_x.serving.websocket_policy_server import WebsocketPolicyServer
from wall_x.serving.vqa_policy import VQAPolicy, VQAPolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class Args:
    backend: str = "wallx"
    model_path: str = ""
    train_config_path: str = ""
    edge_backend_config: str = ""
    edge_backend_config_qwen25: str = ""
    edge_backend_config_qwen3: str = ""
    image_keys: List[str] = field(default_factory=lambda: ["image", "front_view", "face_view"])
    default_prompt: str = "Describe what you see in this image."
    max_new_tokens: int = 20
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a VQA websocket server")
    parser.add_argument("--backend", choices=["wallx", "edge", "edge_router"], default="wallx")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--train-config-path", default="")
    parser.add_argument("--edge-backend-config", default="")
    parser.add_argument("--edge-backend-config-qwen25", default="")
    parser.add_argument("--edge-backend-config-qwen3", default="")
    parser.add_argument("--image-keys", nargs="*", default=["image", "front_view", "face_view"])
    parser.add_argument("--default-prompt", default="Describe what you see in this image.")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = VQAPolicyConfig(
        backend=args.backend,
        model_path=args.model_path or None,
        train_config_path=args.train_config_path or None,
        edge_backend_config=args.edge_backend_config or None,
        edge_backend_config_qwen25=args.edge_backend_config_qwen25 or None,
        edge_backend_config_qwen3=args.edge_backend_config_qwen3 or None,
        default_prompt=args.default_prompt,
        max_new_tokens=args.max_new_tokens,
        image_keys=args.image_keys,
    )
    policy = VQAPolicy(config)
    metadata = policy.metadata
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logger.info(f"Serving VQA backend={args.backend} on ws://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
