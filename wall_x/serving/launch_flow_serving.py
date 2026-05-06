#!/usr/bin/env python3
"""Launch a websocket Flow serving endpoint backed by the Edge-LLM custom bridge."""

from __future__ import annotations

import argparse
import logging
import socket

from wall_x.serving.flow_policy import FlowPolicy, FlowPolicyConfig
from wall_x.serving.websocket_policy_server import WebsocketPolicyServer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Flow websocket server")
    parser.add_argument(
        "--engine-dir",
        default="/data/wy/wall-x/workspace/edge_llm_exp/edge_engines/wallx_flow_action_step",
    )
    parser.add_argument(
        "--ref-path",
        default="/data/wy/wall-x/workspace/trt_spike/tmp/flow_dummy_1/flow_dummy_reference.safetensors",
    )
    parser.add_argument(
        "--model-path",
        default="/data/wy/models/wall-oss-flow",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    policy = FlowPolicy(
        FlowPolicyConfig(
            engine_dir=args.engine_dir,
            ref_path=args.ref_path,
            model_path=args.model_path,
        )
    )
    metadata = policy.metadata

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "unknown"

    logger.info("Starting Flow bridge server")
    logger.info(f"Server hostname: {hostname}")
    logger.info(f"Server IP: {local_ip}")
    logger.info(f"Server will be available at: ws://{args.host}:{args.port}")
    logger.info(f"Health check endpoint: http://{args.host}:{args.port}/healthz")

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
