#!/usr/bin/env python3
"""
Minimal policy adapter for wall-x Flow Action on top of the existing
TensorRT-Edge-LLM custom bridge.

This is intentionally small: it reuses the same bridge + host Euler loop
that already benchmarks on Orin, but exposes a policy-like interface so it
can be wired into wall_x.serving without changing the core bridge code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from safetensors.torch import load_file

from run_wallx_action_engine import run_once_full


@dataclass
class WallXFlowPolicyConfig:
    engine_dir: str
    ref_path: str
    model_path: str = "/data/wy/models/wall-oss-flow"


class WallXFlowPolicyBridge:
    def __init__(self, config: WallXFlowPolicyConfig):
        self.config = config
        self.refs = load_file(config.ref_path, device="cpu")
        self.engine_dir = config.engine_dir
        self.model_path = config.model_path

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # This policy adapter does not consume observation dicts yet; it
        # exposes the existing Flow bridge in policy form so we can wire the
        # custom runtime into serving first, then add observation mapping.
        pred = run_once_full(
            engine_dir=self.engine_dir,
            ref_path=self.config.ref_path,
            model_path=self.model_path,
        )
        return {
            "action": pred["flow_final_action"].detach().cpu().numpy(),
            "server_timing": pred.get("timing", {}),
        }

    def reset(self) -> None:
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "edge_flow_bridge",
            "engine_dir": self.engine_dir,
            "model_path": self.model_path,
        }
