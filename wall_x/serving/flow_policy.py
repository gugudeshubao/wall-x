from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from wall_x.serving.websocket_policy_server import BasePolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOW_BRIDGE_DIR = REPO_ROOT / "workspace" / "edge_llm_wallx_flow"
if str(FLOW_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(FLOW_BRIDGE_DIR))

from run_wallx_action_engine import (  # noqa: E402
    WallXFlowActionRunner,
    WallXFlowActionRunnerConfig,
)


@dataclass
class FlowPolicyConfig:
    engine_dir: str
    ref_path: str
    model_path: str = "/data/wy/models/wall-oss-flow"


class FlowPolicy(BasePolicy):
    """Policy wrapper for the Edge-LLM custom Flow bridge."""

    def __init__(self, config: FlowPolicyConfig):
        self.config = config
        self.runner = WallXFlowActionRunner(
            WallXFlowActionRunnerConfig(
                engine_dir=config.engine_dir,
                ref_path=config.ref_path,
                model_path=config.model_path,
            )
        )

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        result = self.runner.run_once_full()
        action = result["flow_final_action"].detach().cpu().numpy()
        return {
            "action": action,
            "predict_action": action,
            "flow": {
                "action_step_ms": result["action_step_ms"],
                "flow_total_ms": result["flow_total_ms"],
                "flow_step_ms_mean": float(
                    result["flow_step_times_ms"].mean()
                    if result["flow_step_times_ms"].size
                    else 0.0
                ),
            },
            "server_timing": {
                "infer_ms": result["flow_total_ms"],
            },
        }

    def reset(self) -> None:
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "edge_flow_bridge",
            "engine_dir": self.config.engine_dir,
            "ref_path": self.config.ref_path,
            "model_path": self.config.model_path,
        }
