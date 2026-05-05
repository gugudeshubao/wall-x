from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from PIL import Image

from wall_x.serving.vqa_backend import build_vqa_backend

try:
    from wall_x.serving.websocket_policy_server import BasePolicy
except ImportError:
    class BasePolicy:  # type: ignore[override]
        def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
            raise NotImplementedError

        def reset(self) -> None:
            return None

        @property
        def metadata(self) -> Dict[str, Any]:
            return {}


@dataclass
class VQAPolicyConfig:
    backend: str = "wallx"
    model_path: str | None = None
    train_config_path: str | None = None
    edge_backend_config: str | None = None
    edge_backend_config_qwen25: str | None = None
    edge_backend_config_qwen3: str | None = None
    default_prompt: str = "Describe what you see in this image."
    max_new_tokens: int = 20
    image_keys: List[str] = field(default_factory=lambda: ["image", "front_view", "face_view"])


class VQAPolicy(BasePolicy):
    """A lightweight VQA policy that can back wall-x or TensorRT-Edge-LLM."""

    def __init__(self, config: VQAPolicyConfig):
        self.config = config
        self.backend = build_vqa_backend(
            backend=config.backend,
            model_path=config.model_path,
            train_config_path=config.train_config_path,
            edge_backend_config=config.edge_backend_config,
            edge_backend_config_qwen25=config.edge_backend_config_qwen25,
            edge_backend_config_qwen3=config.edge_backend_config_qwen3,
        )

    def _extract_image(self, obs: Dict[str, Any]) -> Any:
        if "image" in obs:
            return obs["image"]
        for key in self.config.image_keys:
            if key in obs:
                return obs[key]
        raise ValueError(f"Cannot find image in obs. Tried keys: {self.config.image_keys}")

    def _extract_prompt(self, obs: Dict[str, Any]) -> str:
        prompt = obs.get("prompt", None)
        if prompt is None or prompt == "":
            prompt = self.config.default_prompt
        return prompt

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        image = self._extract_image(obs)
        prompt = self._extract_prompt(obs)
        result = self.backend.generate_with_metadata(
            image,
            prompt,
            max_new_tokens=self.config.max_new_tokens,
        )
        return {"vqa": result}

    def reset(self) -> None:
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = {
            "backend": self.config.backend,
            "max_new_tokens": self.config.max_new_tokens,
            "default_prompt": self.config.default_prompt,
        }
        if self.config.model_path:
            meta["model_path"] = self.config.model_path
        if self.config.edge_backend_config:
            meta["edge_backend_config"] = self.config.edge_backend_config
        if self.config.edge_backend_config_qwen25:
            meta["edge_backend_config_qwen25"] = self.config.edge_backend_config_qwen25
        if self.config.edge_backend_config_qwen3:
            meta["edge_backend_config_qwen3"] = self.config.edge_backend_config_qwen3
        return meta
