from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib


@dataclass(slots=True)
class CheckpointSpec:
    module_name: str
    backend: str
    checkpoint: str


class CheckpointManager:
    """Unified checkpoint presence validation with clear error messages."""

    @staticmethod
    def ensure_exists(spec: CheckpointSpec) -> Path:
        path = Path(spec.checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"{spec.module_name} checkpoint is missing: '{path}'. "
                f"Set a valid checkpoint path for backend={spec.backend}."
            )
        if path.is_dir():
            raise FileNotFoundError(
                f"{spec.module_name} checkpoint path points to a directory: '{path}'."
            )
        return path


class BackendInferenceEngine:
    """Tiny backend wrapper that executes real torch/onnxruntime calls when available."""

    def __init__(self, module_name: str, backend: str, checkpoint: str) -> None:
        self.module_name = module_name
        self.backend = backend
        self.checkpoint = checkpoint
        self._session: Any = None
        self._torch_model: Any = None
        self._loaded = False

    def _load(self) -> None:
        ckpt_path = CheckpointManager.ensure_exists(
            CheckpointSpec(self.module_name, self.backend, self.checkpoint)
        )
        if self.backend == "onnx":
            try:
                import onnxruntime as ort  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("onnxruntime is not installed for ONNX backend") from exc
            self._session = ort.InferenceSession(str(ckpt_path), providers=["CPUExecutionProvider"])
            self._loaded = True
            return

        if self.backend == "torch":
            try:
                import torch
                import torch.nn as nn
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("torch is not installed for Torch backend") from exc

            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8), nn.Sigmoid())
            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                model.load_state_dict(state, strict=False)
            self._torch_model = model.eval()
            self._loaded = True
            return

        raise ValueError(f"Unsupported backend '{self.backend}' for module {self.module_name}")

    def infer(self, features: list[float]) -> list[float]:
        if not self._loaded:
            self._load()

        x = features[:8] + [0.0] * max(0, 8 - len(features))
        if self.backend == "onnx":
            import numpy as np  # type: ignore

            assert self._session is not None
            name = self._session.get_inputs()[0].name
            out = self._session.run(None, {name: np.array([x], dtype=np.float32)})
            arr = out[0][0].tolist()
            return [float(v) for v in arr]

        if self.backend == "torch":
            import torch

            assert self._torch_model is not None
            with torch.no_grad():
                y = self._torch_model(torch.tensor([x], dtype=torch.float32)).squeeze(0)
            return [float(v) for v in y.tolist()]

        raise ValueError(self.backend)


def image_ref_to_features(image_ref: str) -> list[float]:
    digest = hashlib.sha256(image_ref.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:8]]
