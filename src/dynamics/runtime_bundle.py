from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dynamics.model import DynamicsModel, DynamicsModelError

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DynamicsCheckpointStatus = Literal[
    "torch_unavailable",
    "bootstrap_only",
    "checkpoint_directory_missing",
    "checkpoint_file_missing",
    "checkpoint_invalid",
    "checkpoint_loaded",
]


@dataclass(slots=True)
class DynamicsRuntimeStatus:
    checkpoint_status: DynamicsCheckpointStatus
    usable_for_inference: bool
    runtime_status: str
    fallback_reason: str | None
    details: dict[str, object]


class DynamicsRuntimeBundle:
    """Centralized runtime/readiness policy for dynamics learned backend."""

    def __init__(self, *, checkpoint_dir: str = "artifacts/checkpoints/dynamics", checkpoint_file: str = "dynamics_weights.json", allow_random_init_for_dev: bool = False) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.allow_random_init_for_dev = allow_random_init_for_dev
        self.torch_available = torch is not None
        self.model = DynamicsModel()
        self.checkpoint_status: DynamicsCheckpointStatus = "torch_unavailable" if not self.torch_available else "bootstrap_only"
        self.checkpoint_details: dict[str, object] = {}

    def runtime_status(self) -> DynamicsRuntimeStatus:
        usable = self.checkpoint_status == "checkpoint_loaded" or (self.allow_random_init_for_dev and self.torch_available)
        return DynamicsRuntimeStatus(
            checkpoint_status=self.checkpoint_status,
            usable_for_inference=bool(usable),
            runtime_status="usable" if usable else "fallback_required",
            fallback_reason=None if usable else self.checkpoint_status,
            details={
                **self.checkpoint_details,
                "checkpoint_dir": self.checkpoint_dir,
                "checkpoint_file": str(Path(self.checkpoint_dir) / self.checkpoint_file),
                "allow_random_init_for_dev": self.allow_random_init_for_dev,
                "torch_available": self.torch_available,
            },
        )

    def load_checkpoint(self) -> DynamicsRuntimeStatus:
        if not self.torch_available:
            self.checkpoint_status = "torch_unavailable"
            self.checkpoint_details = {"reason": "torch_unavailable"}
            return self.runtime_status()
        root = Path(self.checkpoint_dir)
        if not root.exists():
            self.checkpoint_status = "checkpoint_directory_missing"
            self.checkpoint_details = {"reason": "checkpoint_directory_missing"}
            return self.runtime_status()
        path = root / self.checkpoint_file
        if not path.exists():
            self.checkpoint_status = "checkpoint_file_missing"
            self.checkpoint_details = {"reason": "checkpoint_file_missing"}
            return self.runtime_status()
        try:
            self.model = DynamicsModel.load(str(path))
            self.checkpoint_status = "checkpoint_loaded"
            self.checkpoint_details = {"reason": "checkpoint_loaded"}
        except Exception as exc:
            self.checkpoint_status = "checkpoint_invalid"
            self.checkpoint_details = {"reason": "checkpoint_invalid", "error": str(exc)}
            self.model = DynamicsModel()
        return self.runtime_status()

    def save_checkpoint(self) -> str:
        if not self.torch_available:
            raise DynamicsModelError("torch_unavailable")
        root = Path(self.checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / self.checkpoint_file
        self.model.save(str(path))
        self.checkpoint_status = "checkpoint_loaded"
        self.checkpoint_details = {"reason": "checkpoint_saved", "path": str(path)}
        return str(path)
