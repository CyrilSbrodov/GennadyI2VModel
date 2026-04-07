from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class TrainingConfig:
    stage_order: list[str] = field(
        default_factory=lambda: [
            "perception",
            "representation",
            "text_encoder",
            "dynamics",
            "dynamics_transition",
            "renderer",
            "patch_synthesis",
            "temporal_refinement",
        ]
    )
    epochs: int = 2
    train_size: int = 8
    val_size: int = 4
    batch_size: int = 2
    learning_rate: float = 1e-3
    checkpoint_dir: str = "artifacts/checkpoints"
    dataset_mix: dict[str, float] = field(default_factory=lambda: {"synthetic": 0.3, "real": 0.7})
    freeze_base: bool = True
    train_head_only: bool = True


@dataclass(slots=True)
class StageResult:
    stage_name: str
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    checkpoint_path: str


class StageTrainer(Protocol):
    stage_name: str

    def train(self, config: TrainingConfig) -> StageResult:
        ...


class DistillationHook(Protocol):
    def __call__(self, stage_name: str, result: StageResult) -> dict[str, float]:
        ...
