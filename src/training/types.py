from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class TrainingConfig:
    stage_order: list[str] = field(
        default_factory=lambda: ["perception", "representation", "dynamics", "renderer"]
    )
    epochs: int = 2
    train_size: int = 8
    val_size: int = 4
    batch_size: int = 2
    learning_rate: float = 1e-3
    checkpoint_dir: str = "artifacts/checkpoints"


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
