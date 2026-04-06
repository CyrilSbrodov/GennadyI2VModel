from __future__ import annotations

from training.dynamics_trainer import DynamicsTrainer
from training.perception_trainer import PerceptionTrainer
from training.renderer_trainer import RendererTrainer
from training.representation_trainer import RepresentationTrainer
from training.types import StageResult, StageTrainer, TrainingConfig


def _build_stage_trainers() -> dict[str, StageTrainer]:
    trainers: list[StageTrainer] = [
        PerceptionTrainer(),
        RepresentationTrainer(),
        DynamicsTrainer(),
        RendererTrainer(),
    ]
    return {trainer.stage_name: trainer for trainer in trainers}


def train_stage(stage_name: str, config: TrainingConfig) -> StageResult:
    trainers = _build_stage_trainers()
    if stage_name not in trainers:
        known = ", ".join(sorted(trainers))
        raise ValueError(f"Unknown stage '{stage_name}'. Available stages: {known}")
    return trainers[stage_name].train(config)


def train_pipeline(config: TrainingConfig) -> list[StageResult]:
    return [train_stage(stage_name, config) for stage_name in config.stage_order]
