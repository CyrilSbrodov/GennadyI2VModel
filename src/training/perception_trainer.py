from __future__ import annotations

from training.base_trainer import BaseTrainer
from training.datasets import PerceptionDataset
from training.types import TrainingConfig


class PerceptionTrainer(BaseTrainer):
    stage_name = "perception"

    def build_datasets(self, config: TrainingConfig) -> tuple[PerceptionDataset, PerceptionDataset]:
        return PerceptionDataset.synthetic(config.train_size), PerceptionDataset.synthetic(config.val_size)
