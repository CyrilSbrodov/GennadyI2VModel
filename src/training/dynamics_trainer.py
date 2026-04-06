from __future__ import annotations

from training.base_trainer import BaseTrainer
from training.datasets import DynamicsDataset
from training.types import TrainingConfig


class DynamicsTrainer(BaseTrainer):
    stage_name = "dynamics"

    def build_datasets(self, config: TrainingConfig) -> tuple[DynamicsDataset, DynamicsDataset]:
        return DynamicsDataset.synthetic(config.train_size), DynamicsDataset.synthetic(config.val_size)
