from __future__ import annotations

from training.base_trainer import BaseTrainer
from training.datasets import RepresentationDataset
from training.types import TrainingConfig


class RepresentationTrainer(BaseTrainer):
    stage_name = "representation"

    def build_datasets(self, config: TrainingConfig) -> tuple[RepresentationDataset, RepresentationDataset]:
        return RepresentationDataset.synthetic(config.train_size), RepresentationDataset.synthetic(config.val_size)
