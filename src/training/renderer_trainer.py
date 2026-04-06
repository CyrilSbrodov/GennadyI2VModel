from __future__ import annotations

from training.base_trainer import BaseTrainer
from training.datasets import RendererDataset
from training.types import TrainingConfig


class RendererTrainer(BaseTrainer):
    stage_name = "renderer"

    def build_datasets(self, config: TrainingConfig) -> tuple[RendererDataset, RendererDataset]:
        return RendererDataset.synthetic(config.train_size), RendererDataset.synthetic(config.val_size)
