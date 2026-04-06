from __future__ import annotations

from dynamics.model import DynamicsInputs, DynamicsModel
from dynamics.targets import build_training_targets
from training.base_trainer import BaseTrainer
from training.datasets import DynamicsDataset
from training.types import TrainingConfig


class DynamicsTrainer(BaseTrainer):
    stage_name = "dynamics"

    def build_datasets(self, config: TrainingConfig) -> tuple[DynamicsDataset, DynamicsDataset]:
        train = DynamicsDataset.synthetic(config.train_size)
        val = DynamicsDataset.synthetic(config.val_size)
        model = DynamicsModel()

        for split in (train, val):
            for sample in split.samples:
                graphs = sample.get("graphs", [])
                if graphs:
                    sample["targets"] = build_training_targets(graphs)
                sample["sanity_metrics"] = model.forward(
                    DynamicsInputs(
                        serialized_scene_graph="graph",
                        action_tokens=["move"],
                        planner_context=[0.3],
                        memory_features=[0.4],
                    )
                )

        return train, val
