from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dynamics.model import DynamicsInputs, DynamicsModel, DynamicsTargets, decode_prediction, featurize_runtime, targets_from_delta
from planning.transition_engine import PlannedState
from training.base_trainer import BaseTrainer
from training.datasets import DynamicsDataset, TrainingSample
from training.types import StageResult, TrainingConfig


@dataclass(slots=True)
class DynamicsBatch:
    inputs: DynamicsInputs
    targets: DynamicsTargets
    graph_before: object
    graph_before_source: str
    action_tokens: list[str]


class DynamicsDatasetAdapter:
    """Builds project-like dynamics batches from dataset samples (synthetic bootstrap + manifest-backed)."""

    @staticmethod
    def sample_to_batch(sample: TrainingSample, step_index: int = 1) -> DynamicsBatch:
        graph_before = sample["graphs"][0]
        actions = sample.get("actions", [])
        action_tokens = [a.type for a in actions] or ["micro_adjust"]
        target_state = PlannedState(step_index=step_index, labels=action_tokens)
        planner_context = {
            "step_index": float(step_index),
            "total_steps": float(max(2, len(sample.get("graphs", [])) + 1)),
            "phase": "mid" if step_index > 1 else "early",
            "target_duration": 1.5,
        }
        inputs = featurize_runtime(graph_before, target_state, planner_context, None)
        delta = sample.get("deltas", [])[0]
        targets = targets_from_delta(delta)
        return DynamicsBatch(
            inputs=inputs,
            targets=targets,
            graph_before=graph_before,
            graph_before_source=str(sample.get("source", "unknown")),
            action_tokens=action_tokens,
        )


class DynamicsTrainer(BaseTrainer):
    stage_name = "dynamics"

    def build_datasets(self, config: TrainingConfig) -> tuple[DynamicsDataset, DynamicsDataset]:
        if config.learned_dataset_path:
            manifest_ds = DynamicsDataset.from_transition_manifest(config.learned_dataset_path)
            if len(manifest_ds) > 1:
                split = max(1, int(0.8 * len(manifest_ds)))
                return DynamicsDataset(samples=manifest_ds.samples[:split]), DynamicsDataset(samples=manifest_ds.samples[split:])
            return manifest_ds, DynamicsDataset.synthetic(max(1, config.val_size))
        train = DynamicsDataset.synthetic(config.train_size)
        val = DynamicsDataset.synthetic(config.val_size)
        return train, val

    def _iter_batches(self, dataset: DynamicsDataset) -> list[DynamicsBatch]:
        return [DynamicsDatasetAdapter.sample_to_batch(sample, step_index=max(1, idx + 1)) for idx, sample in enumerate(dataset.samples)]

    def _evaluate(self, model: DynamicsModel, batches: list[DynamicsBatch]) -> dict[str, float]:
        metrics = {
            "pose_mse": 0.0,
            "garment_mse": 0.0,
            "visibility_mse": 0.0,
            "expression_mse": 0.0,
            "interaction_mse": 0.0,
            "region_mse": 0.0,
            "contract_valid_ratio": 0.0,
            "conditioning_sensitivity": 0.0,
        }
        if not batches:
            metrics["score"] = 0.0
            return metrics

        for batch in batches:
            prediction = model.forward(batch.inputs)
            losses = model.compute_losses(prediction, batch.targets)
            for head in ("pose", "garment", "visibility", "expression", "interaction", "region"):
                metrics[f"{head}_mse"] += float(losses[f"{head}_loss"])

            decoded = decode_prediction(prediction, scene_graph=batch.graph_before, phase="mid", semantic_reasons=batch.action_tokens)
            contract_ok = bool(decoded.pose_deltas and decoded.region_transition_mode and decoded.affected_regions)
            metrics["contract_valid_ratio"] += 1.0 if contract_ok else 0.0

            # planner conditioning probe
            probe_ctx = dict(step_index=3.0, total_steps=4.0, phase="late", target_duration=2.0)
            probe_inputs = featurize_runtime(batch.graph_before, PlannedState(step_index=3, labels=batch.action_tokens + ["intensity=0.9"]), probe_ctx, None)
            probe_pred = model.forward(probe_inputs)
            metrics["conditioning_sensitivity"] += float(abs(probe_pred.pose[0] - prediction.pose[0]))

        denom = float(len(batches))
        for key in list(metrics):
            metrics[key] = round(metrics[key] / denom, 6)
        metrics["score"] = round(max(0.0, 1.0 - metrics["pose_mse"]), 6)
        return metrics

    def train(self, config: TrainingConfig) -> StageResult:
        train_dataset, val_dataset = self.build_datasets(config)
        train_batches = self._iter_batches(train_dataset)
        val_batches = self._iter_batches(val_dataset)
        model = DynamicsModel()

        train_metrics: dict[str, float] = {}
        for _ in range(config.epochs):
            accum = {"pose_loss": 0.0, "garment_loss": 0.0, "visibility_loss": 0.0, "expression_loss": 0.0, "interaction_loss": 0.0, "region_loss": 0.0, "total_loss": 0.0}
            for batch in train_batches:
                losses = model.train_step(batch.inputs, batch.targets, lr=config.learning_rate)
                for key in accum:
                    accum[key] += float(losses[key])
            denom = max(1.0, float(len(train_batches)))
            train_metrics = {k.replace("_loss", ""): round(v / denom, 6) for k, v in accum.items()}

        val_metrics = self._evaluate(model, val_batches)
        val_metrics["score"] = round(max(0.0, 1.0 - val_metrics["pose_mse"]), 6)

        stage_dir = Path(config.checkpoint_dir) / self.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = stage_dir / "latest.json"
        weights_path = stage_dir / "dynamics_weights.json"
        model.save(str(weights_path))
        checkpoint_path.write_text(
            json.dumps(
                {
                    "stage": self.stage_name,
                    "config": asdict(config),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "weights_path": str(weights_path),
                    "dataset_profile": {
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "source": "manifest" if config.learned_dataset_path else "synthetic_bootstrap",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(stage_name=self.stage_name, train_metrics=train_metrics, val_metrics=val_metrics, checkpoint_path=str(checkpoint_path))

