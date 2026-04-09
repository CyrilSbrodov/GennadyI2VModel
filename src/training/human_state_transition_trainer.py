from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dynamics.human_state_transition import FAMILIES, PHASES, HumanStateTransitionModel, HumanStateTransitionTargets
from training.datasets import HumanStateTransitionDataset, TrainingSample
from training.types import StageResult, TrainingConfig


@dataclass(slots=True)
class HumanStateTransitionBatch:
    features: np.ndarray
    targets: HumanStateTransitionTargets
    previous_state_hint: np.ndarray | None
    source: str


class HumanStateTransitionDatasetAdapter:
    @staticmethod
    def sample_to_batch(sample: TrainingSample) -> HumanStateTransitionBatch:
        features = np.asarray(sample.get("human_state_transition_features", [0.0] * 160), dtype=np.float64)
        tgt = sample.get("human_state_transition_target", {}) if isinstance(sample.get("human_state_transition_target"), dict) else {}
        family = str(tgt.get("family", "pose_transition"))
        phase = str(tgt.get("phase", "transition"))
        region_targets = [float(x) for x in tgt.get("region_state_targets", [0.0] * 8)]
        visibility_targets = [float(x) for x in tgt.get("visibility_targets", [0.0] * 8)]
        targets = HumanStateTransitionTargets(
            family_index=FAMILIES.index(family) if family in FAMILIES else 0,
            phase_index=PHASES.index(phase) if phase in PHASES else 1,
            region_state_targets=region_targets[:8],
            visibility_targets=visibility_targets[:8],
            reveal_memory_target=float(tgt.get("reveal_memory_target", 0.0)),
            support_contact_target=float(tgt.get("support_contact_target", 0.0)),
        )
        hist = sample.get("human_state_history", {}) if isinstance(sample.get("human_state_history"), dict) else {}
        prev = hist.get("previous_state_hint") if isinstance(hist.get("previous_state_hint"), list) else None
        prev_np = np.asarray(prev, dtype=np.float64) if isinstance(prev, list) and prev else None
        return HumanStateTransitionBatch(features=features, targets=targets, previous_state_hint=prev_np, source=str(sample.get("source", "unknown")))


class HumanStateTransitionTrainer:
    stage_name = "human_state_transition"

    def __init__(self) -> None:
        self.model = HumanStateTransitionModel()
        self.dataset_source = "synthetic_human_state_transition_bootstrap"
        self.dataset_diagnostics: dict[str, object] = {}

    def build_datasets(self, config: TrainingConfig) -> tuple[HumanStateTransitionDataset, HumanStateTransitionDataset]:
        if config.learned_dataset_path:
            payload = json.loads(Path(config.learned_dataset_path).read_text(encoding="utf-8"))
            if payload.get("manifest_type") == "video_transition_manifest":
                manifest_ds = HumanStateTransitionDataset.from_video_transition_manifest(
                    config.learned_dataset_path,
                    strict=False,
                )
                self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
                if len(manifest_ds) > 1:
                    split = max(1, int(0.8 * len(manifest_ds)))
                    self.dataset_source = "manifest_video_human_state_transition_primary"
                    train_ds = HumanStateTransitionDataset(samples=manifest_ds.samples[:split])
                    val_ds = HumanStateTransitionDataset(samples=manifest_ds.samples[split:])
                    train_ds.diagnostics = dict(self.dataset_diagnostics, split="train")
                    val_ds.diagnostics = dict(self.dataset_diagnostics, split="val")
                    return train_ds, val_ds
                if len(manifest_ds) == 1:
                    self.dataset_source = "manifest_video_human_state_transition_primary_single_sample_reused_for_val"
                    return manifest_ds, manifest_ds

        self.dataset_diagnostics = {}
        self.dataset_source = "synthetic_human_state_transition_bootstrap"

        total = max(2, int(config.train_size) + int(config.val_size))
        synthetic_samples: list[TrainingSample] = []
        families = ["pose_transition", "garment_transition", "expression_transition", "interaction_transition",
                    "visibility_transition"]
        phases = ["prepare", "transition", "contact_or_reveal", "stabilize"]

        for i in range(total):
            family = families[i % len(families)]
            phase = phases[i % len(phases)]

            sample: TrainingSample = {
                "source": "synthetic_human_state_transition_bootstrap",
                "human_state_transition_features": [0.01 * float(i + j + 1) for j in range(160)],
                "human_state_transition_target": {
                    "family": family,
                    "phase": phase,
                    "target_profile": {
                        "primary_regions": ["torso"],
                        "secondary_regions": ["face"],
                        "context_regions": ["legs"],
                    },
                    "region_state_targets": [0.25] * 8,
                    "visibility_targets": [0.75] * 8,
                    "reveal_memory_target": 0.35,
                    "support_contact_target": 0.1,
                },
                "human_state_history": {
                    "has_history": i > 0,
                    "previous_state_hint": [0.0] * 24,
                },
            }
            synthetic_samples.append(sample)

        train_count = max(1, int(config.train_size))
        val_count = max(1, int(config.val_size))

        train_ds = HumanStateTransitionDataset(samples=synthetic_samples[:train_count])
        val_ds = HumanStateTransitionDataset(samples=synthetic_samples[train_count: train_count + val_count])

        train_ds.diagnostics = {
            "source": self.dataset_source,
            "split": "train",
            "usable": len(train_ds.samples),
            "invalid": 0,
            "skipped": 0,
        }
        val_ds.diagnostics = {
            "source": self.dataset_source,
            "split": "val",
            "usable": len(val_ds.samples),
            "invalid": 0,
            "skipped": 0,
        }

        return train_ds, val_ds

    def _iter_batches(self, dataset: HumanStateTransitionDataset) -> list[HumanStateTransitionBatch]:
        return [HumanStateTransitionDatasetAdapter.sample_to_batch(sample) for sample in dataset.samples]

    @staticmethod
    def evaluate_model(model: HumanStateTransitionModel, batches: list[HumanStateTransitionBatch], diagnostics: dict[str, object] | None = None) -> dict[str, float]:
        metrics = {
            "state_transition_consistency": 0.0,
            "reveal_memory_quality": 0.0,
            "visibility_state_accuracy": 0.0,
            "region_state_accuracy_proxy": 0.0,
            "temporal_smoothness_proxy": 0.0,
            "human_state_summary_score": 0.0,
            "usable_sample_count": float(len(batches)),
            "invalid_records": float((diagnostics or {}).get("invalid", 0)),
            "skipped_records": float((diagnostics or {}).get("skipped", 0)),
        }
        if not batches:
            metrics["score"] = 0.0
            return metrics

        for batch in batches:
            pred = model.forward(batch.features)
            losses = model.compute_losses(pred, batch.targets, previous_state_embedding=batch.previous_state_hint)
            metrics["state_transition_consistency"] += float(max(0.0, 1.0 - losses["state_transition_consistency_loss"]))
            metrics["reveal_memory_quality"] += float(max(0.0, 1.0 - losses["reveal_memory_alignment_loss"]))
            metrics["visibility_state_accuracy"] += float(max(0.0, 1.0 - losses["visibility_state_loss"]))
            metrics["region_state_accuracy_proxy"] += float(max(0.0, 1.0 - losses["region_state_prediction_loss"]))
            metrics["temporal_smoothness_proxy"] += float(max(0.0, 1.0 - losses["temporal_smoothness_loss"]))

        n = float(len(batches))
        for k in ("state_transition_consistency", "reveal_memory_quality", "visibility_state_accuracy", "region_state_accuracy_proxy", "temporal_smoothness_proxy"):
            metrics[k] = round(metrics[k] / n, 6)
        metrics["human_state_summary_score"] = round(
            np.clip(
                0.30 * metrics["state_transition_consistency"]
                + 0.20 * metrics["reveal_memory_quality"]
                + 0.20 * metrics["visibility_state_accuracy"]
                + 0.20 * metrics["region_state_accuracy_proxy"]
                + 0.10 * metrics["temporal_smoothness_proxy"],
                0.0,
                1.0,
            ).item(),
            6,
        )
        metrics["score"] = metrics["human_state_summary_score"]
        return metrics

    def train(self, config: TrainingConfig) -> StageResult:
        train_ds, val_ds = self.build_datasets(config)
        train_batches = self._iter_batches(train_ds)
        val_batches = self._iter_batches(val_ds)
        train_metrics: dict[str, float] = {}
        for _ in range(config.epochs):
            accum = {
                "state_transition_consistency_loss": 0.0,
                "reveal_memory_alignment_loss": 0.0,
                "visibility_state_loss": 0.0,
                "region_state_prediction_loss": 0.0,
                "temporal_smoothness_loss": 0.0,
                "support_contact_state_loss": 0.0,
                "total_loss": 0.0,
            }
            for b in train_batches:
                losses = self.model.train_step(b.features, b.targets, lr=config.learning_rate, previous_state_embedding=b.previous_state_hint)
                for k in accum:
                    accum[k] += float(losses[k])
            denom = max(1.0, float(len(train_batches)))
            train_metrics = {k.replace("_loss", ""): round(v / denom, 6) for k, v in accum.items()}

        val_metrics = self.evaluate_model(self.model, val_batches, diagnostics=self.dataset_diagnostics)
        stage_dir = Path(config.checkpoint_dir) / self.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        weights_path = stage_dir / "human_state_transition_weights.json"
        self.model.save(str(weights_path))
        ckpt = stage_dir / "latest.json"
        ckpt.write_text(
            json.dumps(
                {
                    "stage": self.stage_name,
                    "config": asdict(config),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "weights_path": str(weights_path),
                    "dataset_profile": {
                        "source": self.dataset_source,
                        "train_samples": len(train_ds),
                        "val_samples": len(val_ds),
                        "diagnostics": self.dataset_diagnostics,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(stage_name=self.stage_name, train_metrics=train_metrics, val_metrics=val_metrics, checkpoint_path=str(ckpt))
