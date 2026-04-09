from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dynamics.temporal_transition_encoder import FAMILIES, PHASES, REGION_KEYS, TemporalTransitionEncoder, TemporalTransitionTargets
from training.datasets import TemporalTransitionDataset, TrainingSample
from training.types import StageResult, TrainingConfig


@dataclass(slots=True)
class TemporalTransitionBatch:
    features: np.ndarray
    targets: TemporalTransitionTargets
    source: str


class TemporalTransitionDatasetAdapter:
    @staticmethod
    def sample_to_batch(sample: TrainingSample) -> TemporalTransitionBatch:
        features = np.asarray(sample.get("temporal_transition_features", [0.0] * 128), dtype=np.float64)
        tgt = sample.get("temporal_transition_target", {}) if isinstance(sample.get("temporal_transition_target"), dict) else {}
        family = str(tgt.get("family", "pose_transition"))
        phase = str(tgt.get("phase", "transition"))
        profile = tgt.get("target_profile", {}) if isinstance(tgt.get("target_profile"), dict) else {}
        region_targets = [
            1.0
            if region in (profile.get("primary_regions", []) or [])
            or region in (profile.get("secondary_regions", []) or [])
            or region in (profile.get("context_regions", []) or [])
            else 0.0
            for region in REGION_KEYS
        ]
        targets = TemporalTransitionTargets(
            family_index=FAMILIES.index(family) if family in FAMILIES else 0,
            phase_index=PHASES.index(phase) if phase in PHASES else 1,
            target_profile_regions=region_targets,
            reveal_score=float(tgt.get("reveal_score", 0.0)),
            occlusion_score=float(tgt.get("occlusion_score", 0.0)),
            support_contact_score=float(tgt.get("support_contact_score", 0.0)),
        )
        return TemporalTransitionBatch(features=features, targets=targets, source=str(sample.get("source", "unknown")))


class TemporalTransitionTrainer:
    stage_name = "temporal_transition"

    def __init__(self) -> None:
        self.model = TemporalTransitionEncoder()
        self.dataset_source = "synthetic_temporal_transition_bootstrap"
        self.dataset_diagnostics: dict[str, object] = {}

    def build_datasets(self, config: TrainingConfig) -> tuple[TemporalTransitionDataset, TemporalTransitionDataset]:
        if config.learned_dataset_path:
            payload = json.loads(Path(config.learned_dataset_path).read_text(encoding="utf-8"))
            is_video_manifest = payload.get("manifest_type") == "video_transition_manifest"
            if is_video_manifest:
                manifest_ds = TemporalTransitionDataset.from_video_transition_manifest(config.learned_dataset_path, strict=False)
                self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
                if len(manifest_ds) > 1:
                    split = max(1, int(0.8 * len(manifest_ds)))
                    self.dataset_source = "manifest_video_temporal_transition_primary"
                    train_ds = TemporalTransitionDataset(samples=manifest_ds.samples[:split])
                    val_ds = TemporalTransitionDataset(samples=manifest_ds.samples[split:])
                    train_ds.diagnostics = dict(self.dataset_diagnostics, split="train")
                    val_ds.diagnostics = dict(self.dataset_diagnostics, split="val")
                    return train_ds, val_ds
                if len(manifest_ds) == 1:
                    self.dataset_source = "manifest_video_temporal_transition_primary_single_sample_reused_for_val"
                    return manifest_ds, manifest_ds
                self.dataset_source = "synthetic_temporal_transition_bootstrap_fallback_manifest_empty"
        self.dataset_diagnostics = {}
        return TemporalTransitionDataset.synthetic(config.train_size), TemporalTransitionDataset.synthetic(config.val_size)

    def _iter_batches(self, dataset: TemporalTransitionDataset) -> list[TemporalTransitionBatch]:
        return [TemporalTransitionDatasetAdapter.sample_to_batch(sample) for sample in dataset.samples]

    @staticmethod
    def evaluate_model(model: TemporalTransitionEncoder, batches: list[TemporalTransitionBatch], diagnostics: dict[str, object] | None = None) -> dict[str, float]:
        metrics = {
            "family_accuracy": 0.0,
            "phase_accuracy": 0.0,
            "family_coverage": 0.0,
            "phase_coverage": 0.0,
            "target_profile_precision_proxy": 0.0,
            "target_profile_recall_proxy": 0.0,
            "reveal_score_calibration": 0.0,
            "occlusion_score_calibration": 0.0,
            "support_contact_calibration": 0.0,
            "usable_sample_count": 0.0,
            "invalid_records": float((diagnostics or {}).get("invalid_records", 0)),
            "skipped_records": float((diagnostics or {}).get("skipped_records", 0)),
            "fallback_free_ratio": float((diagnostics or {}).get("fallback_free_ratio", 0.0)),
        }
        if not batches:
            metrics["summary_score"] = 0.0
            metrics["score"] = 0.0
            return metrics

        fam_seen: set[int] = set()
        phase_seen: set[int] = set()
        fam_pred_seen: set[int] = set()
        phase_pred_seen: set[int] = set()
        for batch in batches:
            pred = model.forward(batch.features)
            family_pred = int(np.argmax(pred.family_logits))
            phase_pred = int(np.argmax(pred.phase_logits))
            fam_pred_seen.add(family_pred)
            phase_pred_seen.add(phase_pred)
            fam_seen.add(batch.targets.family_index)
            phase_seen.add(batch.targets.phase_index)
            metrics["family_accuracy"] += 1.0 if family_pred == batch.targets.family_index else 0.0
            metrics["phase_accuracy"] += 1.0 if phase_pred == batch.targets.phase_index else 0.0

            pred_region = pred.target_profile_scores > 0.5
            tgt_region = np.asarray(batch.targets.target_profile_regions, dtype=np.float64) > 0.5
            tp = float(np.sum(pred_region & tgt_region))
            fp = float(np.sum(pred_region & ~tgt_region))
            fn = float(np.sum(~pred_region & tgt_region))
            metrics["target_profile_precision_proxy"] += tp / max(1.0, tp + fp)
            metrics["target_profile_recall_proxy"] += tp / max(1.0, tp + fn)
            metrics["reveal_score_calibration"] += abs(float(pred.reveal_score) - float(batch.targets.reveal_score))
            metrics["occlusion_score_calibration"] += abs(float(pred.occlusion_score) - float(batch.targets.occlusion_score))
            metrics["support_contact_calibration"] += abs(float(pred.support_contact_score) - float(batch.targets.support_contact_score))
            metrics["usable_sample_count"] += 1.0

        n = float(len(batches))
        for key in (
            "family_accuracy",
            "phase_accuracy",
            "target_profile_precision_proxy",
            "target_profile_recall_proxy",
            "reveal_score_calibration",
            "occlusion_score_calibration",
            "support_contact_calibration",
            "usable_sample_count",
        ):
            metrics[key] = round(metrics[key] / n, 6)
        metrics["family_coverage"] = round(float(len(fam_seen)) / max(1.0, float(len(FAMILIES))), 6)
        metrics["phase_coverage"] = round(float(len(phase_seen)) / max(1.0, float(len(PHASES))), 6)
        pred_cov_bonus = 0.5 * (float(len(fam_pred_seen)) / max(1.0, float(len(FAMILIES))) + float(len(phase_pred_seen)) / max(1.0, float(len(PHASES))))
        metrics["summary_score"] = round(
            max(
                0.0,
                0.3 * metrics["family_accuracy"]
                + 0.2 * metrics["phase_accuracy"]
                + 0.15 * metrics["target_profile_precision_proxy"]
                + 0.15 * metrics["target_profile_recall_proxy"]
                + 0.1 * (1.0 - metrics["reveal_score_calibration"])
                + 0.1 * (1.0 - metrics["occlusion_score_calibration"])
                + 0.05 * pred_cov_bonus,
            ),
            6,
        )
        metrics["score"] = metrics["summary_score"]
        return metrics

    def train(self, config: TrainingConfig) -> StageResult:
        train_dataset, val_dataset = self.build_datasets(config)
        train_batches = self._iter_batches(train_dataset)
        val_batches = self._iter_batches(val_dataset)

        train_metrics: dict[str, float] = {}
        for _ in range(config.epochs):
            accum = {
                "family_loss": 0.0,
                "phase_loss": 0.0,
                "target_profile_loss": 0.0,
                "reveal_loss": 0.0,
                "occlusion_loss": 0.0,
                "support_contact_loss": 0.0,
                "total_loss": 0.0,
            }
            for batch in train_batches:
                losses = self.model.train_step(batch.features, batch.targets, lr=config.learning_rate)
                for key in accum:
                    accum[key] += float(losses[key])
            denom = max(1.0, float(len(train_batches)))
            train_metrics = {k.replace("_loss", ""): round(v / denom, 6) for k, v in accum.items()}

        val_metrics = self.evaluate_model(self.model, val_batches, diagnostics=self.dataset_diagnostics)
        stage_dir = Path(config.checkpoint_dir) / self.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        weights_path = stage_dir / "temporal_transition_weights.json"
        self.model.save(str(weights_path))
        checkpoint_path = stage_dir / "latest.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "stage": self.stage_name,
                    "config": asdict(config),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "weights_path": str(weights_path),
                    "dataset_profile": {
                        "source": self.dataset_source,
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "diagnostics": self.dataset_diagnostics,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(stage_name=self.stage_name, train_metrics=train_metrics, val_metrics=val_metrics, checkpoint_path=str(checkpoint_path))
