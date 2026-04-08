from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.schema import BBox, RegionRef, SceneGraph
from learned.interfaces import TemporalRefinementRequest
from memory.video_memory import MemoryManager
from training.datasets import TemporalDataset
from training.types import StageResult, TrainingConfig
from rendering.trainable_temporal_consistency import TrainableTemporalConsistencyModel, TemporalBatch, build_temporal_batch


class TemporalBatchAdapter:
    """Runtime-aligned temporal adapter: previous/composed/target + changed-regions + alpha/confidence/drift context."""

    @staticmethod
    def _as_regions(raw: object) -> list[RegionRef]:
        regions: list[RegionRef] = []
        for idx, region in enumerate(raw if isinstance(raw, list) else []):
            if not isinstance(region, dict):
                continue
            bb = region.get("bbox", {}) if isinstance(region.get("bbox", {}), dict) else {}
            regions.append(
                RegionRef(
                    region_id=str(region.get("region_id", f"scene:region_{idx}")),
                    reason=str(region.get("reason", "temporal_drift")),
                    bbox=BBox(float(bb.get("x", 0.2)), float(bb.get("y", 0.2)), float(bb.get("w", 0.3)), float(bb.get("h", 0.3))),
                )
            )
        if not regions:
            regions = [RegionRef(region_id="scene:region_0", reason="temporal_drift", bbox=BBox(0.2, 0.2, 0.3, 0.3))]
        return regions

    def adapt(self, sample: dict[str, object]) -> TemporalBatch:
        contract = sample.get("temporal_consistency_contract", {}) if isinstance(sample.get("temporal_consistency_contract", {}), dict) else {}
        frames = sample.get("frames", []) if isinstance(sample.get("frames", []), list) else []
        prev = contract.get("previous_frame", frames[0] if len(frames) > 0 else [])
        cur = contract.get("composed_frame", frames[1] if len(frames) > 1 else prev)
        target = contract.get("target_frame", frames[2] if len(frames) > 2 else cur)
        changed_regions = self._as_regions(contract.get("changed_regions", []))

        transition = contract.get("transition_context", contract.get("scene_transition_context", {}))
        transition = transition if isinstance(transition, dict) else {}
        memory_ctx = contract.get("memory_context", contract.get("memory_transition_context", {}))
        memory_ctx = memory_ctx if isinstance(memory_ctx, dict) else {}
        region_meta = contract.get("region_consistency_metadata", {}) if isinstance(contract.get("region_consistency_metadata", {}), dict) else {}
        patch_history = contract.get("patch_history", []) if isinstance(contract.get("patch_history", []), list) else []
        alpha_hint = contract.get("alpha_hint")
        confidence_hint = contract.get("confidence_hint")
        changed_region_mask = contract.get("changed_region_mask")

        graph = SceneGraph(frame_index=int(transition.get("step_index", 0)))
        memory = MemoryManager().initialize(graph)
        req = TemporalRefinementRequest(
            previous_frame=prev,
            current_composed_frame=cur,
            changed_regions=changed_regions,
            scene_state=graph,
            memory_state=memory,
            memory_channels={
                "identity": memory_ctx.get("identity", {}),
                "body_regions": {"roi_count": len(changed_regions), **(memory_ctx.get("body_regions", {}) if isinstance(memory_ctx.get("body_regions", {}), dict) else {})},
                "hidden_regions": {
                    "drift": float(memory_ctx.get("drift", transition.get("drift", 0.0) or 0.0)),
                    **(memory_ctx.get("hidden_regions", {}) if isinstance(memory_ctx.get("hidden_regions", {}), dict) else {}),
                },
                "patch_alpha": {
                    "mean_alpha": float(np.mean(np.asarray(alpha_hint, dtype=np.float32))) if alpha_hint is not None else float(region_meta.get("alpha_mean", 0.45)),
                    "edge_alpha": float(region_meta.get("alpha_edge_mean", region_meta.get("alpha_mean", 0.4))),
                },
                "patch_confidence": {
                    "mean_confidence": float(np.mean(np.asarray(confidence_hint, dtype=np.float32))) if confidence_hint is not None else float(region_meta.get("confidence_mean", 0.7)),
                    "min_confidence": float(region_meta.get("confidence_min", 0.55)),
                },
            },
        )
        history_frame = patch_history[-1].get("frame") if patch_history and isinstance(patch_history[-1], dict) else None
        batch = build_temporal_batch(req, target_frame=target, history_frame=history_frame if isinstance(history_frame, list) else None)
        if changed_region_mask is not None:
            mask = np.asarray(changed_region_mask, dtype=np.float32)
            if mask.ndim == 2:
                mask = mask[..., None]
            if mask.shape == batch.changed_mask.shape:
                batch.changed_mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if alpha_hint is not None:
            ah = np.asarray(alpha_hint, dtype=np.float32)
            if ah.ndim == 2:
                ah = ah[..., None]
            if ah.shape == batch.alpha_hint.shape:
                batch.alpha_hint = np.clip(ah, 0.0, 1.0).astype(np.float32)
        if confidence_hint is not None:
            ch = np.asarray(confidence_hint, dtype=np.float32)
            if ch.ndim == 2:
                ch = ch[..., None]
            if ch.shape == batch.confidence_hint.shape:
                batch.confidence_hint = np.clip(ch, 0.0, 1.0).astype(np.float32)
        return batch


class TemporalTrainer:
    stage_name = "temporal_refinement"

    def __init__(self) -> None:
        self.model = TrainableTemporalConsistencyModel()
        self.adapter = TemporalBatchAdapter()
        self.dataset_source = "synthetic_temporal_bootstrap"
        self.dataset_diagnostics: dict[str, object] = {}

    def build_datasets(self, config: TrainingConfig) -> tuple[TemporalDataset, TemporalDataset]:
        if config.learned_dataset_path:
            manifest_ds = TemporalDataset.from_temporal_manifest(config.learned_dataset_path, strict=False)
            if len(manifest_ds) > 1:
                split = max(1, int(0.8 * len(manifest_ds)))
                train_ds = TemporalDataset(samples=manifest_ds.samples[:split])
                val_ds = TemporalDataset(samples=manifest_ds.samples[split:])
                train_ds.diagnostics = dict(getattr(manifest_ds, "diagnostics", {}), split="train")
                val_ds.diagnostics = dict(getattr(manifest_ds, "diagnostics", {}), split="val")
                self.dataset_source = "manifest_temporal_primary"
                self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
                return train_ds, val_ds
            if len(manifest_ds) == 1:
                self.dataset_source = "manifest_temporal_primary_single_sample_reused_for_val"
                self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
                return manifest_ds, TemporalDataset(samples=list(manifest_ds.samples))
            self.dataset_source = "synthetic_temporal_fallback_manifest_empty"
            self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
        return TemporalDataset.synthetic(config.train_size), TemporalDataset.synthetic(config.val_size)

    def _iter_batches(self, dataset: TemporalDataset):
        for sample in dataset.samples:
            yield self.adapter.adapt(sample)

    @staticmethod
    def evaluate_model(model: TrainableTemporalConsistencyModel, batches: list[TemporalBatch], diagnostics: dict[str, object] | None = None) -> dict[str, float]:
        diagnostics = diagnostics or {}
        if not batches:
            return {
                "reconstruction_mae": 1.0,
                "flicker_delta_mae": 1.0,
                "region_consistency_mae": 1.0,
                "seam_temporal_mae": 1.0,
                "confidence_alignment_mae": 1.0,
                "contract_validity": 0.0,
                "fallback_free_happy_path_ratio": 0.0,
                "usable_sample_count": 0.0,
                "invalid_records": float(diagnostics.get("invalid_records", 0)),
                "skipped_records": float(diagnostics.get("skipped_records", 0)),
                "tag_coverage": 0.0,
                "scenario_coverage": 0.0,
                "score": 0.0,
            }
        entries = [model.eval_step(batch) for batch in batches]
        recon_mae = float(sum(e["reconstruction_mae"] for e in entries) / len(entries))
        flicker_mae = float(sum(e["flicker_delta_mae"] for e in entries) / len(entries))
        region_mae = float(sum(e["region_consistency_mae"] for e in entries) / len(entries))
        seam_mae = float(sum(e["seam_temporal_loss"] for e in entries) / len(entries))
        conf_align_mae = float(sum(e["confidence_calibration_loss"] for e in entries) / len(entries))

        contract_validity = float(
            sum(1.0 for b in batches if b.previous_frame.shape == b.current_frame.shape == b.target_frame.shape and b.changed_mask.shape[:2] == b.previous_frame.shape[:2]) / len(batches)
        )
        tag_coverage = float(len((diagnostics.get("tag_counts") or {}).keys()))
        scenario_coverage = float(len((diagnostics.get("scenario_counts") or {}).keys()))
        score = float(max(0.0, 1.0 - (recon_mae + 0.7 * flicker_mae + 0.4 * region_mae + 0.12 * seam_mae + 0.08 * conf_align_mae)))
        return {
            "reconstruction_mae": recon_mae,
            "flicker_delta_mae": flicker_mae,
            "region_consistency_mae": region_mae,
            "seam_temporal_mae": seam_mae,
            "confidence_alignment_mae": conf_align_mae,
            "contract_validity": contract_validity,
            "fallback_free_happy_path_ratio": 1.0,
            "usable_sample_count": float(len(batches)),
            "invalid_records": float(diagnostics.get("invalid_records", 0)),
            "skipped_records": float(diagnostics.get("skipped_records", 0)),
            "tag_coverage": tag_coverage,
            "scenario_coverage": scenario_coverage,
            "score": score,
        }

    def train(self, config: TrainingConfig) -> StageResult:
        train_dataset, val_dataset = self.build_datasets(config)
        stage_dir = Path(config.checkpoint_dir) / self.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        lr = config.learning_rate
        history: list[dict[str, object]] = []
        train_batches = list(self._iter_batches(train_dataset))
        val_batches = list(self._iter_batches(val_dataset))
        last_train: dict[str, float] = {}
        last_val: dict[str, float] = {}

        for epoch in range(config.epochs):
            train_losses = [self.model.train_step(batch, lr=lr) for batch in train_batches]
            val_metrics = self.evaluate_model(self.model, val_batches, diagnostics=self.dataset_diagnostics)

            def m(items: list[dict[str, float]], key: str) -> float:
                return float(sum(i[key] for i in items) / max(1, len(items)))

            last_train = {
                "loss": m(train_losses, "total_loss"),
                "reconstruction_loss": m(train_losses, "reconstruction_loss"),
                "flicker_loss": m(train_losses, "flicker_loss"),
                "region_stability_loss": m(train_losses, "region_stability_loss"),
                "seam_temporal_loss": m(train_losses, "seam_temporal_loss"),
                "confidence_calibration_loss": m(train_losses, "confidence_calibration_loss"),
                "progress": (epoch + 1) / max(1, config.epochs),
            }
            last_val = val_metrics
            history.append({"epoch": epoch + 1, "train": last_train, "val": last_val})
            lr *= 0.95

        model_path = stage_dir / "temporal_model.json"
        self.model.save(str(model_path))
        ckpt = stage_dir / "latest.json"
        ckpt.write_text(
            json.dumps(
                {
                    "stage": self.stage_name,
                    "config": asdict(config),
                    "history": history,
                    "final_train": last_train,
                    "final_val": last_val,
                    "model_path": str(model_path),
                    "eval": last_val,
                    "dataset_profile": {
                        "source": self.dataset_source,
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "diagnostics": self.dataset_diagnostics,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(stage_name=self.stage_name, train_metrics=last_train, val_metrics=last_val, checkpoint_path=str(ckpt))
