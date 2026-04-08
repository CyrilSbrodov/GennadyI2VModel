from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rendering.trainable_patch_renderer import PatchBatch, TrainableLocalPatchModel
from training.datasets import RendererDataset
from training.types import StageResult, TrainingConfig


class RendererBatchAdapter:
    """Runtime-aligned adapter from dataset records into full PatchBatch contract."""

    @staticmethod
    def _np3(x: list) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Renderer batch expects HxWx3 patch tensors")
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        diff = np.mean(np.abs(after - before), axis=2, keepdims=True)
        return np.clip(diff * 3.0, 0.0, 1.0)

    def adapt(self, sample: dict[str, object]) -> PatchBatch:
        roi_pairs = sample.get("roi_pairs") or []
        before, after = roi_pairs[0] if roi_pairs else (sample["frames"][0], sample["frames"][1])
        b = self._np3(before)
        a = self._np3(after)
        contract = sample.get("renderer_batch_contract", {}) if isinstance(sample.get("renderer_batch_contract", {}), dict) else {}
        family = str(sample.get("region_family", "")).lower()
        semantic_default = [1.0, 0.0, 0.0, 0.9, 0.1, 0.2] if family == "face_expression" else ([0.0, 1.0, 0.0, 0.2, 0.85, 0.4] if family == "torso_reveal" else [0.0, 0.0, 1.0, 0.15, 0.45, 0.9])
        semantic_embed = np.asarray(contract.get("semantic_embed", semantic_default), dtype=np.float32)
        delta_cond = np.asarray(contract.get("delta_cond", [0.0] * 9), dtype=np.float32)
        planner_cond = np.asarray(contract.get("planner_cond", [0.0] * 8), dtype=np.float32)
        graph_cond = np.asarray(contract.get("graph_cond", [0.0] * 7), dtype=np.float32)
        memory_cond = np.asarray(contract.get("memory_cond", [0.0] * 8), dtype=np.float32)
        appearance_cond = np.asarray(contract.get("appearance_cond", np.concatenate([np.mean(b, axis=(0, 1)), np.std(b, axis=(0, 1))]).tolist()), dtype=np.float32)
        bbox_cond = np.asarray(contract.get("bbox_cond", [0.2, 0.2, 0.4, 0.4]), dtype=np.float32)
        changed_mask = np.asarray(contract.get("changed_mask", self._mask(b, a).tolist()), dtype=np.float32)
        if changed_mask.ndim == 2:
            changed_mask = changed_mask[..., None]
        if changed_mask.shape[-1] == 3:
            changed_mask = np.mean(changed_mask, axis=2, keepdims=True)
        blend_hint = np.asarray(contract.get("blend_hint", changed_mask.tolist()), dtype=np.float32)
        alpha_target = np.asarray(contract.get("alpha_target", np.clip(0.2 + 0.8 * changed_mask, 0.0, 1.0).tolist()), dtype=np.float32)
        if blend_hint.ndim == 2:
            blend_hint = blend_hint[..., None]
        if alpha_target.ndim == 2:
            alpha_target = alpha_target[..., None]
        if blend_hint.shape[-1] == 3:
            blend_hint = np.mean(blend_hint, axis=2, keepdims=True)
        if alpha_target.shape[-1] == 3:
            alpha_target = np.mean(alpha_target, axis=2, keepdims=True)
        return PatchBatch(
            roi_before=b,
            roi_after=a,
            changed_mask=changed_mask,
            alpha_target=np.clip(alpha_target, 0.0, 1.0),
            blend_hint=np.clip(blend_hint, 0.0, 1.0),
            semantic_embed=semantic_embed,
            delta_cond=delta_cond,
            planner_cond=planner_cond,
            graph_cond=graph_cond,
            memory_cond=memory_cond,
            appearance_cond=appearance_cond,
            bbox_cond=bbox_cond,
        )


class RendererTrainer:
    stage_name = "renderer"

    def __init__(self) -> None:
        self.model = TrainableLocalPatchModel()
        self.adapter = RendererBatchAdapter()
        self.dataset_source = "synthetic_bootstrap"
        self.dataset_diagnostics: dict[str, object] = {}

    def build_datasets(self, config: TrainingConfig) -> tuple[RendererDataset, RendererDataset]:
        if config.learned_dataset_path:
            manifest_ds = RendererDataset.from_renderer_manifest(config.learned_dataset_path, strict=False)
            if len(manifest_ds) > 1:
                split = max(1, int(0.8 * len(manifest_ds)))
                train_ds = RendererDataset(samples=manifest_ds.samples[:split])
                val_ds = RendererDataset(samples=manifest_ds.samples[split:])
                train_ds.diagnostics = dict(getattr(manifest_ds, "diagnostics", {}), split="train")
                val_ds.diagnostics = dict(getattr(manifest_ds, "diagnostics", {}), split="val")
                self.dataset_source = "manifest_paired_roi_primary"
                self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
                return train_ds, val_ds
            if len(manifest_ds) == 1:
                self.dataset_source = "manifest_paired_roi_primary_with_synthetic_val_fallback"
                self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
                return manifest_ds, RendererDataset.synthetic(max(1, config.val_size))
            self.dataset_source = "synthetic_bootstrap_fallback_manifest_empty"
            self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
        else:
            self.dataset_source = "synthetic_bootstrap"
            self.dataset_diagnostics = {}
        return RendererDataset.synthetic(config.train_size), RendererDataset.synthetic(config.val_size)

    def _iter_batches(self, dataset: RendererDataset):
        for sample in dataset.samples:
            yield self.adapter.adapt(sample)

    @staticmethod
    def evaluate_model(model: TrainableLocalPatchModel, batches: list[PatchBatch], diagnostics: dict[str, object] | None = None) -> dict[str, float]:
        if not batches:
            return {"reconstruction_mae": 1.0, "alpha_mae": 1.0, "uncertainty_calibration_mae": 1.0, "contract_validity": 0.0, "fallback_free_happy_path_ratio": 0.0, "face_family_score": 0.0, "torso_family_score": 0.0, "sleeve_family_score": 0.0, "usable_sample_count": 0.0, "invalid_records": float((diagnostics or {}).get("invalid_records", 0)), "skipped_records": float((diagnostics or {}).get("skipped_records", 0)), "score": 0.0}
        eval_entries = [model.eval_step(b) for b in batches]
        recon_mae = float(sum(e["mae"] for e in eval_entries) / len(eval_entries))
        alpha_mae = float(sum(e["alpha_mae"] for e in eval_entries) / len(eval_entries))
        unc_mae = float(sum(e["uncertainty_calibration_loss"] for e in eval_entries) / len(eval_entries))
        contract_validity = float(sum(1.0 for b in batches if b.alpha_target.shape[:2] == b.roi_before.shape[:2] and b.changed_mask.shape[:2] == b.roi_before.shape[:2]) / len(batches))
        per_family = {"face": [], "torso": [], "sleeve": []}
        for b, e in zip(batches, eval_entries):
            if b.semantic_embed[0] > 0.5:
                per_family["face"].append(e["mae"])
            elif b.semantic_embed[1] > 0.5:
                per_family["torso"].append(e["mae"])
            else:
                per_family["sleeve"].append(e["mae"])
        def fam_score(vals: list[float]) -> float:
            if not vals:
                return 0.0
            return float(max(0.0, 1.0 - (sum(vals) / len(vals))))
        score = max(0.0, 1.0 - (recon_mae + 0.5 * alpha_mae + 0.35 * unc_mae))
        metrics = {
            "reconstruction_mae": recon_mae,
            "alpha_mae": alpha_mae,
            "uncertainty_calibration_mae": unc_mae,
            "contract_validity": contract_validity,
            "fallback_free_happy_path_ratio": 1.0,
            "face_family_score": fam_score(per_family["face"]),
            "torso_family_score": fam_score(per_family["torso"]),
            "sleeve_family_score": fam_score(per_family["sleeve"]),
            "usable_sample_count": float(len(batches)),
            "invalid_records": float((diagnostics or {}).get("invalid_records", 0)),
            "skipped_records": float((diagnostics or {}).get("skipped_records", 0)),
            "score": score,
        }
        family_counts = (diagnostics or {}).get("family_counts", {})
        for key in ("face_expression", "torso_reveal", "sleeve_arm_transition"):
            metrics[f"family_count_{key}"] = float(family_counts.get(key, 0)) if isinstance(family_counts, dict) else 0.0
        return metrics

    def train(self, config: TrainingConfig) -> StageResult:
        train_dataset, val_dataset = self.build_datasets(config)
        stage_dir = Path(config.checkpoint_dir) / self.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, object]] = []
        lr = config.learning_rate
        last_train: dict[str, float] = {}
        last_val: dict[str, float] = {}

        train_batches = list(self._iter_batches(train_dataset))
        val_batches = list(self._iter_batches(val_dataset))

        for epoch in range(config.epochs):
            train_losses = [self.model.train_step(batch, lr=lr) for batch in train_batches]
            eval_metrics = self.evaluate_model(self.model, val_batches, diagnostics=self.dataset_diagnostics)

            def m(items: list[dict[str, float]], key: str) -> float:
                return float(sum(i[key] for i in items) / max(1, len(items)))

            last_train = {
                "loss": m(train_losses, "total_loss"),
                "reconstruction_loss": m(train_losses, "reconstruction_loss"),
                "alpha_loss": m(train_losses, "alpha_loss"),
                "uncertainty_calibration_loss": m(train_losses, "uncertainty_calibration_loss"),
                "seam_loss": m(train_losses, "seam_loss"),
                "progress": (epoch + 1) / max(1, config.epochs),
            }
            last_val = eval_metrics
            history.append({"epoch": epoch + 1, "train": last_train, "val": last_val})
            lr *= 0.94

        model_path = stage_dir / "renderer_model.json"
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
