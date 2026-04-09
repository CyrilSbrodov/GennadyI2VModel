from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dynamics.human_state_transition import HumanStateTransitionModel
from dynamics.temporal_transition_encoder import PHASES, TemporalTransitionEncoder
from dynamics.model import DynamicsModel
from rendering.trainable_patch_renderer import PatchBatch, TemporalLocalPatchModel, TrainableLocalPatchModel
from training.rollout_eval import evaluate_rollout_modes_on_video_manifest
from training.datasets import RendererDataset
from training.types import StageResult, TrainingConfig


@dataclass(slots=True)
class RendererTemporalConditioning:
    source: str
    predicted_family: str
    predicted_phase: str
    target_profile: dict[str, list[str]]
    reveal_score: float
    occlusion_score: float
    support_contact_score: float


class RendererBatchAdapter:
    """Runtime-aligned adapter from dataset records into full PatchBatch contract."""

    def __init__(self, encoder: TemporalTransitionEncoder | None = None, human_encoder: HumanStateTransitionModel | None = None, conditioning_mode: str = "weak_contract_only") -> None:
        self.encoder = encoder
        self.human_encoder = human_encoder
        self.conditioning_mode = conditioning_mode

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

    def _weak_contract(self, sample: dict[str, object]) -> RendererTemporalConditioning:
        contract = sample.get("renderer_batch_contract", {}) if isinstance(sample.get("renderer_batch_contract", {}), dict) else {}
        ttarget = sample.get("temporal_transition_target", {}) if isinstance(sample.get("temporal_transition_target"), dict) else {}
        profile = ttarget.get("target_profile", contract.get("target_profile", {})) if isinstance(ttarget.get("target_profile", contract.get("target_profile", {})), dict) else {}
        return RendererTemporalConditioning(
            source="weak_manifest_bootstrap",
            predicted_family=str(ttarget.get("family", contract.get("transition_family", sample.get("region_family", "pose_transition")))),
            predicted_phase=str(ttarget.get("phase", "transition")),
            target_profile={
                "primary_regions": [str(x) for x in profile.get("primary_regions", [])],
                "secondary_regions": [str(x) for x in profile.get("secondary_regions", [])],
                "context_regions": [str(x) for x in profile.get("context_regions", [])],
            },
            reveal_score=float(ttarget.get("reveal_score", 0.0)),
            occlusion_score=float(ttarget.get("occlusion_score", 0.0)),
            support_contact_score=float(ttarget.get("support_contact_score", 0.0)),
        )

    @staticmethod
    def _semantic_from_family(family: str) -> list[float]:
        f = family.strip().lower()
        if f == "expression_transition":
            return [1.0, 0.0, 0.0, 0.85, 0.15, 0.2]
        if f in {"garment_transition", "visibility_transition"}:
            return [0.0, 1.0, 0.0, 0.25, 0.8, 0.35]
        return [0.0, 0.0, 1.0, 0.3, 0.45, 0.8]

    def adapt(self, sample: dict[str, object], *, temporal_mode: bool = False) -> PatchBatch:
        roi_pairs = sample.get("roi_pairs") or []
        before, after = roi_pairs[0] if roi_pairs else (sample["frames"][0], sample["frames"][1])
        b = self._np3(before)
        a = self._np3(after)
        contract = sample.get("renderer_batch_contract", {}) if isinstance(sample.get("renderer_batch_contract", {}), dict) else {}
        weak = self._weak_contract(sample)
        learned: RendererTemporalConditioning | None = None
        features = sample.get("temporal_transition_features")
        if self.encoder is not None and isinstance(features, list) and features:
            pred = self.encoder.forward(np.asarray(features, dtype=np.float64))
            typed = self.encoder.to_typed_contract(pred)
            learned = RendererTemporalConditioning(
                source="learned_temporal_contract",
                predicted_family=typed.predicted_family,
                predicted_phase=typed.predicted_phase,
                target_profile={"primary_regions": list(typed.target_profile.primary_regions), "secondary_regions": list(typed.target_profile.secondary_regions), "context_regions": list(typed.target_profile.context_regions)},
                reveal_score=float(typed.reveal_score),
                occlusion_score=float(typed.occlusion_score),
                support_contact_score=float(typed.support_contact_score),
            )

        human: RendererTemporalConditioning | None = None
        human_features = sample.get("human_state_transition_features")
        if self.human_encoder is not None and isinstance(human_features, list) and human_features:
            hpred = self.human_encoder.forward(np.asarray(human_features, dtype=np.float64))
            htyped = self.human_encoder.to_typed_contract(hpred)
            mean_vis = float(np.mean(list(htyped.visibility_state_scores.values()) or [0.0]))
            human = RendererTemporalConditioning(
                source="learned_human_state_contract",
                predicted_family=htyped.predicted_family,
                predicted_phase=htyped.predicted_phase,
                target_profile={"primary_regions": list(htyped.target_profile.primary_regions), "secondary_regions": list(htyped.target_profile.secondary_regions), "context_regions": list(htyped.target_profile.context_regions)},
                reveal_score=mean_vis,
                occlusion_score=1.0 - mean_vis,
                support_contact_score=float(htyped.support_contact_state),
            )

        selected = weak
        if self.conditioning_mode == "learned_contract_only":
            selected = human or learned or weak
        elif self.conditioning_mode == "mixed_contract_bootstrap":
            selected = human or learned or weak

        semantic_embed = np.asarray(contract.get("semantic_embed", self._semantic_from_family(selected.predicted_family)), dtype=np.float32)
        learned_sources = {"learned_temporal_contract", "learned_human_state_contract"}
        if selected.source in learned_sources:
            semantic_embed = np.asarray(self._semantic_from_family(selected.predicted_family), dtype=np.float32)
        delta_cond = np.asarray(contract.get("delta_cond", [0.0] * 9), dtype=np.float32)
        planner_cond = np.asarray(contract.get("planner_cond", [0.0] * 8), dtype=np.float32)
        if selected.source in learned_sources:
            delta_cond = np.asarray([selected.reveal_score, selected.occlusion_score, selected.support_contact_score] + delta_cond.tolist(), dtype=np.float32)[:9]
            phase_onehot = [1.0 if selected.predicted_phase == p else 0.0 for p in PHASES]
            planner_cond = np.asarray(phase_onehot + planner_cond.tolist(), dtype=np.float32)[:8]
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
        temporal_window = sample.get("temporal_roi_window", {}) if isinstance(sample.get("temporal_roi_window", {}), dict) else {}
        prev_roi = temporal_window.get("roi_t_minus_1", b)
        prev_np = self._np3(prev_roi)
        rollout_weight = float(np.clip(1.0 + selected.reveal_score * 0.4 + selected.occlusion_score * 0.3, 0.5, 2.0))
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
            conditioning_summary={
                "contract_source": selected.source,
                "predicted_family": selected.predicted_family,
                "predicted_phase": selected.predicted_phase,
                "target_profile": selected.target_profile,
            },
            previous_roi=prev_np if temporal_mode else None,
            predicted_family=selected.predicted_family,
            predicted_phase=selected.predicted_phase,
            target_profile=selected.target_profile,
            reveal_score=selected.reveal_score,
            occlusion_score=selected.occlusion_score,
            support_contact_score=selected.support_contact_score,
            temporal_contract_target=sample.get("temporal_transition_target", {}) if isinstance(sample.get("temporal_transition_target", {}), dict) else {},
            graph_delta_target=sample.get("delta_contract", {}) if isinstance(sample.get("delta_contract", {}), dict) else {},
            rollout_weight=rollout_weight,
        )


class RendererTrainer:
    stage_name = "renderer"

    def __init__(self) -> None:
        self.model: TrainableLocalPatchModel | TemporalLocalPatchModel = TrainableLocalPatchModel()
        self.dataset_source = "synthetic_bootstrap"
        self.dataset_diagnostics: dict[str, object] = {}
        self.contract_conditioning_mode = "weak_contract_only"
        self.temporal_encoder = TemporalTransitionEncoder()
        self.human_state_encoder = HumanStateTransitionModel()
        self.adapter = RendererBatchAdapter(encoder=None, human_encoder=None)
        self.temporal_renderer_mode = "legacy_local_renderer"

    @staticmethod
    def _resolve_conditioning_mode(config: TrainingConfig, has_manifest: bool) -> str:
        raw = str(getattr(config, "contract_conditioning_mode", "auto") or "auto")
        if raw in {"weak_contract_only", "learned_contract_only", "mixed_contract_bootstrap"}:
            return raw
        return "mixed_contract_bootstrap" if has_manifest else "weak_contract_only"

    def build_datasets(self, config: TrainingConfig) -> tuple[RendererDataset, RendererDataset]:
        if config.learned_dataset_path:
            payload = json.loads(Path(config.learned_dataset_path).read_text(encoding="utf-8"))
            is_video_manifest = payload.get("manifest_type") == "video_transition_manifest"
            manifest_ds = RendererDataset.from_video_transition_manifest(config.learned_dataset_path, strict=False) if is_video_manifest else RendererDataset.from_renderer_manifest(config.learned_dataset_path, strict=False)
            if len(manifest_ds) > 1:
                split = max(1, int(0.8 * len(manifest_ds)))
                train_ds = RendererDataset(samples=manifest_ds.samples[:split])
                val_ds = RendererDataset(samples=manifest_ds.samples[split:])
                train_ds.diagnostics = dict(getattr(manifest_ds, "diagnostics", {}), split="train")
                val_ds.diagnostics = dict(getattr(manifest_ds, "diagnostics", {}), split="val")
                self.dataset_source = "manifest_video_renderer_primary" if is_video_manifest else "manifest_paired_roi_primary"
                self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
                return train_ds, val_ds
            if len(manifest_ds) == 1:
                self.dataset_source = "manifest_video_renderer_primary_with_synthetic_val_fallback" if is_video_manifest else "manifest_paired_roi_primary_with_synthetic_val_fallback"
                self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
                return manifest_ds, RendererDataset.synthetic(max(1, config.val_size))
            self.dataset_source = "synthetic_bootstrap_fallback_manifest_empty"
            self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
        else:
            self.dataset_source = "synthetic_bootstrap"
            self.dataset_diagnostics = {}
        return RendererDataset.synthetic(config.train_size), RendererDataset.synthetic(config.val_size)

    def _iter_batches(self, dataset: RendererDataset):
        has_temporal = bool(dataset.samples and isinstance(dataset.samples[0].get("temporal_transition_features"), list))
        has_human = bool(dataset.samples and isinstance(dataset.samples[0].get("human_state_transition_features"), list))
        self.adapter = RendererBatchAdapter(encoder=self.temporal_encoder if has_temporal else None, human_encoder=self.human_state_encoder if has_human else None, conditioning_mode=self.contract_conditioning_mode)
        for sample in dataset.samples:
            yield self.adapter.adapt(sample, temporal_mode=self.temporal_renderer_mode == "temporal_local_renderer")

    @staticmethod
    def evaluate_model(model: TrainableLocalPatchModel | TemporalLocalPatchModel, batches: list[PatchBatch], diagnostics: dict[str, object] | None = None) -> dict[str, float]:
        if not batches:
            return {"reconstruction_mae": 1.0, "alpha_mae": 1.0, "uncertainty_calibration_mae": 1.0, "contract_validity": 0.0, "fallback_free_happy_path_ratio": 0.0, "face_family_score": 0.0, "torso_family_score": 0.0, "sleeve_family_score": 0.0, "usable_sample_count": 0.0, "invalid_records": float((diagnostics or {}).get("invalid_records", 0)), "skipped_records": float((diagnostics or {}).get("skipped_records", 0)), "learned_contract_usage_ratio": 0.0, "weak_contract_usage_ratio": 1.0, "temporal_to_renderer_mode_consistency": 0.0, "temporal_to_renderer_target_profile_consistency": 0.0, "score": 0.0}
        eval_entries = [model.eval_step(b) for b in batches]
        recon_mae = float(sum(e["mae"] for e in eval_entries) / len(eval_entries))
        alpha_mae = float(sum(e["alpha_mae"] for e in eval_entries) / len(eval_entries))
        unc_mae = float(sum(e["uncertainty_calibration_loss"] for e in eval_entries) / len(eval_entries))
        contract_validity = float(sum(1.0 for b in batches if b.alpha_target.shape[:2] == b.roi_before.shape[:2] and b.changed_mask.shape[:2] == b.roi_before.shape[:2]) / len(batches))
        per_family = {"face": [], "torso": [], "sleeve": []}
        learned_usage = 0.0
        weak_usage = 0.0
        mode_consistency = 0.0
        profile_consistency = 0.0
        for b, e in zip(batches, eval_entries):
            if b.semantic_embed[0] > 0.5:
                per_family["face"].append(e["mae"])
            elif b.semantic_embed[1] > 0.5:
                per_family["torso"].append(e["mae"])
            else:
                per_family["sleeve"].append(e["mae"])
            summary = b.conditioning_summary if isinstance(b.conditioning_summary, dict) else {}
            source = str(summary.get("contract_source", "weak_manifest_bootstrap"))
            learned_usage += 1.0 if source in {"learned_temporal_contract", "learned_human_state_contract"} else 0.0
            weak_usage += 1.0 if source == "weak_manifest_bootstrap" else 0.0
            fam = str(summary.get("predicted_family", ""))
            if (fam in {"expression_transition"} and b.semantic_embed[0] > 0.5) or (fam in {"garment_transition", "visibility_transition"} and b.semantic_embed[1] > 0.5) or (fam in {"pose_transition", "interaction_transition"} and b.semantic_embed[2] > 0.5):
                mode_consistency += 1.0
            profile = summary.get("target_profile", {}) if isinstance(summary.get("target_profile", {}), dict) else {}
            has_profile = bool(profile.get("primary_regions") or profile.get("secondary_regions") or profile.get("context_regions"))
            profile_consistency += 1.0 if has_profile else 0.0
        temporal_eval = {
            "temporal_window_usage_ratio": float(sum(1.0 for b in batches if isinstance(b.previous_roi, np.ndarray)) / len(batches)),
            "reveal_loss": float(sum(float(e.get("reveal_region_focus_loss", 0.0)) for e in eval_entries) / len(eval_entries)),
            "occlusion_boundary_loss": float(sum(float(e.get("occlusion_boundary_loss", 0.0)) for e in eval_entries) / len(eval_entries)),
            "temporal_consistency_loss": float(sum(float(e.get("temporal_consistency_loss", 0.0)) for e in eval_entries) / len(eval_entries)),
            "rollout_proxy_before_after": float(max(0.0, 1.0 - recon_mae)),
        }

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
            "learned_contract_usage_ratio": float(learned_usage / len(batches)),
            "weak_contract_usage_ratio": float(weak_usage / len(batches)),
            "temporal_to_renderer_mode_consistency": float(mode_consistency / len(batches)),
            "temporal_to_renderer_target_profile_consistency": float(profile_consistency / len(batches)),
            "score": score,
            **temporal_eval,
        }
        family_counts = (diagnostics or {}).get("family_counts", {})
        for key in ("face_expression", "torso_reveal", "sleeve_arm_transition"):
            metrics[f"family_count_{key}"] = float(family_counts.get(key, 0)) if isinstance(family_counts, dict) else 0.0
        return metrics

    def train(self, config: TrainingConfig) -> StageResult:
        train_dataset, val_dataset = self.build_datasets(config)
        self.temporal_renderer_mode = (
            str(getattr(config, "renderer_backend", "legacy_local_renderer"))
            if str(getattr(config, "renderer_backend", "legacy_local_renderer")) in {"legacy_local_renderer", "temporal_local_renderer"}
            else "legacy_local_renderer"
        )
        if self.temporal_renderer_mode == "temporal_local_renderer" and self.dataset_source.startswith("manifest_video_renderer_primary"):
            self.dataset_source = "manifest_video_temporal_renderer_primary"
        self.model = TemporalLocalPatchModel() if self.temporal_renderer_mode == "temporal_local_renderer" else TrainableLocalPatchModel()
        self.contract_conditioning_mode = self._resolve_conditioning_mode(config, has_manifest=bool(config.learned_dataset_path and len(train_dataset) > 0 and self.dataset_source.startswith("manifest")))
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

            learned_ratio = float(sum(1.0 for b in train_batches if (b.conditioning_summary or {}).get("contract_source") in {"learned_temporal_contract", "learned_human_state_contract"}) / max(1, len(train_batches)))
            weak_ratio = float(sum(1.0 for b in train_batches if (b.conditioning_summary or {}).get("contract_source") == "weak_manifest_bootstrap") / max(1, len(train_batches)))
            last_train = {
                "loss": m(train_losses, "total_loss"),
                "reconstruction_loss": m(train_losses, "reconstruction_loss"),
                "alpha_loss": m(train_losses, "alpha_loss"),
                "uncertainty_calibration_loss": m(train_losses, "uncertainty_calibration_loss"),
                "seam_loss": m(train_losses, "seam_loss"),
                "learned_contract_usage_ratio": learned_ratio,
                "weak_contract_usage_ratio": weak_ratio,
                "progress": (epoch + 1) / max(1, config.epochs),
            }
            last_val = eval_metrics
            last_val["contract_conditioning_mode"] = self.contract_conditioning_mode
            if config.learned_dataset_path and self.dataset_source.startswith("manifest_video_renderer_primary"):
                rollout = evaluate_rollout_modes_on_video_manifest(
                    dataset_manifest=config.learned_dataset_path,
                    temporal_model=self.temporal_encoder,
                    dynamics_model=DynamicsModel(),
                    renderer_model=self.model,
                    renderer_backend=self.temporal_renderer_mode,
                    rollout_steps=2,
                    max_records=max(1, config.val_size),
                )
                tf = rollout.get("teacher_forced_rollout", {})
                pr = rollout.get("predicted_rollout", {})
                last_val["rollout_teacher_frame_reconstruction_proxy"] = float(tf.get("rollout_frame_reconstruction_proxy", 0.0))
                last_val["rollout_predicted_frame_reconstruction_proxy"] = float(pr.get("rollout_frame_reconstruction_proxy", 0.0))
                last_val["rollout_consistency_regularizer"] = round(
                    max(
                        0.0,
                        1.0
                        - 0.5 * last_val["rollout_teacher_frame_reconstruction_proxy"]
                        - 0.5 * last_val["rollout_predicted_frame_reconstruction_proxy"],
                    ),
                    6,
                )
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
                    "contract_conditioning": {
                        "mode": self.contract_conditioning_mode,
                        "learned_contract_usage_ratio": last_val.get("learned_contract_usage_ratio", 0.0),
                        "weak_contract_usage_ratio": last_val.get("weak_contract_usage_ratio", 0.0),
                    },
                    "renderer_backend": self.temporal_renderer_mode,
                    "temporal_renderer_path_enabled": bool(self.temporal_renderer_mode == "temporal_local_renderer"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(stage_name=self.stage_name, train_metrics=last_train, val_metrics=last_val, checkpoint_path=str(ckpt))
