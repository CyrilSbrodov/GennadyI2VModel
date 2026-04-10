from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dynamics.model import FAMILIES, DynamicsInputs, DynamicsModel, DynamicsTargets, decode_prediction, dynamics_inputs_from_tensor_batch, featurize_runtime, targets_from_delta, tensorize_dynamics_inputs
from dynamics.human_state_transition import HumanStateTransitionModel
from dynamics.temporal_transition_encoder import PHASES, REGION_KEYS, TemporalTransitionEncoder
from planning.transition_engine import PlannedState
from rendering.trainable_patch_renderer import TrainableLocalPatchModel
from training.base_trainer import BaseTrainer
from training.datasets import DynamicsDataset, TrainingSample
from training.rollout_eval import evaluate_rollout_modes_on_video_manifest
from training.types import StageResult, TrainingConfig
from training.dynamics_family_training import DynamicsDatasetSurface, DynamicsTrainingSample, FamilyAwareDynamicsTrainingModule


@dataclass(slots=True)
class TemporalContractConditioning:
    source: str
    predicted_family: str
    predicted_phase: str
    target_profile: dict[str, list[str]]
    reveal_score: float
    occlusion_score: float
    support_contact_score: float


@dataclass(slots=True)
class DynamicsBatch:
    inputs: DynamicsInputs
    targets: DynamicsTargets
    graph_before: object
    graph_before_source: str
    action_tokens: list[str]
    planner_context: dict[str, float | str]
    target_transition_context: dict[str, object]
    memory_context: dict[str, object]
    delta_groups: dict[str, float]
    temporal_contract_conditioning: TemporalContractConditioning
    tensor_batch: object


class DynamicsDatasetAdapter:
    """Builds project-like dynamics batches from dataset samples (synthetic bootstrap + manifest-backed)."""

    @staticmethod
    def _weak_contract(sample: TrainingSample, step_index: int) -> TemporalContractConditioning:
        contract = sample.get("graph_transition_contract", {}) if isinstance(sample.get("graph_transition_contract"), dict) else {}
        planner = contract.get("planner_context", {}) if isinstance(contract.get("planner_context"), dict) else {}
        metadata = contract.get("metadata", {}) if isinstance(contract.get("metadata"), dict) else {}
        tp = metadata.get("target_profile", {}) if isinstance(metadata.get("target_profile"), dict) else {}
        ttarget = sample.get("temporal_transition_target", {}) if isinstance(sample.get("temporal_transition_target"), dict) else {}
        return TemporalContractConditioning(
            source="weak_manifest_bootstrap",
            predicted_family=str(metadata.get("transition_family", ttarget.get("family", "pose_transition"))),
            predicted_phase=str(planner.get("phase", ttarget.get("phase", "transition"))),
            target_profile={
                "primary_regions": [str(x) for x in tp.get("primary_regions", [])],
                "secondary_regions": [str(x) for x in tp.get("secondary_regions", [])],
                "context_regions": [str(x) for x in tp.get("context_regions", [])],
            },
            reveal_score=float(ttarget.get("reveal_score", 0.0)),
            occlusion_score=float(ttarget.get("occlusion_score", 0.0)),
            support_contact_score=float(ttarget.get("support_contact_score", 0.0)),
        )

    @classmethod
    def sample_to_batch(cls, sample: TrainingSample, *, encoder: TemporalTransitionEncoder | None, human_encoder: HumanStateTransitionModel | None = None, conditioning_mode: str, step_index: int = 1) -> DynamicsBatch:
        graph_before = sample["graphs"][0]
        actions = sample.get("actions", [])
        action_tokens = [a.type for a in actions] or ["micro_adjust"]
        contract = sample.get("graph_transition_contract", {}) if isinstance(sample.get("graph_transition_contract"), dict) else {}
        base_ctx = contract.get("planner_context", {}) if isinstance(contract.get("planner_context"), dict) else {}
        weak = cls._weak_contract(sample, step_index=step_index)

        learned: TemporalContractConditioning | None = None
        features = sample.get("temporal_transition_features")
        if encoder is not None and isinstance(features, list) and features:
            pred = encoder.forward(np.asarray(features, dtype=np.float64))
            typed = encoder.to_typed_contract(pred)
            learned = TemporalContractConditioning(
                source="learned_temporal_contract",
                predicted_family=typed.predicted_family,
                predicted_phase=typed.predicted_phase,
                target_profile={
                    "primary_regions": list(typed.target_profile.primary_regions),
                    "secondary_regions": list(typed.target_profile.secondary_regions),
                    "context_regions": list(typed.target_profile.context_regions),
                },
                reveal_score=float(typed.reveal_score),
                occlusion_score=float(typed.occlusion_score),
                support_contact_score=float(typed.support_contact_score),
            )

        human: TemporalContractConditioning | None = None
        human_features = sample.get("human_state_transition_features")
        if human_encoder is not None and isinstance(human_features, list) and human_features:
            hp = human_encoder.forward(np.asarray(human_features, dtype=np.float64))
            hc = human_encoder.to_typed_contract(hp)
            mean_vis = float(np.mean(list(hc.visibility_state_scores.values()) or [0.0]))
            human = TemporalContractConditioning(
                source="learned_human_state_contract",
                predicted_family=hc.predicted_family,
                predicted_phase=hc.predicted_phase,
                target_profile={
                    "primary_regions": list(hc.target_profile.primary_regions),
                    "secondary_regions": list(hc.target_profile.secondary_regions),
                    "context_regions": list(hc.target_profile.context_regions),
                },
                reveal_score=mean_vis,
                occlusion_score=float(1.0 - mean_vis),
                support_contact_score=float(hc.support_contact_state),
            )

        selected = weak
        if conditioning_mode == "learned_contract_only":
            selected = human or learned or weak
        elif conditioning_mode == "mixed_contract_bootstrap":
            selected = human or learned or weak
        elif conditioning_mode == "weak_contract_only":
            selected = weak

        target_state = PlannedState(step_index=step_index, labels=action_tokens + [f"temporal_family={selected.predicted_family}", f"temporal_phase={selected.predicted_phase}"])
        planner_context = {
            "step_index": float(base_ctx.get("step_index", step_index)),
            "total_steps": float(base_ctx.get("total_steps", max(2, len(sample.get("graphs", [])) + 1))),
            "phase": str(selected.predicted_phase),
            "target_duration": float(base_ctx.get("target_duration", 1.5)),
            "temporal_reveal": float(selected.reveal_score),
            "temporal_occlusion": float(selected.occlusion_score),
            "temporal_support": float(selected.support_contact_score),
        }
        inputs = featurize_runtime(graph_before, target_state, planner_context, None)
        # explicit structured conditioning lane projected into target features tail.
        phase_onehot = [1.0 if selected.predicted_phase == p else 0.0 for p in PHASES]
        region_scores = [
            1.0 if k in set(selected.target_profile.get("primary_regions", []) + selected.target_profile.get("secondary_regions", []) + selected.target_profile.get("context_regions", [])) else 0.0
            for k in REGION_KEYS
        ]
        tail = phase_onehot + region_scores + [selected.reveal_score, selected.occlusion_score, selected.support_contact_score]
        inputs.target_features = (inputs.target_features + tail)[: len(inputs.target_features)]

        delta = sample.get("deltas", [])[0]
        family = str(selected.predicted_family if selected.predicted_family in FAMILIES else "pose_transition")
        targets = targets_from_delta(delta, family=family)
        tensor_batch = tensorize_dynamics_inputs(inputs, family=family, phase=str(selected.predicted_phase))
        return DynamicsBatch(
            inputs=inputs,
            targets=targets,
            graph_before=graph_before,
            graph_before_source=str(sample.get("source", "unknown")),
            action_tokens=action_tokens,
            planner_context=dict(planner_context, transition_family=family),
            target_transition_context=contract.get("target_transition_context", {}) if isinstance(contract, dict) else {},
            memory_context=contract.get("memory_context", {}) if isinstance(contract, dict) else {},
            delta_groups={
                "pose": 1.0 if delta.pose_deltas else 0.0,
                "garment": 1.0 if delta.garment_deltas else 0.0,
                "visibility": 1.0 if delta.visibility_deltas else 0.0,
                "expression": 1.0 if delta.expression_deltas else 0.0,
                "interaction": 1.0 if delta.interaction_deltas else 0.0,
                "region": 1.0 if delta.region_transition_mode else 0.0,
            },
            temporal_contract_conditioning=selected,
            tensor_batch=tensor_batch,
        )


class DynamicsTrainer(BaseTrainer):
    stage_name = "dynamics"
    dataset_source: str = "synthetic_dynamics_bootstrap"
    dataset_diagnostics: dict[str, object] = {}

    def __init__(self) -> None:
        self.temporal_encoder = TemporalTransitionEncoder()
        self.human_state_encoder = HumanStateTransitionModel()
        self.contract_conditioning_mode = "weak_contract_only"
        self.family_trainer = FamilyAwareDynamicsTrainingModule()

    @staticmethod
    def _resolve_conditioning_mode(config: TrainingConfig, has_manifest: bool) -> str:
        raw = str(getattr(config, "contract_conditioning_mode", "auto") or "auto")
        if raw in {"weak_contract_only", "learned_contract_only", "mixed_contract_bootstrap"}:
            return raw
        return "mixed_contract_bootstrap" if has_manifest else "weak_contract_only"

    def build_datasets(self, config: TrainingConfig) -> tuple[DynamicsDataset, DynamicsDataset]:
        self.dataset_source = "synthetic_dynamics_bootstrap"
        self.dataset_diagnostics = {}
        if config.learned_dataset_path:
            payload = json.loads(Path(config.learned_dataset_path).read_text(encoding="utf-8"))
            is_video_manifest = payload.get("manifest_type") == "video_transition_manifest"
            manifest_ds = DynamicsDataset.from_video_transition_manifest(config.learned_dataset_path, strict=False) if is_video_manifest else DynamicsDataset.from_transition_manifest(config.learned_dataset_path, strict=False)
            self.dataset_diagnostics = getattr(manifest_ds, "diagnostics", {})
            if len(manifest_ds) > 1:
                split = max(1, int(0.8 * len(manifest_ds)))
                self.dataset_source = "manifest_video_dynamics_primary" if is_video_manifest else "manifest_dynamics_primary"
                train_ds = DynamicsDataset(samples=manifest_ds.samples[:split])
                val_ds = DynamicsDataset(samples=manifest_ds.samples[split:])
                train_ds.diagnostics = dict(self.dataset_diagnostics, split="train")
                val_ds.diagnostics = dict(self.dataset_diagnostics, split="val")
                return train_ds, val_ds
            if len(manifest_ds) == 1:
                self.dataset_source = "manifest_video_dynamics_primary_with_synthetic_val_fallback" if is_video_manifest else "manifest_dynamics_primary_with_synthetic_val_fallback"
                return manifest_ds, DynamicsDataset.synthetic(max(1, config.val_size))
            self.dataset_source = "synthetic_dynamics_bootstrap_fallback_manifest_empty"
        train = DynamicsDataset.synthetic(config.train_size)
        val = DynamicsDataset.synthetic(config.val_size)
        return train, val

    def _iter_batches(self, dataset: DynamicsDataset) -> list[DynamicsBatch]:
        has_temporal = bool(dataset.samples and isinstance(dataset.samples[0].get("temporal_transition_features"), list))
        has_human = bool(dataset.samples and isinstance(dataset.samples[0].get("human_state_transition_features"), list))
        encoder = self.temporal_encoder if has_temporal else None
        human_encoder = self.human_state_encoder if has_human else None
        return [DynamicsDatasetAdapter.sample_to_batch(sample, encoder=encoder, human_encoder=human_encoder, conditioning_mode=self.contract_conditioning_mode, step_index=max(1, idx + 1)) for idx, sample in enumerate(dataset.samples)]

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
            "usable_sample_count": 0.0,
            "invalid_records": 0.0,
            "skipped_records": 0.0,
            "pose_group_coverage": 0.0,
            "garment_group_coverage": 0.0,
            "visibility_group_coverage": 0.0,
            "expression_group_coverage": 0.0,
            "interaction_group_coverage": 0.0,
            "region_group_coverage": 0.0,
            "family_coverage_count": 0.0,
            "region_coverage_count": 0.0,
            "phase_coverage_count": 0.0,
            "fallback_free_ratio": 0.0,
            "learned_contract_usage_ratio": 0.0,
            "weak_contract_usage_ratio": 0.0,
            "temporal_to_dynamics_phase_consistency": 0.0,
            "temporal_to_dynamics_region_consistency": 0.0,
            "pose_transition_samples": 0.0,
            "garment_transition_samples": 0.0,
            "interaction_transition_samples": 0.0,
            "expression_transition_samples": 0.0,
        }
        if not batches:
            metrics["score"] = 0.0
            return metrics

        for batch in batches:
            family = str(batch.temporal_contract_conditioning.predicted_family if batch.temporal_contract_conditioning.predicted_family in FAMILIES else "pose_transition")
            prediction = model.forward(dynamics_inputs_from_tensor_batch(batch.tensor_batch), family=family)
            losses = model.compute_losses(prediction, batch.targets)
            for head in ("pose", "garment", "visibility", "expression", "interaction", "region"):
                metrics[f"{head}_mse"] += float(losses[f"{head}_loss"])

            decoded = decode_prediction(prediction, scene_graph=batch.graph_before, phase="mid", semantic_reasons=batch.action_tokens)
            contract_ok = bool(decoded.pose_deltas and decoded.region_transition_mode and decoded.affected_regions)
            metrics["contract_valid_ratio"] += 1.0 if contract_ok else 0.0

            probe_ctx = dict(step_index=3.0, total_steps=4.0, phase="late", target_duration=2.0)
            probe_inputs = featurize_runtime(batch.graph_before, PlannedState(step_index=3, labels=batch.action_tokens + ["intensity=0.9"]), probe_ctx, None)
            probe_tensor = tensorize_dynamics_inputs(probe_inputs, family=family, phase="transition")
            probe_pred = model.forward(dynamics_inputs_from_tensor_batch(probe_tensor), family=family)
            metrics["conditioning_sensitivity"] += float(abs(probe_pred.pose[0] - prediction.pose[0]))
            metrics["usable_sample_count"] += 1.0
            for group, present in batch.delta_groups.items():
                metrics[f"{group}_group_coverage"] += present
            csrc = batch.temporal_contract_conditioning.source
            metrics[f"{family}_samples"] += 1.0
            is_learned = csrc in {"learned_temporal_contract", "learned_human_state_contract"}
            metrics["learned_contract_usage_ratio"] += 1.0 if is_learned else 0.0
            metrics["weak_contract_usage_ratio"] += 1.0 if csrc == "weak_manifest_bootstrap" else 0.0
            metrics["temporal_to_dynamics_phase_consistency"] += 1.0 if batch.temporal_contract_conditioning.predicted_phase == str(batch.planner_context.get("phase", "")) else 0.0
            target_regions = set(
                batch.temporal_contract_conditioning.target_profile.get("primary_regions", [])
                + batch.temporal_contract_conditioning.target_profile.get("secondary_regions", [])
                + batch.temporal_contract_conditioning.target_profile.get("context_regions", [])
            )
            delta_regions = set(decoded.affected_regions)
            overlap = float(len(target_regions & delta_regions)) / max(1.0, float(len(target_regions | delta_regions)))
            metrics["temporal_to_dynamics_region_consistency"] += overlap

        denom = float(len(batches))
        for key in list(metrics):
            metrics[key] = round(metrics[key] / denom, 6)
        if self.dataset_diagnostics:
            metrics["invalid_records"] = float(self.dataset_diagnostics.get("invalid_records", 0))
            metrics["skipped_records"] = float(self.dataset_diagnostics.get("skipped_records", 0))
            family_cov = self.dataset_diagnostics.get("family_coverage") or self.dataset_diagnostics.get("family_counts", {})
            region_cov = self.dataset_diagnostics.get("region_coverage", {})
            phase_cov = self.dataset_diagnostics.get("phase_coverage", {})
            metrics["family_coverage_count"] = float(len(family_cov)) if isinstance(family_cov, dict) else 0.0
            metrics["region_coverage_count"] = float(len(region_cov)) if isinstance(region_cov, dict) else 0.0
            metrics["phase_coverage_count"] = float(len(phase_cov)) if isinstance(phase_cov, dict) else 0.0
            metrics["fallback_free_ratio"] = float(self.dataset_diagnostics.get("fallback_free_ratio", 0.0))
        metrics["score"] = round(max(0.0, 1.0 - metrics["pose_mse"]), 6)
        return metrics

    def train(self, config: TrainingConfig) -> StageResult:
        train_dataset, val_dataset = self.build_datasets(config)
        self.contract_conditioning_mode = self._resolve_conditioning_mode(config, has_manifest=bool(config.learned_dataset_path and len(train_dataset) > 0 and self.dataset_source.startswith("manifest")))
        train_batches = self._iter_batches(train_dataset)
        val_batches = self._iter_batches(val_dataset)
        model = DynamicsModel()

        train_surface = DynamicsDatasetSurface(
            samples=[DynamicsTrainingSample(tensor_batch=b.tensor_batch, targets=b.targets, graph_before=b.graph_before, action_tokens=b.action_tokens, source=b.graph_before_source) for b in train_batches],
            source=self.dataset_source,
            diagnostics={"mode": "family_aware_surface", "bootstrap": self.dataset_source.startswith("synthetic"), **(self.dataset_diagnostics or {})},
        )
        val_surface = DynamicsDatasetSurface(
            samples=[DynamicsTrainingSample(tensor_batch=b.tensor_batch, targets=b.targets, graph_before=b.graph_before, action_tokens=b.action_tokens, source=b.graph_before_source) for b in val_batches],
            source=self.dataset_source,
            diagnostics={"mode": "family_aware_surface", "bootstrap": self.dataset_source.startswith("synthetic"), **(self.dataset_diagnostics or {})},
        )
        train_metrics: dict[str, float] = {}
        for _ in range(config.epochs):
            train_metrics = self.family_trainer.train_epoch(model, train_surface, lr=config.learning_rate)
            learned_usage = float(sum(1.0 for b in train_batches if b.temporal_contract_conditioning.source in {"learned_temporal_contract", "learned_human_state_contract"}))
            weak_usage = float(sum(1.0 for b in train_batches if b.temporal_contract_conditioning.source == "weak_manifest_bootstrap"))
            denom = max(1.0, float(len(train_batches)))
            train_metrics["learned_contract_usage_ratio"] = round(learned_usage / denom, 6)
            train_metrics["weak_contract_usage_ratio"] = round(weak_usage / denom, 6)
            for fam in FAMILIES:
                train_metrics[f"{fam}_batch_ratio"] = round(sum(1.0 for b in train_batches if b.targets.family == fam) / denom, 6)

        val_metrics = self._evaluate(model, val_batches)
        val_metrics.update(self.family_trainer.validate_epoch(model, val_surface))
        val_metrics["score"] = round(max(0.0, 1.0 - val_metrics["pose_mse"]), 6)
        val_metrics["contract_conditioning_mode"] = self.contract_conditioning_mode
        if config.learned_dataset_path and self.dataset_source.startswith("manifest_video_dynamics_primary"):
            rollout = evaluate_rollout_modes_on_video_manifest(
                dataset_manifest=config.learned_dataset_path,
                temporal_model=self.temporal_encoder,
                dynamics_model=model,
                renderer_model=TrainableLocalPatchModel(),
                rollout_steps=2,
                max_records=max(1, config.val_size),
            )
            tf = rollout.get("teacher_forced_rollout", {})
            pr = rollout.get("predicted_rollout", {})
            val_metrics["rollout_teacher_frame_reconstruction_proxy"] = float(tf.get("rollout_frame_reconstruction_proxy", 0.0))
            val_metrics["rollout_predicted_frame_reconstruction_proxy"] = float(pr.get("rollout_frame_reconstruction_proxy", 0.0))
            val_metrics["rollout_consistency_regularizer"] = round(
                max(
                    0.0,
                    1.0
                    - 0.5 * val_metrics["rollout_teacher_frame_reconstruction_proxy"]
                    - 0.5 * val_metrics["rollout_predicted_frame_reconstruction_proxy"],
                ),
                6,
            )

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
                    "runtime_compatible": True,
                    "checkpoint_status": "trained",
                    "dataset_profile": {
                        "surface_type": "DynamicsDatasetSurface",
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "source": self.dataset_source,
                        "diagnostics": self.dataset_diagnostics,
                    },
                    "contract_conditioning": {
                        "mode": self.contract_conditioning_mode,
                        "learned_contract_usage_ratio": val_metrics.get("learned_contract_usage_ratio", 0.0),
                        "weak_contract_usage_ratio": val_metrics.get("weak_contract_usage_ratio", 0.0),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(stage_name=self.stage_name, train_metrics=train_metrics, val_metrics=val_metrics, checkpoint_path=str(checkpoint_path))
