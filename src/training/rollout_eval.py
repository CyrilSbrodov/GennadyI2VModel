from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dynamics.model import DynamicsModel, decode_prediction, featurize_runtime
from dynamics.state_update import apply_delta
from dynamics.human_state_transition import HumanStateTransitionModel
from dynamics.temporal_transition_encoder import TemporalTransitionEncoder
from planning.transition_engine import PlannedState
from rendering.trainable_patch_renderer import TemporalLocalPatchModel, TrainableLocalPatchModel
from training.datasets import _build_temporal_transition_features, _deserialize_graph, _serialize_delta_contract, _serialize_graph


@dataclass(slots=True)
class RolloutStepPayload:
    step_index: int
    mode: str
    rollout_payload: dict[str, object]


def _as_hw3(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("expected HxWx3 tensor")
    return np.clip(arr, 0.0, 1.0)


def _as_hw1(value: object, shape_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = np.mean(arr, axis=2, keepdims=True)
    if arr.ndim != 3 or arr.shape[-1] != 1:
        arr = np.zeros((shape_hw[0], shape_hw[1], 1), dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def _target_regions(target_profile: dict[str, object]) -> set[str]:
    out: set[str] = set()
    for key in ("primary_regions", "secondary_regions", "context_regions"):
        for item in target_profile.get(key, []) if isinstance(target_profile.get(key, []), list) else []:
            out.add(str(item))
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return float(len(a & b)) / max(1.0, float(len(a | b)))


def _record_rois(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    roi_records = record.get("roi_records", []) if isinstance(record.get("roi_records", []), list) else []
    if roi_records and isinstance(roi_records[0], dict):
        rr = roi_records[0]
        before = _as_hw3(rr.get("roi_before", []))
        after = _as_hw3(rr.get("roi_after", []))
        changed = _as_hw1(rr.get("changed_mask", np.mean(np.abs(after - before), axis=2, keepdims=True)), (before.shape[0], before.shape[1]))
        preservation = _as_hw1(rr.get("preservation_mask", 1.0 - changed), (before.shape[0], before.shape[1]))
        return before, after, changed, preservation
    if "roi_before" in record and "roi_after" in record:
        before = _as_hw3(record["roi_before"])
        after = _as_hw3(record["roi_after"])
        changed = np.clip(np.mean(np.abs(after - before), axis=2, keepdims=True), 0.0, 1.0)
        return before, after, changed, np.clip(1.0 - changed, 0.0, 1.0)
    raise ValueError("record needs roi_records or roi_before/roi_after")


def evaluate_rollout_on_video_manifest(
    *,
    dataset_manifest: str,
    temporal_model: TemporalTransitionEncoder | None = None,
    dynamics_model: DynamicsModel | None = None,
    renderer_model: TrainableLocalPatchModel | TemporalLocalPatchModel | None = None,
    temporal_weights_path: str = "",
    dynamics_weights_path: str = "",
    renderer_weights_path: str = "",
    mode: str = "teacher_forced_rollout",
    rollout_steps: int = 1,
    max_records: int | None = None,
    renderer_backend: str = "legacy_local_renderer",
    use_human_state_contract: bool = False,
    human_state_model: HumanStateTransitionModel | None = None,
) -> dict[str, object]:
    payload = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
    if payload.get("manifest_type") != "video_transition_manifest":
        return {"dataset_source": "synthetic_rollout_bootstrap_eval_non_video_manifest", "records_used": 0, "payloads": [], "rollout_frame_reconstruction_proxy": 0.0}

    records_all = payload.get("records", []) if isinstance(payload.get("records", []), list) else []
    records = records_all[: max_records if isinstance(max_records, int) and max_records > 0 else len(records_all)]
    if not records:
        return {"dataset_source": "synthetic_rollout_bootstrap_fallback_manifest_empty", "records_used": 0, "payloads": [], "rollout_frame_reconstruction_proxy": 0.0}

    temporal = temporal_model or (TemporalTransitionEncoder.load(temporal_weights_path) if temporal_weights_path else TemporalTransitionEncoder())
    human_state = human_state_model or HumanStateTransitionModel()
    dynamics = dynamics_model or (DynamicsModel.load(dynamics_weights_path) if dynamics_weights_path else DynamicsModel())
    renderer = renderer_model or (
        TemporalLocalPatchModel.load(renderer_weights_path)
        if (renderer_weights_path and renderer_backend == "temporal_local_renderer")
        else (TrainableLocalPatchModel.load(renderer_weights_path) if renderer_weights_path else (TemporalLocalPatchModel() if renderer_backend == "temporal_local_renderer" else TrainableLocalPatchModel()))
    )

    traces: list[RolloutStepPayload] = []
    metrics = {
        "rollout_frame_reconstruction_proxy": 0.0,
        "rollout_roi_reconstruction_proxy": 0.0,
        "rollout_phase_accuracy": 0.0,
        "rollout_family_accuracy": 0.0,
        "rollout_target_profile_consistency": 0.0,
        "rollout_temporal_contract_alignment": 0.0,
        "rollout_dynamics_contract_validity": 0.0,
        "rollout_renderer_contract_validity": 0.0,
        "rollout_reveal_quality": 0.0,
        "rollout_occlusion_quality": 0.0,
    }

    total_steps = 0
    for start_idx in range(len(records)):
        start_record = records[start_idx]
        try:
            current_graph = _deserialize_graph(start_record.get("scene_graph_before", {}))
            current_roi, _, _, _ = _record_rois(start_record)
        except Exception:
            continue

        for step in range(max(1, rollout_steps)):
            rec_idx = start_idx + step
            if rec_idx >= len(records):
                break
            rec = records[rec_idx]
            gt_graph_after = _deserialize_graph(rec.get("scene_graph_after", {}))
            gt_before_roi, gt_after_roi, changed_mask, preservation = _record_rois(rec)
            if mode == "teacher_forced_rollout":
                cur_roi = gt_before_roi
                cur_graph = _deserialize_graph(rec.get("scene_graph_before", {}))
            else:
                cur_roi = current_roi
                cur_graph = current_graph

            planner = rec.get("planner_context", {}) if isinstance(rec.get("planner_context", {}), dict) else {}
            phase_gt = str(rec.get("phase_estimate", planner.get("phase", "transition")))
            family_gt = str(rec.get("transition_family", rec.get("runtime_semantic_transition", "pose_transition")))
            target_profile_gt = rec.get("target_profile", {}) if isinstance(rec.get("target_profile", {}), dict) else {}

            feature_vector = _build_temporal_transition_features(
                graph_before=cur_graph,
                graph_after=gt_graph_after,
                roi_records=rec.get("roi_records", []) if isinstance(rec.get("roi_records", []), list) else [],
                graph_delta_target=rec.get("graph_delta_target", {}) if isinstance(rec.get("graph_delta_target", {}), dict) else {},
                planner_context=planner,
                target_profile=target_profile_gt,
                runtime_semantic_transition=str(rec.get("runtime_semantic_transition", family_gt)),
                phase_estimate=phase_gt,
                reveal_score=float(np.clip(rec.get("reveal_score", float(np.mean(changed_mask))), 0.0, 1.0)),
                occlusion_score=float(np.clip(rec.get("occlusion_score", float(np.mean(changed_mask)) * 0.5), 0.0, 1.0)),
                support_score=float(np.clip(((rec.get("graph_delta_target", {}) or {}).get("interaction_deltas", {}) or {}).get("support_contact", 0.0), 0.0, 1.0)),
                transition_confidence=float(np.clip(rec.get("transition_confidence", 1.0), 0.0, 1.0)),
            )
            temporal_pred = temporal.forward(np.asarray(feature_vector, dtype=np.float64))
            temporal_contract = temporal.to_typed_contract(temporal_pred)
            selected_family = temporal_contract.predicted_family
            selected_phase = temporal_contract.predicted_phase
            selected_profile = {
                "primary_regions": list(temporal_contract.target_profile.primary_regions),
                "secondary_regions": list(temporal_contract.target_profile.secondary_regions),
                "context_regions": list(temporal_contract.target_profile.context_regions),
            }
            reveal_score = float(temporal_contract.reveal_score)
            occlusion_score = float(temporal_contract.occlusion_score)
            support_score = float(temporal_contract.support_contact_score)
            human_contract_payload = {}
            if use_human_state_contract:
                human_input = np.asarray(feature_vector + list(temporal_contract.transition_embedding) + [reveal_score, occlusion_score, support_score], dtype=np.float64)
                if human_input.shape[0] < human_state.input_dim:
                    human_input = np.concatenate([human_input, np.zeros((human_state.input_dim - human_input.shape[0],), dtype=np.float64)])
                human_pred = human_state.forward(human_input[: human_state.input_dim])
                human_contract = human_state.to_typed_contract(human_pred)
                human_contract_payload = human_contract.to_metadata()
                selected_family = human_contract.predicted_family
                selected_phase = human_contract.predicted_phase
                selected_profile = {
                    "primary_regions": list(human_contract.target_profile.primary_regions),
                    "secondary_regions": list(human_contract.target_profile.secondary_regions),
                    "context_regions": list(human_contract.target_profile.context_regions),
                }
                mean_vis = float(np.mean(list(human_contract.visibility_state_scores.values()) or [0.0]))
                reveal_score = mean_vis
                occlusion_score = 1.0 - mean_vis
                support_score = float(human_contract.support_contact_state)

            labels = [str(x) for x in ((rec.get("graph_delta_target", {}) or {}).get("semantic_reasons", []))] or [family_gt]
            labels += [f"temporal_family={selected_family}", f"temporal_phase={selected_phase}"]
            plan_state = PlannedState(step_index=int(planner.get("step_index", rec_idx + 1)), labels=labels)
            dyn_inputs = featurize_runtime(
                cur_graph,
                plan_state,
                {
                    "step_index": float(planner.get("step_index", rec_idx + 1)),
                    "total_steps": float(planner.get("total_steps", max(2, len(records)))),
                    "phase": selected_phase,
                    "target_duration": float(planner.get("target_duration", 1.0)),
                },
                None,
            )
            dyn_pred = dynamics.forward(dyn_inputs)
            pred_delta = decode_prediction(
                dyn_pred,
                scene_graph=copy.deepcopy(cur_graph),
                phase=selected_phase,
                semantic_reasons=labels,
                planner_context={"phase": selected_phase},
            )

            render_batch = type("_RolloutBatch", (), {})()
            render_batch.roi_before = cur_roi
            render_batch.roi_after = gt_after_roi
            render_batch.changed_mask = changed_mask
            render_batch.alpha_target = np.clip(0.2 + 0.8 * changed_mask, 0.0, 1.0)
            render_batch.blend_hint = preservation
            render_batch.semantic_embed = np.asarray([1.0 if selected_family == "expression_transition" else 0.0, 1.0 if selected_family in {"garment_transition", "visibility_transition"} else 0.0, 1.0 if selected_family in {"pose_transition", "interaction_transition"} else 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            render_batch.delta_cond = np.asarray([reveal_score, occlusion_score, support_score] + [0.0] * 9, dtype=np.float32)[:12]
            render_batch.planner_cond = np.asarray([1.0 if selected_phase == p else 0.0 for p in ("prepare", "transition", "contact_or_reveal", "stabilize")] + [0.0] * 10, dtype=np.float32)[:10]
            render_batch.graph_cond = np.zeros((8,), dtype=np.float32)
            render_batch.memory_cond = np.zeros((10,), dtype=np.float32)
            render_batch.appearance_cond = np.asarray(np.concatenate([np.mean(cur_roi, axis=(0, 1)), np.std(cur_roi, axis=(0, 1))]), dtype=np.float32)
            render_batch.bbox_cond = np.asarray([0.2, 0.2, 0.4, 0.4, 0.0, 0.0], dtype=np.float32)
            render_batch.mode_cond = None
            render_batch.role_cond = None
            render_batch.preservation_mask = None
            render_batch.uncertainty_target = None
            render_batch.seam_prior = None
            render_batch.transition_mode = "stable"
            render_batch.profile_role = "primary"
            render_batch.conditioning_summary = {
                "contract_source": "rollout_eval",
                "predicted_family": selected_family,
                "predicted_phase": selected_phase,
            }
            render_batch.previous_roi = cur_roi if renderer_backend == "temporal_local_renderer" else None
            render_batch.predicted_family = selected_family
            render_batch.predicted_phase = selected_phase
            render_batch.target_profile = selected_profile
            render_batch.reveal_score = float(reveal_score)
            render_batch.occlusion_score = float(occlusion_score)
            render_batch.support_contact_score = float(support_score)
            render_batch.rollout_weight = float(np.clip(1.0 + render_batch.reveal_score * 0.4 + render_batch.occlusion_score * 0.3, 0.5, 2.0))
            render_out = renderer.forward(render_batch)
            pred_roi = np.asarray(render_out["rgb"], dtype=np.float32)

            gt_delta_raw = rec.get("graph_delta_target", {}) if isinstance(rec.get("graph_delta_target", {}), dict) else {}
            gt_regions = _target_regions(target_profile_gt)
            pred_regions = _target_regions(selected_profile)

            frame_proxy = float(max(0.0, 1.0 - np.mean(np.abs(pred_roi - gt_after_roi))))
            roi_proxy = float(max(0.0, 1.0 - np.mean(np.abs(pred_roi - gt_after_roi) * (0.25 + 0.75 * changed_mask))))
            phase_acc = 1.0 if selected_phase == phase_gt else 0.0
            family_acc = 1.0 if selected_family == family_gt else 0.0
            profile_consistency = _jaccard(pred_regions, gt_regions)
            alignment = float(np.clip(0.4 * family_acc + 0.3 * phase_acc + 0.3 * profile_consistency, 0.0, 1.0))
            dyn_valid = 1.0 if (pred_delta.region_transition_mode and pred_delta.state_after and pred_delta.affected_regions) else 0.0
            renderer_valid = 1.0 if pred_roi.shape == gt_after_roi.shape else 0.0
            reveal_q = float(max(0.0, 1.0 - abs(float(reveal_score) - float(np.clip(rec.get("reveal_score", 0.0), 0.0, 1.0)))))
            occlusion_q = float(max(0.0, 1.0 - abs(float(occlusion_score) - float(np.clip(rec.get("occlusion_score", 0.0), 0.0, 1.0)))))

            metrics["rollout_frame_reconstruction_proxy"] += frame_proxy
            metrics["rollout_roi_reconstruction_proxy"] += roi_proxy
            metrics["rollout_phase_accuracy"] += phase_acc
            metrics["rollout_family_accuracy"] += family_acc
            metrics["rollout_target_profile_consistency"] += profile_consistency
            metrics["rollout_temporal_contract_alignment"] += alignment
            metrics["rollout_dynamics_contract_validity"] += dyn_valid
            metrics["rollout_renderer_contract_validity"] += renderer_valid
            metrics["rollout_reveal_quality"] += reveal_q
            metrics["rollout_occlusion_quality"] += occlusion_q

            traces.append(
                RolloutStepPayload(
                    step_index=step,
                    mode=mode,
                    rollout_payload={
                        "start_frame": cur_roi.tolist(),
                        "start_graph": _serialize_graph(cur_graph),
                        "predicted_temporal_contract": temporal_contract.to_metadata(),
                        "predicted_human_state_contract": human_contract_payload,
                        "active_conditioning_source": "human_state" if human_contract_payload else "temporal",
                        "active_target_profile": selected_profile,
                        "predicted_graph_delta": _serialize_delta_contract(pred_delta),
                        "predicted_rendered_roi": {
                            "rgb": pred_roi.tolist(),
                            "alpha": np.asarray(render_out.get("alpha", np.zeros(pred_roi.shape[:2], dtype=np.float32))).tolist(),
                            "uncertainty": np.asarray(render_out.get("uncertainty", np.zeros(pred_roi.shape[:2], dtype=np.float32))).tolist(),
                        },
                        "gt_next_frame": gt_after_roi.tolist(),
                        "gt_roi": gt_after_roi.tolist(),
                        "gt_delta": gt_delta_raw,
                        "gt_target_profile": target_profile_gt,
                        "gt_phase": phase_gt,
                        "diagnostics": {
                            "frame_reconstruction_proxy": frame_proxy,
                            "roi_reconstruction_proxy": roi_proxy,
                            "temporal_alignment": alignment,
                            "dynamics_contract_validity": dyn_valid,
                            "renderer_contract_validity": renderer_valid,
                        },
                        "provenance": {
                            "manifest_path": dataset_manifest,
                            "record_id": rec.get("record_id", f"record_{rec_idx}"),
                            "mode": mode,
                            "rollout_step": step,
                        },
                    },
                )
            )

            if mode == "predicted_rollout":
                current_graph = apply_delta(copy.deepcopy(cur_graph), pred_delta)
                current_roi = pred_roi
            else:
                current_graph = _deserialize_graph(rec.get("scene_graph_after", {}))
                current_roi = gt_after_roi
            total_steps += 1

    denom = max(1, total_steps)
    out_metrics = {k: round(float(v) / float(denom), 6) for k, v in metrics.items()}
    out_metrics.update(
        {
            "records_used": len(records),
            "steps_evaluated": total_steps,
            "dataset_source": "manifest_video_rollout_eval_primary",
            "rollout_mode": mode,
            "renderer_backend": renderer_backend,
            "conditioning_path": "temporal_plus_human_state" if use_human_state_contract else "temporal_only",
            "payloads": [t.rollout_payload for t in traces],
        }
    )
    return out_metrics


def evaluate_rollout_modes_on_video_manifest(
    *,
    dataset_manifest: str,
    temporal_model: TemporalTransitionEncoder | None = None,
    dynamics_model: DynamicsModel | None = None,
    renderer_model: TrainableLocalPatchModel | TemporalLocalPatchModel | None = None,
    rollout_steps: int = 1,
    max_records: int | None = None,
    renderer_backend: str = "legacy_local_renderer",
    use_human_state_contract: bool = False,
    human_state_model: HumanStateTransitionModel | None = None,
) -> dict[str, object]:
    teacher = evaluate_rollout_on_video_manifest(
        dataset_manifest=dataset_manifest,
        temporal_model=temporal_model,
        dynamics_model=dynamics_model,
        renderer_model=renderer_model,
        mode="teacher_forced_rollout",
        rollout_steps=rollout_steps,
        max_records=max_records,
        renderer_backend=renderer_backend,
    )
    predicted_human = evaluate_rollout_on_video_manifest(
        dataset_manifest=dataset_manifest,
        temporal_model=temporal_model,
        dynamics_model=dynamics_model,
        renderer_model=renderer_model,
        mode="predicted_rollout",
        rollout_steps=rollout_steps,
        max_records=max_records,
        renderer_backend=renderer_backend,
        use_human_state_contract=True,
        human_state_model=human_state_model,
    )
    predicted_human = evaluate_rollout_on_video_manifest(
        dataset_manifest=dataset_manifest,
        temporal_model=temporal_model,
        dynamics_model=dynamics_model,
        renderer_model=renderer_model,
        mode="predicted_rollout",
        rollout_steps=rollout_steps,
        max_records=max_records,
        renderer_backend=renderer_backend,
        use_human_state_contract=True,
    )
    return {
        "teacher_forced_rollout": teacher,
        "predicted_rollout": predicted,
        "predicted_rollout_with_human_state": predicted_human,
        "path_comparison": {
            "temporal_only": predicted.get("rollout_frame_reconstruction_proxy", 0.0),
            "temporal_plus_human_state": predicted_human.get("rollout_frame_reconstruction_proxy", 0.0),
        },
    }


def tiny_video_overfit_harness(
    dataset_manifest: str,
    *,
    tiny_subset_records: int = 3,
    epochs: int = 4,
    rollout_steps: int = 1,
    renderer_backend: str = "legacy_local_renderer",
    use_human_state_contract: bool = False,
    human_state_model: HumanStateTransitionModel | None = None,
) -> dict[str, object]:
    from training.datasets import DynamicsDataset, RendererDataset, TemporalTransitionDataset
    from training.dynamics_trainer import DynamicsDatasetAdapter
    from training.renderer_trainer import RendererBatchAdapter
    from training.temporal_transition_trainer import TemporalTransitionDatasetAdapter

    temporal_ds = TemporalTransitionDataset.from_video_transition_manifest(dataset_manifest, strict=False)
    dynamics_ds = DynamicsDataset.from_video_transition_manifest(dataset_manifest, strict=False)
    renderer_ds = RendererDataset.from_video_transition_manifest(dataset_manifest, strict=False)

    temporal = TemporalTransitionEncoder()
    dynamics = DynamicsModel()
    renderer = TemporalLocalPatchModel() if renderer_backend == "temporal_local_renderer" else TrainableLocalPatchModel()

    before = evaluate_rollout_on_video_manifest(
        dataset_manifest=dataset_manifest,
        temporal_model=temporal,
        dynamics_model=dynamics,
        renderer_model=renderer,
        mode="teacher_forced_rollout",
        rollout_steps=rollout_steps,
        max_records=tiny_subset_records,
        renderer_backend=renderer_backend,
    )
    best_after = dict(before)

    t_samples = temporal_ds.samples[:tiny_subset_records]
    d_samples = dynamics_ds.samples[:tiny_subset_records]
    r_samples = renderer_ds.samples[:tiny_subset_records]

    for _ in range(max(1, epochs)):
        for sample in t_samples:
            batch = TemporalTransitionDatasetAdapter.sample_to_batch(sample)
            temporal.train_step(batch.features, batch.targets, lr=5e-3)
        for idx, sample in enumerate(d_samples):
            batch = DynamicsDatasetAdapter.sample_to_batch(sample, encoder=temporal, conditioning_mode="learned_contract_only", step_index=idx + 1)
            dynamics.train_step(batch.inputs, batch.targets, lr=2e-3)
        adapter = RendererBatchAdapter(encoder=temporal, conditioning_mode="learned_contract_only")
        for sample in r_samples:
            batch = adapter.adapt(sample, temporal_mode=renderer_backend == "temporal_local_renderer")
            renderer.train_step(batch, lr=2e-3)
        candidate = evaluate_rollout_on_video_manifest(
            dataset_manifest=dataset_manifest,
            temporal_model=temporal,
            dynamics_model=dynamics,
            renderer_model=renderer,
            mode="teacher_forced_rollout",
            rollout_steps=rollout_steps,
            max_records=tiny_subset_records,
            renderer_backend=renderer_backend,
        )
        if float(candidate.get("rollout_frame_reconstruction_proxy", 0.0)) >= float(best_after.get("rollout_frame_reconstruction_proxy", 0.0)):
            best_after = candidate

    after = best_after
    return {
        "before": before,
        "after": after,
        "improved": bool(after.get("rollout_frame_reconstruction_proxy", 0.0) > before.get("rollout_frame_reconstruction_proxy", 0.0)),
        "delta_frame_proxy": float(after.get("rollout_frame_reconstruction_proxy", 0.0) - before.get("rollout_frame_reconstruction_proxy", 0.0)),
    }
