from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

from core.schema import (
    ActionStep,
    BBox,
    BodyPartNode,
    GarmentNode,
    GlobalSceneContext,
    GraphDelta,
    Keypoint,
    PersonNode,
    RegionRef,
    RelationEdge,
    SceneGraph,
    SceneObjectNode,
)
from dynamics.state_update import apply_delta
from evaluation.contracts import (
    build_graph_eval_payload,
    build_hidden_reconstruction_payload,
    build_patch_eval_payload,
    build_temporal_eval_payload,
    build_text_eval_payload,
    graph_transition_eval,
    hidden_region_reconstruction_eval,
    patch_synthesis_eval,
    temporal_consistency_eval,
    text_action_alignment_eval,
)
from learned.factory import BackendBundle, BackendConfig, LearnedBackendFactory
from learned.interfaces import DynamicsTransitionRequest, PatchSynthesisRequest, TemporalRefinementRequest
from training.datasets import _serialize_graph
from memory.video_memory import MemoryManager
from training.learned_contracts import (
    build_graph_transition_contract,
    build_patch_synthesis_contract,
    build_temporal_consistency_contract,
    build_text_action_state_contract,
)


@dataclass(slots=True)
class StageScaffoldConfig:
    stage_name: str
    model_backend: str = "baseline"
    dataset_path: str = ""
    batch_size: int = 2
    learning_rate: float = 1e-4
    epochs: int = 1
    checkpoint_path: str = "artifacts/checkpoints/stage.ckpt"
    backend_config: BackendConfig | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StageScaffoldResult:
    stage_name: str
    backend: str
    checkpoint_path: str
    expected_inputs: list[str]
    expected_outputs: list[str]
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    samples_processed: int = 0
    ingestion_warnings: list[dict[str, object]] = field(default_factory=list)


class LearnedStageDatasetRouter:
    @staticmethod
    def _warn(
        warnings: list[dict[str, object]],
        issue: str,
        *,
        source: str = "dataset_router",
        severity: str = "warning",
        field_path: str | None = None,
    ) -> None:
        warnings.append(
            {
                "source": source,
                "issue": issue,
                "severity": severity,
                "field_path": field_path,
            }
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _as_float(value: object, fallback: float, *, warnings: list[dict[str, object]], issue: str, field_path: str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        LearnedStageDatasetRouter._warn(warnings, issue, field_path=field_path)
        return fallback

    @staticmethod
    def _sanitize_label(value: object, fallback: str, *, warnings: list[dict[str, object]], issue: str, field_path: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace(" ", "_")
        LearnedStageDatasetRouter._warn(warnings, issue, field_path=field_path)
        return fallback

    @staticmethod
    def _coerce_bbox(data: object, fallback: BBox | None = None, *, warnings: list[dict[str, object]] | None = None, scope: str = "bbox") -> BBox:
        local_warnings = warnings if warnings is not None else []
        if isinstance(data, dict):
            x = LearnedStageDatasetRouter._as_float(data.get("x", 0.1), 0.1, warnings=local_warnings, issue=f"{scope}.x_invalid_fallback", field_path=f"{scope}.x")
            y = LearnedStageDatasetRouter._as_float(data.get("y", 0.1), 0.1, warnings=local_warnings, issue=f"{scope}.y_invalid_fallback", field_path=f"{scope}.y")
            w = LearnedStageDatasetRouter._as_float(data.get("w", 0.6), 0.6, warnings=local_warnings, issue=f"{scope}.w_invalid_fallback", field_path=f"{scope}.w")
            h = LearnedStageDatasetRouter._as_float(data.get("h", 0.8), 0.8, warnings=local_warnings, issue=f"{scope}.h_invalid_fallback", field_path=f"{scope}.h")
            clamped = BBox(
                LearnedStageDatasetRouter._clamp(x, 0.0, 1.0),
                LearnedStageDatasetRouter._clamp(y, 0.0, 1.0),
                LearnedStageDatasetRouter._clamp(w, 0.0, 1.0),
                LearnedStageDatasetRouter._clamp(h, 0.0, 1.0),
            )
            if (x, y, w, h) != (clamped.x, clamped.y, clamped.w, clamped.h):
                LearnedStageDatasetRouter._warn(local_warnings, f"{scope}.clamped", field_path=scope)
            return clamped
        LearnedStageDatasetRouter._warn(local_warnings, f"{scope}.missing_fallback", field_path=scope)
        return fallback or BBox(0.1, 0.1, 0.6, 0.8)

    @staticmethod
    def _coerce_region(region_id: str, payload: object, reason: str = "motion", *, warnings: list[dict[str, object]] | None = None, scope: str = "region") -> RegionRef:
        local_warnings = warnings if warnings is not None else []
        bbox = LearnedStageDatasetRouter._coerce_bbox(payload if isinstance(payload, dict) else {}, warnings=local_warnings, scope=f"{scope}.bbox")
        resolved_reason = reason
        if isinstance(payload, dict):
            resolved_reason = LearnedStageDatasetRouter._sanitize_label(payload.get("reason", reason), reason, warnings=local_warnings, issue=f"{scope}.reason_sanitized", field_path=f"{scope}.reason")
            region_id = str(payload.get("region_id", region_id))
        return RegionRef(region_id=region_id, bbox=bbox, reason=resolved_reason)

    @staticmethod
    def _coerce_action_steps(payload: object, fallback_token: str = "micro_adjust") -> list[ActionStep]:
        if isinstance(payload, list) and payload:
            steps: list[ActionStep] = []
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                steps.append(
                    ActionStep(
                        type=str(item.get("type", fallback_token)),
                        priority=int(item.get("priority", idx + 1)),
                        target_entity=item.get("target_entity"),
                        target_object=item.get("target_object"),
                        body_part=item.get("body_part"),
                        can_run_parallel=bool(item.get("can_run_parallel", False)),
                        start_after=[int(v) for v in item.get("start_after", []) if isinstance(v, int)],
                        constraints=[str(c) for c in item.get("constraints", []) if isinstance(c, str)],
                    )
                )
            if steps:
                return steps
        return [ActionStep(type=fallback_token, priority=1)]

    @staticmethod
    def _coerce_graph(record: dict[str, object], key: str, frame_index: int, track: str, *, warnings: list[str] | None = None) -> SceneGraph:
        local_warnings = warnings if warnings is not None else []
        raw = record.get(key)
        if isinstance(raw, dict):
            persons: list[PersonNode] = []
            for idx, p in enumerate(raw.get("persons", []) if isinstance(raw.get("persons"), list) else []):
                if not isinstance(p, dict):
                    LearnedStageDatasetRouter._warn(local_warnings, f"{key}.person_{idx}_invalid_skipped")
                    continue
                pid = str(p.get("person_id", f"p{track}_{idx}"))
                body_parts: list = []
                for bp_idx, bp in enumerate(p.get("body_parts", []) if isinstance(p.get("body_parts"), list) else []):
                    if not isinstance(bp, dict):
                        LearnedStageDatasetRouter._warn(local_warnings, f"{key}.{pid}.body_part_{bp_idx}_invalid_skipped")
                        continue
                    keypoints: list[Keypoint] = []
                    for k_idx, kp in enumerate(bp.get("keypoints", []) if isinstance(bp.get("keypoints"), list) else []):
                        if not isinstance(kp, dict):
                            LearnedStageDatasetRouter._warn(local_warnings, f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}_invalid_skipped")
                            continue
                        kx = LearnedStageDatasetRouter._clamp(
                            LearnedStageDatasetRouter._as_float(
                                kp.get("x", 0.0),
                                0.0,
                                warnings=local_warnings,
                                issue=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.x_invalid_fallback",
                                field_path=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.x",
                            ),
                            0.0,
                            1.0,
                        )
                        ky = LearnedStageDatasetRouter._clamp(
                            LearnedStageDatasetRouter._as_float(
                                kp.get("y", 0.0),
                                0.0,
                                warnings=local_warnings,
                                issue=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.y_invalid_fallback",
                                field_path=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.y",
                            ),
                            0.0,
                            1.0,
                        )
                        conf_raw = LearnedStageDatasetRouter._as_float(
                            kp.get("confidence", 0.0),
                            0.0,
                            warnings=local_warnings,
                            issue=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.confidence_invalid_fallback",
                            field_path=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.confidence",
                        )
                        conf = LearnedStageDatasetRouter._clamp(conf_raw, 0.0, 1.0)
                        if conf != conf_raw:
                            LearnedStageDatasetRouter._warn(local_warnings, f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.confidence_clamped")
                        keypoints.append(Keypoint(name=str(kp.get("name", f"kp_{bp_idx}_{k_idx}")), x=kx, y=ky, confidence=conf))
                    body_parts.append(
                        BodyPartNode(
                            part_id=str(bp.get("part_id", f"{pid}_part_{bp_idx}")),
                            part_type=LearnedStageDatasetRouter._sanitize_label(bp.get("part_type", "unknown"), "unknown", warnings=local_warnings, issue=f"{key}.{pid}.body_part_{bp_idx}.part_type_sanitized", field_path=f"{key}.{pid}.body_part_{bp_idx}.part_type"),
                            keypoints=keypoints,
                            mask_ref=bp.get("mask_ref"),
                            visibility=LearnedStageDatasetRouter._sanitize_label(bp.get("visibility", "unknown"), "unknown", warnings=local_warnings, issue=f"{key}.{pid}.body_part_{bp_idx}.visibility_sanitized", field_path=f"{key}.{pid}.body_part_{bp_idx}.visibility"),
                            occluded_by=[str(v) for v in bp.get("occluded_by", []) if isinstance(v, str)],
                            depth_order=LearnedStageDatasetRouter._as_float(bp.get("depth_order", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.body_part_{bp_idx}.depth_order_invalid_fallback", field_path=f"{key}.{pid}.body_part_{bp_idx}.depth_order"),
                            canonical_slot=str(bp.get("canonical_slot", "")),
                            confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(bp.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.body_part_{bp_idx}.confidence_invalid_fallback", field_path=f"{key}.{pid}.body_part_{bp_idx}.confidence"), 0.0, 1.0),
                            source=str(bp.get("source", "dataset")),
                            frame_index=int(bp.get("frame_index", raw.get("frame_index", frame_index))),
                            timestamp=float(bp["timestamp"]) if isinstance(bp.get("timestamp"), (int, float)) else None,
                        )
                    )
                garments = [
                    GarmentNode(
                        garment_id=str(g.get("garment_id", f"{pid}_garment_{g_idx}")),
                        garment_type=LearnedStageDatasetRouter._sanitize_label(g.get("garment_type", g.get("type", "unknown")), "unknown", warnings=local_warnings, issue=f"{key}.{pid}.garment_{g_idx}.garment_type_sanitized", field_path=f"{key}.{pid}.garment_{g_idx}.garment_type"),
                        mask_ref=g.get("mask_ref"),
                        attachment_targets=[str(v) for v in g.get("attachment_targets", []) if isinstance(v, str)],
                        coverage_targets=[str(v) for v in g.get("coverage_targets", []) if isinstance(v, str)],
                        garment_state=LearnedStageDatasetRouter._sanitize_label(g.get("garment_state", g.get("state", "worn")), "worn", warnings=local_warnings, issue=f"{key}.{pid}.garment_{g_idx}.garment_state_sanitized", field_path=f"{key}.{pid}.garment_{g_idx}.garment_state"),
                        visibility=LearnedStageDatasetRouter._sanitize_label(g.get("visibility", "unknown"), "unknown", warnings=local_warnings, issue=f"{key}.{pid}.garment_{g_idx}.visibility_sanitized", field_path=f"{key}.{pid}.garment_{g_idx}.visibility"),
                        appearance_ref=g.get("appearance_ref"),
                        confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(g.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.garment_{g_idx}.confidence_invalid_fallback", field_path=f"{key}.{pid}.garment_{g_idx}.confidence"), 0.0, 1.0),
                        source=str(g.get("source", "dataset")),
                        frame_index=int(g.get("frame_index", raw.get("frame_index", frame_index))),
                        timestamp=float(g["timestamp"]) if isinstance(g.get("timestamp"), (int, float)) else None,
                    )
                    for g_idx, g in enumerate(p.get("garments", []) if isinstance(p.get("garments"), list) else [])
                    if isinstance(g, dict)
                ]
                persons.append(
                    PersonNode(
                        person_id=pid,
                        track_id=str(p.get("track_id", f"t{track}_{idx}")),
                        bbox=LearnedStageDatasetRouter._coerce_bbox(p.get("bbox"), warnings=local_warnings, scope=f"{key}.{pid}.bbox"),
                        mask_ref=p.get("mask_ref"),
                        body_parts=body_parts,
                        garments=garments,
                        confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(p.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.confidence_invalid_fallback", field_path=f"{key}.{pid}.confidence"), 0.0, 1.0),
                        source=str(p.get("source", "dataset")),
                        frame_index=int(p.get("frame_index", raw.get("frame_index", frame_index))),
                        timestamp=float(p["timestamp"]) if isinstance(p.get("timestamp"), (int, float)) else None,
                    )
                )
            objects = [
                SceneObjectNode(
                    object_id=str(o.get("object_id", f"obj_{track}_{idx}")),
                    object_type=LearnedStageDatasetRouter._sanitize_label(o.get("object_type", "unknown"), "unknown", warnings=local_warnings, issue=f"{key}.object_{idx}.type_sanitized", field_path=f"{key}.object_{idx}.object_type"),
                    bbox=LearnedStageDatasetRouter._coerce_bbox(o.get("bbox"), warnings=local_warnings, scope=f"{key}.object_{idx}.bbox"),
                    mask_ref=o.get("mask_ref"),
                    confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(o.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.object_{idx}.confidence_invalid_fallback", field_path=f"{key}.object_{idx}.confidence"), 0.0, 1.0),
                    source=str(o.get("source", "dataset")),
                    frame_index=int(o.get("frame_index", raw.get("frame_index", frame_index))),
                    timestamp=float(o["timestamp"]) if isinstance(o.get("timestamp"), (int, float)) else None,
                )
                for idx, o in enumerate(raw.get("objects", []) if isinstance(raw.get("objects"), list) else [])
                if isinstance(o, dict)
            ]
            relations = [
                RelationEdge(
                    source=str(r.get("source", "")),
                    relation=LearnedStageDatasetRouter._sanitize_label(r.get("relation", "near"), "near", warnings=local_warnings, issue=f"{key}.relation.relation_sanitized", field_path=f"{key}.relation.relation"),
                    target=str(r.get("target", "")),
                    confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(r.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.relation.confidence_invalid_fallback", field_path=f"{key}.relation.confidence"), 0.0, 1.0),
                    provenance=str(r.get("provenance", "dataset")),
                    frame_index=int(r.get("frame_index", raw.get("frame_index", frame_index))),
                    timestamp=float(r["timestamp"]) if isinstance(r.get("timestamp"), (int, float)) else None,
                )
                for r in raw.get("relations", []) if isinstance(r, dict)
            ]
            gc_raw = raw.get("global_context", {}) if isinstance(raw.get("global_context"), dict) else {}
            frame_size_raw = gc_raw.get("frame_size")
            frame_size = (0, 0)
            if isinstance(frame_size_raw, list) and len(frame_size_raw) >= 2 and all(isinstance(v, (int, float)) for v in frame_size_raw[:2]):
                frame_size = (max(1, int(frame_size_raw[0])), max(1, int(frame_size_raw[1])))
            else:
                LearnedStageDatasetRouter._warn(local_warnings, f"{key}.global_context.frame_size_invalid_fallback", field_path=f"{key}.global_context.frame_size")
            fps_raw = gc_raw.get("fps", 16)
            fps = int(fps_raw) if isinstance(fps_raw, (int, float)) and int(fps_raw) > 0 else 16
            if fps == 16 and fps_raw != 16:
                LearnedStageDatasetRouter._warn(local_warnings, f"{key}.global_context.fps_invalid_fallback", field_path=f"{key}.global_context.fps")
            context = GlobalSceneContext(
                frame_size=frame_size,
                fps=fps,
                source_type=str(gc_raw.get("source_type", "dataset")),
            )
            graph = SceneGraph(
                frame_index=int(raw.get("frame_index", frame_index)),
                persons=persons,
                objects=objects,
                relations=relations,
                global_context=context,
                timestamp=float(raw["timestamp"]) if isinstance(raw.get("timestamp"), (int, float)) else None,
            )
            return graph
        LearnedStageDatasetRouter._warn(local_warnings, f"{key}.missing_graph_fallback_base_graph", field_path=key)
        return LearnedStageDatasetRouter._base_graph(frame_index, track)

    @staticmethod
    def _coerce_delta(record: dict[str, object], before: SceneGraph, action_tokens: list[str], phase: str = "motion", *, warnings: list[dict[str, object]] | None = None) -> GraphDelta:
        local_warnings = warnings if warnings is not None else []
        payload = record.get("delta", {})
        if isinstance(payload, dict):
            def sanitize_dict(raw: object, scope: str) -> dict[str, float]:
                if not isinstance(raw, dict):
                    LearnedStageDatasetRouter._warn(local_warnings, f"{scope}.malformed_fallback_empty", field_path=scope)
                    return {}
                out: dict[str, float] = {}
                for k, v in raw.items():
                    out[str(k)] = LearnedStageDatasetRouter._as_float(v, 0.0, warnings=local_warnings, issue=f"{scope}.{k}_invalid_fallback", field_path=f"{scope}.{k}")
                return out

            def sanitize_state_dict(raw: object, scope: str) -> dict[str, str]:
                if not isinstance(raw, dict):
                    LearnedStageDatasetRouter._warn(local_warnings, f"{scope}.malformed_fallback_empty", field_path=scope)
                    return {}
                out: dict[str, str] = {}
                for k, v in raw.items():
                    out[str(k)] = LearnedStageDatasetRouter._sanitize_label(v, "unknown", warnings=local_warnings, issue=f"{scope}.{k}_sanitized", field_path=f"{scope}.{k}")
                return out

            revealed_raw = payload.get("newly_revealed_regions", [])
            occluded_raw = payload.get("newly_occluded_regions", [])
            if not isinstance(revealed_raw, list):
                LearnedStageDatasetRouter._warn(local_warnings, "delta.newly_revealed_regions_malformed_fallback_empty", field_path="delta.newly_revealed_regions")
                revealed_raw = []
            if not isinstance(occluded_raw, list):
                LearnedStageDatasetRouter._warn(local_warnings, "delta.newly_occluded_regions_malformed_fallback_empty", field_path="delta.newly_occluded_regions")
                occluded_raw = []
            revealed = [LearnedStageDatasetRouter._coerce_region(region_id=f"{before.persons[0].person_id}:revealed_{idx}", payload=item, reason="reveal", warnings=local_warnings, scope=f"delta.newly_revealed_regions[{idx}]") for idx, item in enumerate(revealed_raw) if isinstance(item, dict)] if before.persons else []
            occluded = [LearnedStageDatasetRouter._coerce_region(region_id=f"{before.persons[0].person_id}:occluded_{idx}", payload=item, reason="occlude", warnings=local_warnings, scope=f"delta.newly_occluded_regions[{idx}]") for idx, item in enumerate(occluded_raw) if isinstance(item, dict)] if before.persons else []
            return GraphDelta(
                pose_deltas=sanitize_dict(payload.get("pose_deltas", {}), "delta.pose_deltas"),
                interaction_deltas=sanitize_dict(payload.get("interaction_deltas", {}), "delta.interaction_deltas"),
                semantic_reasons=[str(v) for v in payload.get("semantic_reasons", action_tokens)],
                affected_entities=[str(v) for v in payload.get("affected_entities", [before.persons[0].person_id] if before.persons else [])],
                affected_regions=[str(v) for v in payload.get("affected_regions", [])],
                region_transition_mode=sanitize_state_dict(payload.get("region_transition_mode", {}), "delta.region_transition_mode"),
                predicted_visibility_changes=sanitize_state_dict(payload.get("predicted_visibility_changes", {}), "delta.predicted_visibility_changes"),
                state_before=sanitize_state_dict(payload.get("state_before", {}), "delta.state_before"),
                state_after=sanitize_state_dict(payload.get("state_after", {}), "delta.state_after"),
                transition_phase=LearnedStageDatasetRouter._sanitize_label(payload.get("transition_phase", phase), phase, warnings=local_warnings, issue="delta.transition_phase_sanitized", field_path="delta.transition_phase"),
                newly_revealed_regions=revealed,
                newly_occluded_regions=occluded,
            )
        LearnedStageDatasetRouter._warn(local_warnings, "delta.malformed_fallback_default_delta", field_path="delta")
        return GraphDelta(
            pose_deltas={"torso_pitch": 0.04},
            semantic_reasons=action_tokens or ["micro_adjust"],
            affected_entities=[before.persons[0].person_id] if before.persons else [],
            affected_regions=["torso"],
            region_transition_mode={"torso": "deform"},
            state_before={"pose_phase": "stable"},
            state_after={"pose_phase": "transition"},
            transition_phase=phase,
        )

    @staticmethod
    def _base_graph(frame_index: int, track: str, with_relation: bool = False) -> SceneGraph:
        person = PersonNode(person_id=f"p{track}", track_id=f"t{track}", bbox=BBox(0.1, 0.1, 0.7, 0.8), mask_ref=None)
        graph = SceneGraph(frame_index=frame_index, persons=[person])
        if with_relation:
            graph.relations = []
        return graph

    @staticmethod
    def _synthetic_text(index: int) -> dict[str, object]:
        actions = [
            ("sit down on chair_1", [ActionStep(type="sit_down", priority=1, target_entity="person_1", target_object="chair_1")]),
            ("wave right hand then turn left", [ActionStep(type="wave", priority=1, body_part="right_hand", target_entity="person_1"), ActionStep(type="turn_left", priority=2, start_after=[1])]),
            ("raise left arm and nod head in parallel", [ActionStep(type="raise_arm", priority=1, body_part="left_arm", can_run_parallel=True), ActionStep(type="nod_head", priority=1, body_part="head", can_run_parallel=True)]),
            ("slowly step back then sit while keeping balance and avoiding table", [ActionStep(type="step_back", priority=1), ActionStep(type="sit_down", priority=2, constraints=["keep_balance", "avoid_contact:table_1"], start_after=[1])]),
            ("adjust pose near object", [ActionStep(type="micro_adjust", priority=1, constraints=["underspecified_prompt"]), ActionStep(type="stabilize", priority=2, target_object="nearby_object", start_after=[1])]),
            ("person_2 should place cup_1 on table_1 then wave", [ActionStep(type="place_object", priority=1, target_entity="person_2", target_object="cup_1", constraints=["destination:table_1"]), ActionStep(type="wave", priority=2, target_entity="person_2", body_part="right_hand", start_after=[1])]),
        ]
        text, steps = actions[index % len(actions)]
        return {"text": text, "actions": steps}

    @staticmethod
    def _synthetic_dynamics(index: int) -> dict[str, object]:
        before = LearnedStageDatasetRouter._base_graph(index, str(index))
        token = ["sit_down", "wave", "turn", "stand_up", "garment_change", "micro_adjust"][index % 6]
        region_pool = {
            "sit_down": ["torso", "hips"],
            "wave": ["right_hand", "right_arm"],
            "turn": ["torso", "left_arm"],
            "stand_up": ["torso", "legs"],
            "garment_change": ["coat", "torso"],
            "micro_adjust": ["neck"],
        }
        delta = GraphDelta(
            pose_deltas={"torso_pitch": 0.08 * (index + 1)},
            interaction_deltas={"chair_contact": 1.0 if token == "sit_down" else 0.2},
            semantic_reasons=[token],
            affected_entities=[before.persons[0].person_id],
            affected_regions=region_pool[token],
            region_transition_mode={region: ("appearance" if "coat" in region else "motion") for region in region_pool[token]},
            predicted_visibility_changes={"torso": "visible"},
            state_before={"pose_phase": "stable"},
            state_after={"pose_phase": "transition", "gesture_phase": "executing"},
            transition_phase=["prepare", "motion", "settle"][index % 3],
        )
        return {"graph_before": before, "text_tokens": [token], "delta": delta}

    @staticmethod
    def _synthetic_patch(index: int) -> dict[str, object]:
        graph = LearnedStageDatasetRouter._base_graph(index, str(index))
        reasons = ["face_expression", "garment_adjustment", "arm_motion", "occlusion", "hidden_reveal", "known_hidden_refresh"]
        rid = ["face", "coat", "right_hand", "left_arm", "torso", "left_leg"][index % 6]
        region = RegionRef(region_id=f"{graph.persons[0].person_id}:{rid}", bbox=BBox(0.2, 0.1 + 0.05 * (index % 3), 0.25, 0.2), reason=reasons[index % len(reasons)])
        v = 0.2 + 0.05 * index
        frame = [[[v + ((x + y) % 5) * 0.01, 0.2, 0.1] for x in range(32)] for y in range(32)]
        hidden = {
            "state": "emerging" if index % 2 else "stable",
            "slot": region.region_id,
            "hidden_type": "unknown_hidden" if index % 2 else "known_hidden",
            "retrieval_profile": "rich" if index % 3 else "poor",
            "lifecycle": ["known_hidden", "unknown_hidden_synthesis", "known_hidden_reveal"][index % 3],
        }
        return {"region": region, "graph": graph, "frame": frame, "hidden_state": hidden, "retrieval_summary": {"profile": hidden["retrieval_profile"]}}

    @staticmethod
    def _synthetic_temporal(index: int) -> dict[str, object]:
        graph = LearnedStageDatasetRouter._base_graph(index, str(index))
        roi_count = 1 + (index % 3)
        regions = [
            RegionRef(
                region_id=f"{graph.persons[0].person_id}:region_{r}",
                bbox=BBox(0.15 + 0.1 * r, 0.1 + 0.05 * (index % 2), 0.2, 0.2),
                reason="temporal_drift" if r else "pose_update",
            )
            for r in range(roi_count)
        ]
        base = 0.25 + 0.03 * index
        prev = [[[base, base, base] for _ in range(32)] for _ in range(32)]
        cur = [[[min(1.0, base + 0.02 + (x % 3) * 0.005), base, base] for x in range(32)] for _ in range(32)]
        return {
            "graph": graph,
            "frame_prev": prev,
            "frame_cur": cur,
            "regions": regions,
            "drift": [0.01, 0.05, 0.11][index % 3],
            "temporal_profile": ["reveal", "occlusion", "multi_roi_sync"][index % 3],
        }

    @staticmethod
    def _from_dataset_path(dataset_path: str, stage_name: str, size: int) -> list[dict[str, object]]:
        path = Path(dataset_path)
        if not path.exists():
            return []
        payload = json.loads(path.read_text())
        records = payload.get("records", []) if isinstance(payload, dict) else []
        filtered: list[dict[str, object]] = []
        skipped_non_dict = 0
        for rec in records:
            if not isinstance(rec, dict):
                skipped_non_dict += 1
                continue
            if stage_name and rec.get("stage") not in {None, stage_name}:
                continue
            filtered.append(rec)
            if len(filtered) >= size:
                break
        parsed: list[dict[str, object]] = [LearnedStageDatasetRouter._parse_stage_record(stage_name, rec, idx) for idx, rec in enumerate(filtered)]
        if skipped_non_dict and parsed:
            parsed[0].setdefault("_ingestion_warnings", []).append(
                {
                    "source": "dataset_router",
                    "issue": f"non_dict_records_skipped={skipped_non_dict}",
                    "severity": "warning",
                    "field_path": "records",
                }
            )
        return parsed

    @staticmethod
    def _parse_stage_record(stage_name: str, record: dict[str, object], index: int) -> dict[str, object]:
        warnings: list[dict[str, object]] = []
        if stage_name == "text_encoder":
            text = str(record.get("text", record.get("prompt", f"micro adjust #{index}")))
            actions = LearnedStageDatasetRouter._coerce_action_steps(record.get("actions"), fallback_token="micro_adjust")
            return {"text": text, "actions": actions, "metadata": record.get("metadata", {}), "_ingestion_warnings": warnings}
        if stage_name == "dynamics_transition":
            before = LearnedStageDatasetRouter._coerce_graph(record, "graph_before", frame_index=index, track=f"dataset_{index}", warnings=warnings)
            graph_after = None
            if isinstance(record.get("graph_after"), dict):
                graph_after = LearnedStageDatasetRouter._coerce_graph(record, "graph_after", frame_index=index + 1, track=f"dataset_{index}_after", warnings=warnings)
            else:
                LearnedStageDatasetRouter._warn(warnings, "graph_after_missing_fallback_to_inference_path", field_path="graph_after")
            text_tokens = [a.type for a in LearnedStageDatasetRouter._coerce_action_steps(record.get("actions"), fallback_token="micro_adjust")]
            delta = LearnedStageDatasetRouter._coerce_delta(record, before=before, action_tokens=text_tokens, phase=str(record.get("transition_phase", "motion")), warnings=warnings)
            return {"graph_before": before, "ground_truth_graph_after": graph_after, "text_tokens": text_tokens, "delta": delta, "metadata": record.get("metadata", {}), "_ingestion_warnings": warnings}
        if stage_name == "patch_synthesis":
            graph = LearnedStageDatasetRouter._coerce_graph(record, "graph", frame_index=index, track=f"dataset_{index}", warnings=warnings)
            region_data = record.get("region", {})
            region_id = f"{graph.persons[0].person_id if graph.persons else 'scene'}:{record.get('region_name', 'torso')}"
            region = LearnedStageDatasetRouter._coerce_region(region_id, region_data, reason=str(record.get("reason", "motion")), warnings=warnings, scope="patch.region")
            frame = record.get("frame")
            if not isinstance(frame, list):
                LearnedStageDatasetRouter._warn(warnings, "patch.frame_invalid_fallback", field_path="frame")
                base = 0.2 + 0.03 * index
                frame = [[[base, base, base] for _ in range(32)] for _ in range(32)]
            return {
                "region": region,
                "graph": graph,
                "frame": frame,
                "hidden_state": record.get("hidden_state", {}),
                "retrieval_summary": record.get("retrieval_summary", {"source": "dataset_manifest"}),
                "target_selected_strategy": record.get("expected_selected_strategy"),
                "target_synthesis_mode": record.get("expected_synthesis_mode"),
                "target_hidden_lifecycle": record.get("hidden_lifecycle_target"),
                "target_retrieval_richness": record.get("retrieval_richness_target"),
                "target_region_metadata": record.get("expected_region_metadata"),
                "metadata": record.get("metadata", {}),
                "_ingestion_warnings": warnings,
            }
        if stage_name == "temporal_refinement":
            graph = LearnedStageDatasetRouter._coerce_graph(record, "graph", frame_index=index, track=f"dataset_{index}", warnings=warnings)
            regions_raw = record.get("regions")
            if isinstance(regions_raw, list):
                regions = [LearnedStageDatasetRouter._coerce_region(f"{graph.persons[0].person_id if graph.persons else 'scene'}:region_{i}", rr, reason="temporal_drift", warnings=warnings, scope=f"temporal.region_{i}") for i, rr in enumerate(regions_raw) if isinstance(rr, dict)]
            else:
                regions = []
            if not regions:
                LearnedStageDatasetRouter._warn(warnings, "temporal.regions_missing_fallback_default_region", field_path="regions")
                regions = [RegionRef(region_id=f"{graph.persons[0].person_id if graph.persons else 'scene'}:region_0", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="temporal_drift")]
            frame_prev = record.get("frame_prev") if isinstance(record.get("frame_prev"), list) else [[[0.2, 0.2, 0.2] for _ in range(32)] for _ in range(32)]
            frame_cur = record.get("frame_cur") if isinstance(record.get("frame_cur"), list) else [[[0.23, 0.2, 0.2] for _ in range(32)] for _ in range(32)]
            if not isinstance(record.get("frame_prev"), list):
                LearnedStageDatasetRouter._warn(warnings, "temporal.frame_prev_invalid_fallback", field_path="frame_prev")
            if not isinstance(record.get("frame_cur"), list):
                LearnedStageDatasetRouter._warn(warnings, "temporal.frame_cur_invalid_fallback", field_path="frame_cur")
            return {
                "graph": graph,
                "frame_prev": frame_prev,
                "frame_cur": frame_cur,
                "regions": regions,
                "drift": float(record.get("drift", 0.04)),
                "temporal_profile": str(record.get("temporal_profile", "dataset")),
                "target_temporal_profile": record.get("expected_temporal_profile"),
                "target_drift_regime": record.get("expected_drift_regime"),
                "target_region_consistency": record.get("region_consistency_target_scores"),
                "target_multi_roi_sync": record.get("multi_roi_sync_hints"),
                "_ingestion_warnings": warnings,
            }
        return record

    @staticmethod
    def build(stage_name: str, size: int = 3, dataset_path: str = "") -> list[dict[str, object]]:
        if dataset_path:
            loaded = LearnedStageDatasetRouter._from_dataset_path(dataset_path, stage_name, size)
            if loaded:
                return loaded
        if stage_name == "text_encoder":
            return [LearnedStageDatasetRouter._synthetic_text(i) for i in range(size)]
        if stage_name == "dynamics_transition":
            return [LearnedStageDatasetRouter._synthetic_dynamics(i) for i in range(size)]
        if stage_name == "patch_synthesis":
            return [LearnedStageDatasetRouter._synthetic_patch(i) for i in range(size)]
        if stage_name == "temporal_refinement":
            return [LearnedStageDatasetRouter._synthetic_temporal(i) for i in range(size)]
        raise ValueError(f"Unknown learned stage {stage_name}")


class _BaseStageRunner:
    def __init__(self, backend: str = "baseline", backend_config: BackendConfig | None = None, backends: BackendBundle | None = None) -> None:
        self.backend = backend
        self.backends = backends or LearnedBackendFactory(backend_config or BackendConfig()).build()

    def _write_checkpoint(self, config: StageScaffoldConfig, payload: dict[str, object]) -> str:
        out = Path(config.checkpoint_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return str(out)


def _validate_contract_fields(payload: dict[str, object], required_fields: list[str]) -> list[str]:
    return [name for name in required_fields if name not in payload]


def _build_parity_result(**kwargs: object) -> dict[str, object]:
    from learned.parity import build_parity_result

    return build_parity_result(**kwargs)


def _empty_structured_parity() -> dict[str, list[str]]:
    from learned.parity import empty_structured_parity

    return empty_structured_parity()


def _build_checkpoint_payload(
    *,
    stage_name: str,
    backend_name: str,
    backend_config: dict[str, str],
    contract_version: str,
    dataset_source: str,
    mean_score: float,
    samples_processed: int,
    expected_inputs: list[str],
    expected_outputs: list[str],
    last_contract: dict[str, object],
    eval_summary: dict[str, object],
    parity_summary: dict[str, object],
    warnings_or_fallbacks: list[dict[str, object]],
    stage_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = {
        "stage_name": stage_name,
        "backend_name": backend_name,
        "resolved_backend_config": backend_config,
        "schema_version": "learned_ready.v2",
        "contract_version": contract_version,
        "dataset_source": dataset_source,
        "train_metrics": {"progress": 1.0, "score_mean": mean_score},
        "val_metrics": {f"{stage_name}_mean": mean_score},
        "samples_processed": samples_processed,
        "expected_inputs": expected_inputs,
        "expected_outputs": expected_outputs,
        "contract_payload_shape": {"keys": sorted(last_contract.keys())},
        "eval_summary": eval_summary,
        "parity_summary": parity_summary,
        "warnings_or_fallbacks": warnings_or_fallbacks,
    }
    if stage_fields:
        payload.update(stage_fields)
    return payload


class TextEncoderStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("text_encoder", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        text_parity_log: list[dict[str, object]] = []
        ingestion_warnings: list[dict[str, object]] = []
        for sample in samples:
            encoded = self.backends.text_encoder.encode(str(sample["text"]))
            actions = sample.get("actions") or [ActionStep(type="micro_adjust", priority=1)]
            encoding_payload = asdict(encoded) if is_dataclass(encoded) else vars(encoded)
            contract = build_text_action_state_contract(str(sample["text"]), actions, encoding_payload)
            last_contract = contract
            eval_payload = build_text_eval_payload(contract)
            scores.append(text_action_alignment_eval(eval_payload).metrics["alignment_score"])
            parity = _build_parity_result(
                contract=contract,
                required_fields=["text", "parsed_actions", "action_embedding", "target_entities", "target_objects", "temporal_decomposition", "constraints"],
                stage="text",
                request={"text": sample["text"], "actions": actions},
                output=encoded,
            )
            text_parity_log.append(parity)
            ingestion_warnings.extend([w for w in sample.get("_ingestion_warnings", []) if isinstance(w, dict)])
        mean_score = (sum(scores) / len(scores)) if scores else 0.0
        ckpt = self._write_checkpoint(
            config,
            _build_checkpoint_payload(
                stage_name="text_encoder",
                backend_name=self.backends.backend_names.get("text_encoder", self.backend),
                backend_config=self.backends.backend_names,
                contract_version="text_action_state.v1",
                dataset_source=config.dataset_path or "synthetic_router",
                mean_score=mean_score,
                samples_processed=len(samples),
                expected_inputs=["text", "scene_graph(optional)", "action_plan(optional)"],
                expected_outputs=["action_embedding", "structured_action_tokens", "alignment"],
                last_contract=last_contract,
                eval_summary={"text_alignment_score": mean_score, "text_parity": text_parity_log[-1] if text_parity_log else {}},
                parity_summary=text_parity_log[-1] if text_parity_log else _empty_structured_parity(),
                warnings_or_fallbacks=ingestion_warnings,
                stage_fields={"val_metrics": {"alignment_score_mean": mean_score}},
            ),
        )
        mean = mean_score
        return StageScaffoldResult(
            stage_name="text_encoder",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["text", "scene_graph(optional)", "action_plan(optional)"],
            expected_outputs=["action_embedding", "structured_action_tokens", "alignment"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract, "parity": text_parity_log, "ingestion_warnings": ingestion_warnings},
            samples_processed=len(samples),
            ingestion_warnings=ingestion_warnings,
        )


class DynamicsTransitionStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("dynamics_transition", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        parity_log: list[dict[str, object]] = []
        warnings_or_fallbacks: list[dict[str, object]] = []
        for idx, sample in enumerate(samples):
            before = sample["graph_before"]
            memory = mm.initialize(before)
            text = self.backends.text_encoder.encode(" ".join(sample.get("text_tokens", ["micro_adjust"])))
            graph_enc = self.backends.graph_encoder.encode(before)
            entity_id = before.persons[0].person_id if before.persons else "scene"
            request = DynamicsTransitionRequest(
                graph_state=before,
                memory_summary={},
                memory_channels={},
                text_action_summary=text,
                graph_encoding=graph_enc,
                identity_embeddings={entity_id: [0.05 * (idx + 1)] * 8},
                step_context={"step_index": idx + 1, "memory": memory, "ground_truth_graph_after": sample.get("ground_truth_graph_after")},
            )
            out = self.backends.dynamics_backend.predict_transition(request)
            predicted_after = out.metadata.get("predicted_graph_after") if isinstance(out.metadata, dict) else None
            if not predicted_after:
                from copy import deepcopy
                predicted_after = apply_delta(deepcopy(before), out.delta)
                warnings_or_fallbacks.append({"source": "dynamics_runner", "issue": "graph_after_fallback_to_apply_delta", "severity": "warning", "field_path": f"sample_{idx}.graph_after"})
            ground_truth_after = sample.get("ground_truth_graph_after") if isinstance(sample.get("ground_truth_graph_after"), SceneGraph) else predicted_after
            contract = build_graph_transition_contract(
                before,
                ground_truth_after,
                out.delta,
                {
                    "step_index": idx,
                    "diagnostics": out.diagnostics,
                    "predicted_graph_after": _serialize_graph(predicted_after),
                    "ground_truth_graph_after": _serialize_graph(ground_truth_after),
                    "ground_truth_source": "dataset_manifest" if isinstance(sample.get("ground_truth_graph_after"), SceneGraph) else "apply_delta_fallback",
                    "supervision_mode": "supervision" if isinstance(sample.get("ground_truth_graph_after"), SceneGraph) else "inference",
                    "has_ground_truth_targets": isinstance(sample.get("ground_truth_graph_after"), SceneGraph),
                },
            )
            last_contract = contract
            eval_payload = build_graph_eval_payload(contract)
            scores.append(graph_transition_eval(eval_payload).metrics["transition_correctness"])
            parity_entry = _build_parity_result(
                contract=contract,
                required_fields=["graph_before", "graph_after", "delta_contract", "transition_context"],
                stage="dynamics",
                request=request,
                output=out,
            )
            for severity in ("errors", "warnings", "traces"):
                warnings_or_fallbacks.extend([{"source": "parity", "issue": issue, "severity": severity, "field_path": f"sample_{idx}"} for issue in parity_entry.get(severity, [])])
            parity_log.append(parity_entry)
            warnings_or_fallbacks.extend([dict(w, source=f"sample_{idx}:{w.get('source', 'ingestion')}") for w in sample.get("_ingestion_warnings", []) if isinstance(w, dict)])
        ckpt = self._write_checkpoint(
            config,
            _build_checkpoint_payload(
                stage_name="dynamics_transition",
                backend_name=self.backends.backend_names.get("dynamics_backend", self.backend),
                backend_config=self.backends.backend_names,
                contract_version="graph_transition.v2",
                dataset_source=config.dataset_path or "synthetic_router",
                mean_score=(sum(scores) / len(scores)) if scores else 0.0,
                samples_processed=len(samples),
                expected_inputs=["graph_state", "memory_summary", "text_action_summary", "step_context"],
                expected_outputs=["graph_delta", "confidence", "transition_metadata"],
                last_contract=last_contract,
                eval_summary={"transition_correctness": (sum(scores) / len(scores)) if scores else 0.0, "graph_after_target": "ground_truth_graph_after"},
                parity_summary=parity_log[-1] if parity_log else _empty_structured_parity(),
                warnings_or_fallbacks=warnings_or_fallbacks,
                stage_fields={"val_metrics": {"transition_correctness_mean": (sum(scores) / len(scores)) if scores else 0.0}},
            ),
        )
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="dynamics_transition",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["graph_state", "memory_summary", "text_action_summary", "step_context"],
            expected_outputs=["graph_delta", "confidence", "transition_metadata"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract, "parity": parity_log, "ingestion_warnings": warnings_or_fallbacks},
            samples_processed=len(samples),
            ingestion_warnings=warnings_or_fallbacks,
        )


class PatchSynthesisStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("patch_synthesis", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        parity_log: list[dict[str, list[str]]] = []
        warnings_or_fallbacks: list[dict[str, object]] = []
        supervision_mismatches: list[dict[str, object]] = []
        for idx, sample in enumerate(samples):
            memory = mm.initialize(sample["graph"])
            graph_enc = self.backends.graph_encoder.encode(sample["graph"])
            transition_context = {
                "graph_delta": GraphDelta(affected_entities=["p1"], affected_regions=[sample["region"].region_id], region_transition_mode={sample["region"].region_id: "motion"}),
                "video_memory": memory,
                "supervision_mode": "supervision" if any(sample.get(k) is not None for k in ("target_selected_strategy", "target_synthesis_mode", "target_hidden_lifecycle", "target_retrieval_richness", "target_region_metadata")) else "inference",
                "has_ground_truth_targets": any(sample.get(k) is not None for k in ("target_selected_strategy", "target_synthesis_mode", "target_hidden_lifecycle", "target_retrieval_richness", "target_region_metadata")),
                "expected_selected_strategy": sample.get("target_selected_strategy"),
                "expected_synthesis_mode": sample.get("target_synthesis_mode"),
                "hidden_lifecycle_target": sample.get("target_hidden_lifecycle"),
                "retrieval_richness_target": sample.get("target_retrieval_richness"),
                "expected_region_metadata": sample.get("target_region_metadata"),
            }
            req = PatchSynthesisRequest(
                region=sample["region"],
                scene_state=sample["graph"],
                memory_summary={},
                transition_context=transition_context,
                retrieval_summary=sample.get("retrieval_summary", {"stage": "patch", "case": idx}),
                current_frame=sample["frame"],
                memory_channels={"identity": {"requested": True}, "garments": {}, "hidden_regions": sample.get("hidden_state", {})},
                graph_encoding=graph_enc,
                identity_embedding=[0.1 + 0.01 * idx] * 8,
            )
            out = self.backends.patch_backend.synthesize_patch(req)
            contract = build_patch_synthesis_contract(sample["frame"], out.rgb_patch, sample["region"], "baseline", str(out.execution_trace.get("selected_render_strategy", "unknown")), sample.get("hidden_state", {}), str(out.execution_trace.get("synthesis_mode", "deterministic")), req.transition_context)
            last_contract = contract
            eval_payload = build_patch_eval_payload(contract)
            patch_eval = patch_synthesis_eval(eval_payload)
            hidden_eval = hidden_region_reconstruction_eval(build_hidden_reconstruction_payload(contract))
            scores.append((patch_eval.metrics["patch_quality"] + hidden_eval.metrics["reconstruction_quality"]) / 2.0)
            actual_strategy = str(out.execution_trace.get("selected_render_strategy", "unknown"))
            actual_mode = str(out.execution_trace.get("synthesis_mode", "deterministic"))
            if sample.get("target_selected_strategy") and sample.get("target_selected_strategy") != actual_strategy:
                supervision_mismatches.append({"field": "expected_selected_strategy", "expected": sample.get("target_selected_strategy"), "actual": actual_strategy, "sample_index": idx})
            if sample.get("target_synthesis_mode") and sample.get("target_synthesis_mode") != actual_mode:
                supervision_mismatches.append({"field": "expected_synthesis_mode", "expected": sample.get("target_synthesis_mode"), "actual": actual_mode, "sample_index": idx})
            if sample.get("target_hidden_lifecycle") and sample.get("target_hidden_lifecycle") != sample.get("hidden_state", {}).get("lifecycle"):
                supervision_mismatches.append({"field": "hidden_lifecycle_target", "expected": sample.get("target_hidden_lifecycle"), "actual": sample.get("hidden_state", {}).get("lifecycle"), "sample_index": idx})
            if sample.get("target_retrieval_richness") and sample.get("target_retrieval_richness") != sample.get("hidden_state", {}).get("retrieval_profile"):
                supervision_mismatches.append({"field": "retrieval_richness_target", "expected": sample.get("target_retrieval_richness"), "actual": sample.get("hidden_state", {}).get("retrieval_profile"), "sample_index": idx})
            parity = _build_parity_result(
                contract=contract,
                required_fields=["roi_before", "roi_after", "region_metadata", "selected_strategy", "transition_context"],
                stage="patch",
                request=req,
                output=out,
            )
            parity_log.append(parity)
            for severity in ("errors", "warnings", "traces"):
                warnings_or_fallbacks.extend([{"source": "parity", "issue": issue, "severity": severity, "field_path": f"sample_{idx}"} for issue in parity.get(severity, [])])
            warnings_or_fallbacks.extend([dict(w, source=f"sample_{idx}:{w.get('source', 'ingestion')}") for w in sample.get("_ingestion_warnings", []) if isinstance(w, dict)])
        ckpt = self._write_checkpoint(
            config,
            _build_checkpoint_payload(
                stage_name="patch_synthesis",
                backend_name=self.backends.backend_names.get("patch_backend", self.backend),
                backend_config=self.backends.backend_names,
                contract_version="patch_synthesis.v2",
                dataset_source=config.dataset_path or "synthetic_router",
                mean_score=(sum(scores) / len(scores)) if scores else 0.0,
                samples_processed=len(samples),
                expected_inputs=["region", "scene_state", "memory_summary", "transition_context", "retrieval_summary"],
                expected_outputs=["rgb_patch", "confidence", "uncertainty_map"],
                last_contract=last_contract,
                eval_summary={"composite_patch_hidden_score": (sum(scores) / len(scores)) if scores else 0.0, "supervision_mismatches": supervision_mismatches},
                parity_summary=parity_log[-1] if parity_log else _empty_structured_parity(),
                warnings_or_fallbacks=warnings_or_fallbacks,
                stage_fields={"val_metrics": {"patch_quality_mean": (sum(scores) / len(scores)) if scores else 0.0}},
            ),
        )
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="patch_synthesis",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["region", "scene_state", "memory_summary", "transition_context", "retrieval_summary"],
            expected_outputs=["rgb_patch", "confidence", "uncertainty_map"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract, "hidden_reconstruction_payload": build_hidden_reconstruction_payload(last_contract), "parity": parity_log, "warnings_or_fallbacks": warnings_or_fallbacks},
            samples_processed=len(samples),
            ingestion_warnings=warnings_or_fallbacks,
        )


class TemporalRefinementStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("temporal_refinement", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        parity_log: list[dict[str, list[str]]] = []
        warnings_or_fallbacks: list[dict[str, object]] = []
        supervision_mismatches: list[dict[str, object]] = []
        for idx, sample in enumerate(samples):
            memory = mm.initialize(sample["graph"])
            has_targets = any(sample.get(k) is not None for k in ("target_temporal_profile", "target_drift_regime", "target_region_consistency", "target_multi_roi_sync"))
            req = TemporalRefinementRequest(
                previous_frame=sample["frame_prev"],
                current_composed_frame=sample["frame_cur"],
                changed_regions=sample["regions"],
                scene_state=sample["graph"],
                memory_state=memory,
                memory_channels={"identity": {}, "body_regions": {"roi_count": len(sample["regions"])}, "hidden_regions": {"drift": sample["drift"]}},
            )
            out = self.backends.temporal_backend.refine_temporal(req)
            contract = build_temporal_consistency_contract(
                sample["frame_prev"],
                sample["frame_cur"],
                out.refined_frame,
                sample["regions"],
                {"scores": out.region_consistency_scores, "temporal_drift": sample["drift"], "roi_count": len(sample["regions"])},
                {
                    "stage": "temporal",
                    "step_index": idx + 1,
                    "supervision_mode": "supervision" if has_targets else "inference",
                    "has_ground_truth_targets": has_targets,
                    "expected_temporal_profile": sample.get("target_temporal_profile"),
                    "expected_drift_regime": sample.get("target_drift_regime"),
                    "region_consistency_target_scores": sample.get("target_region_consistency"),
                    "multi_roi_sync_hints": sample.get("target_multi_roi_sync"),
                },
                {"channels": list(req.memory_channels.keys())},
            )
            last_contract = contract
            eval_payload = build_temporal_eval_payload(contract)
            scores.append(temporal_consistency_eval(eval_payload).metrics["temporal_consistency"])
            actual_profile = sample.get("temporal_profile")
            actual_drift_regime = "high_drift" if float(sample.get("drift", 0.0)) > 0.08 else "low_drift"
            if sample.get("target_temporal_profile") and sample.get("target_temporal_profile") != actual_profile:
                supervision_mismatches.append({"field": "expected_temporal_profile", "expected": sample.get("target_temporal_profile"), "actual": actual_profile, "sample_index": idx})
            if sample.get("target_drift_regime") and sample.get("target_drift_regime") != actual_drift_regime:
                supervision_mismatches.append({"field": "expected_drift_regime", "expected": sample.get("target_drift_regime"), "actual": actual_drift_regime, "sample_index": idx})
            if sample.get("target_region_consistency"):
                supervision_mismatches.append({"field": "region_consistency_target_scores", "expected": sample.get("target_region_consistency"), "actual": out.region_consistency_scores, "sample_index": idx})
            parity = _build_parity_result(
                contract=contract,
                required_fields=["previous_frame", "composed_frame", "target_frame", "changed_regions", "scene_transition_context"],
                stage="temporal",
                request=req,
                output=out,
                changed_regions_count=len(sample["regions"]),
            )
            parity_log.append(parity)
            for severity in ("errors", "warnings", "traces"):
                warnings_or_fallbacks.extend([{"source": "parity", "issue": issue, "severity": severity, "field_path": f"sample_{idx}"} for issue in parity.get(severity, [])])
            warnings_or_fallbacks.extend([dict(w, source=f"sample_{idx}:{w.get('source', 'ingestion')}") for w in sample.get("_ingestion_warnings", []) if isinstance(w, dict)])
        ckpt = self._write_checkpoint(
            config,
            _build_checkpoint_payload(
                stage_name="temporal_refinement",
                backend_name=self.backends.backend_names.get("temporal_backend", self.backend),
                backend_config=self.backends.backend_names,
                contract_version="temporal_consistency.v2",
                dataset_source=config.dataset_path or "synthetic_router",
                mean_score=(sum(scores) / len(scores)) if scores else 0.0,
                samples_processed=len(samples),
                expected_inputs=["previous_frame", "current_composed_frame", "changed_regions", "scene_state", "memory_state"],
                expected_outputs=["refined_frame", "region_consistency_scores"],
                last_contract=last_contract,
                eval_summary={"temporal_consistency": (sum(scores) / len(scores)) if scores else 0.0, "supervision_mismatches": supervision_mismatches},
                parity_summary=parity_log[-1] if parity_log else _empty_structured_parity(),
                warnings_or_fallbacks=warnings_or_fallbacks,
                stage_fields={"val_metrics": {"temporal_consistency_mean": (sum(scores) / len(scores)) if scores else 0.0}},
            ),
        )
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="temporal_refinement",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["previous_frame", "current_composed_frame", "changed_regions", "scene_state", "memory_state"],
            expected_outputs=["refined_frame", "region_consistency_scores"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract, "parity": parity_log, "warnings_or_fallbacks": warnings_or_fallbacks},
            samples_processed=len(samples),
            ingestion_warnings=warnings_or_fallbacks,
        )


def build_stage_runner(stage_name: str, backend: str = "baseline", backend_config: BackendConfig | None = None, backends: BackendBundle | None = None):
    mapping = {
        "text_encoder": TextEncoderStageRunner,
        "dynamics_transition": DynamicsTransitionStageRunner,
        "patch_synthesis": PatchSynthesisStageRunner,
        "temporal_refinement": TemporalRefinementStageRunner,
    }
    if stage_name not in mapping:
        known = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown learned stage {stage_name}. Known: {known}")
    return mapping[stage_name](backend=backend, backend_config=backend_config, backends=backends)
