from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    ingestion_warnings: list[str] = field(default_factory=list)


class LearnedStageDatasetRouter:
    @staticmethod
    def _warn(warnings: list[str], issue: str) -> None:
        warnings.append(issue)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _as_float(value: object, fallback: float, *, warnings: list[str], issue: str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        LearnedStageDatasetRouter._warn(warnings, issue)
        return fallback

    @staticmethod
    def _sanitize_label(value: object, fallback: str, *, warnings: list[str], issue: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace(" ", "_")
        LearnedStageDatasetRouter._warn(warnings, issue)
        return fallback

    @staticmethod
    def _coerce_bbox(data: object, fallback: BBox | None = None, *, warnings: list[str] | None = None, scope: str = "bbox") -> BBox:
        local_warnings = warnings if warnings is not None else []
        if isinstance(data, dict):
            x = LearnedStageDatasetRouter._as_float(data.get("x", 0.1), 0.1, warnings=local_warnings, issue=f"{scope}.x_invalid_fallback")
            y = LearnedStageDatasetRouter._as_float(data.get("y", 0.1), 0.1, warnings=local_warnings, issue=f"{scope}.y_invalid_fallback")
            w = LearnedStageDatasetRouter._as_float(data.get("w", 0.6), 0.6, warnings=local_warnings, issue=f"{scope}.w_invalid_fallback")
            h = LearnedStageDatasetRouter._as_float(data.get("h", 0.8), 0.8, warnings=local_warnings, issue=f"{scope}.h_invalid_fallback")
            clamped = BBox(
                LearnedStageDatasetRouter._clamp(x, 0.0, 1.0),
                LearnedStageDatasetRouter._clamp(y, 0.0, 1.0),
                LearnedStageDatasetRouter._clamp(w, 0.0, 1.0),
                LearnedStageDatasetRouter._clamp(h, 0.0, 1.0),
            )
            if (x, y, w, h) != (clamped.x, clamped.y, clamped.w, clamped.h):
                LearnedStageDatasetRouter._warn(local_warnings, f"{scope}.clamped")
            return clamped
        LearnedStageDatasetRouter._warn(local_warnings, f"{scope}.missing_fallback")
        return fallback or BBox(0.1, 0.1, 0.6, 0.8)

    @staticmethod
    def _coerce_region(region_id: str, payload: object, reason: str = "motion", *, warnings: list[str] | None = None, scope: str = "region") -> RegionRef:
        local_warnings = warnings if warnings is not None else []
        bbox = LearnedStageDatasetRouter._coerce_bbox(payload if isinstance(payload, dict) else {}, warnings=local_warnings, scope=f"{scope}.bbox")
        resolved_reason = reason
        if isinstance(payload, dict):
            resolved_reason = LearnedStageDatasetRouter._sanitize_label(payload.get("reason", reason), reason, warnings=local_warnings, issue=f"{scope}.reason_sanitized")
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
                        kx = LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(kp.get("x", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.x_invalid_fallback"), 0.0, 1.0)
                        ky = LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(kp.get("y", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.y_invalid_fallback"), 0.0, 1.0)
                        conf_raw = LearnedStageDatasetRouter._as_float(kp.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.confidence_invalid_fallback")
                        conf = LearnedStageDatasetRouter._clamp(conf_raw, 0.0, 1.0)
                        if conf != conf_raw:
                            LearnedStageDatasetRouter._warn(local_warnings, f"{key}.{pid}.keypoint_{bp_idx}_{k_idx}.confidence_clamped")
                        keypoints.append(Keypoint(name=str(kp.get("name", f"kp_{bp_idx}_{k_idx}")), x=kx, y=ky, confidence=conf))
                    body_parts.append(
                        BodyPartNode(
                            part_id=str(bp.get("part_id", f"{pid}_part_{bp_idx}")),
                            part_type=LearnedStageDatasetRouter._sanitize_label(bp.get("part_type", "unknown"), "unknown", warnings=local_warnings, issue=f"{key}.{pid}.body_part_{bp_idx}.part_type_sanitized"),
                            keypoints=keypoints,
                            mask_ref=bp.get("mask_ref"),
                            visibility=LearnedStageDatasetRouter._sanitize_label(bp.get("visibility", "unknown"), "unknown", warnings=local_warnings, issue=f"{key}.{pid}.body_part_{bp_idx}.visibility_sanitized"),
                            occluded_by=[str(v) for v in bp.get("occluded_by", []) if isinstance(v, str)],
                            depth_order=LearnedStageDatasetRouter._as_float(bp.get("depth_order", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.body_part_{bp_idx}.depth_order_invalid_fallback"),
                            canonical_slot=str(bp.get("canonical_slot", "")),
                            confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(bp.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.body_part_{bp_idx}.confidence_invalid_fallback"), 0.0, 1.0),
                            source=str(bp.get("source", "dataset")),
                            frame_index=int(bp.get("frame_index", raw.get("frame_index", frame_index))),
                            timestamp=float(bp["timestamp"]) if isinstance(bp.get("timestamp"), (int, float)) else None,
                        )
                    )
                garments = [
                    GarmentNode(
                        garment_id=str(g.get("garment_id", f"{pid}_garment_{g_idx}")),
                        garment_type=LearnedStageDatasetRouter._sanitize_label(g.get("garment_type", g.get("type", "unknown")), "unknown", warnings=local_warnings, issue=f"{key}.{pid}.garment_{g_idx}.garment_type_sanitized"),
                        mask_ref=g.get("mask_ref"),
                        attachment_targets=[str(v) for v in g.get("attachment_targets", []) if isinstance(v, str)],
                        coverage_targets=[str(v) for v in g.get("coverage_targets", []) if isinstance(v, str)],
                        garment_state=LearnedStageDatasetRouter._sanitize_label(g.get("garment_state", g.get("state", "worn")), "worn", warnings=local_warnings, issue=f"{key}.{pid}.garment_{g_idx}.garment_state_sanitized"),
                        visibility=LearnedStageDatasetRouter._sanitize_label(g.get("visibility", "unknown"), "unknown", warnings=local_warnings, issue=f"{key}.{pid}.garment_{g_idx}.visibility_sanitized"),
                        appearance_ref=g.get("appearance_ref"),
                        confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(g.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.garment_{g_idx}.confidence_invalid_fallback"), 0.0, 1.0),
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
                        confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(p.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.{pid}.confidence_invalid_fallback"), 0.0, 1.0),
                        source=str(p.get("source", "dataset")),
                        frame_index=int(p.get("frame_index", raw.get("frame_index", frame_index))),
                        timestamp=float(p["timestamp"]) if isinstance(p.get("timestamp"), (int, float)) else None,
                    )
                )
            objects = [
                SceneObjectNode(
                    object_id=str(o.get("object_id", f"obj_{track}_{idx}")),
                    object_type=LearnedStageDatasetRouter._sanitize_label(o.get("object_type", "unknown"), "unknown", warnings=local_warnings, issue=f"{key}.object_{idx}.type_sanitized"),
                    bbox=LearnedStageDatasetRouter._coerce_bbox(o.get("bbox"), warnings=local_warnings, scope=f"{key}.object_{idx}.bbox"),
                    mask_ref=o.get("mask_ref"),
                    confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(o.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.object_{idx}.confidence_invalid_fallback"), 0.0, 1.0),
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
                    relation=LearnedStageDatasetRouter._sanitize_label(r.get("relation", "near"), "near", warnings=local_warnings, issue=f"{key}.relation.relation_sanitized"),
                    target=str(r.get("target", "")),
                    confidence=LearnedStageDatasetRouter._clamp(LearnedStageDatasetRouter._as_float(r.get("confidence", 0.0), 0.0, warnings=local_warnings, issue=f"{key}.relation.confidence_invalid_fallback"), 0.0, 1.0),
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
                LearnedStageDatasetRouter._warn(local_warnings, f"{key}.global_context.frame_size_invalid_fallback")
            fps_raw = gc_raw.get("fps", 16)
            fps = int(fps_raw) if isinstance(fps_raw, (int, float)) and int(fps_raw) > 0 else 16
            if fps == 16 and fps_raw != 16:
                LearnedStageDatasetRouter._warn(local_warnings, f"{key}.global_context.fps_invalid_fallback")
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
        LearnedStageDatasetRouter._warn(local_warnings, f"{key}.missing_graph_fallback_base_graph")
        return LearnedStageDatasetRouter._base_graph(frame_index, track)

    @staticmethod
    def _coerce_delta(record: dict[str, object], before: SceneGraph, action_tokens: list[str], phase: str = "motion") -> GraphDelta:
        payload = record.get("delta", {})
        if isinstance(payload, dict):
            revealed = [LearnedStageDatasetRouter._coerce_region(region_id=f"{before.persons[0].person_id}:revealed_{idx}", payload=item, reason="reveal") for idx, item in enumerate(payload.get("newly_revealed_regions", [])) if isinstance(item, dict)] if before.persons else []
            occluded = [LearnedStageDatasetRouter._coerce_region(region_id=f"{before.persons[0].person_id}:occluded_{idx}", payload=item, reason="occlude") for idx, item in enumerate(payload.get("newly_occluded_regions", [])) if isinstance(item, dict)] if before.persons else []
            return GraphDelta(
                pose_deltas={str(k): float(v) for k, v in payload.get("pose_deltas", {}).items()} if isinstance(payload.get("pose_deltas"), dict) else {},
                interaction_deltas={str(k): float(v) for k, v in payload.get("interaction_deltas", {}).items()} if isinstance(payload.get("interaction_deltas"), dict) else {},
                semantic_reasons=[str(v) for v in payload.get("semantic_reasons", action_tokens)],
                affected_entities=[str(v) for v in payload.get("affected_entities", [before.persons[0].person_id] if before.persons else [])],
                affected_regions=[str(v) for v in payload.get("affected_regions", [])],
                region_transition_mode={str(k): str(v) for k, v in payload.get("region_transition_mode", {}).items()} if isinstance(payload.get("region_transition_mode"), dict) else {},
                predicted_visibility_changes={str(k): str(v) for k, v in payload.get("predicted_visibility_changes", {}).items()} if isinstance(payload.get("predicted_visibility_changes"), dict) else {},
                state_before={str(k): str(v) for k, v in payload.get("state_before", {}).items()} if isinstance(payload.get("state_before"), dict) else {},
                state_after={str(k): str(v) for k, v in payload.get("state_after", {}).items()} if isinstance(payload.get("state_after"), dict) else {},
                transition_phase=str(payload.get("transition_phase", phase)),
                newly_revealed_regions=revealed,
                newly_occluded_regions=occluded,
            )
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
            parsed[0].setdefault("_ingestion_warnings", []).append(f"non_dict_records_skipped={skipped_non_dict}")
        return parsed

    @staticmethod
    def _parse_stage_record(stage_name: str, record: dict[str, object], index: int) -> dict[str, object]:
        warnings: list[str] = []
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
                LearnedStageDatasetRouter._warn(warnings, "graph_after_missing_fallback_to_inference_path")
            text_tokens = [a.type for a in LearnedStageDatasetRouter._coerce_action_steps(record.get("actions"), fallback_token="micro_adjust")]
            delta = LearnedStageDatasetRouter._coerce_delta(record, before=before, action_tokens=text_tokens, phase=str(record.get("transition_phase", "motion")))
            return {"graph_before": before, "ground_truth_graph_after": graph_after, "text_tokens": text_tokens, "delta": delta, "metadata": record.get("metadata", {}), "_ingestion_warnings": warnings}
        if stage_name == "patch_synthesis":
            graph = LearnedStageDatasetRouter._coerce_graph(record, "graph", frame_index=index, track=f"dataset_{index}", warnings=warnings)
            region_data = record.get("region", {})
            region_id = f"{graph.persons[0].person_id if graph.persons else 'scene'}:{record.get('region_name', 'torso')}"
            region = LearnedStageDatasetRouter._coerce_region(region_id, region_data, reason=str(record.get("reason", "motion")), warnings=warnings, scope="patch.region")
            frame = record.get("frame")
            if not isinstance(frame, list):
                LearnedStageDatasetRouter._warn(warnings, "patch.frame_invalid_fallback")
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
                LearnedStageDatasetRouter._warn(warnings, "temporal.regions_missing_fallback_default_region")
                regions = [RegionRef(region_id=f"{graph.persons[0].person_id if graph.persons else 'scene'}:region_0", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="temporal_drift")]
            frame_prev = record.get("frame_prev") if isinstance(record.get("frame_prev"), list) else [[[0.2, 0.2, 0.2] for _ in range(32)] for _ in range(32)]
            frame_cur = record.get("frame_cur") if isinstance(record.get("frame_cur"), list) else [[[0.23, 0.2, 0.2] for _ in range(32)] for _ in range(32)]
            if not isinstance(record.get("frame_prev"), list):
                LearnedStageDatasetRouter._warn(warnings, "temporal.frame_prev_invalid_fallback")
            if not isinstance(record.get("frame_cur"), list):
                LearnedStageDatasetRouter._warn(warnings, "temporal.frame_cur_invalid_fallback")
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


def _text_semantic_checks(contract: dict[str, object]) -> list[str]:
    issues: list[str] = []
    parsed = contract.get("parsed_actions", [])
    embedding = contract.get("action_embedding", [])
    if not parsed:
        issues.append("structured_action_tokens_empty")
    if parsed and not embedding:
        issues.append("embedding_empty_for_non_empty_actions")
    if isinstance(parsed, list) and len(parsed) > 1 and not contract.get("temporal_decomposition"):
        issues.append("decomposition_hints_missing_for_multi_action_input")
    has_targets = isinstance(parsed, list) and any(isinstance(a, dict) and (a.get("target_entity") or a.get("target_object")) for a in parsed)
    if has_targets and not (contract.get("target_entities") or contract.get("target_objects")):
        issues.append("target_hints_missing_despite_action_targets")
    return issues


def _empty_parity_result() -> dict[str, list[str]]:
    return {"errors": [], "warnings": [], "traces": []}


class TextEncoderStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("text_encoder", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        text_parity_log: list[dict[str, object]] = []
        ingestion_warnings: list[str] = []
        for sample in samples:
            encoded = self.backends.text_encoder.encode(str(sample["text"]))
            actions = sample.get("actions") or [ActionStep(type="micro_adjust", priority=1)]
            contract = build_text_action_state_contract(str(sample["text"]), actions, vars(encoded))
            last_contract = contract
            eval_payload = build_text_eval_payload(contract)
            scores.append(text_action_alignment_eval(eval_payload).metrics["alignment_score"])
            missing = _validate_contract_fields(contract, ["text", "parsed_actions", "action_embedding", "target_entities", "target_objects", "temporal_decomposition", "constraints"])
            semantic_text = _text_semantic_checks(contract)
            parity = _empty_parity_result()
            parity["errors"].extend([f"missing_field:{m}" for m in missing])
            parity["warnings"].extend(semantic_text)
            text_parity_log.append(parity)
            ingestion_warnings.extend([str(w) for w in sample.get("_ingestion_warnings", []) if isinstance(w, str)])
        ckpt = self._write_checkpoint(
            config,
            {
                "stage_name": "text_encoder",
                "backend_name": self.backends.backend_names.get("text_encoder", self.backend),
                "resolved_backend_config": self.backends.backend_names,
                "schema_version": "learned_ready.v2",
                "contract_version": "text_action_state.v1",
                "dataset_source": config.dataset_path or "synthetic_router",
                "train_metrics": {"progress": 1.0, "score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "val_metrics": {"alignment_score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "samples_processed": len(samples),
                "expected_inputs": ["text", "scene_graph(optional)", "action_plan(optional)"],
                "expected_outputs": ["action_embedding", "structured_action_tokens", "alignment"],
                "contract_payload_shape": {"keys": sorted(last_contract.keys())},
                "eval_summary": {"text_alignment_score": (sum(scores) / len(scores)) if scores else 0.0, "text_parity": text_parity_log[-1] if text_parity_log else {}},
                "parity_summary": text_parity_log[-1] if text_parity_log else _empty_parity_result(),
                "warnings_or_fallbacks": ingestion_warnings,
            },
        )
        mean = sum(scores) / len(scores)
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
        warnings_or_fallbacks: list[str] = []
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
                warnings_or_fallbacks.append(f"sample_{idx}:graph_after_fallback_to_apply_delta")
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
            missing = _validate_contract_fields(contract, ["graph_before", "graph_after", "delta_contract", "transition_context"])
            from learned.parity import semantic_parity_checks

            semantic = semantic_parity_checks(stage="dynamics", contract=contract, request=request, output=out)
            parity_entry = _empty_parity_result()
            parity_entry["errors"].extend([f"missing_field:{m}" for m in missing])
            for severity in ("errors", "warnings", "traces"):
                parity_entry[severity].extend(semantic.get(severity, []))
                warnings_or_fallbacks.extend([f"sample_{idx}:{severity}:{issue}" for issue in semantic.get(severity, [])])
            parity_log.append(parity_entry)
            warnings_or_fallbacks.extend([f"sample_{idx}:ingestion:{w}" for w in sample.get("_ingestion_warnings", []) if isinstance(w, str)])
        ckpt = self._write_checkpoint(
            config,
            {
                "stage_name": "dynamics_transition",
                "backend_name": self.backends.backend_names.get("dynamics_backend", self.backend),
                "resolved_backend_config": self.backends.backend_names,
                "schema_version": "learned_ready.v2",
                "contract_version": "graph_transition.v2",
                "dataset_source": config.dataset_path or "synthetic_router",
                "train_metrics": {"progress": 1.0, "score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "val_metrics": {"transition_correctness_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "samples_processed": len(samples),
                "expected_inputs": ["graph_state", "memory_summary", "text_action_summary", "step_context"],
                "expected_outputs": ["graph_delta", "confidence", "transition_metadata"],
                "contract_payload_shape": {"keys": sorted(last_contract.keys())},
                "eval_summary": {"transition_correctness": (sum(scores) / len(scores)) if scores else 0.0, "graph_after_target": "ground_truth_graph_after"},
                "parity_summary": parity_log[-1] if parity_log else _empty_parity_result(),
                "warnings_or_fallbacks": warnings_or_fallbacks,
            },
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
        warnings_or_fallbacks: list[str] = []
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
            from learned.parity import semantic_parity_checks

            missing = _validate_contract_fields(contract, ["roi_before", "roi_after", "region_metadata", "selected_strategy", "transition_context"])
            semantic = semantic_parity_checks(stage="patch", contract=contract, request=req, output=out)
            parity = _empty_parity_result()
            parity["errors"].extend([f"missing_field:{m}" for m in missing])
            for severity in ("errors", "warnings", "traces"):
                parity[severity].extend(semantic.get(severity, []))
            parity_log.append(parity)
            warnings_or_fallbacks.extend([f"sample_{idx}:{severity}:{issue}" for severity in ("errors", "warnings", "traces") for issue in semantic.get(severity, [])])
            warnings_or_fallbacks.extend([f"sample_{idx}:ingestion:{w}" for w in sample.get("_ingestion_warnings", []) if isinstance(w, str)])
        ckpt = self._write_checkpoint(
            config,
            {
                "stage_name": "patch_synthesis",
                "backend_name": self.backends.backend_names.get("patch_backend", self.backend),
                "resolved_backend_config": self.backends.backend_names,
                "schema_version": "learned_ready.v2",
                "contract_version": "patch_synthesis.v2",
                "dataset_source": config.dataset_path or "synthetic_router",
                "train_metrics": {"progress": 1.0, "score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "val_metrics": {"patch_quality_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "samples_processed": len(samples),
                "expected_inputs": ["region", "scene_state", "memory_summary", "transition_context", "retrieval_summary"],
                "expected_outputs": ["rgb_patch", "confidence", "uncertainty_map"],
                "contract_payload_shape": {"keys": sorted(last_contract.keys())},
                "eval_summary": {"composite_patch_hidden_score": (sum(scores) / len(scores)) if scores else 0.0},
                "parity_summary": parity_log[-1] if parity_log else _empty_parity_result(),
                "warnings_or_fallbacks": warnings_or_fallbacks,
            },
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
        warnings_or_fallbacks: list[str] = []
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
            from learned.parity import semantic_parity_checks

            missing = _validate_contract_fields(contract, ["previous_frame", "composed_frame", "target_frame", "changed_regions", "scene_transition_context"])
            semantic = semantic_parity_checks(stage="temporal", contract=contract, request=req, output=out, changed_regions_count=len(sample["regions"]))
            parity = _empty_parity_result()
            parity["errors"].extend([f"missing_field:{m}" for m in missing])
            for severity in ("errors", "warnings", "traces"):
                parity[severity].extend(semantic.get(severity, []))
            parity_log.append(parity)
            warnings_or_fallbacks.extend([f"sample_{idx}:{severity}:{issue}" for severity in ("errors", "warnings", "traces") for issue in semantic.get(severity, [])])
            warnings_or_fallbacks.extend([f"sample_{idx}:ingestion:{w}" for w in sample.get("_ingestion_warnings", []) if isinstance(w, str)])
        ckpt = self._write_checkpoint(
            config,
            {
                "stage_name": "temporal_refinement",
                "backend_name": self.backends.backend_names.get("temporal_backend", self.backend),
                "resolved_backend_config": self.backends.backend_names,
                "schema_version": "learned_ready.v2",
                "contract_version": "temporal_consistency.v2",
                "dataset_source": config.dataset_path or "synthetic_router",
                "train_metrics": {"progress": 1.0, "score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "val_metrics": {"temporal_consistency_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "samples_processed": len(samples),
                "expected_inputs": ["previous_frame", "current_composed_frame", "changed_regions", "scene_state", "memory_state"],
                "expected_outputs": ["refined_frame", "region_consistency_scores"],
                "contract_payload_shape": {"keys": sorted(last_contract.keys())},
                "eval_summary": {"temporal_consistency": (sum(scores) / len(scores)) if scores else 0.0},
                "parity_summary": parity_log[-1] if parity_log else _empty_parity_result(),
                "warnings_or_fallbacks": warnings_or_fallbacks,
            },
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
