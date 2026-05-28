from __future__ import annotations

import copy
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from core.input_layer import InputAssetLayer
from core.schema import (
    ActionStep,
    BBox,
    BodyPartNode,
    ExpressionState,
    GarmentNode,
    GlobalSceneContext,
    GraphDelta,
    Keypoint,
    OrientationState,
    PersonNode,
    PoseState,
    RelationEdge,
    RegionRef,
    SceneGraph,
    SceneObjectNode,
)
from memory.video_memory import MemoryManager
from perception.pipeline import ObjectFacts, PerceptionOutput, PerceptionPipeline, PersonFacts
from representation.graph_builder import SceneGraphBuilder
from rendering.roi_renderer import ROISelector
from learned.interfaces import PatchSynthesisRequest
from rendering.target_provenance_policy import (
    ALLOWED_TARGET_SOURCES as POLICY_ALLOWED_TARGET_SOURCES,
    ALLOWED_TRAINING_TARGET_QUALITIES as POLICY_ALLOWED_TRAINING_TARGET_QUALITIES,
    classify_target_training_role,
    target_quality_warning,
    target_role_allowed_by_policy,
)

PRIMARY_DYNAMICS_FAMILIES = {"pose_transition", "garment_transition", "interaction_transition", "expression_transition"}
AUXILIARY_SEMANTIC_FAMILIES = {"visibility_transition"}


def _resolve_manifest_primary_family(record: dict[str, object], family_field: str) -> tuple[str, str, list[str], dict[str, object]]:
    manifest_family = str(record.get(family_field, "pose_transition")).strip().lower()
    primary_raw = record.get("primary_family")
    auxiliary_families: list[str] = []
    metadata: dict[str, object] = {"manifest_family": manifest_family}

    if manifest_family in PRIMARY_DYNAMICS_FAMILIES:
        metadata["primary_family"] = manifest_family
        return manifest_family, manifest_family, auxiliary_families, metadata

    if manifest_family in AUXILIARY_SEMANTIC_FAMILIES:
        if primary_raw is None or not str(primary_raw).strip():
            raise ValueError("visibility_transition_requires_primary_family")
        primary_family = str(primary_raw).strip().lower()
        if primary_family not in PRIMARY_DYNAMICS_FAMILIES:
            raise ValueError(f"primary_family must be supported primary dynamics family, got {primary_family}")
        auxiliary_families.append(manifest_family)
        metadata.update({"primary_family": primary_family, "auxiliary_families": auxiliary_families})
        return primary_family, manifest_family, auxiliary_families, metadata

    raise ValueError(f"{family_field} must be canonical primary or auxiliary semantic family, got {manifest_family}")


def _attach_auxiliary_family_metadata(delta: GraphDelta, auxiliary_families: list[str], metadata: dict[str, object]) -> None:
    for family in auxiliary_families:
        if family not in delta.semantic_reasons:
            delta.semantic_reasons.append(family)
    if metadata:
        delta.transition_diagnostics.update(metadata)


class TrainingSample(TypedDict, total=False):
    frames: list[list]
    graphs: list[SceneGraph]
    actions: list[ActionStep]
    deltas: list[GraphDelta]
    roi_pairs: list[tuple[list, list]]
    targets: list[GraphDelta]
    text: str
    text_alignment: dict[str, object]
    memory_records: list[dict[str, object]]
    source: str
    sanity_metrics: dict[str, float]
    delta_contract: dict[str, object]
    text_action_contract: dict[str, object]
    graph_transition_contract: dict[str, object]
    patch_synthesis_contract: dict[str, object]
    temporal_consistency_contract: dict[str, object]

    temporal_transition_features: list[float]
    temporal_transition_target: dict[str, object]

    human_state_transition_features: list[float]
    human_state_transition_target: dict[str, object]
    human_state_history: dict[str, object]

    renderer_batch_contract: dict[str, object]
    temporal_roi_window: dict[str, object]
    region_family: str

    target_profile: dict[str, object]
    phase_family: dict[str, object]
    reveal_occlusion_support_cues: dict[str, float]
    visibility_targets: list[float]

    region_id: str
    target_source: str
    training_target_quality: str
    target_training_role: str
    target_quality_warning: str


@dataclass(slots=True)
class BaseStageDataset:
    samples: list[TrainingSample]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        return self.samples[index]


class PerceptionDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "PerceptionDataset":
        from utils_tensor import zeros

        return cls(samples=[{"frames": [zeros(64, 64, 3)], "source": "synthetic"} for _ in range(size)])

    @classmethod
    def from_image_manifest(cls, manifest_path: str, quality_profile: str = "balanced") -> "PerceptionDataset":
        manifest = json.loads(Path(manifest_path).read_text())
        layer = InputAssetLayer()
        samples: list[TrainingSample] = []
        for rec in manifest.get("records", []):
            req = layer.build_request(images=[rec["image"]], text=rec.get("text", ""), quality_profile=quality_profile)
            frame = req.unified_asset.frames[0].tensor
            samples.append({"frames": [frame], "source": rec["image"], "text": rec.get("text", "")})
        return cls(samples=samples)


class RepresentationDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "RepresentationDataset":
        from utils_tensor import zeros

        samples: list[TrainingSample] = []
        for idx in range(size):
            graph = SceneGraph(frame_index=idx)
            samples.append({"frames": [zeros(64, 64, 3)], "graphs": [graph], "source": "synthetic"})
        return cls(samples=samples)

    @classmethod
    def from_perception_dataset(cls, source: PerceptionDataset) -> "RepresentationDataset":
        perception = PerceptionPipeline()
        builder = SceneGraphBuilder()
        out: list[TrainingSample] = []
        for idx, sample in enumerate(source.samples):
            text = sample.get("text", "")
            frame = sample.get("frames", [[]])[0]
            p = perception.analyze(frame)
            graph = builder.build(p, frame_index=idx)
            out.append({"frames": sample.get("frames", []), "graphs": [graph], "text": text, "source": sample.get("source", "")})
        return cls(samples=out)

    @classmethod
    def _from_cached_facts(cls, rec: dict[str, object], frame_index: int) -> PerceptionOutput:
        persons: list[PersonFacts] = []
        for idx, p in enumerate(rec.get("persons", [])):
            bbox_data = p.get("bbox")
            if not bbox_data:
                raise ValueError("perception cache person requires bbox")
            bbox = BBox(float(bbox_data["x"]), float(bbox_data["y"]), float(bbox_data["w"]), float(bbox_data["h"]))
            persons.append(
                PersonFacts(
                    bbox=bbox,
                    bbox_confidence=float(p.get("bbox_confidence", 0.0)),
                    bbox_source=str(p.get("bbox_source", "cache")),
                    mask_ref=p.get("mask_ref"),
                    mask_confidence=float(p.get("mask_confidence", 0.0)),
                    mask_source=str(p.get("mask_source", "cache")),
                    pose=PoseState(coarse_pose=str(p.get("coarse_pose", "unknown"))),
                    pose_confidence=float(p.get("pose_confidence", 0.0)),
                    pose_source=str(p.get("pose_source", "cache")),
                    expression=ExpressionState(label=str(p.get("expression_label", "neutral"))),
                    expression_confidence=float(p.get("expression_confidence", 0.0)),
                    expression_source=str(p.get("expression_source", "cache")),
                    orientation=OrientationState(),
                    orientation_confidence=float(p.get("orientation_confidence", 0.0)),
                    orientation_source=str(p.get("orientation_source", "cache")),
                    track_id=str(p.get("track_id", f"track_cached_{idx}")),
                    track_confidence=float(p.get("track_confidence", 0.0)),
                    track_source=str(p.get("track_source", "cache")),
                    garments=list(p.get("garments", [])),
                )
            )

        objects = [
            ObjectFacts(
                object_type=str(o.get("object_type", "unknown")),
                bbox=BBox(float(o["bbox"]["x"]), float(o["bbox"]["y"]), float(o["bbox"]["w"]), float(o["bbox"]["h"])),
                confidence=float(o.get("confidence", 0.0)),
                source=str(o.get("source", "cache")),
            )
            for o in rec.get("objects", [])
            if o.get("bbox")
        ]

        return PerceptionOutput(persons=persons, objects=objects, frame_size=tuple(rec.get("frame_size", [0, 0])), warnings=[], module_confidence={"cache": 1.0}, module_latency_ms={"cache": 0.0}, module_fallbacks={"source": "perception_cache"}, depth_score=rec.get("depth_score"))

    @classmethod
    def from_perception_cache(cls, cache_path: str, strict: bool = True) -> "RepresentationDataset":
        payload = json.loads(Path(cache_path).read_text())
        builder = SceneGraphBuilder()
        out: list[TrainingSample] = []
        for idx, rec in enumerate(payload.get("records", [])):
            if "persons" not in rec:
                if strict:
                    raise ValueError("perception cache record missing 'persons'")
                continue
            p = cls._from_cached_facts(rec, frame_index=int(rec.get("frame_index", idx)))
            graph = builder.build(p, frame_index=int(rec.get("frame_index", idx)))
            out.append({"frames": [rec.get("frame", [])] if isinstance(rec.get("frame"), list) else [], "graphs": [graph], "source": rec.get("source", "perception_cache")})
        return cls(samples=out)


class DynamicsDataset(BaseStageDataset):
    CANONICAL_PHASES = {"prepare", "transition", "contact_or_reveal", "stabilize"}
    CANONICAL_FAMILIES = {"pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition"}

    @classmethod
    def from_video_transition_manifest(cls, manifest_path: str, strict: bool = False) -> "DynamicsDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if payload.get("manifest_type") == "dynamics_graph_delta_manifest":
            return cls._from_observed_graph_delta_manifest(payload, manifest_path=manifest_path, strict=strict)
        records = payload.get("records", [])
        out: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_video_dynamics_primary",
            "manifest_path": manifest_path,
            "manifest_type": payload.get("manifest_type", "unknown"),
            "total_records": len(records),
            "loaded_records": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
            "family_coverage": {},
            "region_coverage": {},
            "phase_coverage": {},
            "fallback_free_ratio": 0.0,
        }
        fallback_free = 0
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be object")
                graph_before = _deserialize_graph(rec["scene_graph_before"])
                graph_after = _deserialize_graph(rec["scene_graph_after"])
                delta_raw = rec.get("graph_delta_target", {})
                if not isinstance(delta_raw, dict):
                    raise ValueError("graph_delta_target must be object")
                delta = GraphDelta(
                    pose_deltas={str(k): float(v) for k, v in (delta_raw.get("pose_deltas", {}) or {}).items()},
                    garment_deltas={str(k): float(v) for k, v in (delta_raw.get("garment_deltas", {}) or {}).items()},
                    visibility_deltas={str(k): str(v) for k, v in (delta_raw.get("visibility_deltas", {}) or {}).items()},
                    expression_deltas={str(k): float(v) for k, v in (delta_raw.get("expression_deltas", {}) or {}).items()},
                    interaction_deltas={str(k): float(v) for k, v in (delta_raw.get("interaction_deltas", {}) or {}).items()},
                    affected_entities=[str(x) for x in delta_raw.get("affected_entities", [str(rec.get("tracked_entity_id", "scene"))])],
                    affected_regions=[str(x) for x in delta_raw.get("affected_regions", ["torso"])],
                    semantic_reasons=[str(x) for x in delta_raw.get("semantic_reasons", [str(rec.get("transition_family", "pose_transition"))])],
                    region_transition_mode={str(k): str(v) for k, v in (delta_raw.get("region_transition_mode", {}) or {}).items()},
                    transition_phase=str(delta_raw.get("transition_phase", rec.get("phase_estimate", "transition"))),
                )
                if not (delta.pose_deltas or delta.garment_deltas or delta.visibility_deltas or delta.expression_deltas or delta.interaction_deltas):
                    raise ValueError("graph_delta_target requires at least one delta group")
                family, manifest_family, auxiliary_families, family_metadata = _resolve_manifest_primary_family(rec, "transition_family")
                _attach_auxiliary_family_metadata(delta, auxiliary_families, family_metadata)
                planner_context = rec.get("planner_context", {}) if isinstance(rec.get("planner_context", {}), dict) else {}
                phase = str(rec.get("phase_estimate", planner_context.get("phase", delta.transition_phase)))
                if phase not in cls.CANONICAL_PHASES:
                    raise ValueError(f"phase must be canonical, got {phase}")
                target_profile = rec.get("target_profile", {}) if isinstance(rec.get("target_profile"), dict) else {}
                reveal_cov = 1 if any("revealed" in str(k) for k in delta.visibility_deltas.keys()) else 0
                occlusion_cov = 1 if any("occluded" in str(k) for k in delta.visibility_deltas.keys()) else 0
                temporal_features = _build_temporal_transition_features(
                    graph_before=graph_before,
                    graph_after=graph_after,
                    roi_records=rec.get("roi_records", []) if isinstance(rec.get("roi_records"), list) else [],
                    graph_delta_target=delta_raw,
                    planner_context=planner_context,
                    target_profile=target_profile,
                    runtime_semantic_transition=str(rec.get("runtime_semantic_transition", family)),
                    phase_estimate=phase,
                    reveal_score=float(np.clip(rec.get("reveal_score", reveal_cov), 0.0, 1.0)),
                    occlusion_score=float(np.clip(rec.get("occlusion_score", occlusion_cov), 0.0, 1.0)),
                    support_score=float(np.clip((delta_raw.get("interaction_deltas", {}) or {}).get("support_contact", 0.0), 0.0, 1.0)),
                    transition_confidence=float(np.clip(rec.get("transition_confidence", 1.0), 0.0, 1.0)),
                )
                human_features, human_target, human_history = _build_human_state_transition_payload(
                    record=rec,
                    graph_before=graph_before,
                    graph_after=graph_after,
                    family=family,
                    phase=phase,
                    target_profile=target_profile,
                    roi_records=rec.get("roi_records", []) if isinstance(rec.get("roi_records"), list) else [],
                    planner_context=planner_context,
                    graph_delta_target=delta_raw,
                    has_previous_in_sequence=idx > 0,
                )
                out.append(
                    {
                        "graphs": [graph_before, graph_after],
                        "actions": [ActionStep(type=family, priority=1, target_entity=str(rec.get("tracked_entity_id", "scene")))],
                        "deltas": [delta],
                        "source": "manifest_video_dynamics_primary",
                        "delta_contract": _serialize_delta_contract(delta),
                        "temporal_transition_features": temporal_features,
                        "temporal_transition_target": {
                            "family": family,
                            "phase": phase,
                            "target_profile": target_profile,
                            "reveal_score": float(np.clip(rec.get("reveal_score", reveal_cov), 0.0, 1.0)),
                            "occlusion_score": float(np.clip(rec.get("occlusion_score", occlusion_cov), 0.0, 1.0)),
                            "support_contact_score": float(np.clip((delta_raw.get("interaction_deltas", {}) or {}).get("support_contact", 0.0), 0.0, 1.0)),
                        },
                        "human_state_transition_features": human_features,
                        "human_state_transition_target": human_target,
                        "human_state_history": human_history,
                        "graph_transition_contract": {
                            "planner_context": {
                                "step_index": float(planner_context.get("step_index", idx + 1)),
                                "total_steps": float(planner_context.get("total_steps", max(2, len(records)))),
                                "phase": str(planner_context.get("phase", rec.get("phase_estimate", "transition"))),
                                "target_duration": float(planner_context.get("target_duration", 1.0)),
                            },
                            "target_transition_context": rec.get("target_transition_context", {}) if isinstance(rec.get("target_transition_context"), dict) else {},
                            "memory_context": rec.get("memory_context", {}) if isinstance(rec.get("memory_context"), dict) else {},
                            "metadata": {"record_id": rec.get("record_id", f"video_transition_{idx}"), "transition_family": family, "manifest_family": manifest_family, "primary_family": family, "auxiliary_families": auxiliary_families, "target_profile": target_profile},
                        },
                    }
                )
                diagnostics["loaded_records"] = int(diagnostics["loaded_records"]) + 1
                diagnostics["family_coverage"][family] = int(diagnostics["family_coverage"].get(family, 0)) + 1
                for region in delta.affected_regions:
                    diagnostics["region_coverage"][region] = int(diagnostics["region_coverage"].get(region, 0)) + 1
                diagnostics["phase_coverage"][phase] = int(diagnostics["phase_coverage"].get(phase, 0)) + 1
                tp_cov = diagnostics.setdefault("target_profile_coverage", {"primary_regions": {}, "secondary_regions": {}, "context_regions": {}})
                for slot in ("primary_regions", "secondary_regions", "context_regions"):
                    for region in target_profile.get(slot, []) if isinstance(target_profile.get(slot, []), list) else []:
                        tp_cov[slot][str(region)] = int(tp_cov[slot].get(str(region), 0)) + 1
                roc_cov = diagnostics.setdefault("reveal_occlusion_coverage", {"revealed": 0, "occluded": 0})
                roc_cov["revealed"] = int(roc_cov.get("revealed", 0)) + reveal_cov
                roc_cov["occluded"] = int(roc_cov.get("occluded", 0)) + occlusion_cov
                support_cov = diagnostics.setdefault("support_contact_coverage", 0)
                diagnostics["support_contact_coverage"] = int(support_cov) + (1 if delta.interaction_deltas else 0)
                flags = rec.get("fallback_flags", {}) if isinstance(rec.get("fallback_flags"), dict) else {}
                if not bool(flags.get("heuristic_priors_used", False)):
                    fallback_free += 1
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "reason": str(exc)})
                if strict:
                    raise ValueError(f"video transition manifest record {idx} invalid: {exc}") from exc
        diagnostics["fallback_free_ratio"] = round(float(fallback_free / max(1, len(out))), 6)
        ds = cls(samples=out)
        ds.diagnostics = diagnostics
        return ds

    @classmethod
    def synthetic(cls, size: int) -> "DynamicsDataset":
        from utils_tensor import zeros

        samples: list[TrainingSample] = []
        for idx in range(size):
            action = ActionStep(type="move", priority=1, target_entity=f"person_{idx}")
            delta = GraphDelta(pose_deltas={"torso_pitch": 0.1 * (idx + 1)})
            samples.append({"graphs": [SceneGraph(frame_index=idx)], "actions": [action], "deltas": [delta], "frames": [zeros(64, 64, 3), [[[1.0, 1.0, 1.0] for _ in range(64)] for _ in range(64)]], "source": "synthetic_dynamics_bootstrap", "delta_contract": _serialize_delta_contract(delta)})
        ds = cls(samples=samples)
        ds.diagnostics = {"source": "synthetic_dynamics_bootstrap", "total_records": len(samples), "loaded_records": len(samples), "invalid_records": 0, "skipped_records": 0, "family_counts": {}, "tag_counts": {}}
        return ds

    @classmethod
    def from_graph_sequence(cls, graphs: list[SceneGraph], actions: list[ActionStep]) -> "DynamicsDataset":
        from training.learned_contracts import build_graph_transition_contract

        samples: list[TrainingSample] = []
        for idx in range(max(0, len(graphs) - 1)):
            delta = GraphDelta(pose_deltas={"frame_delta": float(graphs[idx + 1].frame_index - graphs[idx].frame_index)})
            graph_contract = build_graph_transition_contract(
                before=graphs[idx],
                after=graphs[idx + 1],
                delta=delta,
                transition_context={"source": "graph_sequence", "step_index": idx},
            )
            sample: TrainingSample = {
                "graphs": [graphs[idx], graphs[idx + 1]],
                "actions": actions,
                "deltas": [delta],
                "source": "graph_sequence",
                "delta_contract": _serialize_delta_contract(delta),
                "graph_transition_contract": graph_contract,
            }
            samples.append(sample)
        return cls(samples=samples)

    @classmethod
    def from_transition_manifest(cls, manifest_path: str, strict: bool = False) -> "DynamicsDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if payload.get("manifest_type") == "dynamics_graph_delta_manifest":
            return cls._from_observed_graph_delta_manifest(payload, manifest_path=manifest_path, strict=strict)
        records = payload.get("records", [])
        out: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_dynamics_transition_primary",
            "manifest_path": manifest_path,
            "manifest_type": payload.get("manifest_type", "dynamics_transition_manifest"),
            "total_records": len(records),
            "usable_samples": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
            "family_counts": {},
            "phase_counts": {},
            "transition_mode_counts": {},
            "affected_region_coverage": {},
            "region_coverage": {},
            "source_modes": {"manifest_explicit": 0, "manifest_derived": 0},
            "supervision_mode_counts": {"explicit": 0, "derived": 0},
            "missing_optional_fields": {},
            "notes_preview": [],
        }
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be json object")
                legacy_upgraded = False
                required = ("sample_id", "video_id", "frame_before_index", "frame_after_index", "family", "transition_mode", "phase")
                missing_required = [field for field in required if field not in rec]
                if missing_required:
                    legacy_compatible = isinstance(rec.get("scene_graph"), dict) or isinstance(rec.get("graph_delta_target"), dict)
                    if not legacy_compatible:
                        raise ValueError(f"missing required fields: {', '.join(missing_required)}")
                    graph_delta = rec.get("graph_delta_target", {}) if isinstance(rec.get("graph_delta_target"), dict) else {}
                    inferred_family = "pose_transition"
                    if graph_delta.get("garment_deltas"):
                        inferred_family = "garment_transition"
                    elif graph_delta.get("interaction_deltas"):
                        inferred_family = "interaction_transition"
                    elif graph_delta.get("expression_deltas"):
                        inferred_family = "expression_transition"
                    rec = dict(rec)
                    rec.setdefault("sample_id", str(rec.get("record_id", f"legacy_{idx}")))
                    rec.setdefault("video_id", str(rec.get("video_id", "legacy_video")))
                    rec.setdefault("frame_before_index", int(idx))
                    rec.setdefault("frame_after_index", int(idx + 1))
                    rec.setdefault("family", inferred_family)
                    rec.setdefault("transition_mode", "deform")
                    rec.setdefault("phase", str((rec.get("planner_context") or {}).get("phase", "transition")) if isinstance(rec.get("planner_context"), dict) else "transition")
                    rec.setdefault("state_before", {})
                    rec.setdefault("state_after", {})
                    legacy_upgraded = True
                    diagnostics["missing_optional_fields"]["legacy_schema_upgrade"] = int(diagnostics["missing_optional_fields"].get("legacy_schema_upgrade", 0)) + 1
                for field in ("memory_context", "target_transition_context", "planner_context"):
                    if rec.get(field) is not None and not isinstance(rec.get(field), dict):
                        raise ValueError(f"{field} must be object")

                supervision_mode = str(rec.get("supervision_mode", "explicit" if isinstance(rec.get("graph_delta_target"), dict) else "derived")).strip().lower()
                if supervision_mode not in {"explicit", "derived"}:
                    raise ValueError(f"supervision_mode must be explicit|derived, got {supervision_mode}")

                family, manifest_family, auxiliary_families, family_metadata = _resolve_manifest_primary_family(rec, "family")
                phase = str(rec["phase"]).strip().lower()
                if phase not in cls.CANONICAL_PHASES:
                    raise ValueError(f"phase must be canonical, got {phase}")
                transition_mode = str(rec["transition_mode"]).strip().lower()
                if not transition_mode:
                    raise ValueError("transition_mode must be non-empty")

                state_before = rec.get("state_before")
                state_after = rec.get("state_after")
                if not isinstance(state_before, dict) or not isinstance(state_after, dict):
                    raise ValueError("state_before/state_after must be objects")
                delta_raw = rec.get("graph_delta_target")
                if supervision_mode == "explicit" and not isinstance(delta_raw, dict):
                    raise ValueError("explicit mode requires graph_delta_target object")
                if supervision_mode == "derived":
                    delta_raw = _derive_graph_delta_target(state_before=state_before, state_after=state_after, family=family, phase=phase, transition_mode=transition_mode, record=rec)
                if not isinstance(delta_raw, dict):
                    raise ValueError("graph_delta_target must be object after derivation")

                graph_before_raw = rec.get("scene_graph_before") if isinstance(rec.get("scene_graph_before"), dict) else rec.get("source_graph_before")
                graph_after_raw = rec.get("scene_graph_after") if isinstance(rec.get("scene_graph_after"), dict) else rec.get("source_graph_after")
                if isinstance(graph_before_raw, dict) and isinstance(graph_after_raw, dict):
                    prev = _deserialize_graph(graph_before_raw)
                    nxt = _deserialize_graph(graph_after_raw)
                else:
                    tracked_entity = str(rec.get("tracked_entity_id", "person_1"))
                    person = PersonNode(person_id=tracked_entity, track_id=str(rec.get("track_id", f"track_{tracked_entity}")), bbox=BBox(0.2, 0.1, 0.6, 0.8), mask_ref=None)
                    prev = SceneGraph(frame_index=int(rec.get("frame_before_index", idx)), persons=[person])
                    nxt = SceneGraph(frame_index=int(rec.get("frame_after_index", idx + 1)), persons=[person])

                action_tokens = rec.get("action_tokens")
                if not isinstance(action_tokens, list) or not action_tokens:
                    action_tokens = rec.get("labels", [])
                if isinstance(rec.get("actions"), list) and rec["actions"]:
                    action_tokens = [str(a.get("type", "micro_adjust")) for a in rec["actions"] if isinstance(a, dict)]
                action_tokens = [str(x) for x in action_tokens] if isinstance(action_tokens, list) and action_tokens else ["micro_adjust"]
                actions = [ActionStep(type=str(label), priority=i + 1) for i, label in enumerate(action_tokens)]

                delta = GraphDelta(
                    pose_deltas={str(k): float(v) for k, v in (delta_raw.get("pose_deltas", {}) or {}).items()},
                    garment_deltas={str(k): float(v) for k, v in (delta_raw.get("garment_deltas", {}) or {}).items()},
                    visibility_deltas={str(k): str(v) for k, v in (delta_raw.get("visibility_deltas", {}) or {}).items()},
                    expression_deltas={str(k): float(v) for k, v in (delta_raw.get("expression_deltas", {}) or {}).items()},
                    interaction_deltas={str(k): float(v) for k, v in (delta_raw.get("interaction_deltas", {}) or {}).items()},
                    predicted_visibility_changes={str(k): str(v) for k, v in (delta_raw.get("predicted_visibility_changes", delta_raw.get("visibility_deltas", {})) or {}).items()},
                    affected_entities=[str(x) for x in delta_raw.get("affected_entities", rec.get("affected_entities", [str(rec.get("tracked_entity_id", "scene"))]))],
                    affected_regions=[str(x) for x in delta_raw.get("affected_regions", rec.get("affected_regions", []))],
                    semantic_reasons=[str(x) for x in delta_raw.get("semantic_reasons", action_tokens)],
                    region_transition_mode={str(k): str(v) for k, v in (delta_raw.get("region_transition_mode", rec.get("region_transition_mode", {})) or {}).items()},
                    transition_phase=phase,
                    state_before={str(k): str(v) for k, v in state_before.items()},
                    state_after={str(k): str(v) for k, v in state_after.items()},
                    transition_diagnostics={"supervision_mode": supervision_mode, "transition_mode": transition_mode},
                )
                delta.newly_revealed_regions = [RegionRef(region_id=str(x), bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="manifest") for x in rec.get("newly_revealed_regions", []) if isinstance(x, str)]
                delta.newly_occluded_regions = [RegionRef(region_id=str(x), bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="manifest") for x in rec.get("newly_occluded_regions", []) if isinstance(x, str)]
                _attach_auxiliary_family_metadata(delta, auxiliary_families, family_metadata)
                if not delta.affected_regions:
                    raise ValueError("affected_regions must be provided")
                if not delta.region_transition_mode:
                    raise ValueError("region_transition_mode must be provided")
                family_group_checks = {
                    "pose_transition": bool(delta.pose_deltas),
                    "garment_transition": bool(delta.garment_deltas),
                    "interaction_transition": bool(delta.interaction_deltas),
                    "expression_transition": bool(delta.expression_deltas),
                }
                if not family_group_checks[family]:
                    raise ValueError(f"{family} requires corresponding deltas")

                planner_context = rec.get("planner_context", {}) if isinstance(rec.get("planner_context"), dict) else {}
                target_profile = rec.get("target_profile", {}) if isinstance(rec.get("target_profile"), dict) else {}
                reveal_score = float(rec.get("reveal_score", 1.0 if delta.newly_revealed_regions else 0.0))
                occlusion_score = float(rec.get("occlusion_score", 1.0 if delta.newly_occluded_regions else 0.0))
                support_contact_score = float(rec.get("support_contact_score", delta.interaction_deltas.get("support_contact", 0.0)))
                temporal_features = rec.get("precomputed_temporal_transition_features")
                if not isinstance(temporal_features, list):
                    temporal_features = _build_temporal_transition_features(
                        graph_before=prev,
                        graph_after=nxt,
                        roi_records=rec.get("roi_records", []) if isinstance(rec.get("roi_records"), list) else [],
                        graph_delta_target=delta_raw,
                        planner_context=planner_context,
                        target_profile=target_profile,
                        runtime_semantic_transition=family,
                        phase_estimate=phase,
                        reveal_score=reveal_score,
                        occlusion_score=occlusion_score,
                        support_score=support_contact_score,
                        transition_confidence=float(rec.get("transition_confidence", 1.0)),
                    )
                human_features, human_target, human_history = _build_human_state_transition_payload(
                    record=rec,
                    graph_before=prev,
                    graph_after=nxt,
                    family=family,
                    phase=phase,
                    target_profile=target_profile,
                    roi_records=rec.get("roi_records", []) if isinstance(rec.get("roi_records"), list) else [],
                    planner_context=planner_context,
                    graph_delta_target=delta_raw,
                    has_previous_in_sequence=idx > 0,
                )
                graph_contract = {
                    "planner_context": {
                        "step_index": float(planner_context.get("step_index", idx + 1)),
                        "total_steps": float(planner_context.get("total_steps", max(2, idx + 2))),
                        "phase": phase,
                        "target_duration": float(planner_context.get("target_duration", 1.5)),
                    },
                    "target_transition_context": rec.get("target_transition_context", {}) if isinstance(rec.get("target_transition_context"), dict) else {},
                    "memory_context": rec.get("memory_context", {}) if isinstance(rec.get("memory_context"), dict) else {},
                    "metadata": {
                        "record_id": rec.get("sample_id", rec.get("record_id", f"record_{idx}")),
                        "video_id": rec.get("video_id"),
                        "transition_family": family,
                        "manifest_family": manifest_family,
                        "primary_family": family,
                        "auxiliary_families": auxiliary_families,
                        "transition_mode": transition_mode,
                        "target_profile": target_profile,
                        "supervision_mode": supervision_mode,
                        "tags": rec.get("tags", []),
                        "notes": rec.get("notes", ""),
                        "source_graph_paths": {
                            "before": rec.get("source_graph_path_before"),
                            "after": rec.get("source_graph_path_after"),
                            "before_frame": rec.get("source_frame_path_before"),
                            "after_frame": rec.get("source_frame_path_after"),
                        },
                    },
                }
                out.append(
                    {
                        "graphs": [prev, nxt],
                        "actions": actions,
                        "deltas": [delta],
                        "source": rec.get("source", "manifest_dynamics_transition_primary"),
                        "delta_contract": _serialize_delta_contract(delta),
                        "graph_transition_contract": graph_contract,
                        "temporal_transition_features": [float(x) for x in temporal_features],
                        "temporal_transition_target": {
                            "family": family,
                            "phase": phase,
                            "target_profile": target_profile,
                            "reveal_score": reveal_score,
                            "occlusion_score": occlusion_score,
                            "support_contact_score": support_contact_score,
                        },
                        "human_state_transition_features": human_features,
                        "human_state_transition_target": human_target,
                        "human_state_history": human_history,
                    }
                )
                diagnostics["usable_samples"] = int(diagnostics["usable_samples"]) + 1
                diagnostics["family_counts"][family] = int(diagnostics["family_counts"].get(family, 0)) + 1
                diagnostics["phase_counts"][phase] = int(diagnostics["phase_counts"].get(phase, 0)) + 1
                diagnostics["transition_mode_counts"][transition_mode] = int(diagnostics["transition_mode_counts"].get(transition_mode, 0)) + 1
                diagnostics["supervision_mode_counts"][supervision_mode] = int(diagnostics["supervision_mode_counts"].get(supervision_mode, 0)) + 1
                sm_key = "manifest_explicit" if supervision_mode == "explicit" else "manifest_derived"
                diagnostics["source_modes"][sm_key] = int(diagnostics["source_modes"].get(sm_key, 0)) + 1
                if legacy_upgraded:
                    diagnostics["source_modes"]["legacy_schema_upgrade"] = int(diagnostics["source_modes"].get("legacy_schema_upgrade", 0)) + 1
                for region in delta.affected_regions:
                    diagnostics["affected_region_coverage"][region] = int(diagnostics["affected_region_coverage"].get(region, 0)) + 1
                    diagnostics["region_coverage"][region] = int(diagnostics["region_coverage"].get(region, 0)) + 1
                for optional in ("planner_context", "target_profile", "planner_context_summary", "memory_context_summary", "planner_context", "segment_id"):
                    if rec.get(optional) is None:
                        diagnostics["missing_optional_fields"][optional] = int(diagnostics["missing_optional_fields"].get(optional, 0)) + 1
                if len(diagnostics["notes_preview"]) < 5 and rec.get("notes"):
                    diagnostics["notes_preview"].append(str(rec.get("notes")))
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "reason": str(exc)})
                if strict:
                    raise ValueError(f"dynamics manifest record {idx} invalid: {exc}") from exc
        diagnostics["loaded_records"] = int(diagnostics["usable_samples"])
        ds = cls(samples=out)
        ds.diagnostics = diagnostics
        return ds


class TemporalTransitionDataset(BaseStageDataset):
    CANONICAL_PHASES = {"prepare", "transition", "contact_or_reveal", "stabilize"}
    CANONICAL_FAMILIES = {"pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition"}
    REGION_KEYS = ("face", "torso", "left_arm", "right_arm", "legs", "garments", "inner_garment", "outer_garment")

    @classmethod
    def from_video_transition_manifest(cls, manifest_path: str, strict: bool = False) -> "TemporalTransitionDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        samples: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_video_temporal_transition_primary",
            "manifest_path": manifest_path,
            "manifest_type": payload.get("manifest_type", "unknown"),
            "total_records": len(records),
            "loaded_records": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
            "family_coverage": {},
            "phase_coverage": {},
            "region_coverage": {},
            "fallback_free_ratio": 0.0,
        }
        fallback_free = 0
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be object")
                graph_before = _deserialize_graph(rec["scene_graph_before"])
                graph_after = _deserialize_graph(rec["scene_graph_after"])
                family = str(rec.get("transition_family", rec.get("runtime_semantic_transition", "pose_transition"))).strip().lower()
                if family not in cls.CANONICAL_FAMILIES:
                    raise ValueError(f"transition_family must be canonical, got {family}")
                planner_context = rec.get("planner_context", {}) if isinstance(rec.get("planner_context"), dict) else {}
                phase = str(rec.get("phase_estimate", planner_context.get("phase", "transition"))).strip().lower()
                if phase not in cls.CANONICAL_PHASES:
                    raise ValueError(f"phase must be canonical, got {phase}")
                target_profile = rec.get("target_profile", {}) if isinstance(rec.get("target_profile"), dict) else {}
                graph_delta_target = rec.get("graph_delta_target", {}) if isinstance(rec.get("graph_delta_target"), dict) else {}
                roi_records = rec.get("roi_records", [])
                if not isinstance(roi_records, list):
                    roi_records = []
                changed_ratio = float(np.mean([float(r.get("changed_ratio", 0.0)) for r in roi_records if isinstance(r, dict)] or [0.0]))
                reveal_score = float(graph_delta_target.get("visibility_deltas", {}).get("revealed_regions_score", rec.get("reveal_score", changed_ratio)) if isinstance(graph_delta_target.get("visibility_deltas", {}), dict) else changed_ratio)
                occlusion_score = float(graph_delta_target.get("visibility_deltas", {}).get("occluded_regions_score", rec.get("occlusion_score", changed_ratio * 0.5)) if isinstance(graph_delta_target.get("visibility_deltas", {}), dict) else changed_ratio * 0.5)
                support_score = float(graph_delta_target.get("interaction_deltas", {}).get("support_contact", rec.get("support_contact_score", 0.0)) if isinstance(graph_delta_target.get("interaction_deltas", {}), dict) else 0.0)
                transition_confidence = float(rec.get("transition_confidence", changed_ratio))

                feature_vector = _build_temporal_transition_features(
                    graph_before=graph_before,
                    graph_after=graph_after,
                    roi_records=roi_records,
                    graph_delta_target=graph_delta_target,
                    planner_context=planner_context,
                    target_profile=target_profile,
                    runtime_semantic_transition=str(rec.get("runtime_semantic_transition", family)),
                    phase_estimate=phase,
                    reveal_score=reveal_score,
                    occlusion_score=occlusion_score,
                    support_score=support_score,
                    transition_confidence=transition_confidence,
                )
                samples.append(
                    {
                        "graphs": [graph_before, graph_after],
                        "source": "manifest_video_temporal_transition_primary",
                        "temporal_transition_features": feature_vector,
                        "temporal_transition_target": {
                            "family": family,
                            "phase": phase,
                            "target_profile": target_profile,
                            "reveal_score": float(np.clip(reveal_score, 0.0, 1.0)),
                            "occlusion_score": float(np.clip(occlusion_score, 0.0, 1.0)),
                            "support_contact_score": float(np.clip(support_score, 0.0, 1.0)),
                        },
                        "graph_transition_contract": {
                            "planner_context": planner_context,
                            "target_transition_context": rec.get("target_transition_context", {}) if isinstance(rec.get("target_transition_context"), dict) else {},
                            "memory_context": rec.get("memory_context", {}) if isinstance(rec.get("memory_context"), dict) else {},
                            "metadata": {"record_id": rec.get("record_id", f"video_transition_{idx}"), "transition_family": family},
                        },
                    }
                )
                diagnostics["loaded_records"] = int(diagnostics["loaded_records"]) + 1
                diagnostics["family_coverage"][family] = int(diagnostics["family_coverage"].get(family, 0)) + 1
                diagnostics["phase_coverage"][phase] = int(diagnostics["phase_coverage"].get(phase, 0)) + 1
                for slot in ("primary_regions", "secondary_regions", "context_regions"):
                    for region in target_profile.get(slot, []) if isinstance(target_profile.get(slot, []), list) else []:
                        diagnostics["region_coverage"][str(region)] = int(diagnostics["region_coverage"].get(str(region), 0)) + 1
                flags = rec.get("fallback_flags", {}) if isinstance(rec.get("fallback_flags"), dict) else {}
                if not bool(flags.get("heuristic_priors_used", False)):
                    fallback_free += 1
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "error": str(exc)})
                if strict:
                    raise ValueError(f"video transition temporal record {idx} invalid: {exc}") from exc
        diagnostics["fallback_free_ratio"] = round(float(fallback_free / max(1, len(samples))), 6)
        ds = cls(samples=samples)
        ds.diagnostics = diagnostics
        return ds

    @classmethod
    def synthetic(cls, size: int) -> "TemporalTransitionDataset":
        samples: list[TrainingSample] = []
        for i in range(size):
            family = ("pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition")[i % 5]
            phase = ("prepare", "transition", "contact_or_reveal", "stabilize")[i % 4]
            samples.append(
                {
                    "source": "synthetic_temporal_transition_bootstrap",
                    "temporal_transition_features": [0.01 * float((j + 1) * (i + 1)) for j in range(128)],
                    "temporal_transition_target": {
                        "family": family,
                        "phase": phase,
                        "target_profile": {"primary_regions": ["torso"], "secondary_regions": ["face"], "context_regions": ["legs"]},
                        "reveal_score": 0.35,
                        "occlusion_score": 0.2,
                        "support_contact_score": 0.1,
                    },
                }
            )
        ds = cls(samples=samples)
        ds.diagnostics = {"source": "synthetic_temporal_transition_bootstrap", "total_records": len(samples), "loaded_records": len(samples), "invalid_records": 0, "skipped_records": 0, "fallback_free_ratio": 1.0}
        return ds


class HumanStateTransitionDataset(BaseStageDataset):
    CANONICAL_PHASES = {"prepare", "transition", "contact_or_reveal", "stabilize"}
    CANONICAL_FAMILIES = {"pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition"}
    REGION_KEYS = ("face", "torso", "left_arm", "right_arm", "legs", "garments", "inner_garment", "outer_garment")

    @classmethod
    def from_video_transition_manifest(cls, manifest_path: str, strict: bool = False) -> "HumanStateTransitionDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        samples: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_video_human_state_transition_primary",
            "manifest_path": manifest_path,
            "manifest_type": payload.get("manifest_type", "unknown"),
            "total_records": len(records),
            "usable": 0,
            "invalid": 0,
            "skipped": 0,
            "family_coverage": {},
            "phase_coverage": {},
            "region_coverage": {},
            "reveal_coverage": 0,
            "occlusion_coverage": 0,
            "history_available": 0,
            "history_availability_ratio": 0.0,
            "invalid_examples": [],
        }
        record_index = {str(r.get("record_id", f"record_{idx}")): idx for idx, r in enumerate(records) if isinstance(r, dict)}
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be object")
                family = str(rec.get("transition_family", rec.get("runtime_semantic_transition", "pose_transition"))).strip().lower()
                if family not in cls.CANONICAL_FAMILIES:
                    raise ValueError(f"transition_family must be canonical, got {family}")
                planner = rec.get("planner_context", {}) if isinstance(rec.get("planner_context"), dict) else {}
                phase = str(rec.get("phase_estimate", planner.get("phase", "transition"))).strip().lower()
                if phase not in cls.CANONICAL_PHASES:
                    raise ValueError(f"phase must be canonical, got {phase}")
                before = _deserialize_graph(rec.get("scene_graph_before", {}))
                after = _deserialize_graph(rec.get("scene_graph_after", {}))
                roi_records = rec.get("roi_records", []) if isinstance(rec.get("roi_records"), list) else []
                target_profile = rec.get("target_profile", {}) if isinstance(rec.get("target_profile"), dict) else {}
                delta_target = rec.get("graph_delta_target", {}) if isinstance(rec.get("graph_delta_target"), dict) else {}
                human_features, human_target, human_history = _build_human_state_transition_payload(
                    record=rec,
                    graph_before=before,
                    graph_after=after,
                    family=family,
                    phase=phase,
                    target_profile=target_profile,
                    roi_records=roi_records,
                    planner_context=planner,
                    graph_delta_target=delta_target,
                    has_previous_in_sequence=idx > 0,
                )
                reveal_cue = float(human_target.get("reveal_memory_target", 0.0))
                occlusion_cue = float(1.0 - float(np.mean(human_target.get("visibility_targets", [1.0]))))

                sample: TrainingSample = {
                    "source": "manifest_video_human_state_transition_primary",
                    "graphs": [before, after],
                    "human_state_transition_features": human_features,
                    "human_state_transition_target": human_target,
                    "human_state_history": human_history,
                    "graph_transition_contract": {
                        "planner_context": planner,
                        "target_transition_context": rec.get("target_transition_context", {}) if isinstance(rec.get("target_transition_context"), dict) else {},
                        "memory_context": rec.get("memory_context", {}) if isinstance(rec.get("memory_context"), dict) else {},
                        "metadata": {"record_id": rec.get("record_id", f"video_transition_{idx}")},
                    },
                    "target_profile": target_profile,
                    "phase_family": {"phase": phase, "family": family},
                    "reveal_occlusion_support_cues": {
                        "reveal": reveal_cue,
                        "occlusion": occlusion_cue,
                        "support": float(human_target.get("support_contact_target", 0.0)),
                    },
                    "visibility_targets": human_target.get("visibility_targets", []),
                }
                if bool(human_history.get("has_history", False)):
                    diagnostics["history_available"] = int(diagnostics["history_available"]) + 1
                samples.append(sample)
                diagnostics["usable"] = int(diagnostics["usable"]) + 1
                diagnostics["family_coverage"][family] = int(diagnostics["family_coverage"].get(family, 0)) + 1
                diagnostics["phase_coverage"][phase] = int(diagnostics["phase_coverage"].get(phase, 0)) + 1
                active_regions = set(str(x) for x in (human_target.get("target_profile", {}) or {}).get("primary_regions", []))
                active_regions.update(str(x) for x in (human_target.get("target_profile", {}) or {}).get("secondary_regions", []))
                active_regions.update(str(x) for x in (human_target.get("target_profile", {}) or {}).get("context_regions", []))
                for r in active_regions:
                    diagnostics["region_coverage"][r] = int(diagnostics["region_coverage"].get(r, 0)) + 1
                diagnostics["reveal_coverage"] = int(diagnostics["reveal_coverage"]) + (1 if reveal_cue > 0.1 else 0)
                diagnostics["occlusion_coverage"] = int(diagnostics["occlusion_coverage"]) + (1 if occlusion_cue > 0.1 else 0)
            except Exception as exc:
                diagnostics["invalid"] = int(diagnostics["invalid"]) + 1
                diagnostics["skipped"] = int(diagnostics["skipped"]) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "error": str(exc)})
                if strict:
                    raise ValueError(f"video transition human-state record {idx} invalid: {exc}") from exc
        diagnostics["history_availability_ratio"] = round(float(diagnostics["history_available"]) / max(1.0, float(diagnostics["usable"])), 6)
        ds = cls(samples=samples)
        ds.diagnostics = diagnostics
        return ds



class RendererDataset(BaseStageDataset):
    CANONICAL_PHASES = {"prepare", "transition", "contact_or_reveal", "stabilize"}
    CANONICAL_FAMILIES = {"pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition"}
    ALLOWED_MEMORY_SUPPORT_LEVELS = {"none", "weak", "medium", "strong"}
    ALLOWED_TARGET_SOURCES = POLICY_ALLOWED_TARGET_SOURCES
    ALLOWED_TRAINING_TARGET_QUALITIES = POLICY_ALLOWED_TRAINING_TARGET_QUALITIES

    @classmethod
    def _normalize_target_source(cls, value: object) -> str:
        source = str(value if value is not None else "unknown").strip().lower()
        return source if source in cls.ALLOWED_TARGET_SOURCES else "unknown"

    @classmethod
    def _normalize_training_target_quality(cls, value: object) -> str:
        quality = str(value if value is not None else "unknown").strip().lower()
        return quality if quality in cls.ALLOWED_TRAINING_TARGET_QUALITIES else "unknown"

    @classmethod
    def _normalize_renderer_memory_bundle(
        cls,
        rec: dict[str, object],
        diagnostics: dict[str, object],
        idx: int,
    ) -> dict[str, object]:
        contract = rec.get("renderer_batch_contract", {}) if isinstance(rec.get("renderer_batch_contract"), dict) else {}
        raw = rec.get("renderer_memory_bundle")
        if raw is None:
            raw = contract.get("region_memory_bundle_serialized")
        if not isinstance(raw, dict) or not raw:
            return {
                "memory_bundle_present": False,
                "memory_support_level": "none",
                "reveal_lifecycle": "unknown",
                "has_current_reuse": False,
                "has_identity_reference": False,
                "has_appearance_reference": False,
                "has_garment_reference": False,
                "has_hidden_slot": False,
                "hidden_type": "none",
                "hidden_support_active": False,
                "retrieval_reasons": [],
            }

        support = str(raw.get("memory_support_level", "none")).strip().lower()
        if support not in cls.ALLOWED_MEMORY_SUPPORT_LEVELS:
            diagnostics["invalid_memory_bundle_records"] = int(diagnostics.get("invalid_memory_bundle_records", 0)) + 1
            warning = {"index": idx, "field": "memory_support_level", "value": support, "normalized": "none"}
            warnings = diagnostics.get("memory_bundle_warnings")
            if isinstance(warnings, list) and len(warnings) < 8:
                warnings.append(warning)
            support = "none"

        reasons = raw.get("retrieval_reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        hidden_slot = raw.get("hidden_slot") if isinstance(raw.get("hidden_slot"), dict) else {}
        hidden_type = str(raw.get("hidden_type", hidden_slot.get("hidden_type", "none"))).strip().lower()
        hidden_support_active = bool(raw.get("hidden_support_active", False))
        if hidden_type in {"revealed", "revealed_history"}:
            hidden_support_active = False
        return {
            "memory_bundle_present": bool(raw.get("memory_bundle_present", True)),
            "memory_support_level": support,
            "reveal_lifecycle": str(raw.get("reveal_lifecycle", "unknown")).strip().lower(),
            "has_current_reuse": bool(raw.get("has_current_reuse", False)),
            "has_identity_reference": bool(raw.get("has_identity_reference", False)),
            "has_appearance_reference": bool(raw.get("has_appearance_reference", False)),
            "has_garment_reference": bool(raw.get("has_garment_reference", False)),
            "has_hidden_slot": bool(raw.get("has_hidden_slot", False)),
            "hidden_type": hidden_type,
            "hidden_support_active": hidden_support_active,
            "retrieval_reasons": [str(x) for x in reasons if isinstance(x, str)],
        }

    @staticmethod
    def _target_role_from_sample(sample: TrainingSample) -> str:
        role = str(sample.get("target_training_role", "") or "").strip().lower()
        if role not in {"supervised_external", "bootstrap_self_generated", "weak_unknown"}:
            contract = sample.get("renderer_batch_contract", {}) if isinstance(sample.get("renderer_batch_contract", {}), dict) else {}
            role = str(contract.get("target_training_role", "") or "").strip().lower()
        return role if role in {"supervised_external", "bootstrap_self_generated", "weak_unknown"} else "weak_unknown"

    def filtered_by_target_role_policy(self, policy: str) -> "RendererDataset":
        """Return a non-mutating renderer dataset view filtered by explicit target-role policy."""
        role_keys = ("supervised_external", "bootstrap_self_generated", "weak_unknown")
        retained_by_role = {key: 0 for key in role_keys}
        filtered_out_by_role = {key: 0 for key in role_keys}
        filtered_samples: list[TrainingSample] = []
        for sample in self.samples:
            role = self._target_role_from_sample(sample)
            if target_role_allowed_by_policy(role, policy):
                filtered_samples.append(sample)
                retained_by_role[role] += 1
            else:
                filtered_out_by_role[role] += 1

        diagnostics = copy.deepcopy(getattr(self, "diagnostics", {}))
        pre_count = len(self.samples)
        post_count = len(filtered_samples)
        sq_present = 0
        sq_warning_records = 0
        sq_changed_sum = 0.0
        sq_delta_sum = 0.0
        sq_family_counts: dict[str, int] = {}
        sq_warning_counts: dict[str, int] = {}
        for sample in filtered_samples:
            quality = sample.get("supervised_quality") if isinstance(sample.get("supervised_quality"), dict) else None
            if quality is None:
                contract = sample.get("renderer_batch_contract", {}) if isinstance(sample.get("renderer_batch_contract", {}), dict) else {}
                quality = contract.get("supervised_quality") if isinstance(contract.get("supervised_quality"), dict) else None
            if quality is None:
                continue
            sq_present += 1
            sq_changed_sum += float(quality.get("changed_ratio", 0.0) or 0.0)
            sq_delta_sum += float(quality.get("mean_abs_delta", 0.0) or 0.0)
            family = str(quality.get("semantic_family", ""))
            if family:
                sq_family_counts[family] = sq_family_counts.get(family, 0) + 1
            q_warnings = quality.get("warnings", []) if isinstance(quality.get("warnings", []), list) else []
            if q_warnings:
                sq_warning_records += 1
            for warning in q_warnings:
                warning_text = str(warning)
                sq_warning_counts[warning_text] = sq_warning_counts.get(warning_text, 0) + 1
        diagnostics.update(
            {
                "target_role_policy": policy,
                "pre_filter_sample_count": pre_count,
                "post_filter_sample_count": post_count,
                "filtered_out_sample_count": pre_count - post_count,
                "filtered_out_by_role": filtered_out_by_role,
                "retained_by_role": retained_by_role,
                "supervised_quality_present_count": sq_present,
                "supervised_quality_warning_record_count": sq_warning_records,
                "supervised_quality_avg_changed_ratio": round(sq_changed_sum / max(1, sq_present), 6),
                "supervised_quality_avg_mean_abs_delta": round(sq_delta_sum / max(1, sq_present), 6),
                "supervised_quality_semantic_family_counts": sq_family_counts,
                "supervised_quality_warnings_count_by_type": sq_warning_counts,
            }
        )
        warnings = diagnostics.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            diagnostics["warnings"] = warnings
        if post_count == 0:
            diagnostics["filter_warning"] = f"renderer_target_role_policy={policy!r} retained zero samples"
            warnings.append({"type": "empty_after_target_role_policy", "message": diagnostics["filter_warning"]})
        if policy in {"supervised_plus_bootstrap", "bootstrap_only", "all_non_unknown", "all"}:
            warnings.append(
                {
                    "type": "bootstrap_self_generated_not_ground_truth",
                    "message": "bootstrap_self_generated targets are not ground truth; weighted/bootstrapped training only",
                }
            )
        if policy == "all":
            warnings.append({"type": "weak_unknown_included", "message": "weak_unknown targets included by policy='all'"})

        ds = RendererDataset(samples=filtered_samples)
        ds.diagnostics = diagnostics
        return ds

    @classmethod
    def from_video_transition_manifest(cls, manifest_path: str, strict: bool = False) -> "RendererDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        samples: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_video_renderer_primary",
            "manifest_path": manifest_path,
            "manifest_type": payload.get("manifest_type", "unknown"),
            "total_records": len(records),
            "loaded_records": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
            "invalid_memory_bundle_records": 0,
            "memory_bundle_warnings": [],
            "family_counts": {},
            "region_coverage": {},
            "phase_coverage": {},
            "fallback_free_ratio": 0.0,
            "temporal_window_records": 0,
            "temporal_window_usage_ratio": 0.0,
            "missing_target_provenance_count": 0,
            "contains_legacy_unknown_target_provenance": True,
        }
        fallback_free = 0
        for idx, rec in enumerate(records):
            try:
                family = str(rec.get("transition_family", "pose_transition")).strip().lower()
                if family not in cls.CANONICAL_FAMILIES:
                    raise ValueError(f"transition_family must be canonical, got {family}")
                phase = str(rec.get("phase_estimate", (rec.get("planner_context", {}) or {}).get("phase", "transition")))
                if phase not in cls.CANONICAL_PHASES:
                    raise ValueError(f"phase_estimate must be canonical, got {phase}")
                roi_records = rec.get("roi_records", [])
                if not isinstance(roi_records, list) or not roi_records:
                    if "roi_before" not in rec or "roi_after" not in rec:
                        raise ValueError("record requires roi_records or roi_before/roi_after")
                    roi_records = [
                        {
                            "region_type": "torso",
                            "roi_before": rec["roi_before"],
                            "roi_after": rec["roi_after"],
                            "changed_mask": rec.get("changed_mask"),
                            "preservation_mask": rec.get("preservation_mask"),
                            "priors": {},
                        }
                    ]
                for roi_idx, roi_rec in enumerate(roi_records):
                    if not isinstance(roi_rec, dict):
                        raise ValueError(f"roi_records[{roi_idx}] must be object")
                    roi_before = cls._as_hw3_tensor(roi_rec.get("roi_before"), "roi_before")
                    roi_after = cls._as_hw3_tensor(roi_rec.get("roi_after"), "roi_after")
                    b = np.asarray(roi_before, dtype=np.float32)
                    a = np.asarray(roi_after, dtype=np.float32)
                    if b.shape != a.shape:
                        raise ValueError("roi_before and roi_after shape mismatch")
                    changed = roi_rec.get("changed_mask")
                    if changed is None:
                        changed = np.clip(np.mean(np.abs(a - b), axis=2, keepdims=True) * 3.0, 0.0, 1.0).tolist()
                    preservation = roi_rec.get("preservation_mask")
                    if preservation is None:
                        preservation = np.clip(1.0 - np.asarray(changed, dtype=np.float32), 0.0, 1.0).tolist()
                    changed = cls._as_hw1_tensor(changed, "changed_mask")
                    preservation = cls._as_hw1_tensor(preservation, "preservation_mask")
                    region_type = str(roi_rec.get("region_type", "torso"))
                    tp = roi_rec.get("target_profile", rec.get("target_profile", {}))
                    prev_roi_raw = roi_rec.get("roi_prev", rec.get("roi_prev", roi_before))
                    prev_roi = cls._as_hw3_tensor(prev_roi_raw, "roi_prev")
                    temporal_target = {
                        "predicted_family": family,
                        "predicted_phase": phase,
                        "target_profile": tp if isinstance(tp, dict) else {},
                        "reveal_score": float(np.clip(rec.get("reveal_score", float(np.mean(changed))), 0.0, 1.0)),
                        "occlusion_score": float(np.clip(rec.get("occlusion_score", float(np.mean(changed)) * 0.5), 0.0, 1.0)),
                        "support_contact_score": float(
                            np.clip(
                                (rec.get("graph_delta_target", {}) or {}).get("interaction_deltas", {}).get("support_contact", 0.0)
                                if isinstance((rec.get("graph_delta_target", {}) or {}).get("interaction_deltas", {}), dict)
                                else 0.0,
                                0.0,
                                1.0,
                            )
                        ),
                    }
                    target_source = "unknown"
                    training_target_quality = "unknown"
                    target_training_role = "weak_unknown"
                    quality_warning = "unknown_training_target_quality"
                    renderer_contract = {
                        "semantic_embed": rec.get("semantic_embed", cls._semantic_embed_from_family(family)),
                        "delta_cond": rec.get("delta_cond", [0.0] * 9),
                        "planner_cond": rec.get("planner_cond", [0.0] * 8),
                        "graph_cond": rec.get("graph_cond", [0.0] * 7),
                        "memory_cond": rec.get("memory_cond", [0.0] * 8),
                        "appearance_cond": rec.get("appearance_cond", np.concatenate([np.mean(b, axis=(0, 1)), np.std(b, axis=(0, 1))]).tolist()),
                        "bbox_cond": rec.get("bbox_cond", [0.2, 0.2, 0.4, 0.4]),
                        "alpha_target": changed,
                        "blend_hint": preservation,
                        "changed_mask": changed,
                        "preservation_mask": preservation,
                        "region_type": region_type,
                        "transition_family": family,
                        "target_profile": tp,
                        "heuristic_priors": roi_rec.get("priors", rec.get("heuristic_priors", {})),
                        "predicted_family": temporal_target["predicted_family"],
                        "predicted_phase": temporal_target["predicted_phase"],
                        "reveal_score": temporal_target["reveal_score"],
                        "occlusion_score": temporal_target["occlusion_score"],
                        "support_contact_score": temporal_target["support_contact_score"],
                        "target_source": target_source,
                        "training_target_quality": training_target_quality,
                        "target_training_role": target_training_role,
                        "target_quality_warning": quality_warning,
                    }
                    renderer_memory_bundle = cls._normalize_renderer_memory_bundle(rec, diagnostics, idx)
                    renderer_contract["region_memory_bundle_serialized"] = renderer_memory_bundle
                    temporal_features = _build_temporal_transition_features(
                        graph_before=_deserialize_graph(rec.get("scene_graph_before", {})),
                        graph_after=_deserialize_graph(rec.get("scene_graph_after", {})),
                        roi_records=roi_records,
                        graph_delta_target=rec.get("graph_delta_target", {}) if isinstance(rec.get("graph_delta_target"), dict) else {},
                        planner_context=rec.get("planner_context", {}) if isinstance(rec.get("planner_context"), dict) else {},
                        target_profile=tp if isinstance(tp, dict) else {},
                        runtime_semantic_transition=str(rec.get("runtime_semantic_transition", family)),
                        phase_estimate=phase,
                        reveal_score=float(np.clip(rec.get("reveal_score", float(np.mean(changed))), 0.0, 1.0)),
                        occlusion_score=float(np.clip(rec.get("occlusion_score", float(np.mean(changed)) * 0.5), 0.0, 1.0)),
                        support_score=float(np.clip((rec.get("graph_delta_target", {}) or {}).get("interaction_deltas", {}).get("support_contact", 0.0) if isinstance((rec.get("graph_delta_target", {}) or {}).get("interaction_deltas", {}), dict) else 0.0, 0.0, 1.0)),
                        transition_confidence=float(np.clip(rec.get("transition_confidence", 1.0), 0.0, 1.0)),
                    ) if isinstance(rec.get("scene_graph_before"), dict) and isinstance(rec.get("scene_graph_after"), dict) else [0.0] * 128
                    human_features, human_target, human_history = _build_human_state_transition_payload(
                        record=rec,
                        graph_before=_deserialize_graph(rec.get("scene_graph_before", {})),
                        graph_after=_deserialize_graph(rec.get("scene_graph_after", {})),
                        family=family,
                        phase=phase,
                        target_profile=tp if isinstance(tp, dict) else {},
                        roi_records=roi_records,
                        planner_context=rec.get("planner_context", {}) if isinstance(rec.get("planner_context"), dict) else {},
                        graph_delta_target=rec.get("graph_delta_target", {}) if isinstance(rec.get("graph_delta_target"), dict) else {},
                        has_previous_in_sequence=idx > 0,
                    ) if isinstance(rec.get("scene_graph_before"), dict) and isinstance(rec.get("scene_graph_after"), dict) else ([0.0] * 160, {
                        "family": family,
                        "phase": phase,
                        "target_profile": tp if isinstance(tp, dict) else {},
                        "region_state_targets": [0.0] * 8,
                        "visibility_targets": [0.0] * 8,
                        "reveal_memory_target": 0.0,
                        "support_contact_target": 0.0,
                    }, {"has_history": False, "previous_state_hint": [0.0] * 24})
                    samples.append(
                        {
                            "frames": [roi_before, roi_after],
                            "roi_pairs": [(roi_before, roi_after)],
                            "temporal_roi_window": {
                                "roi_t_minus_1": prev_roi,
                                "roi_t": roi_before,
                                "roi_t_plus_1_target": roi_after,
                                "region_type": region_type,
                                "phase": phase,
                                "target_profile": tp if isinstance(tp, dict) else {},
                            },
                            "source": "manifest_video_renderer_primary",
                            "region_family": family,
                            "target_source": target_source,
                            "training_target_quality": training_target_quality,
                            "target_training_role": target_training_role,
                            "target_quality_warning": quality_warning,
                            "renderer_batch_contract": renderer_contract,
                            "delta_contract": rec.get("graph_delta_target", {}),
                            "temporal_transition_features": temporal_features,
                            "temporal_transition_target": {
                                "family": family,
                                "phase": phase,
                                "target_profile": tp if isinstance(tp, dict) else {},
                                "reveal_score": temporal_target["reveal_score"],
                                "occlusion_score": temporal_target["occlusion_score"],
                                "support_contact_score": temporal_target["support_contact_score"],
                            },
                            "human_state_transition_features": human_features,
                            "human_state_transition_target": human_target,
                            "human_state_history": human_history,
                            "graph_transition_contract": {
                                "planner_context": rec.get("planner_context", {}) if isinstance(rec.get("planner_context"), dict) else {},
                                "target_transition_context": rec.get("target_transition_context", {}) if isinstance(rec.get("target_transition_context"), dict) else {},
                                "memory_context": rec.get("memory_context", {}) if isinstance(rec.get("memory_context"), dict) else {},
                                "metadata": {"record_id": rec.get("record_id", f"video_transition_{idx}"), "region_type": region_type, "target_profile": tp},
                            },
                        }
                    )
                    diagnostics["loaded_records"] = int(diagnostics["loaded_records"]) + 1
                    diagnostics["missing_target_provenance_count"] = int(diagnostics.get("missing_target_provenance_count", 0)) + 1
                    diagnostics["family_counts"][family] = int(diagnostics["family_counts"].get(family, 0)) + 1
                    diagnostics["region_coverage"][region_type] = int(diagnostics["region_coverage"].get(region_type, 0)) + 1
                    diagnostics["phase_coverage"][phase] = int(diagnostics["phase_coverage"].get(phase, 0)) + 1
                    diagnostics["temporal_window_records"] = int(diagnostics["temporal_window_records"]) + 1
                flags = rec.get("fallback_flags", {}) if isinstance(rec.get("fallback_flags"), dict) else {}
                if not bool(flags.get("heuristic_priors_used", False)):
                    fallback_free += 1
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "error": str(exc)})
                if strict:
                    raise ValueError(f"video transition renderer record {idx} invalid: {exc}") from exc
        diagnostics["fallback_free_ratio"] = round(float(fallback_free / max(1, len(samples))), 6)
        diagnostics["temporal_window_usage_ratio"] = round(float(diagnostics["temporal_window_records"]) / max(1, len(samples)), 6)
        ds = cls(samples=samples)
        ds.diagnostics = diagnostics
        return ds

    @staticmethod
    def _as_hw3_tensor(value: object, field_name: str) -> list[list[list[float]]]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3:
            raise ValueError(f"{field_name} must be HxWxC tensor")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[2] != 3:
            raise ValueError(f"{field_name} channel count must be 3 or 1")
        return np.clip(arr, 0.0, 1.0).tolist()

    @staticmethod
    def _as_hw1_tensor(value: object, field_name: str) -> list[list[list[float]]]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3 or arr.shape[2] not in {1, 3}:
            raise ValueError(f"{field_name} must be HxW or HxWx1/HxWx3 tensor")
        if arr.shape[2] == 3:
            arr = np.mean(arr, axis=2, keepdims=True)
        return np.clip(arr, 0.0, 1.0).tolist()

    @staticmethod
    def _semantic_embed_from_family(family: str) -> list[float]:
        family = family.lower().strip()
        if family == "expression_transition":
            return [1.0, 0.0, 0.0, 0.9, 0.1, 0.2]
        if family in {"garment_transition", "visibility_transition"}:
            return [0.0, 1.0, 0.0, 0.2, 0.85, 0.4]
        if family == "interaction_transition":
            return [0.0, 0.0, 1.0, 0.4, 0.65, 0.85]
        return [0.0, 0.0, 1.0, 0.15, 0.45, 0.9]  # pose_transition

    @classmethod
    def from_renderer_manifest(cls, manifest_path: str, strict: bool = False) -> "RendererDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        manifest_type_raw = payload.get("manifest_type")
        manifest_type = str(manifest_type_raw).strip() if manifest_type_raw is not None else "unknown"
        contract_version = str(payload.get("contract_version", "")).strip()
        supported_manifest_types = {"renderer_patch_manifest", "manifest_paired_roi", "paired_roi_renderer_manifest", "unknown"}
        samples: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_paired_roi",
            "manifest_path": manifest_path,
            "manifest_type": manifest_type,
            "total_records": len(records),
            "loaded_records": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
            "warnings": [],
            "unsupported_manifest_type": 0,
            "unknown_manifest_type": 0,
            "invalid_memory_bundle_records": 0,
            "memory_bundle_warnings": [],
            "family_counts": {"face_expression": 0, "torso_reveal": 0, "sleeve_arm_transition": 0},
            "target_quality_counts": {"self_generated_runtime_target": 0, "external_or_observed_target": 0, "unknown": 0},
            "target_source_counts": {"runtime_output_patch": 0, "provided_ground_truth_roi": 0, "unknown": 0},
            "target_training_role_counts": {"supervised_external": 0, "bootstrap_self_generated": 0, "weak_unknown": 0},
            "missing_target_provenance_count": 0,
            "contains_legacy_unknown_target_provenance": False,
            "supervised_external_ratio": 0.0,
            "bootstrap_self_generated_ratio": 0.0,
            "weak_unknown_target_ratio": 0.0,
            "contains_self_generated_targets": False,
            "contains_external_targets": False,
            "contains_only_self_generated_targets": False,
            "contains_only_bootstrap_targets": False,
            "contains_no_supervised_external_targets": False,
            "trainable_record_count": 0,
            "supervised_record_count": 0,
            "bootstrap_record_count": 0,
            "weak_unknown_record_count": 0,
            "supervised_quality_present_count": 0,
            "supervised_quality_warning_record_count": 0,
            "supervised_quality_avg_changed_ratio": 0.0,
            "supervised_quality_avg_mean_abs_delta": 0.0,
            "supervised_quality_semantic_family_counts": {},
            "supervised_quality_warnings_count_by_type": {},
            "contract_version": contract_version,
            "v2_record_count": 0,
            "region_metadata_used_count": 0,
            "percent_records_with_region_metadata_used": 0.0,
            "metadata_completeness_score_sum": 0.0,
            "evidence_strength_score_sum": 0.0,
            "average_metadata_completeness_score": 0.0,
            "average_evidence_strength_score": 0.0,
            "roi_source_distribution": {},
            "source_node_type_distribution": {},
            "mask_kind_distribution": {},
            "fallback_person_bbox_record_count": 0,
        }
        if manifest_type not in supported_manifest_types:
            diagnostics["unsupported_manifest_type"] = 1
            diagnostics["unknown_manifest_type"] = 1
            diagnostics["warnings"].append({"type": "unknown_manifest_type", "manifest_type": manifest_type})
        elif manifest_type == "unknown":
            diagnostics["unknown_manifest_type"] = 1
            diagnostics["warnings"].append({"type": "unknown_manifest_type", "manifest_type": manifest_type})
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be a json object")
                if "roi_before" not in rec or "roi_after" not in rec:
                    raise ValueError("record requires roi_before and roi_after")

                roi_before = cls._as_hw3_tensor(rec["roi_before"], "roi_before")
                roi_after = cls._as_hw3_tensor(rec["roi_after"], "roi_after")
                b = np.asarray(roi_before, dtype=np.float32)
                a = np.asarray(roi_after, dtype=np.float32)
                if b.shape != a.shape:
                    raise ValueError("roi_before and roi_after shape mismatch")

                family = str(rec.get("semantic_family", rec.get("region_family", ""))).strip().lower()
                region_id = str(rec.get("region_id", rec.get("region", {}).get("region_id", ""))).strip()
                region_id_for_family = region_id.lower()
                if not family:
                    if "face" in region_id_for_family or "head" in region_id_for_family:
                        family = "face_expression"
                    elif "torso" in region_id_for_family or "inner" in region_id_for_family:
                        family = "torso_reveal"
                    elif "arm" in region_id_for_family or "sleeve" in region_id_for_family or "outer" in region_id_for_family:
                        family = "sleeve_arm_transition"
                    else:
                        family = "sleeve_arm_transition"
                if family not in diagnostics["family_counts"]:
                    raise ValueError(f"unsupported semantic_family={family!r}")
                diagnostics["family_counts"][family] = int(diagnostics["family_counts"][family]) + 1

                rec_contract_version = str(rec.get("contract_version", contract_version)).strip()
                is_v2 = rec_contract_version == "renderer_patch_manifest_v2"
                region_metadata = rec.get("region_metadata") if isinstance(rec.get("region_metadata"), dict) else None
                patch_contract = rec.get("patch_synthesis_contract") if isinstance(rec.get("patch_synthesis_contract"), dict) else {}
                if region_metadata is None and isinstance(patch_contract.get("region_metadata"), dict):
                    region_metadata = copy.deepcopy(patch_contract["region_metadata"])
                if is_v2:
                    diagnostics["v2_record_count"] = int(diagnostics.get("v2_record_count", 0)) + 1
                    required_v2 = {"record_id", "frame_index", "step_index", "region_id", "canonical_region", "entity_id", "roi_before", "roi_after", "alpha_mask", "region_metadata", "transition_context_summary", "selected_render_strategy", "synthesis_mode", "execution_trace_summary", "metadata_completeness_score", "evidence_strength_score", "roi_source", "source_node_type", "mask_kind", "mask_ref_present"}
                    missing_v2 = sorted(required_v2.difference(rec.keys()))
                    if missing_v2:
                        raise ValueError(f"renderer manifest v2 missing required fields: {missing_v2}")
                    if not isinstance(region_metadata, dict) or not region_metadata:
                        raise ValueError("renderer manifest v2 requires non-empty region_metadata")
                    if "metadata_completeness_score" not in rec or "metadata_completeness_score" not in region_metadata:
                        raise ValueError("renderer manifest v2 requires metadata_completeness_score")
                elif strict and contract_version == "renderer_patch_manifest_v2":
                    raise ValueError("manifest declares renderer_patch_manifest_v2 but record is not v2")
                if not isinstance(region_metadata, dict):
                    region_metadata = {"region_id": region_id, "metadata_completeness_score": 0.0, "evidence_strength_score": 0.0, "missing_fields": ["region_metadata"]}
                metadata_completeness = cls._safe_float(rec.get("metadata_completeness_score", region_metadata.get("metadata_completeness_score", 0.0)), 0.0)
                evidence_strength = cls._safe_float(rec.get("evidence_strength_score", region_metadata.get("evidence_strength_score", 0.0)), 0.0)
                roi_source = str(rec.get("roi_source", region_metadata.get("roi_source", "unknown")))
                source_node_type = str(rec.get("source_node_type", region_metadata.get("source_node_type", "unknown")))
                mask_kind = str(rec.get("mask_kind", region_metadata.get("mask_kind", "")))
                mask_ref_present = bool(rec.get("mask_ref_present", bool(region_metadata.get("mask_ref"))))
                if region_metadata and metadata_completeness > 0.0:
                    diagnostics["region_metadata_used_count"] = int(diagnostics.get("region_metadata_used_count", 0)) + 1
                diagnostics["metadata_completeness_score_sum"] = float(diagnostics.get("metadata_completeness_score_sum", 0.0)) + metadata_completeness
                diagnostics["evidence_strength_score_sum"] = float(diagnostics.get("evidence_strength_score_sum", 0.0)) + evidence_strength
                for dist_key, value in (("roi_source_distribution", roi_source), ("source_node_type_distribution", source_node_type), ("mask_kind_distribution", mask_kind)):
                    dist = diagnostics.get(dist_key, {})
                    if isinstance(dist, dict):
                        dist[value] = int(dist.get(value, 0)) + 1
                if roi_source == "person_bbox_fallback":
                    diagnostics["fallback_person_bbox_record_count"] = int(diagnostics.get("fallback_person_bbox_record_count", 0)) + 1

                changed_mask = rec.get("changed_mask")
                if changed_mask is None:
                    changed_mask = np.clip(np.mean(np.abs(a - b), axis=2, keepdims=True) * 3.0, 0.0, 1.0).tolist()
                alpha_target = rec.get("alpha_target", rec.get("alpha_mask"))
                if alpha_target is None:
                    alpha_target = np.clip(0.15 + 0.85 * np.asarray(changed_mask, dtype=np.float32), 0.0, 1.0).tolist()
                blend_hint = rec.get("blend_hint")
                if blend_hint is None:
                    blend_hint = np.clip(0.2 + 0.75 * np.asarray(changed_mask, dtype=np.float32), 0.0, 1.0).tolist()

                changed_mask = cls._as_hw1_tensor(changed_mask, "changed_mask")
                alpha_target = cls._as_hw1_tensor(alpha_target, "alpha_target")
                blend_hint = cls._as_hw1_tensor(blend_hint, "blend_hint")

                missing_target_provenance = "target_source" not in rec and "training_target_quality" not in rec
                if missing_target_provenance:
                    diagnostics["missing_target_provenance_count"] = int(diagnostics.get("missing_target_provenance_count", 0)) + 1
                    diagnostics["contains_legacy_unknown_target_provenance"] = True
                target_source = cls._normalize_target_source(rec.get("target_source", "unknown"))
                training_target_quality = cls._normalize_training_target_quality(rec.get("training_target_quality", "unknown"))
                diagnostics["target_quality_counts"][training_target_quality] = int(diagnostics["target_quality_counts"].get(training_target_quality, 0)) + 1
                diagnostics["target_source_counts"][target_source] = int(diagnostics["target_source_counts"].get(target_source, 0)) + 1
                diagnostics["contains_self_generated_targets"] = bool(diagnostics["contains_self_generated_targets"]) or training_target_quality == "self_generated_runtime_target"
                diagnostics["contains_external_targets"] = bool(diagnostics["contains_external_targets"]) or training_target_quality == "external_or_observed_target"
                target_training_role = classify_target_training_role(training_target_quality)
                quality_warning = target_quality_warning(training_target_quality)
                diagnostics["target_training_role_counts"][target_training_role] = int(diagnostics["target_training_role_counts"].get(target_training_role, 0)) + 1

                supervised_quality = rec.get("supervised_quality") if isinstance(rec.get("supervised_quality"), dict) else None
                if supervised_quality is not None:
                    sq_copy = copy.deepcopy(supervised_quality)
                    diagnostics["supervised_quality_present_count"] = int(diagnostics.get("supervised_quality_present_count", 0)) + 1
                    sq_warnings = sq_copy.get("warnings", []) if isinstance(sq_copy.get("warnings", []), list) else []
                    if sq_warnings:
                        diagnostics["supervised_quality_warning_record_count"] = int(diagnostics.get("supervised_quality_warning_record_count", 0)) + 1
                    family_key = str(sq_copy.get("semantic_family", family))
                    sq_family_counts = diagnostics.get("supervised_quality_semantic_family_counts", {})
                    if isinstance(sq_family_counts, dict) and family_key:
                        sq_family_counts[family_key] = int(sq_family_counts.get(family_key, 0)) + 1
                    sq_warning_counts = diagnostics.get("supervised_quality_warnings_count_by_type", {})
                    if isinstance(sq_warning_counts, dict):
                        for warning in sq_warnings:
                            warning_text = str(warning)
                            sq_warning_counts[warning_text] = int(sq_warning_counts.get(warning_text, 0)) + 1
                    diagnostics["_supervised_quality_changed_ratio_sum"] = float(diagnostics.get("_supervised_quality_changed_ratio_sum", 0.0)) + float(sq_copy.get("changed_ratio", 0.0) or 0.0)
                    diagnostics["_supervised_quality_mean_abs_delta_sum"] = float(diagnostics.get("_supervised_quality_mean_abs_delta_sum", 0.0)) + float(sq_copy.get("mean_abs_delta", 0.0) or 0.0)

                renderer_contract = {
                    "semantic_embed": rec.get("semantic_embed", cls._semantic_embed_from_family(family)),
                    "delta_cond": rec.get("delta_cond", rec.get("graph_delta_cond", [0.0] * 9)),
                    "planner_cond": rec.get("planner_cond", rec.get("transition_context", {}).get("planner_cond", [0.0] * 8)),
                    "graph_cond": rec.get("graph_cond", rec.get("graph_context", [0.0] * 7)),
                    "memory_cond": rec.get("memory_cond", rec.get("memory_context", [0.0] * 8)),
                    "appearance_cond": rec.get("appearance_cond", np.concatenate([np.mean(b, axis=(0, 1)), np.std(b, axis=(0, 1))]).tolist()),
                    "bbox_cond": rec.get("bbox_cond", rec.get("bbox", [0.2, 0.2, 0.4, 0.4])),
                    "alpha_target": alpha_target,
                    "blend_hint": blend_hint,
                    "changed_mask": changed_mask,
                    "preservation_mask": cls._as_hw1_tensor(rec["preservation_mask"], "preservation_mask") if rec.get("preservation_mask") is not None else None,
                    "uncertainty_target": cls._as_hw1_tensor(rec["uncertainty_target"], "uncertainty_target") if rec.get("uncertainty_target") is not None else None,
                    "seam_prior": cls._as_hw1_tensor(rec["seam_prior"], "seam_prior") if rec.get("seam_prior") is not None else None,
                    "target_source": target_source,
                    "training_target_quality": training_target_quality,
                    "target_training_role": target_training_role,
                    "conditioning_summary": {
                        "region_metadata_used": bool(region_metadata and metadata_completeness > 0.0),
                        "metadata_completeness_score": metadata_completeness,
                        "evidence_strength_score": evidence_strength,
                        "metadata_raw_keys": [str(k) for k in region_metadata.keys() if k not in {"missing_fields", "metadata_source_trace"}],
                        "mask_ref_present": mask_ref_present,
                        "roi_source": roi_source,
                        "source_node_type": source_node_type,
                        "mask_kind": mask_kind,
                    },
                }
                if supervised_quality is not None:
                    renderer_contract["supervised_quality"] = sq_copy
                if quality_warning is not None:
                    renderer_contract["target_quality_warning"] = quality_warning
                renderer_memory_bundle = cls._normalize_renderer_memory_bundle(rec, diagnostics, idx)
                renderer_contract["region_memory_bundle_serialized"] = renderer_memory_bundle

                sample: TrainingSample = {
                        "frames": [roi_before, roi_after],
                        "roi_pairs": [(roi_before, roi_after)],
                        "source": str(rec.get("source", "manifest_paired_roi")),
                        "region_family": family,
                        "region_id": region_id,
                        "target_source": target_source,
                        "training_target_quality": training_target_quality,
                        "target_training_role": target_training_role,
                        "renderer_batch_contract": renderer_contract,
                        "delta_contract": rec.get("graph_delta", rec.get("delta_contract", {})),
                        "graph_transition_contract": rec.get("graph_transition_contract", {}),
                        "memory_records": rec.get("memory_records", []),
                        "patch_synthesis_contract": {**(rec.get("patch_synthesis_contract", {}) if isinstance(rec.get("patch_synthesis_contract", {}), dict) else {}), "region_metadata": region_metadata},
                        "region_metadata": region_metadata,
                        "transition_context_summary": rec.get("transition_context_summary", rec.get("transition_context", {})),
                        "selected_render_strategy": rec.get("selected_render_strategy", "unknown"),
                        "synthesis_mode": rec.get("synthesis_mode", "unknown"),
                        "execution_trace_summary": rec.get("execution_trace_summary", {}),
                        "metadata_completeness_score": metadata_completeness,
                        "evidence_strength_score": evidence_strength,
                        "roi_source": roi_source,
                        "source_node_type": source_node_type,
                        "mask_kind": mask_kind,
                        "mask_ref_present": mask_ref_present,
                        "canonical_region": rec.get("canonical_region", region_metadata.get("canonical_region", "")),
                        "entity_id": rec.get("entity_id", region_metadata.get("entity_id", "unknown")),
                        "record_id": rec.get("record_id", f"legacy_renderer_{idx}"),
                        "frame_index": int(rec.get("frame_index", 0) or 0),
                        "step_index": int(rec.get("step_index", -1) if rec.get("step_index") is not None else -1),
                    }
                if supervised_quality is not None:
                    sample["supervised_quality"] = sq_copy
                if quality_warning is not None:
                    sample["target_quality_warning"] = quality_warning
                samples.append(sample)
                diagnostics["loaded_records"] = int(diagnostics["loaded_records"]) + 1
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "error": str(exc)})
                if strict:
                    raise ValueError(f"renderer manifest record {idx} is invalid: {exc}") from exc

        loaded = int(diagnostics.get("loaded_records", 0))
        role_counts = diagnostics.get("target_training_role_counts", {}) if isinstance(diagnostics.get("target_training_role_counts", {}), dict) else {}
        supervised_count = int(role_counts.get("supervised_external", 0))
        bootstrap_count = int(role_counts.get("bootstrap_self_generated", 0))
        weak_unknown_count = int(role_counts.get("weak_unknown", 0))
        self_generated_count = int(diagnostics["target_quality_counts"].get("self_generated_runtime_target", 0))
        diagnostics["supervised_external_ratio"] = round(float(supervised_count) / max(1.0, float(loaded)), 6)
        diagnostics["bootstrap_self_generated_ratio"] = round(float(bootstrap_count) / max(1.0, float(loaded)), 6)
        diagnostics["weak_unknown_target_ratio"] = round(float(weak_unknown_count) / max(1.0, float(loaded)), 6)
        diagnostics["trainable_record_count"] = loaded
        diagnostics["supervised_record_count"] = supervised_count
        diagnostics["bootstrap_record_count"] = bootstrap_count
        diagnostics["weak_unknown_record_count"] = weak_unknown_count
        diagnostics["contains_only_self_generated_targets"] = bool(loaded > 0 and self_generated_count == loaded)
        diagnostics["contains_only_bootstrap_targets"] = bool(loaded > 0 and bootstrap_count == loaded)
        diagnostics["contains_no_supervised_external_targets"] = bool(loaded > 0 and supervised_count == 0)
        sq_present = int(diagnostics.get("supervised_quality_present_count", 0) or 0)
        diagnostics["supervised_quality_avg_changed_ratio"] = round(float(diagnostics.get("_supervised_quality_changed_ratio_sum", 0.0) or 0.0) / max(1.0, float(sq_present)), 6)
        diagnostics["supervised_quality_avg_mean_abs_delta"] = round(float(diagnostics.get("_supervised_quality_mean_abs_delta_sum", 0.0) or 0.0) / max(1.0, float(sq_present)), 6)
        diagnostics["percent_records_with_region_metadata_used"] = round(float(diagnostics.get("region_metadata_used_count", 0) or 0) / max(1.0, float(loaded)), 6)
        diagnostics["average_metadata_completeness_score"] = round(float(diagnostics.get("metadata_completeness_score_sum", 0.0) or 0.0) / max(1.0, float(loaded)), 6)
        diagnostics["average_evidence_strength_score"] = round(float(diagnostics.get("evidence_strength_score_sum", 0.0) or 0.0) / max(1.0, float(loaded)), 6)
        diagnostics.pop("_supervised_quality_changed_ratio_sum", None)
        diagnostics.pop("_supervised_quality_mean_abs_delta_sum", None)
        warnings = diagnostics.get("warnings")
        if diagnostics["contains_no_supervised_external_targets"] and isinstance(warnings, list):
            warnings.append(
                {
                    "type": "no_supervised_external_targets",
                    "message": "renderer manifest has no external/observed targets; use only for bootstrap/eval unless explicitly intended",
                }
            )
        if diagnostics["contains_only_self_generated_targets"]:
            diagnostics["warning"] = "renderer manifest contains only self-generated runtime targets; use for bootstrap/eval cautiously"
            if isinstance(warnings, list):
                warnings.append({"type": "self_generated_only_targets", "message": diagnostics["warning"]})
        ds = cls(samples=samples)
        ds.diagnostics = diagnostics
        return ds

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _bbox_xywh_to_bbox(value: object) -> BBox:
        if isinstance(value, dict):
            missing = [key for key in ("x", "y", "w", "h") if key not in value]
            if missing:
                raise ValueError(f"bbox_xywh dict missing keys: {missing}")
            return BBox(float(value["x"]), float(value["y"]), float(value["w"]), float(value["h"]))
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return BBox(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        raise ValueError("bbox_xywh must be a dict with x/y/w/h or a 4-item list/tuple")

    @staticmethod
    def sample_to_patch_request(sample: TrainingSample) -> PatchSynthesisRequest:
        """Reconstruct a PatchSynthesisRequest-compatible object from a manifest sample."""

        metadata = sample.get("region_metadata", {}) if isinstance(sample.get("region_metadata", {}), dict) else {}
        region_id = str(sample.get("region_id", metadata.get("region_id", "unknown:region")))
        bbox_raw = metadata.get("bbox_xywh", sample.get("canonical_region_bbox_xywh", [0.2, 0.2, 0.4, 0.4]))
        try:
            bbox = RendererDataset._bbox_xywh_to_bbox(bbox_raw)
        except (TypeError, ValueError, OverflowError):
            bbox = BBox(0.2, 0.2, 0.4, 0.4)
        roi_pairs = sample.get("roi_pairs") or []
        before = roi_pairs[0][0] if roi_pairs else (sample.get("frames", [[[]]])[0])
        transition_context = sample.get("transition_context_summary", {}) if isinstance(sample.get("transition_context_summary", {}), dict) else {}
        contract = sample.get("renderer_batch_contract", {}) if isinstance(sample.get("renderer_batch_contract", {}), dict) else {}
        if contract:
            transition_context = dict(transition_context, renderer_batch_contract=contract)
            if "expected_reference_payload" in contract:
                transition_context["expected_reference_payload"] = contract.get("expected_reference_payload")
            if "expected_reference_patch_material" in contract:
                transition_context["expected_reference_patch_material"] = contract.get("expected_reference_patch_material")
            if "reference_patch_material_trace_reasons" in contract:
                transition_context["reference_patch_material_trace_reasons"] = contract.get("reference_patch_material_trace_reasons")
        else:
            if "expected_reference_payload" in sample:
                transition_context = dict(transition_context, expected_reference_payload=sample.get("expected_reference_payload"))
            if "expected_reference_patch_material" in sample:
                transition_context = dict(transition_context, expected_reference_patch_material=sample.get("expected_reference_patch_material"))
        return PatchSynthesisRequest(
            region=RegionRef(region_id=region_id, bbox=bbox, reason=str(metadata.get("roi_reason", sample.get("source", "manifest_renderer_record")))),
            scene_state=SceneGraph(frame_index=int(sample.get("frame_index", 0) or 0)),
            memory_summary={},
            transition_context=transition_context,
            retrieval_summary=sample.get("memory_retrieval_evidence", {}) if isinstance(sample.get("memory_retrieval_evidence", {}), dict) else {},
            current_frame=before,
            region_metadata=dict(metadata),
        )

    @classmethod
    def synthetic(cls, size: int) -> "RendererDataset":
        from utils_tensor import zeros

        samples: list[TrainingSample] = []
        families = ["face_expression", "torso_reveal", "sleeve_arm_transition"]
        for idx in range(size):
            family = families[idx % len(families)]
            before = zeros(16, 16, 3, value=0.25)
            after = zeros(16, 16, 3, value=0.25)
            for y in range(16):
                for x in range(16):
                    if family == "face_expression" and 6 <= y <= 11 and 4 <= x <= 11:
                        after[y][x] = [0.6, 0.35, 0.35]
                    elif family == "torso_reveal" and x in {7, 8}:
                        after[y][x] = [0.8, 0.78, 0.75]
                    elif family == "sleeve_arm_transition" and 3 <= x <= 10 and y >= 6:
                        after[y][x] = [0.45, 0.5, 0.7]
            yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 16), np.linspace(0.0, 1.0, 16), indexing="ij")
            changed = np.clip(np.mean(np.abs(np.asarray(after) - np.asarray(before)), axis=2, keepdims=True) * 3.0, 0.0, 1.0)
            blend_hint = np.clip(0.2 + 0.75 * changed + 0.05 * (1.0 - np.abs(xx - 0.5))[..., None], 0.0, 1.0)
            semantic = [1.0, 0.0, 0.0, 0.9, 0.1, 0.2] if family == "face_expression" else ([0.0, 1.0, 0.0, 0.2, 0.85, 0.4] if family == "torso_reveal" else [0.0, 0.0, 1.0, 0.15, 0.45, 0.9])
            samples.append(
                {
                    "frames": [zeros(64, 64, 3), zeros(64, 64, 3)],
                    "roi_pairs": [(before, after)],
                    "source": "synthetic",
                    "region_family": family,
                    "target_source": "runtime_output_patch",
                    "training_target_quality": "self_generated_runtime_target",
                    "target_training_role": "bootstrap_self_generated",
                    "renderer_batch_contract": {
                        "target_source": "runtime_output_patch",
                        "training_target_quality": "self_generated_runtime_target",
                        "target_training_role": "bootstrap_self_generated",
                        "semantic_embed": semantic,
                        "delta_cond": [0.2 + 0.1 * idx, 0.1, 0.2 if family != "face_expression" else 0.05, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        "planner_cond": [min(1.0, idx / max(1, size)), 0.0, 1.0, 0.0, 1.0 if family == "torso_reveal" else 0.0, 1.0 if family == "sleeve_arm_transition" else 0.0, 1.0 if family == "face_expression" else 0.0, 0.0],
                        "graph_cond": [1.0, 0.2, 0.1, 0.53, 0.5, 0.5, 0.15],
                        "memory_cond": [0.6, 0.4, 0.3, 0.5, 0.2, 0.35, 0.5, 1.0 if family == "torso_reveal" else 0.0],
                        "appearance_cond": [0.25, 0.25, 0.25, 0.02, 0.02, 0.02],
                        "bbox_cond": [0.2, 0.2, 0.4, 0.4],
                        "alpha_target": np.clip(0.15 + 0.85 * changed, 0.0, 1.0).tolist(),
                        "blend_hint": blend_hint.tolist(),
                    },
                }
            )
        ds = cls(samples=samples)
        ds.diagnostics = {
            "source": "synthetic_bootstrap",
            "total_records": size,
            "loaded_records": size,
            "invalid_records": 0,
            "skipped_records": 0,
            "family_counts": {
                "face_expression": sum(1 for s in samples if s["region_family"] == "face_expression"),
                "torso_reveal": sum(1 for s in samples if s["region_family"] == "torso_reveal"),
                "sleeve_arm_transition": sum(1 for s in samples if s["region_family"] == "sleeve_arm_transition"),
            },
            "invalid_examples": [],
            "target_quality_counts": {"self_generated_runtime_target": size, "external_or_observed_target": 0, "unknown": 0},
            "target_source_counts": {"runtime_output_patch": size, "provided_ground_truth_roi": 0, "unknown": 0},
            "target_training_role_counts": {"supervised_external": 0, "bootstrap_self_generated": size, "weak_unknown": 0},
            "supervised_external_ratio": 0.0,
            "bootstrap_self_generated_ratio": 1.0 if size else 0.0,
            "weak_unknown_target_ratio": 0.0,
            "contains_no_supervised_external_targets": bool(size > 0),
            "contains_only_bootstrap_targets": bool(size > 0),
        }
        return ds

    @classmethod
    def from_frame_pairs(cls, before_frames: list[list], after_frames: list[list]) -> "RendererDataset":
        pairs = list(zip(before_frames, after_frames))
        samples: list[TrainingSample] = []
        for before, after in pairs:
            samples.append({"frames": [before, after], "roi_pairs": [(before, after)], "source": "real_pair"})
        return cls(samples=samples)

    @classmethod
    def from_video_manifest(cls, manifest_path: str, quality_profile: str = "debug") -> "RendererDataset":
        from training.learned_contracts import build_patch_synthesis_contract, build_temporal_consistency_contract

        layer = InputAssetLayer()
        selector = ROISelector()
        records = json.loads(Path(manifest_path).read_text()).get("records", [])
        out: list[TrainingSample] = []
        for rec in records:
            req = layer.build_request(images=rec.get("images", []), text=rec.get("text", ""), quality_profile=quality_profile)
            frames = [f.tensor for f in (req.unified_asset.frames if req.unified_asset else [])]
            if len(frames) < 2:
                continue
            person = PersonNode(person_id="person_1", track_id="track_1", bbox=BBox(0.2, 0.1, 0.6, 0.8), mask_ref=None)
            graph = SceneGraph(frame_index=0, persons=[person])
            delta = GraphDelta(affected_entities=[person.person_id], semantic_reasons=rec.get("reasons", []), affected_regions=rec.get("affected_regions", []))
            rois = selector.select(graph, delta)
            roi_pairs = [(frames[0], frames[1]) for _ in rois] or [(frames[0], frames[1])]
            patch_contract = None
            temporal_contract = None
            if rois:
                patch_contract = build_patch_synthesis_contract(
                    roi_before=frames[0],
                    roi_after=frames[1],
                    region=rois[0],
                    retrieval_summary=rec.get("retrieval_summary", "manifest"),
                    selected_render_strategy=rec.get(
                        "selected_render_strategy", rec.get("selected_strategy", "fallback")
                    ),
                    hidden_state=rec.get("hidden_lifecycle_state", {}),
                    synthesis_mode=rec.get("synthesis_mode", "deterministic"),
                    transition_context=rec.get("transition_context", {}),
                )
                temporal_contract = build_temporal_consistency_contract(
                    previous_frame=frames[0],
                    composed_frame=frames[1],
                    target_frame=frames[1],
                    changed_regions=rois,
                    region_consistency_metadata=rec.get("region_consistency_metadata", {}),
                    scene_transition_context=rec.get("scene_transition_context", {}),
                    memory_transition_context=rec.get("memory_transition_context", {}),
                )
            sample: TrainingSample = {
                "frames": [frames[0], frames[1]],
                "roi_pairs": roi_pairs,
                "patch_synthesis_contract": patch_contract,
                "temporal_consistency_contract": temporal_contract,
                "renderer_batch_contract": {
                    "semantic_embed": rec.get("semantic_embed", [0.0, 1.0, 0.0, 0.25, 0.8, 0.35]),
                    "delta_cond": rec.get("delta_cond", [0.3, 0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
                    "planner_cond": rec.get("planner_cond", [0.2, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    "graph_cond": rec.get("graph_cond", [1.0, 0.2, 0.15, 0.5, 0.5, 0.5, 0.2]),
                    "memory_cond": rec.get("memory_cond", [0.5, 0.4, 0.2, 0.6, 0.3, 0.2, 0.4, 0.0]),
                    "appearance_cond": rec.get("appearance_cond", [0.3, 0.3, 0.3, 0.05, 0.05, 0.05]),
                    "bbox_cond": rec.get("bbox_cond", [0.2, 0.2, 0.4, 0.4]),
                },
                "source": rec.get("source", "video_manifest"),
            }
            out.append(sample)
        return cls(samples=out)

    @classmethod
    def _from_observed_graph_delta_manifest(cls, payload: dict[str, object], manifest_path: str, strict: bool = False) -> "DynamicsDataset":
        records = payload.get("records", []) if isinstance(payload.get("records", []), list) else []
        out: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_dynamics_graph_delta_supervised",
            "manifest_path": manifest_path,
            "strict": bool(strict),
            "total_records": len(records),
            "supervised_dynamics_records": 0,
            "skipped_records": 0,
            "skipped_by_reason": {},
            "rejected_runtime_or_heuristic_targets": 0,
            "transition_family_counts": {},
            "affected_region_counts": {},
        }
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be object")
                ts = str(rec.get("target_source", ""))
                tq = str(rec.get("training_target_quality", ""))
                tr = str(rec.get("target_training_role", ""))
                if ts in {"heuristic_runtime_prediction", "legacy_graph_delta_fallback", "self_generated_runtime_target"}:
                    diagnostics["rejected_runtime_or_heuristic_targets"] = int(diagnostics["rejected_runtime_or_heuristic_targets"]) + 1
                    raise ValueError("runtime_or_heuristic_target_forbidden")
                if ts != "provided_ground_truth_graph_transition" or tq != "external_or_observed_graph_transition" or tr != "supervised_dynamics_external":
                    raise ValueError("invalid_supervised_dynamics_target_policy")
                gb = rec.get("graph_before")
                td = rec.get("target_delta")
                if not isinstance(gb, dict):
                    raise ValueError("missing_graph_before")
                if not isinstance(td, dict):
                    raise ValueError("missing_target_delta")
                before = _deserialize_graph(gb)
                delta = GraphDelta(
                    pose_deltas={str(k): float(v) for k, v in (td.get("pose_deltas", {}) or {}).items()},
                    garment_deltas={str(k): float(v) for k, v in (td.get("garment_deltas", {}) or {}).items()},
                    visibility_deltas={str(k): str(v) for k, v in (td.get("visibility_deltas", {}) or {}).items()},
                    expression_deltas={str(k): float(v) for k, v in (td.get("expression_deltas", {}) or {}).items()},
                    interaction_deltas={str(k): float(v) for k, v in (td.get("interaction_deltas", {}) or {}).items()},
                    affected_entities=[str(x) for x in td.get("affected_entities", ["scene"])],
                    affected_regions=[str(x) for x in td.get("affected_regions", rec.get("affected_regions", []))],
                    semantic_reasons=[str(x) for x in td.get("semantic_reasons", [str((rec.get("transition_context", {}) or {}).get("family", "pose_transition"))])],
                    region_transition_mode={str(k): str(v) for k, v in (td.get("region_transition_mode", {}) or {}).items()},
                )
                if not (delta.pose_deltas or delta.garment_deltas or delta.visibility_deltas or delta.expression_deltas or delta.interaction_deltas):
                    raise ValueError("empty_target_delta_groups")
                fam = str((rec.get("transition_context", {}) or {}).get("family", "pose_transition"))
                diagnostics["transition_family_counts"][fam] = int(diagnostics["transition_family_counts"].get(fam, 0)) + 1
                for r in rec.get("affected_regions", []) if isinstance(rec.get("affected_regions", []), list) else []:
                    diagnostics["affected_region_counts"][str(r)] = int(diagnostics["affected_region_counts"].get(str(r), 0)) + 1
                out.append({"graphs": [before, apply_delta(before, delta)], "actions": [ActionStep(type=fam, priority=1)], "deltas": [delta], "source": "manifest_dynamics_graph_delta_supervised", "graph_transition_contract": {"metadata": {"transition_family": fam}}, "target_source": ts, "training_target_quality": tq, "target_training_role": tr})
                diagnostics["supervised_dynamics_records"] = int(diagnostics["supervised_dynamics_records"]) + 1
            except Exception as exc:
                if strict:
                    raise
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                diagnostics["skipped_by_reason"][str(exc)] = int(diagnostics["skipped_by_reason"].get(str(exc), 0)) + 1
        if not out:
            raise ValueError("no supervised_dynamics_external records")
        ds = cls(samples=out)
        ds.diagnostics = diagnostics
        return ds


class TemporalDataset(BaseStageDataset):
    @staticmethod
    def _as_hw3_tensor(value: object, field_name: str) -> list[list[list[float]]]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3:
            raise ValueError(f"{field_name} must be HxWxC tensor")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[2] != 3:
            raise ValueError(f"{field_name} channel count must be 3 or 1")
        return np.clip(arr, 0.0, 1.0).tolist()

    @staticmethod
    def _as_hw1_tensor(value: object, field_name: str, shape_hint: tuple[int, int] | None = None) -> list[list[list[float]]]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f"{field_name} must be HxW or HxWx1 tensor")
        if arr.shape[2] != 1:
            raise ValueError(f"{field_name} channel count must be 1")
        if shape_hint is not None and arr.shape[:2] != shape_hint:
            raise ValueError(f"{field_name} shape {arr.shape[:2]} mismatch expected {shape_hint}")
        return np.clip(arr, 0.0, 1.0).tolist()

    @classmethod
    def synthetic(cls, size: int) -> "TemporalDataset":
        samples: list[TrainingSample] = []
        for idx in range(size):
            base = 0.22 + 0.04 * (idx % 4)
            prev = np.full((24, 24, 3), base, dtype=np.float32)
            cur = prev.copy()
            cur[5:18, 7:20, :] = np.array([min(1.0, base + 0.2), base + 0.05, base + 0.03], dtype=np.float32)
            target = cur.copy()
            target[6:17, 8:19, :] = np.array([min(1.0, base + 0.17), base + 0.04, base + 0.03], dtype=np.float32)
            rid = RegionRef(region_id=f"p1:region_{idx%3}", bbox=BBox(0.25, 0.2, 0.45, 0.45), reason="temporal_drift")
            samples.append(
                {
                    "frames": [prev.tolist(), cur.tolist(), target.tolist()],
                    "temporal_consistency_contract": {
                        "previous_frame": prev.tolist(),
                        "composed_frame": cur.tolist(),
                        "target_frame": target.tolist(),
                        "changed_regions": [{"region_id": rid.region_id, "reason": rid.reason, "bbox": {"x": rid.bbox.x, "y": rid.bbox.y, "w": rid.bbox.w, "h": rid.bbox.h}}],
                        "region_consistency_metadata": {"alpha_mean": 0.55, "confidence_mean": 0.72},
                        "scene_transition_context": {"transition_phase": "motion", "step_index": idx + 1},
                        "memory_transition_context": {"drift": [0.02, 0.05, 0.1][idx % 3]},
                    },
                    "source": "synthetic_temporal",
                }
            )
        ds = cls(samples=samples)
        ds.diagnostics = {"source": "synthetic_temporal", "loaded_records": len(samples), "invalid_records": 0, "skipped_records": 0}
        return ds

    @classmethod
    def from_temporal_manifest(cls, manifest_path: str, strict: bool = False) -> "TemporalDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        samples: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_temporal_consistency",
            "strict": bool(strict),
            "supervised_temporal_records": 0,
            "rejected_self_generated_targets": 0,
            "skipped_by_reason": {},
            "sequence_count": 0,
            "frame_shape_summary": {},
            "manifest_path": manifest_path,
            "total_records": len(records),
            "loaded_records": 0,
            "usable_records": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
            "family_counts": {
                "frame_refinement": 0,
                "flicker": 0,
                "region_stability": 0,
                "alpha_consistency": 0,
                "confidence_calibration": 0,
                "history_conditioning": 0,
                "memory_conditioning": 0,
                "transition_conditioning": 0,
            },
            "tag_counts": {},
            "scenario_counts": {},
        }
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be a json object")
                target_source = rec.get("target_source")
                target_quality = rec.get("training_target_quality")
                target_role = rec.get("target_training_role")
                if target_source is None or target_quality is None or target_role is None:
                    raise ValueError("missing target provenance fields")
                target_source = str(target_source)
                target_quality = str(target_quality)
                target_role = str(target_role)
                if target_source in {"self_generated_runtime_target", "runtime_generated_frame", "bootstrap_temporal_target"}:
                    diagnostics["rejected_self_generated_targets"] = int(diagnostics["rejected_self_generated_targets"]) + 1
                    raise ValueError("self/runtime generated temporal targets are forbidden")
                if target_source != "provided_ground_truth_temporal_frame" or target_quality != "external_or_observed_temporal_target" or target_role != "supervised_temporal_external":
                    raise ValueError("invalid supervised temporal target policy")
                prev = cls._as_hw3_tensor(rec["previous_frame"], "previous_frame")
                cur = cls._as_hw3_tensor(rec.get("current_composed_frame", rec.get("current_frame")), "current_composed_frame")
                target = cls._as_hw3_tensor(rec.get("target_refined_frame", rec.get("target_frame", rec.get("current_composed_frame", rec.get("current_frame")))), "target_frame")
                if np.asarray(prev, dtype=np.float32).shape != np.asarray(cur, dtype=np.float32).shape:
                    raise ValueError("previous_frame and current_composed_frame shape mismatch")
                if np.asarray(target, dtype=np.float32).shape != np.asarray(cur, dtype=np.float32).shape:
                    raise ValueError("target_frame shape mismatch")
                changed_regions_raw = rec.get("changed_regions", [])
                changed_regions: list[dict[str, object]] = []
                for ridx, region in enumerate(changed_regions_raw if isinstance(changed_regions_raw, list) else []):
                    if not isinstance(region, dict):
                        continue
                    bb = region.get("bbox", {}) if isinstance(region.get("bbox", {}), dict) else {}
                    changed_regions.append(
                        {
                            "region_id": str(region.get("region_id", f"scene:region_{ridx}")),
                            "reason": str(region.get("reason", "temporal_drift")),
                            "bbox": {
                                "x": float(bb.get("x", 0.2)),
                                "y": float(bb.get("y", 0.2)),
                                "w": float(bb.get("w", 0.3)),
                                "h": float(bb.get("h", 0.3)),
                            },
                        }
                    )
                changed_region_mask = rec.get("changed_region_mask")
                if changed_region_mask is not None:
                    mask = cls._as_hw1_tensor(changed_region_mask, "changed_region_mask", shape_hint=np.asarray(cur, dtype=np.float32).shape[:2])
                    if not changed_regions:
                        changed_regions = [{"region_id": "scene:mask_region", "reason": "mask_supervision", "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}]
                if not changed_regions:
                    changed_regions = [{"region_id": "scene:region_0", "reason": "temporal_drift", "bbox": {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}}]
                alpha_hint = rec.get("alpha_hint")
                if alpha_hint is not None:
                    alpha_hint = cls._as_hw1_tensor(alpha_hint, "alpha_hint", shape_hint=np.asarray(cur, dtype=np.float32).shape[:2])
                confidence_hint = rec.get("confidence_hint")
                if confidence_hint is not None:
                    confidence_hint = cls._as_hw1_tensor(confidence_hint, "confidence_hint", shape_hint=np.asarray(cur, dtype=np.float32).shape[:2])
                patch_history = rec.get("patch_history", []) if isinstance(rec.get("patch_history", []), list) else []
                transition_context = rec.get("transition_context", rec.get("scene_transition_context", {}))
                if transition_context is not None and not isinstance(transition_context, dict):
                    raise ValueError("transition_context must be object")
                memory_context = rec.get("memory_context", rec.get("memory_transition_context", {}))
                if memory_context is not None and not isinstance(memory_context, dict):
                    raise ValueError("memory_context must be object")
                region_meta = rec.get("region_consistency_metadata", {})
                if region_meta is not None and not isinstance(region_meta, dict):
                    raise ValueError("region_consistency_metadata must be object")
                contract = {
                    "previous_frame": prev,
                    "composed_frame": cur,
                    "target_frame": target,
                    "changed_regions": changed_regions,
                    "changed_region_mask": changed_region_mask if changed_region_mask is None else mask,
                    "alpha_hint": alpha_hint,
                    "confidence_hint": confidence_hint,
                    "patch_history": patch_history,
                    "transition_context": transition_context if isinstance(transition_context, dict) else {},
                    "memory_context": memory_context if isinstance(memory_context, dict) else {},
                    "region_consistency_metadata": region_meta if isinstance(region_meta, dict) else {},
                    "scene_transition_context": transition_context if isinstance(transition_context, dict) else {},
                    "memory_transition_context": memory_context if isinstance(memory_context, dict) else {},
                    "record_id": str(rec.get("record_id", f"temporal_record_{idx}")),
                    "scenario": str(rec.get("scenario", rec.get("scenario_id", "unknown"))),
                    "tags": [str(t) for t in rec.get("tags", [])] if isinstance(rec.get("tags"), list) else [],
                    "notes": str(rec.get("notes", "")),
                }
                samples.append({"frames": [prev, cur, target], "temporal_consistency_contract": contract, "source": rec.get("source", "temporal_manifest")})
                diagnostics["loaded_records"] = int(diagnostics["loaded_records"]) + 1
                diagnostics["usable_records"] = int(diagnostics["usable_records"]) + 1
                diagnostics["supervised_temporal_records"] = int(diagnostics["supervised_temporal_records"]) + 1
                seq_key = str(rec.get("sequence_id", "unknown"))
                shape_key = str(np.asarray(cur, dtype=np.float32).shape)
                diagnostics["frame_shape_summary"][shape_key] = int(diagnostics["frame_shape_summary"].get(shape_key, 0)) + 1
                diagnostics.setdefault("_sequence_keys", set()).add(seq_key)
                diagnostics["family_counts"]["frame_refinement"] = int(diagnostics["family_counts"]["frame_refinement"]) + 1
                delta_prev_target = float(np.mean(np.abs(np.asarray(target, dtype=np.float32) - np.asarray(prev, dtype=np.float32))))
                diagnostics["family_counts"]["flicker"] = int(diagnostics["family_counts"]["flicker"]) + (1 if delta_prev_target > 1e-6 else 0)
                diagnostics["family_counts"]["region_stability"] = int(diagnostics["family_counts"]["region_stability"]) + (1 if changed_regions else 0)
                diagnostics["family_counts"]["alpha_consistency"] = int(diagnostics["family_counts"]["alpha_consistency"]) + (1 if alpha_hint is not None else 0)
                diagnostics["family_counts"]["confidence_calibration"] = int(diagnostics["family_counts"]["confidence_calibration"]) + (1 if confidence_hint is not None else 0)
                diagnostics["family_counts"]["history_conditioning"] = int(diagnostics["family_counts"]["history_conditioning"]) + (1 if patch_history else 0)
                diagnostics["family_counts"]["memory_conditioning"] = int(diagnostics["family_counts"]["memory_conditioning"]) + (1 if isinstance(memory_context, dict) and memory_context != {} else 0)
                diagnostics["family_counts"]["transition_conditioning"] = int(diagnostics["family_counts"]["transition_conditioning"]) + (1 if isinstance(transition_context, dict) and transition_context != {} else 0)
                scenario_key = str(rec.get("scenario", rec.get("scenario_id", "unknown"))).strip().lower() or "unknown"
                diagnostics["scenario_counts"][scenario_key] = int(diagnostics["scenario_counts"].get(scenario_key, 0)) + 1
                if isinstance(rec.get("tags"), list):
                    for tag in rec["tags"]:
                        tk = str(tag).strip().lower()
                        if tk:
                            diagnostics["tag_counts"][tk] = int(diagnostics["tag_counts"].get(tk, 0)) + 1
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                reason = str(exc)
                diagnostics["skipped_by_reason"][reason] = int(diagnostics["skipped_by_reason"].get(reason, 0)) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "reason": str(exc)})
                if strict:
                    raise
        diagnostics["sequence_count"] = len(diagnostics.pop("_sequence_keys", set()))
        if int(diagnostics.get("supervised_temporal_records", 0)) == 0:
            raise ValueError("no valid supervised temporal records")
        ds = cls(samples=samples)
        ds.diagnostics = diagnostics
        return ds



class TextActionDataset(BaseStageDataset):
    @classmethod
    def from_jsonl(cls, path: str) -> "TextActionDataset":
        samples: list[TrainingSample] = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            steps = [ActionStep(type=a["type"], priority=i + 1, target_object=a.get("target_object")) for i, a in enumerate(rec.get("actions", []))]
            samples.append({"text": rec.get("text", ""), "actions": steps, "text_alignment": rec, "source": rec.get("source", "annotation")})
        return cls(samples=samples)

    @classmethod
    def from_annotation_manifest(cls, manifest_path: str) -> "TextActionDataset":
        from text.learned_bridge import BaselineTextEncoderAdapter
        from training.learned_contracts import build_text_action_state_contract

        manifest = json.loads(Path(manifest_path).read_text())
        encoder = BaselineTextEncoderAdapter()
        samples: list[TrainingSample] = []
        for rec in manifest.get("records", []):
            actions = [ActionStep(type=a.get("type", "unknown"), priority=i + 1, target_entity=a.get("target_entity"), target_object=a.get("target_object")) for i, a in enumerate(rec.get("actions", []))]
            encoding = encoder.encode(rec.get("text", ""))
            contract = build_text_action_state_contract(rec.get("text", ""), actions, vars(encoding))
            sample: TrainingSample = {
                "text": rec.get("text", ""),
                "actions": actions,
                "text_alignment": rec,
                "text_action_contract": contract,
                "source": rec.get("source", "annotation_manifest"),
            }
            samples.append(sample)
        return cls(samples=samples)


class MemoryDataset(BaseStageDataset):
    @classmethod
    def from_representation_dataset(cls, ds: RepresentationDataset) -> "MemoryDataset":
        samples: list[TrainingSample] = []
        for sample in ds.samples:
            graphs = sample.get("graphs", [])
            records = []
            for g in graphs:
                records.append({"frame_index": g.frame_index, "person_count": len(g.persons), "object_count": len(g.objects)})
            samples.append({"memory_records": records, "source": sample.get("source", "representation")})
        return cls(samples=samples)

    @classmethod
    def from_graph_sequence(cls, graphs: list[SceneGraph], frames: list[list] | None = None) -> "MemoryDataset":
        manager = MemoryManager()
        records: list[dict[str, object]] = []
        if not graphs:
            return cls(samples=[])
        memory = manager.initialize(graphs[0])
        for idx, graph in enumerate(graphs):
            memory = manager.update_from_graph(memory, graph)
            if frames and idx < len(frames):
                memory = manager.update_from_frame(memory, frames[idx], graph)
            records.append({"frame_index": graph.frame_index, "regions": len(memory.region_descriptors), "patches": len(memory.texture_patches), "hidden_slots": len(memory.hidden_region_slots)})
        return cls(samples=[{"memory_records": records, "source": "graph_sequence"}])


def _serialize_bbox(b: BBox) -> dict[str, float]:
    return {"x": b.x, "y": b.y, "w": b.w, "h": b.h}


def _serialize_graph(graph: SceneGraph) -> dict[str, object]:
    return {
        "frame_index": graph.frame_index,
        "global_context": {"frame_size": list(graph.global_context.frame_size), "fps": graph.global_context.fps, "source_type": graph.global_context.source_type},
        "persons": [
            {
                "person_id": p.person_id,
                "track_id": p.track_id,
                "bbox": _serialize_bbox(p.bbox),
                "confidence": p.confidence,
                "garments": [{"garment_id": g.garment_id, "garment_type": g.garment_type, "coverage_targets": g.coverage_targets, "visibility": g.visibility, "confidence": g.confidence} for g in p.garments],
                "body_parts": [{"part_id": bp.part_id, "part_type": bp.part_type, "visibility": bp.visibility, "confidence": bp.confidence, "keypoints": [{"name": kp.name, "x": kp.x, "y": kp.y, "confidence": kp.confidence} for kp in bp.keypoints]} for bp in p.body_parts],
            }
            for p in graph.persons
        ],
        "objects": [{"object_id": o.object_id, "object_type": o.object_type, "bbox": _serialize_bbox(o.bbox), "confidence": o.confidence} for o in graph.objects],
        "relations": [{"source": r.source, "relation": r.relation, "target": r.target, "confidence": r.confidence, "provenance": r.provenance} for r in graph.relations],
    }


def _build_human_state_transition_payload(
    *,
    record: dict[str, object],
    graph_before: SceneGraph,
    graph_after: SceneGraph,
    family: str,
    phase: str,
    target_profile: dict[str, object],
    roi_records: list[dict[str, object]],
    planner_context: dict[str, object],
    graph_delta_target: dict[str, object],
    feature_dim: int = 160,
    has_previous_in_sequence: bool = False,
) -> tuple[list[float], dict[str, object], dict[str, object]]:
    vis_delta = (graph_delta_target.get("visibility_deltas", {}) or {}) if isinstance((graph_delta_target.get("visibility_deltas", {}) or {}), dict) else {}
    reveal_cue = float(np.clip(record.get("reveal_score", vis_delta.get("revealed_regions_score", 0.0)), 0.0, 1.0))
    occlusion_cue = float(np.clip(record.get("occlusion_score", vis_delta.get("occluded_regions_score", 0.0)), 0.0, 1.0))
    support_cue = float(
        np.clip(
            ((graph_delta_target.get("interaction_deltas", {}) or {}).get("support_contact", record.get("support_contact_score", 0.0)))
            if isinstance((graph_delta_target.get("interaction_deltas", {}) or {}), dict)
            else float(record.get("support_contact_score", 0.0)),
            0.0,
            1.0,
        )
    )
    temporal_feats = _build_temporal_transition_features(
        graph_before=graph_before,
        graph_after=graph_after,
        roi_records=roi_records,
        graph_delta_target=graph_delta_target,
        planner_context=planner_context,
        target_profile=target_profile,
        runtime_semantic_transition=str(record.get("runtime_semantic_transition", family)),
        phase_estimate=phase,
        reveal_score=reveal_cue,
        occlusion_score=occlusion_cue,
        support_score=support_cue,
        transition_confidence=float(np.clip(record.get("transition_confidence", 1.0), 0.0, 1.0)),
    )
    region_keys = ("face", "torso", "left_arm", "right_arm", "legs", "garments", "inner_garment", "outer_garment")
    all_regions: set[str] = set()
    for slot in ("primary_regions", "secondary_regions", "context_regions"):
        for region in target_profile.get(slot, []) if isinstance(target_profile.get(slot, []), list) else []:
            all_regions.add(str(region))
    roi_region_changed = {str(rr.get("region_type", "")): float(rr.get("changed_ratio", 0.0)) for rr in roi_records if isinstance(rr, dict)}
    region_states: list[float] = []
    visibility_targets: list[float] = []
    for region in region_keys:
        on = 1.0 if region in all_regions else 0.0
        region_states.append(float(np.clip(max(on * 0.65, roi_region_changed.get(region, 0.0)), 0.0, 1.0)))
        visibility_targets.append(float(np.clip(1.0 - occlusion_cue * (1.0 if region in all_regions else 0.5), 0.0, 1.0)))

    features = temporal_feats + region_states + visibility_targets + [reveal_cue, occlusion_cue, support_cue]
    if len(features) < feature_dim:
        features.extend([0.0] * (feature_dim - len(features)))
    prev_hint = [0.0] * 24
    return (
        features[:feature_dim],
        {
            "family": family,
            "phase": phase,
            "target_profile": target_profile,
            "region_state_targets": region_states,
            "visibility_targets": visibility_targets,
            "reveal_memory_target": reveal_cue,
            "support_contact_target": support_cue,
        },
        {
            "has_history": bool(record.get("previous_record_id")) or bool(has_previous_in_sequence),
            "previous_state_hint": prev_hint,
        },
    )


def _derive_graph_delta_target(
    *,
    state_before: dict[str, object],
    state_after: dict[str, object],
    family: str,
    phase: str,
    transition_mode: str,
    record: dict[str, object],
) -> dict[str, object]:
    def _float_delta(key: str) -> float:
        before = state_before.get(key, 0.0)
        after = state_after.get(key, before)
        try:
            return float(after) - float(before)
        except Exception:
            return 0.0

    pose_deltas = {"torso_pitch": _float_delta("torso_pitch"), "head_yaw": _float_delta("head_yaw")}
    garment_deltas = {"attachment_delta": _float_delta("garment_attachment"), "coverage_delta": _float_delta("garment_coverage"), "layer_shift": _float_delta("garment_layer_shift")}
    expression_deltas = {"smile_intensity": _float_delta("smile_intensity"), "eye_openness": _float_delta("eye_openness")}
    interaction_deltas = {
        "support_contact": max(0.0, min(1.0, _float_delta("support_contact") + float(state_before.get("support_contact", 0.0)))),
        "hand_contact": max(0.0, min(1.0, _float_delta("hand_contact") + float(state_before.get("hand_contact", 0.0)))),
    }
    visibility_after = state_after.get("visibility", {})
    visibility_before = state_before.get("visibility", {})
    visibility_deltas = visibility_after if isinstance(visibility_after, dict) else (visibility_before if isinstance(visibility_before, dict) else {})
    predicted_visibility_changes = state_after.get("predicted_visibility_changes", visibility_deltas)
    affected_regions = record.get("affected_regions", [])
    if not isinstance(affected_regions, list) or not affected_regions:
        affected_regions = (
            ["face", "head"]
            if family == "expression_transition"
            else (["garments", "torso", "inner_garment"] if family == "garment_transition" else (["arms", "torso"] if family == "interaction_transition" else ["torso", "legs"]))
        )
    region_transition_mode = record.get("region_transition_mode", {})
    if not isinstance(region_transition_mode, dict) or not region_transition_mode:
        default_mode = "reveal" if transition_mode in {"reveal", "contact_or_reveal"} else ("occlude" if transition_mode in {"occlusion", "hide"} else "deform")
        region_transition_mode = {str(region): default_mode for region in affected_regions}
    return {
        "pose_deltas": pose_deltas if family in {"pose_transition", "expression_transition"} else {},
        "garment_deltas": garment_deltas if family == "garment_transition" else {},
        "expression_deltas": expression_deltas if family == "expression_transition" else {},
        "interaction_deltas": interaction_deltas if family == "interaction_transition" else {},
        "visibility_deltas": {str(k): str(v) for k, v in visibility_deltas.items()},
        "predicted_visibility_changes": {str(k): str(v) for k, v in predicted_visibility_changes.items()} if isinstance(predicted_visibility_changes, dict) else {},
        "affected_entities": [str(x) for x in record.get("affected_entities", [str(record.get("tracked_entity_id", "scene"))])],
        "affected_regions": [str(x) for x in affected_regions],
        "semantic_reasons": [str(x) for x in record.get("action_tokens", [family])],
        "region_transition_mode": {str(k): str(v) for k, v in region_transition_mode.items()},
        "transition_phase": phase,
    }


def _build_temporal_transition_features(
    *,
    graph_before: SceneGraph,
    graph_after: SceneGraph,
    roi_records: list[dict[str, object]],
    graph_delta_target: dict[str, object],
    planner_context: dict[str, object],
    target_profile: dict[str, object],
    runtime_semantic_transition: str,
    phase_estimate: str,
    reveal_score: float,
    occlusion_score: float,
    support_score: float,
    transition_confidence: float,
) -> list[float]:
    def _person_graph_features(graph: SceneGraph) -> list[float]:
        person = graph.persons[0] if graph.persons else None
        if not person:
            return [0.0] * 14
        return [
            float(person.bbox.x),
            float(person.bbox.y),
            float(person.bbox.w),
            float(person.bbox.h),
            float(len(person.body_parts)) / 12.0,
            float(len(person.garments)) / 8.0,
            float(person.expression_state.smile_intensity),
            float(person.expression_state.eye_openness),
            float(person.orientation.yaw) / 90.0,
            float(person.orientation.pitch) / 90.0,
            float(person.orientation.roll) / 90.0,
            float(person.pose_state.angles.get("torso_pitch", 0.0)) / 45.0,
            float(person.pose_state.angles.get("head_yaw", 0.0)) / 45.0,
            float(len(graph.objects)) / 8.0,
        ]

    fb = _person_graph_features(graph_before)
    fa = _person_graph_features(graph_after)
    graph_delta = [fa[i] - fb[i] for i in range(len(fa))]

    region_stats = {k: [] for k in TemporalTransitionDataset.REGION_KEYS}
    for rec in roi_records:
        if not isinstance(rec, dict):
            continue
        region = str(rec.get("region_type", "torso"))
        if region in region_stats:
            region_stats[region].append(float(rec.get("changed_ratio", 0.0)))
    roi_features = []
    for k in TemporalTransitionDataset.REGION_KEYS:
        vals = region_stats[k]
        roi_features.extend([float(np.mean(vals or [0.0])), float(len(vals)) / 4.0])

    pose_d = graph_delta_target.get("pose_deltas", {}) if isinstance(graph_delta_target.get("pose_deltas", {}), dict) else {}
    garment_d = graph_delta_target.get("garment_deltas", {}) if isinstance(graph_delta_target.get("garment_deltas", {}), dict) else {}
    expr_d = graph_delta_target.get("expression_deltas", {}) if isinstance(graph_delta_target.get("expression_deltas", {}), dict) else {}
    inter_d = graph_delta_target.get("interaction_deltas", {}) if isinstance(graph_delta_target.get("interaction_deltas", {}), dict) else {}
    vis_d = graph_delta_target.get("visibility_deltas", {}) if isinstance(graph_delta_target.get("visibility_deltas", {}), dict) else {}
    delta_features = [
        float(pose_d.get("torso_motion", pose_d.get("torso_pitch", 0.0))),
        float(pose_d.get("head_motion", pose_d.get("head_yaw", 0.0))),
        float(pose_d.get("arm_motion", pose_d.get("arm_raise", 0.0))),
        float(garment_d.get("coverage_change", garment_d.get("coverage_delta", 0.0))),
        float(garment_d.get("attachment_shift", garment_d.get("attachment_delta", 0.0))),
        float(expr_d.get("face_expression_shift", expr_d.get("smile_intensity", 0.0))),
        float(inter_d.get("support_contact", 0.0)),
        float(inter_d.get("contact_hint", 0.0)),
        float(vis_d.get("revealed_regions_score", reveal_score)),
        float(vis_d.get("occluded_regions_score", occlusion_score)),
    ]
    profile_vec = []
    for slot in ("primary_regions", "secondary_regions", "context_regions"):
        listed = target_profile.get(slot, []) if isinstance(target_profile.get(slot, []), list) else []
        for region in TemporalTransitionDataset.REGION_KEYS:
            profile_vec.append(1.0 if region in listed else 0.0)
    phase_onehot = [1.0 if phase_estimate == p else 0.0 for p in ("prepare", "transition", "contact_or_reveal", "stabilize")]
    family_onehot = [1.0 if runtime_semantic_transition == f else 0.0 for f in ("pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition")]
    planner_vec = [
        float(planner_context.get("step_index", 1.0)) / 16.0,
        float(planner_context.get("total_steps", 4.0)) / 16.0,
        float(planner_context.get("target_duration", 1.0)) / 8.0,
    ]
    clues = [float(np.clip(reveal_score, 0.0, 1.0)), float(np.clip(occlusion_score, 0.0, 1.0)), float(np.clip(support_score, 0.0, 1.0)), float(np.clip(transition_confidence, 0.0, 1.0))]
    merged = fb + fa + graph_delta + roi_features + delta_features + profile_vec + phase_onehot + family_onehot + planner_vec + clues
    if len(merged) < 128:
        merged.extend([0.0] * (128 - len(merged)))
    return [float(v) for v in merged[:128]]


def _deserialize_graph(payload: dict[str, object]) -> SceneGraph:
    persons = []
    for p in payload.get("persons", []):
        body_parts = []
        for bp in p.get("body_parts", []):
            kps = [Keypoint(name=kp.get("name", "kp"), x=float(kp.get("x", 0.0)), y=float(kp.get("y", 0.0)), confidence=float(kp.get("confidence", 0.0))) for kp in bp.get("keypoints", [])]
            body_parts.append(BodyPartNode(part_id=bp.get("part_id", "bp"), part_type=bp.get("part_type", "unknown"), visibility=bp.get("visibility", "unknown"), confidence=float(bp.get("confidence", 0.0)), keypoints=kps))
        garments = [GarmentNode(garment_id=g.get("garment_id", "garment"), garment_type=g.get("garment_type", "unknown"), coverage_targets=list(g.get("coverage_targets", [])), visibility=g.get("visibility", "unknown"), confidence=float(g.get("confidence", 0.0))) for g in p.get("garments", [])]
        bb = p.get("bbox", {})
        persons.append(PersonNode(person_id=p.get("person_id", "person"), track_id=p.get("track_id"), bbox=BBox(float(bb.get("x", 0.0)), float(bb.get("y", 0.0)), float(bb.get("w", 0.0)), float(bb.get("h", 0.0))), mask_ref=None, body_parts=body_parts, garments=garments, confidence=float(p.get("confidence", 0.0))))

    objects = []
    for o in payload.get("objects", []):
        bb = o.get("bbox", {})
        objects.append(SceneObjectNode(object_id=o.get("object_id", "object"), object_type=o.get("object_type", "unknown"), bbox=BBox(float(bb.get("x", 0.0)), float(bb.get("y", 0.0)), float(bb.get("w", 0.0)), float(bb.get("h", 0.0))), confidence=float(o.get("confidence", 0.0))))

    relations = [RelationEdge(source=r.get("source", ""), relation=r.get("relation", "near"), target=r.get("target", ""), confidence=float(r.get("confidence", 0.0)), provenance=r.get("provenance", "cache")) for r in payload.get("relations", [])]
    gc = payload.get("global_context", {})
    context = GlobalSceneContext(frame_size=tuple(gc.get("frame_size", [0, 0])), fps=int(gc.get("fps", 16)), source_type=str(gc.get("source_type", "single_image")))
    return SceneGraph(frame_index=int(payload.get("frame_index", 0)), persons=persons, objects=objects, relations=relations, global_context=context)


def save_graph_cache(graphs: list[SceneGraph], path: str) -> None:
    payload = {"cache_version": 2, "partial_fields": ["persons.bbox", "objects.bbox", "garments", "body_parts", "relations", "global_context"], "graphs": [_serialize_graph(g) for g in graphs]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_graph_cache(path: str) -> list[SceneGraph]:
    payload = json.loads(Path(path).read_text())
    return [_deserialize_graph(rec) for rec in payload.get("graphs", [])]


def _serialize_delta_contract(delta: GraphDelta) -> dict[str, object]:
    magnitude = 0.0
    for payload in (delta.pose_deltas, delta.interaction_deltas, delta.garment_deltas, delta.expression_deltas):
        magnitude += sum(abs(float(v)) for v in payload.values() if isinstance(v, (int, float)))
    return {
        "state_before": dict(delta.state_before),
        "state_after": dict(delta.state_after),
        "transition_phase": delta.transition_phase,
        "region_transition_mode": dict(delta.region_transition_mode),
        "predicted_visibility_changes": dict(delta.predicted_visibility_changes),
        "semantic_reasons": list(delta.semantic_reasons),
        "affected_entities": list(delta.affected_entities),
        "affected_regions": list(delta.affected_regions),
        "pose_deltas": dict(delta.pose_deltas),
        "interaction_deltas": dict(delta.interaction_deltas),
        "garment_deltas": dict(delta.garment_deltas),
        "expression_deltas": dict(delta.expression_deltas),
        "magnitude": magnitude,
    }
