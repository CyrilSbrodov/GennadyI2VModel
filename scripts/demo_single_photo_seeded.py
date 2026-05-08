from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from pprint import pprint
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from core.region_ids import parse_region_id
from core.schema import (
    BBox,
    BodyPartNode,
    ExpressionState,
    GarmentNode,
    HiddenRegionSlot,
    PersonNode,
    PoseState,
    SceneGraph,
    SceneObjectNode,
    TexturePatchMemory,
    VideoMemory,
)
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from dynamics.state_update import apply_delta
from memory.video_memory import MemoryManager
from planning.transition_engine import PlannedState
from rendering.roi_renderer import ROISelector, PatchRenderer
from text.intent_parser import IntentParser
from text.text_encoder import BaselineStructuredTextEncoder


# ============================================================
# Generic helpers
# ============================================================

def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_dataclass_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): safe_dataclass_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_dataclass_dict(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def bbox_to_dict(bbox: BBox | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    return {"x": float(bbox.x), "y": float(bbox.y), "w": float(bbox.w), "h": float(bbox.h)}


def bbox_to_pixels(bbox: BBox, image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x0 = max(0, min(w - 1, int(float(bbox.x) * w)))
    y0 = max(0, min(h - 1, int(float(bbox.y) * h)))
    x1 = max(x0 + 1, min(w, int((float(bbox.x) + float(bbox.w)) * w)))
    y1 = max(y0 + 1, min(h, int((float(bbox.y) + float(bbox.h)) * h)))
    return x0, y0, x1, y1


def draw_bbox(
    image: np.ndarray,
    bbox: BBox,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    x0, y0, x1, y1 = bbox_to_pixels(bbox, image)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    cv2.putText(
        image,
        label,
        (x0, max(16, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def crop_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    x0, y0, x1, y1 = bbox_to_pixels(bbox, image)
    return image[y0:y1, x0:x1].copy()


def patch_rgb_to_bgr_image(rgb_patch: Any) -> np.ndarray:
    arr = np.array(rgb_patch, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def alpha_to_uint8(alpha_mask: Any) -> np.ndarray:
    alpha = np.array(alpha_mask, dtype=np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)
    return (alpha * 255.0).astype(np.uint8)


def composite_patch_onto_image(base_bgr: np.ndarray, patch: Any) -> None:
    patch_bgr = patch_rgb_to_bgr_image(patch.rgb_patch)
    alpha = np.array(patch.alpha_mask, dtype=np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)

    x0, y0, x1, y1 = bbox_to_pixels(patch.region.bbox, base_bgr)
    target_h = max(1, y1 - y0)
    target_w = max(1, x1 - x0)

    patch_bgr = cv2.resize(patch_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    roi = base_bgr[y0:y1, x0:x1].astype(np.float32)
    patch_f = patch_bgr.astype(np.float32)
    alpha_f = alpha[..., None]

    blended = roi * (1.0 - alpha_f) + patch_f * alpha_f
    base_bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# Canonical seeded scene
# ============================================================

CANONICAL_REGION_NAMES = [
    "face",
    "head",
    "neck",
    "torso",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "pelvis",
    "left_leg",
    "right_leg",
    "upper_garment",
    "outer_garment",
    "inner_garment",
]


def canonical_payload(
    canonical_name: str,
    *,
    visibility_state: str = "visible",
    confidence: float = 0.7,
    provenance: str = "seeded_debug_scene",
    lifecycle_state: str = "stable",
    last_transition_mode: str = "stable",
    last_transition_phase: str = "stable",
    reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "canonical_name": canonical_name,
        "raw_sources": [canonical_name],
        "source_regions": [canonical_name],
        "mask_ref": f"debug://mask/{canonical_name}",
        "confidence": float(confidence),
        "visibility_state": visibility_state,
        "provenance": provenance,
        "attachment_hints": ["person"],
        "ownership_hints": ["person"],
        "coverage_hints": [canonical_name],
        "lifecycle_state": lifecycle_state,
        "last_transition_mode": last_transition_mode,
        "last_transition_phase": last_transition_phase,
        "last_semantic_reasons": reasons or ["seeded_debug_initial_state"],
        "last_update_source": provenance,
    }


def make_seeded_scene() -> SceneGraph:
    canonical_regions = {
        "face": canonical_payload("face", confidence=0.86),
        "head": canonical_payload("head", confidence=0.82),
        "neck": canonical_payload("neck", confidence=0.62),
        "torso": canonical_payload("torso", confidence=0.78),
        "left_arm": canonical_payload("left_arm", confidence=0.72),
        "right_arm": canonical_payload("right_arm", confidence=0.72),
        "left_hand": canonical_payload("left_hand", confidence=0.6),
        "right_hand": canonical_payload("right_hand", confidence=0.6),
        "pelvis": canonical_payload("pelvis", confidence=0.64),
        "left_leg": canonical_payload("left_leg", confidence=0.68),
        "right_leg": canonical_payload("right_leg", confidence=0.68),
        "upper_garment": canonical_payload("upper_garment", confidence=0.78),
        "outer_garment": canonical_payload("outer_garment", visibility_state="visible", confidence=0.84),
        "inner_garment": canonical_payload(
            "inner_garment",
            visibility_state="hidden_by_garment",
            confidence=0.45,
            lifecycle_state="currently_hidden",
            last_transition_mode="covered_by_outer_garment",
            reasons=["inner_garment_expected_under_outer_layer"],
        ),
    }

    person = PersonNode(
        person_id="p1",
        track_id="t1",
        bbox=BBox(0.2, 0.08, 0.5, 0.84),
        mask_ref="debug://person/p1",
        pose_state=PoseState(coarse_pose="standing"),
        expression_state=ExpressionState(label="neutral"),
        body_parts=[
            BodyPartNode(part_id="bp_face", part_type="face", visibility="visible"),
            BodyPartNode(part_id="bp_head", part_type="head", visibility="visible"),
            BodyPartNode(part_id="bp_left_arm", part_type="left_arm", visibility="visible"),
            BodyPartNode(part_id="bp_right_arm", part_type="right_arm", visibility="visible"),
            BodyPartNode(part_id="bp_left_hand", part_type="left_hand", visibility="visible"),
            BodyPartNode(part_id="bp_right_hand", part_type="right_hand", visibility="visible"),
            BodyPartNode(part_id="bp_legs", part_type="legs", visibility="visible"),
            BodyPartNode(part_id="bp_torso", part_type="torso", visibility="visible"),
            BodyPartNode(part_id="bp_pelvis", part_type="pelvis", visibility="visible"),
        ],
        garments=[
            GarmentNode(garment_id="g_outer_1", garment_type="outer_garment", garment_state="worn"),
            GarmentNode(garment_id="g_inner_1", garment_type="inner_garment", garment_state="covered"),
        ],
        canonical_regions=canonical_regions,
    )

    chair = SceneObjectNode(
        object_id="chair_1",
        object_type="chair",
        bbox=BBox(0.18, 0.62, 0.42, 0.22),
    )

    return SceneGraph(frame_index=0, persons=[person], objects=[chair])


def infer_scene_from_image(image: np.ndarray, mode: str) -> tuple[SceneGraph, str]:
    _ = image

    if mode == "seeded":
        return make_seeded_scene(), "seeded_debug_scene"

    raise ValueError(f"Unsupported scene mode: {mode!r}")


# ============================================================
# Debug serialization
# ============================================================

def scene_to_debug_dict(scene: SceneGraph) -> dict[str, object]:
    persons = []
    for p in scene.persons:
        canonical_debug: dict[str, object] = {}
        canonical_regions = getattr(p, "canonical_regions", {}) or {}
        if isinstance(canonical_regions, dict):
            for name, payload in canonical_regions.items():
                if not isinstance(payload, dict):
                    canonical_debug[str(name)] = safe_dataclass_dict(payload)
                    continue
                canonical_debug[str(name)] = {
                    "canonical_name": payload.get("canonical_name", name),
                    "visibility_state": payload.get("visibility_state"),
                    "lifecycle_state": payload.get("lifecycle_state"),
                    "confidence": payload.get("confidence"),
                    "provenance": payload.get("provenance"),
                    "mask_ref": payload.get("mask_ref"),
                    "source_regions": payload.get("source_regions", []),
                    "raw_sources": payload.get("raw_sources", []),
                    "last_transition_mode": payload.get("last_transition_mode"),
                    "last_transition_phase": payload.get("last_transition_phase"),
                    "last_semantic_reasons": payload.get("last_semantic_reasons", []),
                    "last_update_source": payload.get("last_update_source"),
                }

        persons.append(
            {
                "person_id": p.person_id,
                "track_id": p.track_id,
                "bbox": bbox_to_dict(p.bbox),
                "mask_ref": getattr(p, "mask_ref", None),
                "confidence": getattr(p, "confidence", None),
                "pose_state": safe_dataclass_dict(getattr(p, "pose_state", None)),
                "expression_state": safe_dataclass_dict(getattr(p, "expression_state", None)),
                "canonical_region_count": len(canonical_debug),
                "canonical_regions": canonical_debug,
                "body_parts": [
                    {
                        "part_id": bp.part_id,
                        "part_type": bp.part_type,
                        "visibility": bp.visibility,
                        "bbox": bbox_to_dict(getattr(bp, "bbox", None)),
                        "mask_ref": getattr(bp, "mask_ref", None),
                    }
                    for bp in getattr(p, "body_parts", [])
                ],
                "garments": [
                    {
                        "garment_id": g.garment_id,
                        "garment_type": g.garment_type,
                        "garment_state": getattr(g, "garment_state", None),
                        "visibility": getattr(g, "visibility", None),
                        "bbox": bbox_to_dict(getattr(g, "bbox", None)),
                        "mask_ref": getattr(g, "mask_ref", None),
                    }
                    for g in getattr(p, "garments", [])
                ],
            }
        )

    return {
        "frame_index": scene.frame_index,
        "persons": persons,
        "objects": [
            {
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "bbox": bbox_to_dict(obj.bbox),
            }
            for obj in scene.objects
        ],
    }


def parsed_intent_to_dict(parsed: Any) -> dict[str, object]:
    return {
        "normalized_text": parsed.normalized_text,
        "parser_confidence": parsed.parser_confidence,
        "clauses": [
            {
                "index": clause.index,
                "text": clause.text,
                "actions": [
                    {
                        "semantic_family": a.semantic_family,
                        "semantic_action": a.semantic_action,
                        "confidence": a.confidence,
                        "lexical_reason": a.lexical_reason,
                    }
                    for a in clause.action_candidates
                ],
                "targets": [
                    {
                        "target_entity_class": t.target_entity_class,
                        "target_entity_id": t.target_entity_id,
                        "target_region": t.target_region,
                        "target_object": t.target_object,
                        "grounding_confidence": t.grounding_confidence,
                        "resolution_reason": t.resolution_reason,
                        "unresolved": t.unresolved,
                    }
                    for t in clause.resolved_targets
                ],
                "constraints": [
                    {
                        "requirement": c.requirement,
                        "reason": c.reason,
                    }
                    for c in clause.constraints
                ],
                "ambiguities": clause.ambiguities[:],
                "modifiers": clause.modifiers.as_dict(),
            }
            for clause in parsed.clauses
        ],
        "temporal_relations": [
            {
                "relation": rel.relation,
                "source_clause": rel.source_clause,
                "target_clause": rel.target_clause,
            }
            for rel in parsed.temporal_relations
        ],
        "global_constraints": [c.requirement for c in parsed.global_constraints],
        "explainability_trace": parsed.explainability_trace,
    }


def encoding_to_dict(encoded: Any) -> dict[str, object]:
    return {
        "structured_action_tokens": encoded.structured_action_tokens,
        "parser_confidence": encoded.parser_confidence,
        "encoder_confidence": encoded.encoder_confidence,
        "scene_alignment_score": encoded.scene_alignment_score,
        "ambiguity_score": encoded.ambiguity_score,
        "target_hints": encoded.target_hints,
        "temporal_hints": encoded.temporal_hints,
        "constraints": encoded.constraints,
        "conditioning_hints": encoded.conditioning_hints,
        "diagnostics": {
            "action_count": encoded.diagnostics.action_count,
            "family_distribution": encoded.diagnostics.family_distribution,
            "target_count": encoded.diagnostics.target_count,
            "resolved_target_count": encoded.diagnostics.resolved_target_count,
            "unresolved_target_count": encoded.diagnostics.unresolved_target_count,
            "weak_grounding_count": encoded.diagnostics.weak_grounding_count,
            "temporal_relation_count": encoded.diagnostics.temporal_relation_count,
            "constraint_count": encoded.diagnostics.constraint_count,
            "ambiguity_count": encoded.diagnostics.ambiguity_count,
            "parser_confidence": encoded.diagnostics.parser_confidence,
            "encoder_confidence": encoded.diagnostics.encoder_confidence,
            "explainability_summary": encoded.diagnostics.explainability_summary,
        },
        "trace": encoded.trace,
    }


def delta_to_dict(delta: Any) -> dict[str, object]:
    def region_to_dict(region: Any) -> dict[str, object]:
        return {
            "region_id": region.region_id,
            "reason": region.reason,
            "bbox": bbox_to_dict(region.bbox),
        }

    return {
        "transition_phase": delta.transition_phase,
        "affected_entities": delta.affected_entities,
        "affected_regions": delta.affected_regions,
        "semantic_reasons": delta.semantic_reasons,
        "pose_deltas": delta.pose_deltas,
        "garment_deltas": delta.garment_deltas,
        "expression_deltas": delta.expression_deltas,
        "interaction_deltas": delta.interaction_deltas,
        "visibility_deltas": delta.visibility_deltas,
        "predicted_visibility_changes": delta.predicted_visibility_changes,
        "region_transition_mode": delta.region_transition_mode,
        "state_before": delta.state_before,
        "state_after": delta.state_after,
        "newly_revealed_regions": [region_to_dict(r) for r in delta.newly_revealed_regions],
        "newly_occluded_regions": [region_to_dict(r) for r in delta.newly_occluded_regions],
        "transition_diagnostics": delta.transition_diagnostics,
    }


def patch_to_dict(patch: Any) -> dict[str, object]:
    return {
        "region_id": patch.region.region_id,
        "reason": patch.region.reason,
        "bbox": bbox_to_dict(patch.region.bbox),
        "confidence": patch.confidence,
        "shape": {
            "height": patch.height,
            "width": patch.width,
            "channels": patch.channels,
        },
        "debug_trace": patch.debug_trace,
        "execution_trace": safe_dataclass_dict(patch.execution_trace),
    }


def memory_summary(memory: VideoMemory) -> dict[str, int]:
    return {
        "canonical_region_memory_count": len(getattr(memory, "canonical_region_memory", {}) or {}),
        "hidden_region_slot_count": len(getattr(memory, "hidden_region_slots", {}) or {}),
        "texture_patch_count": len(getattr(memory, "texture_patches", {}) or {}),
        "patch_cache_count": len(getattr(memory, "patch_cache", {}) or {}),
        "identity_memory_count": len(getattr(memory, "identity_memory", {}) or {}),
        "garment_memory_count": len(getattr(memory, "garment_memory", {}) or {}),
    }


# ============================================================
# Memory seeding
# ============================================================

def make_patch(h: int, w: int, rgb: tuple[float, float, float]) -> list[list[list[float]]]:
    r, g, b = rgb
    return [[[r, g, b] for _ in range(w)] for _ in range(h)]


def seed_memory(memory: VideoMemory) -> None:
    memory.patch_cache["patch_face_1"] = make_patch(64, 64, (0.62, 0.42, 0.36))
    memory.patch_cache["patch_outer_1"] = make_patch(96, 128, (0.68, 0.68, 0.70))
    memory.patch_cache["patch_inner_1"] = make_patch(96, 128, (0.78, 0.77, 0.75))
    memory.patch_cache["patch_torso_1"] = make_patch(128, 128, (0.72, 0.66, 0.61))

    memory.texture_patches["patch_face_1"] = TexturePatchMemory(
        patch_id="patch_face_1",
        region_type="face",
        entity_id="p1",
        source_frame=0,
        patch_ref="debug://patch_face_1",
        confidence=0.72,
        descriptor={"mean": [0.62, 0.42, 0.36], "std": [0.03, 0.03, 0.03], "edge_density": 0.05, "energy": 0.12},
        evidence_score=0.61,
        semantic_family="face",
        coverage_targets=["face"],
        attachment_targets=["face"],
        suitable_for_reveal=False,
    )

    memory.texture_patches["patch_outer_1"] = TexturePatchMemory(
        patch_id="patch_outer_1",
        region_type="outer_garment",
        entity_id="p1",
        source_frame=0,
        patch_ref="debug://patch_outer_1",
        confidence=0.81,
        descriptor={"mean": [0.68, 0.68, 0.70], "std": [0.04, 0.04, 0.04], "edge_density": 0.08, "energy": 0.21},
        evidence_score=0.74,
        semantic_family="garment",
        coverage_targets=["outer_garment", "torso"],
        attachment_targets=["torso"],
        suitable_for_reveal=False,
    )

    memory.texture_patches["patch_inner_1"] = TexturePatchMemory(
        patch_id="patch_inner_1",
        region_type="inner_garment",
        entity_id="p1",
        source_frame=0,
        patch_ref="debug://patch_inner_1",
        confidence=0.84,
        descriptor={"mean": [0.78, 0.77, 0.75], "std": [0.04, 0.04, 0.04], "edge_density": 0.07, "energy": 0.28},
        evidence_score=0.79,
        semantic_family="garment",
        coverage_targets=["inner_garment", "torso"],
        attachment_targets=["torso"],
        suitable_for_reveal=True,
    )

    memory.texture_patches["patch_torso_1"] = TexturePatchMemory(
        patch_id="patch_torso_1",
        region_type="torso",
        entity_id="p1",
        source_frame=0,
        patch_ref="debug://patch_torso_1",
        confidence=0.77,
        descriptor={"mean": [0.72, 0.66, 0.61], "std": [0.03, 0.03, 0.03], "edge_density": 0.06, "energy": 0.18},
        evidence_score=0.68,
        semantic_family="torso",
        coverage_targets=["torso"],
        attachment_targets=["torso"],
        suitable_for_reveal=True,
    )

    memory.hidden_region_slots["p1:inner_garment"] = HiddenRegionSlot(
        slot_id="p1:inner_garment",
        region_type="inner_garment",
        owner_entity="p1",
        candidate_patch_ids=["patch_inner_1"],
        confidence=0.82,
        hidden_type="known_hidden",
        evidence_score=0.76,
    )

    memory.hidden_region_slots["p1:torso"] = HiddenRegionSlot(
        slot_id="p1:torso",
        region_type="torso",
        owner_entity="p1",
        candidate_patch_ids=["patch_torso_1"],
        confidence=0.51,
        hidden_type="unknown_hidden",
        evidence_score=0.48,
    )

    memory.identity_memory["p1"] = {"identity_strength": 0.9}
    memory.garment_memory["g_outer_1"] = {"garment_type": "outer_garment"}
    memory.garment_memory["g_inner_1"] = {"garment_type": "inner_garment"}


# ============================================================
# Layer visualization
# ============================================================

def approximate_canonical_bbox(person_bbox: BBox, canonical_name: str) -> tuple[BBox, str]:
    x, y, w, h = float(person_bbox.x), float(person_bbox.y), float(person_bbox.w), float(person_bbox.h)

    specs: dict[str, tuple[float, float, float, float]] = {
        "head": (0.34, 0.00, 0.32, 0.18),
        "face": (0.37, 0.05, 0.26, 0.13),
        "neck": (0.42, 0.17, 0.16, 0.07),
        "torso": (0.28, 0.22, 0.44, 0.32),
        "upper_garment": (0.25, 0.20, 0.50, 0.36),
        "outer_garment": (0.22, 0.18, 0.56, 0.45),
        "inner_garment": (0.30, 0.24, 0.40, 0.28),
        "left_arm": (0.10, 0.24, 0.20, 0.34),
        "right_arm": (0.70, 0.24, 0.20, 0.34),
        "left_hand": (0.08, 0.54, 0.16, 0.10),
        "right_hand": (0.76, 0.54, 0.16, 0.10),
        "pelvis": (0.34, 0.52, 0.32, 0.13),
        "left_leg": (0.30, 0.62, 0.18, 0.36),
        "right_leg": (0.52, 0.62, 0.18, 0.36),
    }
    rx, ry, rw, rh = specs.get(canonical_name, (0.25, 0.25, 0.5, 0.5))
    return BBox(
        clamp01(x + rx * w),
        clamp01(y + ry * h),
        clamp01(rw * w),
        clamp01(rh * h),
    ), "person_bbox_heuristic_debug"


def canonical_region_bboxes(scene: SceneGraph) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for person in scene.persons:
        canonical_regions = getattr(person, "canonical_regions", {}) or {}
        if not isinstance(canonical_regions, dict):
            continue
        for name, payload in canonical_regions.items():
            bbox_payload = payload.get("bbox") if isinstance(payload, dict) else None
            bbox: BBox
            source: str
            if isinstance(bbox_payload, BBox):
                bbox = bbox_payload
                source = "payload_bbox"
            elif isinstance(bbox_payload, dict) and all(k in bbox_payload for k in ("x", "y", "w", "h")):
                bbox = BBox(
                    float(bbox_payload["x"]),
                    float(bbox_payload["y"]),
                    float(bbox_payload["w"]),
                    float(bbox_payload["h"]),
                )
                source = "payload_bbox"
            else:
                bbox, source = approximate_canonical_bbox(person.bbox, str(name))

            out.append(
                {
                    "person_id": person.person_id,
                    "canonical_name": str(name),
                    "bbox": bbox,
                    "bbox_source": source,
                    "visibility_state": payload.get("visibility_state") if isinstance(payload, dict) else None,
                    "lifecycle_state": payload.get("lifecycle_state") if isinstance(payload, dict) else None,
                }
            )
    return out


def safe_region_name(region_id: str) -> str:
    return (
        region_id.replace(":", "__")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def save_patch_artifacts(outdir: Path, rendered_patches: list[Any]) -> None:
    patches_dir = outdir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    for idx, patch in enumerate(rendered_patches, start=1):
        patch_name = safe_region_name(patch.region.region_id)
        patch_img = patch_rgb_to_bgr_image(patch.rgb_patch)
        alpha_img = alpha_to_uint8(patch.alpha_mask)

        cv2.imwrite(str(patches_dir / f"{idx:02d}_{patch_name}_rgb.jpg"), patch_img)
        cv2.imwrite(str(patches_dir / f"{idx:02d}_{patch_name}_alpha.jpg"), alpha_img)


def save_layer_artifacts(
    outdir: Path,
    image_bgr: np.ndarray,
    scene_before: SceneGraph,
    scene_after: SceneGraph,
    rois: list[Any],
    rendered_patches: list[Any],
) -> dict[str, object]:
    layers_dir = outdir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, object] = {"layers_dir": str(layers_dir), "files": []}

    def add_file(path: Path, kind: str, extra: dict[str, object] | None = None) -> None:
        artifacts["files"].append({"kind": kind, "path": str(path), **(extra or {})})

    entities = image_bgr.copy()
    for person in scene_before.persons:
        draw_bbox(entities, person.bbox, f"person:{person.person_id}", color=(0, 255, 255))
    for obj in scene_before.objects:
        draw_bbox(entities, obj.bbox, f"{obj.object_type}:{obj.object_id}", color=(255, 200, 0))
    path = layers_dir / "00_scene_entities.jpg"
    cv2.imwrite(str(path), entities)
    add_file(path, "scene_entities")

    canonical_overlay = image_bgr.copy()
    canonical_debug = []
    colors = [
        (0, 255, 0),
        (0, 180, 255),
        (255, 0, 200),
        (255, 100, 0),
        (180, 255, 0),
        (180, 0, 255),
    ]
    for idx, item in enumerate(canonical_region_bboxes(scene_after)):
        bbox = item["bbox"]
        color = colors[idx % len(colors)]
        label = f"{item['person_id']}:{item['canonical_name']}"
        draw_bbox(canonical_overlay, bbox, label, color=color, thickness=1)

        crop = crop_bbox(image_bgr, bbox)
        crop_path = layers_dir / f"canonical_{idx:02d}_{safe_region_name(label)}.jpg"
        cv2.imwrite(str(crop_path), crop)

        canonical_debug.append(
            {
                "person_id": item["person_id"],
                "canonical_name": item["canonical_name"],
                "bbox": bbox_to_dict(bbox),
                "bbox_source": item["bbox_source"],
                "visibility_state": item["visibility_state"],
                "lifecycle_state": item["lifecycle_state"],
                "crop_path": str(crop_path),
            }
        )

    path = layers_dir / "01_canonical_regions.jpg"
    cv2.imwrite(str(path), canonical_overlay)
    add_file(path, "canonical_regions_overlay", {"regions": canonical_debug})

    roi_overlay = image_bgr.copy()
    for roi in rois:
        draw_bbox(roi_overlay, roi.bbox, roi.region_id, color=(0, 255, 0), thickness=2)
    path = layers_dir / "02_selected_rois.jpg"
    cv2.imwrite(str(path), roi_overlay)
    add_file(path, "selected_rois_overlay")

    for idx, patch in enumerate(rendered_patches, start=1):
        region_name = safe_region_name(patch.region.region_id)
        before_crop = crop_bbox(image_bgr, patch.region.bbox)
        before_path = layers_dir / f"patch_{idx:02d}_{region_name}_before.jpg"
        cv2.imwrite(str(before_path), before_crop)
        add_file(before_path, "patch_before", {"region_id": patch.region.region_id})

        patch_rgb = patch_rgb_to_bgr_image(patch.rgb_patch)
        rgb_path = layers_dir / f"patch_{idx:02d}_{region_name}_rgb.jpg"
        cv2.imwrite(str(rgb_path), patch_rgb)
        add_file(rgb_path, "patch_rgb", {"region_id": patch.region.region_id})

        alpha_img = alpha_to_uint8(patch.alpha_mask)
        alpha_path = layers_dir / f"patch_{idx:02d}_{region_name}_alpha.jpg"
        cv2.imwrite(str(alpha_path), alpha_img)
        add_file(alpha_path, "patch_alpha", {"region_id": patch.region.region_id})

        region_composite = image_bgr.copy()
        composite_patch_onto_image(region_composite, patch)
        region_crop = crop_bbox(region_composite, patch.region.bbox)
        comp_path = layers_dir / f"patch_{idx:02d}_{region_name}_composited_region.jpg"
        cv2.imwrite(str(comp_path), region_crop)
        add_file(comp_path, "patch_composited_region", {"region_id": patch.region.region_id})

    return artifacts


# ============================================================
# Summary
# ============================================================

def save_summary_txt(outdir: Path, rendered_patches: list[Any]) -> Path:
    summary_path = outdir / "summary.txt"
    lines: list[str] = []

    for idx, patch in enumerate(rendered_patches, start=1):
        execution = patch.execution_trace or {}
        selection = execution.get("selection", {}) if isinstance(execution.get("selection", {}), dict) else {}
        hidden_state = execution.get("hidden_state", {}) if isinstance(execution.get("hidden_state", {}), dict) else {}
        retrieval = execution.get("retrieval", {}) if isinstance(execution.get("retrieval", {}), dict) else {}
        exec_policy = selection.get("execution_policy", {}) if isinstance(selection.get("execution_policy", {}), dict) else {}

        lines.append(f"[{idx}] {patch.region.region_id}")
        lines.append(f"  reason: {patch.region.reason}")
        lines.append(f"  confidence: {patch.confidence:.4f}")
        lines.append(
            f"  bbox: x={patch.region.bbox.x:.4f}, y={patch.region.bbox.y:.4f}, "
            f"w={patch.region.bbox.w:.4f}, h={patch.region.bbox.h:.4f}"
        )
        lines.append(f"  selected_render_strategy: {execution.get('selected_render_strategy', 'unknown')}")
        lines.append(f"  planner_selected_strategy: {selection.get('planner_selected_strategy', 'unknown')}")
        lines.append(f"  selected_render_mode: {selection.get('selected_render_mode', 'unknown')}")
        lines.append(f"  selected_family: {selection.get('selected_family', 'unknown')}")
        lines.append(f"  transition_mode: {selection.get('transition_mode', 'unknown')}")
        lines.append(f"  decision_kind: {exec_policy.get('decision_kind', 'unknown')}")
        lines.append(f"  execution_policy_kind: {exec_policy.get('execution_policy_kind', 'unknown')}")
        lines.append(f"  hidden_reconstruction_mode: {hidden_state.get('hidden_reconstruction_mode', 'unknown')}")
        lines.append(f"  retrieval_mode: {selection.get('retrieval_mode', 'unknown')}")
        lines.append(f"  candidate_count: {retrieval.get('summary', {}).get('candidate_count', 0) if isinstance(retrieval.get('summary', {}), dict) else 0}")
        lines.append(f"  memory_bundle_present: {execution.get('memory_bundle_present', False)}")
        lines.append(f"  memory_support_level: {execution.get('memory_support_level', 'unknown')}")
        lines.append(f"  memory_bundle_has_current_reuse: {execution.get('memory_bundle_has_current_reuse', False)}")
        lines.append(f"  memory_bundle_has_identity_reference: {execution.get('memory_bundle_has_identity_reference', False)}")
        lines.append(f"  memory_bundle_has_garment_reference: {execution.get('memory_bundle_has_garment_reference', False)}")
        lines.append(f"  memory_bundle_has_hidden_slot: {execution.get('memory_bundle_has_hidden_slot', False)}")
        lines.append(f"  memory_bundle_hidden_support_active: {execution.get('memory_bundle_hidden_support_active', False)}")
        lines.append(f"  fallback_reason: {execution.get('fallback_reason', 'none')}")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ============================================================
# Demo-only visible boost
# ============================================================

def add_visible_debug_boost(patch: Any) -> None:
    """
    Demo-only boost.
    Это НЕ production-логика. Нужно только, чтобы визуально было заметно,
    какие ROI были затронуты.
    """
    planner_strategy = (patch.execution_trace or {}).get("selection", {}).get("planner_selected_strategy", "")
    rgb = np.array(patch.rgb_patch, dtype=np.float32)

    if planner_strategy == "face_refine":
        rgb[..., 0] *= 1.08
        rgb[..., 1] *= 1.03
        rgb[..., 2] *= 1.03
    elif planner_strategy == "garment_reveal":
        rgb[..., 0] *= 1.12
        rgb[..., 1] *= 1.10
        rgb[..., 2] *= 1.03
    elif planner_strategy == "garment_surface_update":
        rgb[..., 1] *= 1.12
        rgb[..., 2] *= 1.06
    elif planner_strategy == "pose_local_deform":
        rgb[..., 2] *= 1.12
    elif planner_strategy == "fallback_repair":
        rgb[..., 1] *= 1.08

    rgb = np.clip(rgb, 0.0, 1.0)
    patch.rgb_patch = rgb.tolist()


def entity_region_from_roi(region_id: str, fallback_entity: str = "p1") -> tuple[str, str]:
    try:
        return parse_region_id(region_id)
    except Exception:
        return fallback_entity, region_id


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Путь к одному изображению")
    parser.add_argument("--text", required=True, help="Текст действия")
    parser.add_argument("--outdir", default="output/demo_single_photo_seeded", help="Папка вывода")
    parser.add_argument("--scene-mode", default="seeded", choices=["seeded"])
    parser.add_argument("--step-index", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--target-duration", type=float, default=2.0)
    parser.add_argument("--downscale", type=float, default=0.5, help="Во сколько раз уменьшить вход для скорости, например 0.5")
    parser.add_argument("--no-boost", action="store_true", help="Отключить demo-only визуальный буст patch colors")
    args = parser.parse_args()

    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    if 0.0 < args.downscale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            None,
            fx=args.downscale,
            fy=args.downscale,
            interpolation=cv2.INTER_AREA,
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    scene, scene_source = infer_scene_from_image(image_rgb, args.scene_mode)
    scene_before = copy.deepcopy(scene)

    intent_parser = IntentParser()
    parsed = intent_parser.parse_to_structured_intent(args.text, scene_graph=scene)

    text_encoder = BaselineStructuredTextEncoder()
    encoded = text_encoder.encode(
        raw_text=parsed.normalized_text,
        parsed_intent=parsed,
        scene_graph=scene,
    )

    labels: list[str] = []
    for token in encoded.structured_action_tokens:
        if token == "sit":
            labels.append("sit_down")
        elif token == "remove":
            labels.append("remove_garment")
        elif token == "smile":
            labels.append("smile")
        elif token == "raise_arm":
            labels.append("raise_arm")
        elif token == "turn_head":
            labels.append("turn_head")
        else:
            labels.append(token)

    intensity = float(encoded.conditioning_hints["dynamics"]["modifier_intensities"]["intensity"])
    labels.append(f"intensity={clamp01(intensity):.2f}")

    planned = PlannedState(step_index=args.step_index, labels=labels)

    memory_manager = MemoryManager()
    memory = memory_manager.initialize(scene_before)
    seed_memory(memory)

    predictor = GraphDeltaPredictor()
    delta, metrics = predictor.predict(
        scene_graph=scene_before,
        target_state=planned,
        planner_context={
            "step_index": float(args.step_index),
            "total_steps": float(args.total_steps),
            "target_duration": float(args.target_duration),
            "intensity": float(intensity),
        },
        memory=memory,
    )

    # Authoritative graph-state update stage.
    state_update_applied = True
    try:
        scene_after = apply_delta(copy.deepcopy(scene_before), delta)
    except Exception as exc:
        state_update_applied = False
        scene_after = copy.deepcopy(scene_before)
        delta.transition_diagnostics = dict(delta.transition_diagnostics or {})
        delta.transition_diagnostics["state_update_error"] = str(exc)

    # Sync memory from updated graph.
    try:
        memory = memory_manager.update_from_graph(memory, scene_after)
    except Exception as exc:
        delta.transition_diagnostics = dict(delta.transition_diagnostics or {})
        delta.transition_diagnostics["memory_update_error"] = str(exc)

    roi_selector = ROISelector()
    rois = roi_selector.select(scene_after, delta)

    renderer = PatchRenderer()
    rendered_patches = []
    roi_memory_bundles: list[dict[str, object]] = []

    for roi in rois:
        entity_id, canonical_region = entity_region_from_roi(roi.region_id)
        bundle = memory_manager.get_region_memory_bundle(memory, entity_id, canonical_region)
        bundle_dict = safe_dataclass_dict(bundle)

        roi_memory_bundles.append(
            {
                "region_id": roi.region_id,
                "entity_id": entity_id,
                "canonical_region": canonical_region,
                "bundle": bundle_dict,
            }
        )

        patch = renderer.render(
            current_frame=image_rgb,
            scene_graph=scene_after,
            delta=delta,
            memory=memory,
            region=roi,
            transition_context={
                "region_memory_bundle": bundle,
                "region_memory_bundle_serialized": bundle_dict,
                "region_memory_support_level": getattr(bundle, "memory_support_level", "none"),
                "region_memory_retrieval_reasons": list(getattr(bundle, "retrieval_reasons", []) or []),
            },
        )
        if not args.no_boost:
            add_visible_debug_boost(patch)
        rendered_patches.append(patch)

    # Main overlays.
    overlay = image_bgr.copy()
    for person in scene_after.persons:
        draw_bbox(overlay, person.bbox, f"person:{person.person_id}", color=(0, 255, 255))
    for obj in scene_after.objects:
        draw_bbox(overlay, obj.bbox, f"{obj.object_type}:{obj.object_id}", color=(255, 200, 0))
    for roi in rois:
        draw_bbox(overlay, roi.bbox, roi.region_id, color=(0, 255, 0))

    overlay_path = outdir / "overlay_rois.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    composited = image_bgr.copy()
    for patch in rendered_patches:
        composite_patch_onto_image(composited, patch)
    composited_path = outdir / "composited_preview.jpg"
    cv2.imwrite(str(composited_path), composited)

    save_patch_artifacts(outdir, rendered_patches)
    layer_artifacts = save_layer_artifacts(outdir, image_bgr, scene_before, scene_after, rois, rendered_patches)
    summary_path = save_summary_txt(outdir, rendered_patches)

    canonical_count_by_person = {
        p.person_id: len(getattr(p, "canonical_regions", {}) or {})
        for p in scene_after.persons
    }

    payload = {
        "input": {
            "image": str(image_path),
            "text": args.text,
            "scene_mode": args.scene_mode,
            "scene_source": scene_source,
            "downscale": args.downscale,
            "demo_only_visual_boost": not args.no_boost,
        },
        "scene_source": scene_source,
        "state_update_applied": state_update_applied,
        "scene_before": scene_to_debug_dict(scene_before),
        "scene_after": scene_to_debug_dict(scene_after),
        "canonical_region_count_by_person": canonical_count_by_person,
        "parsed_intent": parsed_intent_to_dict(parsed),
        "text_encoding": encoding_to_dict(encoded),
        "planned_labels": labels,
        "graph_delta": delta_to_dict(delta),
        "dynamics_metrics": {
            "delta_magnitude": metrics.delta_magnitude,
            "constraint_violations": metrics.constraint_violations,
            "temporal_smoothness_proxy": metrics.temporal_smoothness_proxy,
        },
        "memory_summary": memory_summary(memory),
        "roi_memory_bundles": roi_memory_bundles,
        "rois": [
            {
                "region_id": roi.region_id,
                "reason": roi.reason,
                "bbox": bbox_to_dict(roi.bbox),
            }
            for roi in rois
        ],
        "rendered_patches": [patch_to_dict(p) for p in rendered_patches],
        "layer_artifacts": layer_artifacts,
        "outputs": {
            "overlay_rois": str(overlay_path),
            "composited_preview": str(composited_path),
            "patches_dir": str(outdir / "patches"),
            "layers_dir": str(outdir / "layers"),
            "summary_txt": str(summary_path),
        },
    }

    debug_json_path = outdir / "debug.json"
    debug_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    person_count = len(scene_after.persons)
    canonical_total = sum(canonical_count_by_person.values())

    print("\n=== DONE ===")
    print("scene_source:", scene_source)
    print("person_count:", person_count)
    print("canonical_region_count:", canonical_total)
    print("roi_count:", len(rois))
    print("patch_count:", len(rendered_patches))
    print("overlay:", overlay_path)
    print("composited:", composited_path)
    print("layers_dir:", outdir / "layers")
    print("debug_json:", debug_json_path)
    print("patches_dir:", outdir / "patches")
    print("summary_txt:", summary_path)

    print("\nplanned_labels:")
    pprint(labels)

    print("\ncanonical_region_count_by_person:")
    pprint(canonical_count_by_person)

    print("\nmemory_summary:")
    pprint(memory_summary(memory))


if __name__ == "__main__":
    main()