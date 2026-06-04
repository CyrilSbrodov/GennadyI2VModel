from __future__ import annotations

from typing import Any

from core.region_ids import parse_region_id
from core.schema import BBox, GraphDelta, RegionRef, SceneGraph, VideoMemory
from perception.mask_store import DEFAULT_MASK_STORE, InMemoryMaskStore
from representation.scene_graph_queries import SceneGraphQueries
from runtime.region_routing import RegionRoutingDecision

ROI_SOURCES = {"parser_mask_bbox", "body_part_keypoints", "garment_coverage", "person_bbox_fallback", "unknown"}
SOURCE_NODE_TYPES = {"body_part", "garment", "face_region", "canonical_region", "fallback"}
_BODY_REGIONS = {"head", "hair", "neck", "torso", "pelvis", "left_arm", "right_arm", "left_hand", "right_hand", "left_leg", "right_leg", "arms", "hands", "legs"}
_FACE_REGIONS = {"face", "eyes", "mouth", "cheek"}
_GARMENT_REGIONS = {"garments", "outer_garment", "upper_garment", "lower_garment", "inner_garment", "sleeves", "dress", "accessories"}
_CRITICAL_FIELDS = ("region_id", "entity_id", "canonical_region", "bbox_xywh", "roi_source", "source_node_type")
_OPTIONAL_EVIDENCE_FIELDS = (
    "mask_ref",
    "mask_kind",
    "source_confidence",
    "canonical_confidence",
    "memory_entry_exists",
    "route_decision",
)


def _bbox_xywh(bbox: BBox) -> dict[str, float]:
    return {"x": float(bbox.x), "y": float(bbox.y), "w": float(bbox.w), "h": float(bbox.h)}


def _reason_value(reason: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(reason or "").split(";"):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _classify_roi_source(reason: str, mask_ref: str | None, source_node_type: str) -> str:
    explicit = _reason_value(reason, "roi_source")
    if explicit in ROI_SOURCES:
        return explicit
    lower = str(reason or "").lower()
    if "parser_mask_bbox" in lower or mask_ref:
        return "parser_mask_bbox"
    if "keypoints" in lower:
        return "body_part_keypoints"
    if "garment_coverage" in lower:
        return "garment_coverage"
    if "person_bbox_fallback" in lower or "fallback:person_bbox" in lower:
        return "person_bbox_fallback"
    if source_node_type == "fallback":
        return "person_bbox_fallback"
    return "unknown"


def _canonical_payload(person: object | None, canonical_region: str) -> dict[str, Any]:
    payload = getattr(person, "canonical_regions", {}).get(canonical_region, {}) if person is not None else {}
    return dict(payload) if isinstance(payload, dict) else {}


def _extract_alt_value(alternatives: list[str], prefix: str) -> str | None:
    for alt in alternatives:
        if alt.startswith(prefix):
            value = alt.split(":", 1)[1]
            return None if value in {"None", "", "unknown"} else value
    return None


def _resolve_source_node(person: object | None, canonical_region: str, canonical: dict[str, Any]) -> tuple[str, object | None]:
    if person is None:
        return "fallback", None
    for part in getattr(person, "body_parts", []):
        if getattr(part, "part_type", "") == canonical_region or getattr(part, "canonical_slot", "") == canonical_region:
            return "body_part", part
    for garment in getattr(person, "garments", []):
        if getattr(garment, "garment_type", "") == canonical_region:
            return "garment", garment
    if canonical:
        return "canonical_region", None
    return "fallback", None


def _mask_metadata(mask_ref: str | None, mask_store: InMemoryMaskStore | None = None) -> dict[str, Any]:
    if not mask_ref:
        return {}
    store = mask_store or DEFAULT_MASK_STORE
    stored = store.get(mask_ref)
    if stored is None:
        return {"mask_ref": mask_ref, "mask_lookup_missing": True}
    extra = stored.extra if isinstance(stored.extra, dict) else {}
    return {
        "mask_ref": mask_ref,
        "mask_kind": stored.mask_kind,
        "mask_confidence": float(stored.confidence),
        "mask_source": stored.source,
        "mask_backend": stored.backend,
        "mask_pixel_count": extra.get("pixel_count"),
        "mask_bbox_xyxy": extra.get("bbox_xyxy"),
        "mask_frame_size": stored.frame_size,
        "mask_tags": list(stored.tags),
        "parser_class_name": extra.get("parser_class_name"),
        "parser_class_id": extra.get("class_id"),
    }


def _memory_support(memory: VideoMemory, region_id: str) -> dict[str, Any]:
    entry = memory.canonical_region_memory.get(region_id)
    hidden_slot = memory.hidden_region_slots.get(region_id)
    return {
        "memory_entry_exists": entry is not None,
        "memory_confidence": float(entry.confidence) if entry is not None else None,
        "memory_visibility_state": str(entry.visibility_state) if entry is not None else "unknown",
        "memory_reliable_for_reuse": bool(entry.reliable_for_reuse) if entry is not None else False,
        "memory_suitable_for_reveal": bool(entry.suitable_for_reveal) if entry is not None else False,
        "memory_generated": bool(entry.generated) if entry is not None else False,
        "memory_inferred": bool(entry.inferred) if entry is not None else False,
        "memory_support_level": "strong" if entry and entry.reliable_for_reuse and entry.evidence_score >= 0.7 else ("weak" if entry or hidden_slot else "none"),
        "memory_retrieval_reasons": ([f"lifecycle:{entry.reveal_lifecycle}"] if entry is not None else []) + ([f"hidden_slot:{hidden_slot.hidden_type}"] if hidden_slot is not None else []),
    }


def _delta_metadata(delta: GraphDelta | None, region_id: str, canonical_region: str) -> dict[str, Any]:
    if delta is None:
        return {
            "transition_phase": "unknown",
            "region_transition_mode": "unknown",
            "affected_region": False,
            "newly_revealed": False,
            "newly_occluded": False,
            "semantic_reasons": [],
        }
    revealed = {r.region_id for r in delta.newly_revealed_regions}
    occluded = {r.region_id for r in delta.newly_occluded_regions}
    return {
        "transition_phase": delta.transition_phase,
        "region_transition_mode": delta.region_transition_mode.get(canonical_region, delta.region_transition_mode.get(region_id, "unknown")),
        "affected_region": region_id in delta.affected_regions or canonical_region in delta.affected_regions,
        "newly_revealed": region_id in revealed,
        "newly_occluded": region_id in occluded,
        "semantic_reasons": list(delta.semantic_reasons),
    }



def _evidence_strength(metadata: dict[str, Any]) -> float:
    mask_strength = 0.0
    if metadata.get("mask_ref") and not metadata.get("mask_lookup_missing"):
        mask_strength = 0.35 + 0.35 * max(
            _as_float(metadata.get("mask_confidence")),
            _as_float(metadata.get("source_confidence")),
            _as_float(metadata.get("canonical_confidence")),
        )
    memory_strength = 0.0
    if metadata.get("memory_reliable_for_reuse") or metadata.get("memory_suitable_for_reveal"):
        memory_strength = 0.15 + 0.25 * _as_float(metadata.get("memory_confidence"))
    elif metadata.get("memory_entry_exists"):
        memory_strength = 0.10 + 0.15 * _as_float(metadata.get("memory_confidence"))
    route_strength = 0.0
    if metadata.get("route_decision") not in {None, "", "unknown"}:
        route_strength = 0.10 + 0.20 * _as_float(metadata.get("routing_confidence"))
    canonical_strength = 0.15 * _as_float(metadata.get("canonical_confidence"))
    return round(max(0.0, min(1.0, mask_strength + memory_strength + route_strength + canonical_strength)), 4)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _completeness(metadata: dict[str, Any]) -> tuple[float, list[str]]:
    fields = _CRITICAL_FIELDS + _OPTIONAL_EVIDENCE_FIELDS
    missing: list[str] = []
    present = 0
    for field in fields:
        value = metadata.get(field)
        if value is None or value == "" or value == "unknown" or value == []:
            missing.append(field)
        else:
            present += 1
    return round(present / len(fields), 4), missing


def build_region_metadata(
    *,
    scene_graph: SceneGraph,
    memory: VideoMemory,
    region: RegionRef,
    route_decision: RegionRoutingDecision | None,
    delta: GraphDelta | None = None,
    mask_store: InMemoryMaskStore | None = None,
) -> dict[str, object]:
    trace: list[str] = ["region_metadata_bridge:v1"]
    parse_error = ""
    try:
        entity_id, canonical_region = parse_region_id(region.region_id)
    except ValueError as exc:
        entity_id, canonical_region = "unknown", region.region_id
        parse_error = str(exc)
        trace.append(f"parse_error:{parse_error}")

    person = SceneGraphQueries._person(scene_graph, entity_id) if entity_id != "unknown" else None
    canonical = _canonical_payload(person, canonical_region)
    source_node_type, source_node = _resolve_source_node(person, canonical_region, canonical)
    trace.append(f"source_node_type:{source_node_type}")

    node_mask_ref = getattr(source_node, "mask_ref", None) if source_node is not None else None
    canonical_mask_ref = canonical.get("mask_ref") if isinstance(canonical.get("mask_ref"), str) else None
    person_mask_ref = getattr(person, "mask_ref", None) if person is not None else None
    mask_ref = node_mask_ref or canonical_mask_ref
    if not mask_ref and "mask_ref=" in region.reason:
        mask_ref = _reason_value(region.reason, "mask_ref") or None
    roi_source = _classify_roi_source(region.reason, mask_ref, source_node_type)
    if roi_source == "person_bbox_fallback" and not mask_ref:
        source_node_type = "fallback" if not canonical else source_node_type

    alternatives = list(getattr(source_node, "alternatives", [])) if source_node is not None else []
    mask_meta = _mask_metadata(mask_ref, mask_store=mask_store)
    parser_class_name = mask_meta.get("parser_class_name") or _extract_alt_value(alternatives, "parser_class:")
    parser_class_id = mask_meta.get("parser_class_id") or _extract_alt_value(alternatives, "class_id:")

    metadata: dict[str, Any] = {
        "region_id": region.region_id,
        "entity_id": entity_id,
        "canonical_region": canonical_region,
        "bbox_xywh": _bbox_xywh(region.bbox),
        "roi_reason": region.reason,
        "roi_source": roi_source,
        "source_node_type": source_node_type if source_node_type in SOURCE_NODE_TYPES else "fallback",
        "source_node_id": getattr(source_node, "part_id", getattr(source_node, "garment_id", None)) if source_node is not None else None,
        "source_confidence": float(getattr(source_node, "confidence", canonical.get("confidence", 0.0)) or 0.0),
        "source_provenance": getattr(source_node, "source", canonical.get("provenance", "unknown")) if source_node is not None else canonical.get("provenance", "unknown"),
        "source": getattr(source_node, "source", canonical.get("provenance", "unknown")) if source_node is not None else canonical.get("provenance", "unknown"),
        "visibility_state": str(getattr(source_node, "visibility", canonical.get("visibility_state", "unknown"))),
        "lifecycle_state": str(canonical.get("lifecycle_state", "unknown")),
        "parser_class_name": parser_class_name,
        "parser_class_id": parser_class_id,
        "alternatives": alternatives,
        "canonical_visibility_state": str(canonical.get("visibility_state", "unknown")),
        "canonical_lifecycle_state": str(canonical.get("lifecycle_state", "unknown")),
        "canonical_confidence": float(canonical.get("confidence", 0.0) or 0.0),
        "canonical_provenance": str(canonical.get("provenance", "unknown")),
        "canonical_source_regions": list(canonical.get("source_regions", [])) if isinstance(canonical.get("source_regions", []), list) else [],
        "canonical_coverage_hints": list(canonical.get("coverage_hints", [])) if isinstance(canonical.get("coverage_hints", []), list) else [],
        "canonical_attachment_hints": list(canonical.get("attachment_hints", [])) if isinstance(canonical.get("attachment_hints", []), list) else [],
        "route_decision": route_decision.decision if route_decision is not None else "unknown",
        "renderer_mode_hint": route_decision.renderer_mode_hint if route_decision is not None else "unknown",
        "reveal_mode": route_decision.reveal_mode if route_decision is not None else "none",
        "synthesis_required": bool(route_decision.synthesis_required) if route_decision is not None else False,
        "routing_confidence": float(route_decision.confidence) if route_decision is not None else 0.0,
        "routing_reasons": list(route_decision.reasons) if route_decision is not None else [],
        "should_render": bool(getattr(route_decision, "should_render", True)) if route_decision is not None else True,
        "parse_error": parse_error,
        "is_identity_sensitive": canonical_region in _FACE_REGIONS or canonical_region in {"face", "head", "hair"},
        "is_garment_region": canonical_region in _GARMENT_REGIONS,
        "is_body_region": canonical_region in _BODY_REGIONS,
        "is_face_region": canonical_region in _FACE_REGIONS or canonical_region == "face",
    }
    metadata.update({key: value for key, value in mask_meta.items() if value is not None})
    if parser_class_name and not metadata.get("parser_class_name"):
        metadata["parser_class_name"] = parser_class_name
    if parser_class_id is not None and not metadata.get("parser_class_id"):
        metadata["parser_class_id"] = parser_class_id
    if person_mask_ref and roi_source == "person_bbox_fallback":
        metadata["person_mask_ref"] = person_mask_ref
    metadata.update(_memory_support(memory, region.region_id))
    if route_decision is not None:
        metadata["memory_support_level"] = route_decision.memory_support_level or metadata.get("memory_support_level", "none")
    metadata.update(_delta_metadata(delta, region.region_id, canonical_region))
    if metadata.get("mask_ref"):
        trace.append("mask_store:hit" if not metadata.get("mask_lookup_missing") else "mask_store:missing")
    if canonical:
        trace.append("canonical_payload:present")
    if route_decision is not None:
        trace.append("routing_decision:present")
    if region.region_id in memory.canonical_region_memory:
        trace.append("memory_entry:present")
    score, missing = _completeness(metadata)
    fallback_without_mask = metadata.get("roi_source") == "person_bbox_fallback" and not metadata.get("mask_ref")
    if fallback_without_mask:
        score = min(score, 0.45)
        if "mask_ref" not in missing:
            missing.append("mask_ref")
    metadata["metadata_completeness_score"] = score
    evidence_score = _evidence_strength(metadata)
    if fallback_without_mask:
        evidence_score = min(evidence_score, 0.15)
    metadata["evidence_strength_score"] = evidence_score
    metadata["missing_fields"] = missing
    metadata["metadata_source_trace"] = trace
    return metadata
