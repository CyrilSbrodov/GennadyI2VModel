from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from core.schema import BBox, FrameRegionObservation, RegionMaskPropagationResult, SceneGraph
from perception.mask_store import InMemoryMaskStore

HIGH_CONFIDENCE_THRESHOLD = 0.8
MINOR_DRIFT_IOU_THRESHOLD = 0.7
MAJOR_DRIFT_IOU_THRESHOLD = 0.4
STALE_REGION_FRAME_THRESHOLD = 3


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1e-8, a.w * a.h + b.w * b.h - inter)
    return float(inter / union)


def _graph_bbox_by_region(scene_graph: SceneGraph) -> dict[str, BBox]:
    out: dict[str, BBox] = {}
    for person in scene_graph.persons:
        for part in person.body_parts:
            out[f"{person.person_id}:{part.part_type}"] = part.bbox
    return out




def seed_input_region_observations(scene_graph: SceneGraph, mask_store: InMemoryMaskStore) -> list[FrameRegionObservation]:
    seeded: list[FrameRegionObservation] = []
    for person in scene_graph.persons:
        for part in person.body_parts:
            rid = f"{person.person_id}:{part.part_type}"
            mask_ref = part.mask_ref if isinstance(part.mask_ref, str) and part.mask_ref else None
            mask_ref_exists = bool(mask_ref and mask_store.get(mask_ref) is not None)
            if mask_ref_exists:
                meta = mask_store.get(mask_ref)
                source = str((meta.source if meta else "parser")).lower()
                observation_source = "parser_mask" if "parser" in source else "detector_mask"
                mask_provenance = "input_frame_observed"
                confidence = min(1.0, max(0.4, float(part.confidence)))
                evidence = min(1.0, confidence)
            else:
                observation_source = "graph_bbox_projection"
                mask_provenance = "graph_projection_no_mask"
                confidence = 0.35
                evidence = 0.2
                mask_ref = None
            seeded.append(
                FrameRegionObservation(
                    frame_index=scene_graph.frame_index,
                    region_id=rid,
                    bbox=part.bbox,
                    mask_ref=mask_ref,
                    mask_kind="binary" if mask_ref else "none",
                    mask_provenance=mask_provenance,
                    observation_source=observation_source,
                    confidence=confidence,
                    evidence_strength_score=evidence,
                    metadata_completeness_score=1.0 if mask_ref else 0.8,
                    drift_score=0.2 if mask_ref else 0.75,
                    stale_frame_count=0,
                    is_generated_evidence=False,
                    is_carry_forward=False,
                    is_fallback_region=not bool(mask_ref),
                    diagnostics={"seeded_from_input": True},
                )
            )
    return seeded


def _patch_to_observation(frame_index: int, patch_out: Any, mask_store: InMemoryMaskStore) -> FrameRegionObservation | None:
    region = getattr(patch_out, "region", None)
    if region is None:
        return None
    alpha = np.asarray(getattr(patch_out, "alpha_mask", []), dtype=np.float32)
    if alpha.ndim != 2 or alpha.size == 0:
        return None
    coverage = float(np.mean(alpha > 0.05))
    confidence = min(HIGH_CONFIDENCE_THRESHOLD, max(0.3, 0.35 + coverage * 0.45))
    explicit_confidence = getattr(patch_out, "confidence", None)
    if isinstance(explicit_confidence, (int, float)):
        confidence = float(max(0.0, min(1.0, explicit_confidence)))
    mask_ref = mask_store.put(alpha.tolist(), confidence=confidence, source="runtime_region_mask_propagation", prefix="generated_patch_alpha", mask_kind="alpha", roi_bbox=(region.bbox.x, region.bbox.y, region.bbox.w, region.bbox.h))
    return FrameRegionObservation(frame_index=frame_index, region_id=region.region_id, bbox=region.bbox, mask_ref=mask_ref, mask_kind="alpha", mask_provenance="generated_patch_alpha", observation_source="patch_alpha_update", confidence=confidence, evidence_strength_score=min(1.0, 0.4 + coverage * 0.6), metadata_completeness_score=1.0, drift_score=0.35, stale_frame_count=0, is_generated_evidence=True, is_carry_forward=False, is_fallback_region=False, diagnostics={"alpha_coverage": coverage})


def propagate_region_masks_for_frame(frame_index: int, stable_frame: np.ndarray, previous_observations: list[FrameRegionObservation], scene_graph: SceneGraph, changed_regions: list[str], patch_outputs: list[Any], mask_store: InMemoryMaskStore, strict: bool = False) -> RegionMaskPropagationResult:
    _ = stable_frame
    prev_by_region = {o.region_id: o for o in previous_observations}
    graph_bbox_by_region = _graph_bbox_by_region(scene_graph)
    observations: dict[str, FrameRegionObservation] = {}
    violations: list[str] = []
    missing_observation_regions: list[str] = []

    for patch_out in patch_outputs:
        obs = _patch_to_observation(frame_index, patch_out, mask_store)
        if obs is not None:
            observations[obs.region_id] = obs

    for region_id, graph_bbox in graph_bbox_by_region.items():
        if region_id in observations:
            continue
        previous = prev_by_region.get(region_id)
        if previous is not None and region_id not in changed_regions:
            observations[region_id] = FrameRegionObservation(frame_index=frame_index, region_id=region_id, bbox=previous.bbox, mask_ref=previous.mask_ref, mask_kind=previous.mask_kind, mask_provenance="propagated_from_previous_frame", observation_source="carry_forward_previous_frame", confidence=max(0.1, previous.confidence * 0.95), evidence_strength_score=max(0.1, previous.evidence_strength_score * 0.95), metadata_completeness_score=previous.metadata_completeness_score, drift_score=max(previous.drift_score, 0.55), stale_frame_count=previous.stale_frame_count + 1, is_generated_evidence=False, is_carry_forward=True, is_fallback_region=False, diagnostics={"carry_forward_from": previous.frame_index})
        else:
            observations[region_id] = FrameRegionObservation(frame_index=frame_index, region_id=region_id, bbox=graph_bbox, mask_ref=None, mask_kind="none", mask_provenance="graph_projection_no_mask", observation_source="graph_bbox_projection", confidence=0.35, evidence_strength_score=0.2, metadata_completeness_score=0.8, drift_score=0.75, stale_frame_count=0, is_generated_evidence=False, is_carry_forward=False, is_fallback_region=True, diagnostics={"strict": strict})
            missing_observation_regions.append(region_id)

        if region_id in changed_regions and region_id not in observations:
            violations.append("missing_changed_region_mask_evidence")

    for region_id in changed_regions:
        obs = observations.get(region_id)
        if obs is None or obs.observation_source != "patch_alpha_update":
            violations.append("changed_region_without_fresh_mask_evidence")

    statuses: dict[str, list[str]] = {k: [] for k in ["aligned", "minor_drift", "major_drift", "missing_mask", "stale_carry_forward", "fallback_only"]}
    drift_scores: list[float] = []
    per_region: dict[str, Any] = {}
    for region_id, obs in observations.items():
        graph_bbox = graph_bbox_by_region.get(region_id, obs.bbox)
        observed_bbox = obs.bbox
        iou = _bbox_iou(graph_bbox, observed_bbox)
        drift_score = 1.0 - iou if obs.mask_ref else max(0.6, obs.drift_score)
        if obs.mask_ref is None and obs.is_fallback_region:
            status = "fallback_only"
        elif obs.mask_ref is None:
            status = "missing_mask"
        elif obs.is_carry_forward and obs.stale_frame_count >= STALE_REGION_FRAME_THRESHOLD:
            status = "stale_carry_forward"
        elif iou >= MINOR_DRIFT_IOU_THRESHOLD:
            status = "aligned"
        elif iou >= MAJOR_DRIFT_IOU_THRESHOLD:
            status = "minor_drift"
        else:
            status = "major_drift"
        statuses[status].append(region_id)
        drift_scores.append(drift_score)
        per_region[region_id] = {"graph_bbox": asdict(graph_bbox), "observed_bbox": asdict(observed_bbox), "bbox_iou": iou, "mask_available": obs.mask_ref is not None, "mask_area_ratio": obs.diagnostics.get("alpha_coverage") if isinstance(obs.diagnostics, dict) else None, "drift_score": drift_score, "stale_frame_count": obs.stale_frame_count, "region_status": status}

    diagnostics = {"missing_observation_regions": missing_observation_regions, "violations": violations, "missing_changed_region_mask_evidence_count": sum(1 for v in violations if v in {"missing_changed_region_mask_evidence", "changed_region_without_fresh_mask_evidence"})}
    region_drift_summary = {"region_count": len(observations), "aligned_count": len(statuses["aligned"]), "minor_drift_count": len(statuses["minor_drift"]), "major_drift_count": len(statuses["major_drift"]), "missing_mask_count": len(statuses["missing_mask"]), "stale_carry_forward_count": len(statuses["stale_carry_forward"]), "fallback_only_count": len(statuses["fallback_only"]), "mean_drift_score": float(sum(drift_scores) / max(1, len(drift_scores))), "max_drift_score": float(max(drift_scores) if drift_scores else 0.0), "regions_by_status": statuses, "per_region": per_region}
    return RegionMaskPropagationResult(frame_index=frame_index, observations=list(observations.values()), updated_mask_refs={rid: obs.mask_ref for rid, obs in observations.items() if obs.mask_ref}, region_drift_summary=region_drift_summary, fallback_count=sum(1 for obs in observations.values() if obs.is_fallback_region), carry_forward_count=sum(1 for obs in observations.values() if obs.is_carry_forward), generated_evidence_count=sum(1 for obs in observations.values() if obs.is_generated_evidence), high_confidence_observation_count=sum(1 for obs in observations.values() if obs.confidence >= HIGH_CONFIDENCE_THRESHOLD), stale_region_count=sum(1 for obs in observations.values() if obs.stale_frame_count >= STALE_REGION_FRAME_THRESHOLD), diagnostics=diagnostics)
