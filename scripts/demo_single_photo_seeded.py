from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

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
from planning.transition_engine import PlannedState
from rendering.roi_renderer import ROISelector, PatchRenderer
from text.intent_parser import IntentParser
from text.text_encoder import BaselineStructuredTextEncoder


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def bbox_to_pixels(bbox: BBox, image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x0 = max(0, min(w - 1, int(bbox.x * w)))
    y0 = max(0, min(h - 1, int(bbox.y * h)))
    x1 = max(x0 + 1, min(w, int((bbox.x + bbox.w) * w)))
    y1 = max(y0 + 1, min(h, int((bbox.y + bbox.h) * h)))
    return x0, y0, x1, y1


def draw_bbox(image: np.ndarray, bbox: BBox, label: str, color: tuple[int, int, int] = (0, 255, 0)) -> None:
    x0, y0, x1, y1 = bbox_to_pixels(bbox, image)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        image,
        label,
        (x0, max(16, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def patch_rgb_to_bgr_image(rgb_patch) -> np.ndarray:
    arr = np.array(rgb_patch, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def alpha_to_uint8(alpha_mask) -> np.ndarray:
    alpha = np.array(alpha_mask, dtype=np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)
    return (alpha * 255.0).astype(np.uint8)


def composite_patch_onto_image(base_bgr: np.ndarray, patch) -> None:
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


def save_patch_artifacts(outdir: Path, rendered_patches: list) -> None:
    patches_dir = outdir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    for idx, patch in enumerate(rendered_patches, start=1):
        patch_name = patch.region.region_id.replace(":", "__")
        patch_img = patch_rgb_to_bgr_image(patch.rgb_patch)
        alpha_img = alpha_to_uint8(patch.alpha_mask)

        cv2.imwrite(str(patches_dir / f"{idx:02d}_{patch_name}_rgb.jpg"), patch_img)
        cv2.imwrite(str(patches_dir / f"{idx:02d}_{patch_name}_alpha.jpg"), alpha_img)


def save_summary_txt(outdir: Path, rendered_patches: list) -> Path:
    summary_path = outdir / "summary.txt"
    lines: list[str] = []

    for idx, patch in enumerate(rendered_patches, start=1):
        execution = patch.execution_trace or {}
        selection = execution.get("selection", {})
        hidden_state = execution.get("hidden_state", {})
        retrieval = execution.get("retrieval", {})
        fallback_reason = execution.get("fallback_reason", "none")

        lines.append(f"[{idx}] {patch.region.region_id}")
        lines.append(f"  reason: {patch.region.reason}")
        lines.append(f"  confidence: {patch.confidence:.4f}")
        lines.append(
            f"  bbox: x={patch.region.bbox.x:.4f}, y={patch.region.bbox.y:.4f}, "
            f"w={patch.region.bbox.w:.4f}, h={patch.region.bbox.h:.4f}"
        )
        lines.append(f"  strategy: {selection.get('selected_strategy', 'unknown')}")
        lines.append(f"  family: {selection.get('selected_family', 'unknown')}")
        lines.append(f"  transition_mode: {selection.get('transition_mode', 'unknown')}")
        lines.append(f"  hidden_mode: {hidden_state.get('hidden_reconstruction_mode', 'unknown')}")
        lines.append(f"  retrieval_mode: {selection.get('retrieval_mode', 'unknown')}")
        lines.append(f"  candidate_count: {retrieval.get('summary', {}).get('candidate_count', 0)}")
        lines.append(f"  fallback_reason: {fallback_reason}")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def scene_to_debug_dict(scene: SceneGraph) -> dict[str, object]:
    persons = []
    for p in scene.persons:
        persons.append(
            {
                "person_id": p.person_id,
                "track_id": p.track_id,
                "bbox": {"x": p.bbox.x, "y": p.bbox.y, "w": p.bbox.w, "h": p.bbox.h},
                "pose_state": getattr(p.pose_state, "coarse_pose", None),
                "expression_state": getattr(p.expression_state, "label", None),
                "body_parts": [
                    {
                        "part_id": bp.part_id,
                        "part_type": bp.part_type,
                        "visibility": bp.visibility,
                    }
                    for bp in p.body_parts
                ],
                "garments": [
                    {
                        "garment_id": g.garment_id,
                        "garment_type": g.garment_type,
                        "garment_state": getattr(g, "garment_state", None),
                    }
                    for g in p.garments
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
                "bbox": {"x": obj.bbox.x, "y": obj.bbox.y, "w": obj.bbox.w, "h": obj.bbox.h},
            }
            for obj in scene.objects
        ],
    }


def parsed_intent_to_dict(parsed) -> dict[str, object]:
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


def encoding_to_dict(encoded) -> dict[str, object]:
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


def delta_to_dict(delta) -> dict[str, object]:
    def region_to_dict(region):
        return {
            "region_id": region.region_id,
            "reason": region.reason,
            "bbox": {
                "x": region.bbox.x,
                "y": region.bbox.y,
                "w": region.bbox.w,
                "h": region.bbox.h,
            },
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


def patch_to_dict(patch) -> dict[str, object]:
    return {
        "region_id": patch.region.region_id,
        "reason": patch.region.reason,
        "bbox": {
            "x": patch.region.bbox.x,
            "y": patch.region.bbox.y,
            "w": patch.region.bbox.w,
            "h": patch.region.bbox.h,
        },
        "confidence": patch.confidence,
        "shape": {
            "height": patch.height,
            "width": patch.width,
            "channels": patch.channels,
        },
        "debug_trace": patch.debug_trace,
        "execution_trace": patch.execution_trace,
    }


def make_seeded_scene() -> SceneGraph:
    person = PersonNode(
        person_id="p1",
        track_id="t1",
        bbox=BBox(0.2, 0.08, 0.5, 0.84),
        mask_ref=None,
        pose_state=PoseState(coarse_pose="standing"),
        expression_state=ExpressionState(label="neutral"),
        body_parts=[
            BodyPartNode(part_id="bp_face", part_type="face", visibility="visible"),
            BodyPartNode(part_id="bp_arm", part_type="arm", visibility="visible"),
            BodyPartNode(part_id="bp_legs", part_type="legs", visibility="visible"),
            BodyPartNode(part_id="bp_torso", part_type="torso", visibility="visible"),
            BodyPartNode(part_id="bp_pelvis", part_type="pelvis", visibility="visible"),
        ],
        garments=[
            GarmentNode(garment_id="g1", garment_type="coat", garment_state="worn"),
        ],
    )

    chair = SceneObjectNode(
        object_id="chair_1",
        object_type="chair",
        bbox=BBox(0.18, 0.62, 0.42, 0.22),
    )

    return SceneGraph(frame_index=0, persons=[person], objects=[chair])


def infer_scene_from_image(image: np.ndarray, mode: str) -> SceneGraph:
    _ = image
    if mode == "seeded":
        return make_seeded_scene()

    person = PersonNode(
        person_id="p1",
        track_id="t1",
        bbox=BBox(0.18, 0.08, 0.54, 0.84),
        mask_ref=None,
        pose_state=PoseState(coarse_pose="standing"),
        expression_state=ExpressionState(label="neutral"),
        body_parts=[
            BodyPartNode(part_id="bp_face", part_type="face", visibility="visible"),
            BodyPartNode(part_id="bp_arm", part_type="arm", visibility="visible"),
            BodyPartNode(part_id="bp_legs", part_type="legs", visibility="visible"),
            BodyPartNode(part_id="bp_torso", part_type="torso", visibility="visible"),
            BodyPartNode(part_id="bp_pelvis", part_type="pelvis", visibility="visible"),
        ],
        garments=[
            GarmentNode(garment_id="g1", garment_type="coat", garment_state="worn"),
        ],
    )
    return SceneGraph(frame_index=0, persons=[person], objects=[])


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
    memory.garment_memory["g1"] = {"garment_type": "coat"}


def add_visible_debug_boost(patch) -> None:
    """
    Для demo-режима делаем эффект заметнее, иначе визуально почти ничего не видно.
    Это НЕ production-логика, а просто визуальный буст для smoke/demo.
    """
    strategy = (patch.execution_trace or {}).get("selection", {}).get("selected_strategy", "")
    rgb = np.array(patch.rgb_patch, dtype=np.float32)

    if strategy == "face_refine":
        rgb[..., 0] *= 1.06
        rgb[..., 1] *= 1.02
        rgb[..., 2] *= 1.02
    elif strategy == "garment_reveal":
        rgb[..., 0] *= 1.10
        rgb[..., 1] *= 1.08
        rgb[..., 2] *= 1.02
    elif strategy == "garment_surface_update":
        rgb[..., 1] *= 1.10
        rgb[..., 2] *= 1.06
    elif strategy == "pose_local_deform":
        rgb[..., 2] *= 1.10
    elif strategy == "fallback_repair":
        rgb[..., 1] *= 1.06

    rgb = np.clip(rgb, 0.0, 1.0)
    patch.rgb_patch = rgb.tolist()


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

    scene = infer_scene_from_image(image_rgb, args.scene_mode)

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
    memory = VideoMemory()
    seed_memory(memory)

    predictor = GraphDeltaPredictor()
    delta, metrics = predictor.predict(
        scene_graph=scene,
        target_state=planned,
        planner_context={
            "step_index": float(args.step_index),
            "total_steps": float(args.total_steps),
            "target_duration": float(args.target_duration),
            "intensity": float(intensity),
        },
        memory=memory,
    )

    roi_selector = ROISelector()
    rois = roi_selector.select(scene, delta)

    renderer = PatchRenderer()
    rendered_patches = []
    for roi in rois:
        patch = renderer.render(
            current_frame=image_rgb,
            scene_graph=scene,
            delta=delta,
            memory=memory,
            region=roi,
        )
        add_visible_debug_boost(patch)
        rendered_patches.append(patch)

    overlay = image_bgr.copy()
    for person in scene.persons:
        draw_bbox(overlay, person.bbox, f"person:{person.person_id}", color=(0, 255, 255))
    for obj in scene.objects:
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
    summary_path = save_summary_txt(outdir, rendered_patches)

    payload = {
        "input": {
            "image": str(image_path),
            "text": args.text,
            "scene_mode": args.scene_mode,
            "downscale": args.downscale,
        },
        "scene": scene_to_debug_dict(scene),
        "parsed_intent": parsed_intent_to_dict(parsed),
        "text_encoding": encoding_to_dict(encoded),
        "planned_labels": labels,
        "graph_delta": delta_to_dict(delta),
        "dynamics_metrics": {
            "delta_magnitude": metrics.delta_magnitude,
            "constraint_violations": metrics.constraint_violations,
            "temporal_smoothness_proxy": metrics.temporal_smoothness_proxy,
        },
        "rois": [
            {
                "region_id": roi.region_id,
                "reason": roi.reason,
                "bbox": {
                    "x": roi.bbox.x,
                    "y": roi.bbox.y,
                    "w": roi.bbox.w,
                    "h": roi.bbox.h,
                },
            }
            for roi in rois
        ],
        "rendered_patches": [patch_to_dict(p) for p in rendered_patches],
    }

    debug_json_path = outdir / "debug.json"
    debug_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print("overlay:", overlay_path)
    print("composited:", composited_path)
    print("debug_json:", debug_json_path)
    print("patches_dir:", outdir / "patches")
    print("summary_txt:", summary_path)

    print("\nplanned_labels:")
    pprint(labels)
    print("\nroi_count:", len(rois))
    print("patch_count:", len(rendered_patches))


if __name__ == "__main__":
    main()