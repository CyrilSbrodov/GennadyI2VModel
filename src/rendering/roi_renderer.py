from __future__ import annotations

from dataclasses import dataclass, field

from core.region_ids import make_region_id, parse_region_id
from core.schema import BBox, GraphDelta, RegionRef, SceneGraph, VideoMemory
from utils_tensor import alpha_radial, crop, mean_color, shape, zeros


@dataclass(slots=True)
class RenderedPatch:
    region: RegionRef
    rgb_patch: list[list[list[float]]]
    alpha_mask: list[list[float]]
    height: int
    width: int
    channels: int
    uncertainty_map: list[list[float]] | None = None
    confidence: float = 0.0
    z_index: int = 0
    debug_trace: list[str] = field(default_factory=list)


class ROISelector:
    def _person_fallback_bbox(self, person_bbox: BBox, region_type: str) -> BBox:
        if region_type == "head" or "face" in region_type:
            return BBox(person_bbox.x + person_bbox.w * 0.3, person_bbox.y, person_bbox.w * 0.4, person_bbox.h * 0.22)
        if "upper_arm" in region_type or "lower_arm" in region_type or "hand" in region_type or "arm" in region_type:
            return BBox(person_bbox.x, person_bbox.y + person_bbox.h * 0.2, person_bbox.w, person_bbox.h * 0.3)
        if "pelvis" in region_type:
            return BBox(person_bbox.x + person_bbox.w * 0.3, person_bbox.y + person_bbox.h * 0.5, person_bbox.w * 0.4, person_bbox.h * 0.2)
        if "leg" in region_type or "foot" in region_type:
            return BBox(person_bbox.x + person_bbox.w * 0.15, person_bbox.y + person_bbox.h * 0.55, person_bbox.w * 0.7, person_bbox.h * 0.42)
        return BBox(person_bbox.x + person_bbox.w * 0.1, person_bbox.y + person_bbox.h * 0.18, person_bbox.w * 0.8, person_bbox.h * 0.62)

    def _bbox_from_keypoints(self, keypoints: list, margin: float = 0.03) -> BBox | None:
        pts = [(k.x, k.y) for k in keypoints if getattr(k, "confidence", 0.0) > 0.2]
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0.0, min(xs) - margin)
        y0 = max(0.0, min(ys) - margin)
        x1 = min(1.0, max(xs) + margin)
        y1 = min(1.0, max(ys) + margin)
        return BBox(x0, y0, max(0.05, x1 - x0), max(0.05, y1 - y0))

    def semantic_roi_from_graph(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        person = next((p for p in scene_graph.persons if p.person_id == entity_id), None)
        if person is None:
            return None

        if region_type in {"face", "head"}:
            part = next((bp for bp in person.body_parts if bp.part_type == "head"), None)
            if part and part.keypoints:
                bbox = self._bbox_from_keypoints(part.keypoints)
                if bbox:
                    return RegionRef(make_region_id(entity_id, region_type), bbox, "graph_semantic:body_part_keypoints")

        if region_type in {"left_arm", "right_arm", "left_upper_arm", "right_upper_arm", "sleeves"}:
            arm_keys = {"left_arm": {"left_upper_arm", "left_lower_arm", "left_hand"}, "right_arm": {"right_upper_arm", "right_lower_arm", "right_hand"}}
            wanted = arm_keys.get(region_type, {region_type})
            kp = [k for bp in person.body_parts if bp.part_type in wanted for k in bp.keypoints]
            if kp:
                bbox = self._bbox_from_keypoints(kp)
                if bbox:
                    return RegionRef(make_region_id(entity_id, region_type), bbox, "graph_semantic:arm_keypoints")

        if region_type in {"pelvis", "legs", "left_lower_leg", "right_lower_leg"}:
            wanted = {"pelvis"} if region_type == "pelvis" else {"left_upper_leg", "left_lower_leg", "right_upper_leg", "right_lower_leg", "left_foot", "right_foot"}
            kp = [k for bp in person.body_parts if bp.part_type in wanted for k in bp.keypoints]
            if kp:
                bbox = self._bbox_from_keypoints(kp)
                if bbox:
                    return RegionRef(make_region_id(entity_id, region_type), bbox, "graph_semantic:lower_body_keypoints")

        if region_type in {"garments", "torso"} and person.garments:
            covered = {t for g in person.garments for t in g.coverage_targets}
            if covered:
                torso_related = any("torso" in t or "upper_arm" in t for t in covered)
                if torso_related:
                    bbox = BBox(person.bbox.x + person.bbox.w * 0.15, person.bbox.y + person.bbox.h * 0.18, person.bbox.w * 0.7, person.bbox.h * 0.42)
                    return RegionRef(make_region_id(entity_id, region_type), bbox, "graph_semantic:garment_coverage")
        return None

    def fallback_roi_from_person_bbox(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        person = next((p for p in scene_graph.persons if p.person_id == entity_id), None)
        if person is None:
            return None
        return RegionRef(make_region_id(entity_id, region_type), self._person_fallback_bbox(person.bbox, region_type), "fallback:person_bbox_template")

    def _resolve_region(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        return self.semantic_roi_from_graph(scene_graph, entity_id, region_type) or self.fallback_roi_from_person_bbox(scene_graph, entity_id, region_type)

    def select(self, scene_graph: SceneGraph, delta: GraphDelta) -> list[RegionRef]:
        selected = list(delta.newly_revealed_regions)
        selected.extend(delta.newly_occluded_regions)
        entity = delta.affected_entities[0] if delta.affected_entities else (scene_graph.persons[0].person_id if scene_graph.persons else "scene")

        requested: list[str] = list(delta.affected_regions)
        if "expression_change" in delta.semantic_reasons or delta.expression_deltas:
            requested.append("face")
        if "garment_change" in delta.semantic_reasons or delta.garment_deltas:
            requested.append("garments")
        if "raise_arm" in delta.semantic_reasons:
            requested.extend(["left_arm", "sleeves"])
        if "sit_down" in delta.semantic_reasons:
            requested.extend(["pelvis", "legs"])

        for region_type in requested:
            resolved = self._resolve_region(scene_graph, entity, region_type)
            if resolved:
                selected.append(resolved)

        dedup: dict[str, RegionRef] = {}
        for region in selected:
            region_entity, region_type = parse_region_id(region.region_id)
            canonical = make_region_id(region_entity, region_type)
            dedup[canonical] = RegionRef(canonical, region.bbox, region.reason)

        if not dedup and scene_graph.persons:
            person = scene_graph.persons[0]
            dedup[make_region_id(person.person_id, "fallback")] = RegionRef(
                make_region_id(person.person_id, "fallback"), person.bbox, "fallback:delta_low"
            )
        return list(dedup.values())


class PatchRenderer:
    def _bbox_to_pixels(self, bbox: BBox, frame: list) -> tuple[int, int, int, int]:
        h, w, _ = shape(frame)
        x0 = max(0, min(w - 1, int(bbox.x * w)))
        y0 = max(0, min(h - 1, int(bbox.y * h)))
        x1 = max(x0 + 1, min(w, int((bbox.x + bbox.w) * w)))
        y1 = max(y0 + 1, min(h, int((bbox.y + bbox.h) * h)))
        return x0, y0, x1, y1

    def _blend_memory_patch(self, roi: list[list[list[float]]], memory_patch: list[list[list[float]]], weight: float) -> list[list[list[float]]]:
        h, w, c = shape(roi)
        mh, mw, _ = shape(memory_patch)
        out = zeros(h, w, c)
        for y in range(h):
            for x in range(w):
                my = min(mh - 1, int((y / max(1, h - 1)) * max(0, mh - 1))) if mh else 0
                mx = min(mw - 1, int((x / max(1, w - 1)) * max(0, mw - 1))) if mw else 0
                for k in range(c):
                    mv = memory_patch[my][mx][k] if mh and mw else roi[y][x][k]
                    out[y][x][k] = max(0.0, min(1.0, roi[y][x][k] * (1.0 - weight) + mv * weight))
        return out

    def _strategy_memory_candidates(self, memory: VideoMemory, entity: str, region_type: str, semantic_reasons: list[str]) -> tuple[list, str]:
        if "garment_change" in semantic_reasons or region_type in {"garments", "sleeves"}:
            return memory.texture_patches.values(), "garment_memory_blend"
        if "expression_change" in semantic_reasons or region_type in {"face", "head"}:
            return memory.texture_patches.values(), "face_identity_blend"
        if "raise_arm" in semantic_reasons or "sit_down" in semantic_reasons:
            return memory.texture_patches.values(), "entity_region_memory_blend"
        return memory.texture_patches.values(), "entity_region_memory_blend"

    def render(
        self,
        current_frame: list,
        scene_graph: SceneGraph,
        delta: GraphDelta,
        memory: VideoMemory,
        region: RegionRef,
        image_tensor: str | list | None = None,
        crop_tensor: str | list | None = None,
    ) -> RenderedPatch:
        _ = (scene_graph, image_tensor, crop_tensor)
        x0, y0, x1, y1 = self._bbox_to_pixels(region.bbox, current_frame)
        roi = crop(current_frame, x0, y0, x1, y1)
        if not roi or not roi[0]:
            roi = zeros(32, 32, 3)

        h, w, ch = shape(roi)
        alpha = alpha_radial(h, w)
        entity, region_type = parse_region_id(region.region_id)
        debug_trace: list[str] = [f"reason={region.reason}", f"region={region.region_id}"]
        rendered = [[px[:] for px in row] for row in roi]
        confidence = 0.5

        is_revealed = region.region_id in {r.region_id for r in delta.newly_revealed_regions}
        if is_revealed:
            slot = memory.hidden_region_slots.get(region.region_id)
            if slot and slot.candidate_patch_ids:
                hidden_id = slot.candidate_patch_ids[0]
                hidden_patch = memory.patch_cache.get(hidden_id)
                if hidden_patch:
                    rendered = self._blend_memory_patch(rendered, hidden_patch, min(0.8, 0.3 + slot.confidence * 0.5))
                    confidence = min(0.99, 0.65 + slot.confidence * 0.3)
                    debug_trace.append("strategy=known_hidden_reveal")
                    debug_trace.append(f"hidden_candidate={hidden_id}")

        if len(debug_trace) <= 2:
            candidates, strategy = self._strategy_memory_candidates(memory, entity, region_type, delta.semantic_reasons)
            filtered = [p for p in candidates if p.entity_id == entity and (p.region_type == region_type or strategy == "garment_memory_blend" and p.region_type in {"garments", "sleeves"} or strategy == "face_identity_blend" and p.region_type == "face")]
            if filtered:
                top = sorted(filtered, key=lambda c: (c.confidence, c.evidence_score), reverse=True)[0]
                mem_patch = memory.patch_cache.get(top.patch_id)
                if mem_patch:
                    mem_weight = min(0.75, 0.25 + top.confidence * 0.5)
                    rendered = self._blend_memory_patch(rendered, mem_patch, mem_weight)
                    confidence = min(0.98, 0.6 + top.confidence * 0.35)
                    debug_trace.append(f"strategy={strategy}")
                    debug_trace.append(f"memory_patch={top.patch_id}")

        if len(debug_trace) <= 2:
            mc = mean_color(roi)
            for y in range(h):
                for x in range(w):
                    for k in range(ch):
                        rendered[y][x][k] = max(0.0, min(1.0, rendered[y][x][k] * 0.95 + mc[k] * 0.05))
            confidence = 0.45
            debug_trace.append("strategy=deterministic_refine_fallback")

        semantic_tint = 0.02 if ("expression_change" in delta.semantic_reasons or region_type == "face") else 0.01
        for y in range(h):
            for x in range(w):
                rendered[y][x][0] = max(0.0, min(1.0, rendered[y][x][0] + semantic_tint))

        uncertainty = [[1.0 - confidence for _ in range(w)] for _ in range(h)]
        return RenderedPatch(region, rendered, alpha, h, w, ch, uncertainty, confidence, 1, debug_trace)
