from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.semantic_roi import SemanticROIHelper
from core.region_ids import make_region_id, parse_region_id
from core.schema import BBox, GraphDelta, RegionRef, SceneGraph, VideoMemory
from memory.video_memory import MemoryManager
from representation.scene_graph_queries import SceneGraphQueries
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
    def __init__(self) -> None:
        self.roi = SemanticROIHelper()

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
        return self.roi.region_from_graph(scene_graph, entity_id, region_type)

    def fallback_roi_from_person_bbox(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        person = next((p for p in scene_graph.persons if p.person_id == entity_id), None)
        if person is None:
            return None
        return RegionRef(make_region_id(entity_id, region_type), self.roi.fallback_person_bbox(person.bbox, region_type), "fallback:person_bbox_template")

    def _resolve_region(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        return self.roi.resolve_region(scene_graph, entity_id, region_type)

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
    class RenderStrategy(Enum):
        KNOWN_HIDDEN_REVEAL = "KNOWN_HIDDEN_REVEAL"
        UNKNOWN_HIDDEN_SYNTHESIS = "UNKNOWN_HIDDEN_SYNTHESIS"
        IDENTITY_FACE_UPDATE = "IDENTITY_FACE_UPDATE"
        GARMENT_UPDATE = "GARMENT_UPDATE"
        POSE_TRANSFORM = "POSE_TRANSFORM"
        MEMORY_BLEND = "MEMORY_BLEND"
        FALLBACK = "FALLBACK"

    def __init__(self) -> None:
        self.memory_manager = MemoryManager()

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

    def _pick_strategy(self, scene_graph: SceneGraph, delta: GraphDelta, region: RegionRef, memory: VideoMemory) -> "PatchRenderer.RenderStrategy":
        entity, region_type = parse_region_id(region.region_id)
        is_revealed = region.region_id in {r.region_id for r in delta.newly_revealed_regions}
        slot = memory.hidden_region_slots.get(region.region_id)
        state_before = delta.state_before
        state_after = delta.state_after
        transition_phase = delta.transition_phase
        visibility_after = delta.predicted_visibility_changes.get(region_type, state_after.get("visibility_phase", "stable"))
        attached_garments = SceneGraphQueries.get_attached_garments(scene_graph, entity)
        covering = SceneGraphQueries.get_covering_entities(scene_graph, entity, region_type)
        likely_revealed = SceneGraphQueries.is_region_likely_revealed(scene_graph, entity, region_type, delta.predicted_visibility_changes)
        region_mode = delta.region_transition_mode.get(region_type, "")
        if is_revealed and slot and slot.hidden_type == "known_hidden" and slot.candidate_patch_ids:
            return self.RenderStrategy.KNOWN_HIDDEN_REVEAL
        if is_revealed and (slot is None or slot.hidden_type == "unknown_hidden" or not slot.candidate_patch_ids) and likely_revealed:
            return self.RenderStrategy.UNKNOWN_HIDDEN_SYNTHESIS
        if region_mode.startswith("garment_") or (state_before.get("garment_phase") in {"worn", "opening"} and state_after.get("garment_phase") in {"opening", "removed"}):
            return self.RenderStrategy.GARMENT_UPDATE
        if region_mode == "pose_exposure" or transition_phase in {"bend_knees", "lower_pelvis"} or (state_before.get("pose_phase") != state_after.get("pose_phase") and region_type in {"left_arm", "right_arm", "pelvis", "legs"}):
            return self.RenderStrategy.POSE_TRANSFORM
        if visibility_after in {"visible", "partially_visible", "revealing"} and (covering or not attached_garments):
            if slot and slot.hidden_type == "unknown_hidden":
                return self.RenderStrategy.UNKNOWN_HIDDEN_SYNTHESIS
        if "expression_change" in delta.semantic_reasons or region_type in {"face", "head"}:
            return self.RenderStrategy.IDENTITY_FACE_UPDATE
        if "garment_change" in delta.semantic_reasons or region_type in {"garments", "sleeves"}:
            return self.RenderStrategy.GARMENT_UPDATE
        if any(reason in delta.semantic_reasons for reason in {"raise_arm", "sit_down"}) or region_type in {"left_arm", "right_arm", "pelvis", "legs"}:
            return self.RenderStrategy.POSE_TRANSFORM
        active_relations = SceneGraphQueries.relations_for_entity(scene_graph, entity)
        if active_relations or memory.texture_patches:
            return self.RenderStrategy.MEMORY_BLEND
        return self.RenderStrategy.FALLBACK

    def _unknown_hidden_synthesis(
        self,
        roi: list[list[list[float]]],
        region_type: str,
        identity_embedding: list[float] | None,
        descriptor_hint: dict[str, float | list[float]] | None,
        visibility: str,
    ) -> list[list[list[float]]]:
        h, w, c = shape(roi)
        base_mean = mean_color(roi)
        hint_mean = (descriptor_hint or {}).get("mean", base_mean)
        hint_std = (descriptor_hint or {}).get("std", [0.05, 0.05, 0.05])
        tint = [float(hint_mean[0]), float(hint_mean[1]), float(hint_mean[2])]
        if identity_embedding:
            tint = [max(0.0, min(1.0, tint[k] * 0.75 + (identity_embedding[k] + 1.0) * 0.125)) for k in range(3)]
        if region_type in {"face", "head"}:
            tint = [min(1.0, tint[0] + 0.03), min(1.0, tint[1] + 0.015), tint[2]]
        elif region_type in {"garments", "sleeves"}:
            tint = [tint[0] * 0.95, min(1.0, tint[1] * 1.03), min(1.0, tint[2] * 1.04)]
        elif region_type in {"torso", "left_arm", "right_arm", "pelvis", "legs"}:
            tint = [min(1.0, tint[0] * 1.02), tint[1] * 0.98, tint[2] * 0.96]
        visibility_boost = 0.18 if visibility in {"visible", "partially_visible"} else 0.08

        out = zeros(h, w, c)
        for y in range(h):
            for x in range(w):
                radial = abs((x / max(1, w - 1)) - 0.5) + abs((y / max(1, h - 1)) - 0.5)
                smooth = max(0.0, min(1.0, 1.0 - radial))
                for k in range(c):
                    std_term = float(hint_std[k]) * (0.15 - 0.1 * smooth)
                    mixed = roi[y][x][k] * (1.0 - visibility_boost) + tint[k] * visibility_boost + std_term
                    out[y][x][k] = max(0.0, min(1.0, mixed))
        return out

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

        strategy = self._pick_strategy(scene_graph, delta, region, memory)
        debug_trace.append(f"strategy_selected={strategy.value}")
        debug_trace.append(f"state={delta.state_before.get('pose_phase', 'na')}->{delta.state_after.get('pose_phase', 'na')}")
        debug_trace.append(f"visibility_hint={delta.predicted_visibility_changes.get(region_type, 'none')}")
        relation_hints = SceneGraphQueries.relations_for_entity(scene_graph, entity)
        if relation_hints:
            debug_trace.append(f"graph_relations={len(relation_hints)}")
        if strategy == self.RenderStrategy.KNOWN_HIDDEN_REVEAL:
            slot = memory.hidden_region_slots.get(region.region_id)
            if slot and slot.candidate_patch_ids:
                hidden_id = slot.candidate_patch_ids[0]
                hidden_patch = memory.patch_cache.get(hidden_id)
                if hidden_patch:
                    rendered = self._blend_memory_patch(rendered, hidden_patch, min(0.8, 0.3 + slot.confidence * 0.5))
                    confidence = min(0.99, 0.65 + slot.confidence * 0.3)
                    debug_trace.append("strategy=known_hidden_reveal")
                    debug_trace.append(f"hidden_candidate={hidden_id}")
                    debug_trace.append(f"hidden_slot={slot.hidden_type}:{slot.last_transition}")

        if "strategy=known_hidden_reveal" not in debug_trace and strategy == self.RenderStrategy.UNKNOWN_HIDDEN_SYNTHESIS:
            query_desc = self.memory_manager._patch_descriptor(roi)
            route = self.memory_manager.route_region_retrieval(
                memory,
                region.region_id,
                region_type,
                entity,
                query_descriptor=query_desc,
                transition_context={
                    "transition_phase": delta.transition_phase,
                    "visibility_phase": delta.state_after.get("visibility_phase", "stable"),
                },
            )
            identity = memory.identity_memory.get(entity)
            synth_hint = route["candidates"][0].descriptor if route["candidates"] else query_desc
            rendered = self._unknown_hidden_synthesis(
                rendered,
                region_type,
                identity.embedding if identity else None,
                synth_hint,
                delta.predicted_visibility_changes.get(region_type, "unknown"),
            )
            confidence = 0.4
            debug_trace.append("strategy=unknown_hidden_synthesis")
            debug_trace.append(f"retrieval_hint={route['strategy_hint']}")
            debug_trace.append(f"retrieval_debug={route['explanation']}")

        if not any(msg.startswith("strategy=") for msg in debug_trace[2:]):
            query_desc = self.memory_manager._patch_descriptor(roi)
            route = self.memory_manager.route_region_retrieval(
                memory,
                region.region_id,
                region_type,
                entity,
                query_descriptor=query_desc,
                transition_context={
                    "transition_phase": delta.transition_phase,
                    "visibility_phase": delta.state_after.get("visibility_phase", "stable"),
                },
            )
            candidates = route["candidates"]
            if candidates:
                top = candidates[0]
                mem_patch = memory.patch_cache.get(top.patch_id)
                if mem_patch:
                    mem_weight = min(0.75, 0.25 + top.confidence * 0.5)
                    rendered = self._blend_memory_patch(rendered, mem_patch, mem_weight)
                    confidence = min(0.98, 0.6 + top.confidence * 0.35)
                    debug_trace.append(f"strategy={route['strategy_hint']}")
                    debug_trace.append(f"memory_patch={top.patch_id}")
                    debug_trace.append(f"retrieval_source={route['explanation']}")

        if not any(msg.startswith("strategy=") for msg in debug_trace[2:]):
            mc = mean_color(roi)
            for y in range(h):
                for x in range(w):
                    for k in range(ch):
                        rendered[y][x][k] = max(0.0, min(1.0, rendered[y][x][k] * 0.95 + mc[k] * 0.05))
            confidence = 0.45
            debug_trace.append("strategy=deterministic_refine_fallback")
            debug_trace.append("fallback_reason=no_viable_retrieval")

        semantic_tint = 0.02 if ("expression_change" in delta.semantic_reasons or region_type == "face") else 0.01
        for y in range(h):
            for x in range(w):
                rendered[y][x][0] = max(0.0, min(1.0, rendered[y][x][0] + semantic_tint))

        uncertainty = [[1.0 - confidence for _ in range(w)] for _ in range(h)]
        return RenderedPatch(region, rendered, alpha, h, w, ch, uncertainty, confidence, 1, debug_trace)
