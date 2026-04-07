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
    execution_trace: dict[str, object] = field(default_factory=dict)


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

    _REGION_GROUPS = {
        "face_head": {"face", "head"},
        "garment": {"garments", "sleeves"},
        "torso": {"torso", "pelvis"},
        "limbs": {"left_arm", "right_arm", "legs"},
    }

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

    def _region_group(self, region_type: str) -> str:
        for key, members in self._REGION_GROUPS.items():
            if region_type in members:
                return key
        return "generic"

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
        if region_mode == "expression_refine" or "expression_change" in delta.semantic_reasons or region_type in {"face", "head"}:
            return self.RenderStrategy.IDENTITY_FACE_UPDATE
        if "garment_change" in delta.semantic_reasons or region_type in {"garments", "sleeves"}:
            return self.RenderStrategy.GARMENT_UPDATE
        if any(reason in delta.semantic_reasons for reason in {"raise_arm", "sit_down"}) or region_type in {"left_arm", "right_arm", "pelvis", "legs"}:
            return self.RenderStrategy.POSE_TRANSFORM
        active_relations = SceneGraphQueries.relations_for_entity(scene_graph, entity)
        if active_relations or memory.texture_patches:
            return self.RenderStrategy.MEMORY_BLEND
        return self.RenderStrategy.FALLBACK

    def _synthesis_context(self, delta: GraphDelta, region_type: str, route: dict[str, object]) -> dict[str, str]:
        return {
            "visibility_phase": delta.state_after.get("visibility_phase", "stable"),
            "garment_phase": delta.state_after.get("garment_phase", "worn"),
            "pose_phase": delta.state_after.get("pose_phase", "stable"),
            "expression_state": delta.state_after.get("expression_state", "neutral"),
            "support_contact_state": delta.state_after.get("support_contact_state", "free_contact"),
            "region_transition_mode": delta.region_transition_mode.get(region_type, ""),
            "retrieval_hint": str(route.get("strategy_hint", "none")),
        }

    def _unknown_hidden_synthesis(
        self,
        roi: list[list[list[float]]],
        region_type: str,
        identity_embedding: list[float] | None,
        descriptor_hint: dict[str, float | list[float]] | None,
        context_descriptor: dict[str, float | list[float]] | None,
        synth_context: dict[str, str],
    ) -> tuple[list[list[list[float]]], str]:
        h, w, c = shape(roi)
        base_mean = mean_color(roi)
        hint_mean = (descriptor_hint or {}).get("mean", base_mean)
        hint_std = (descriptor_hint or {}).get("std", [0.05, 0.05, 0.05])
        ctx_mean = (context_descriptor or {}).get("mean", base_mean)
        mode = synth_context.get("region_transition_mode", "stable")
        phase = synth_context.get("visibility_phase", "stable")

        tint = [0.58 * float(hint_mean[k]) + 0.32 * float(ctx_mean[k]) + 0.1 * float(base_mean[k]) for k in range(3)]
        if identity_embedding:
            tint = [max(0.0, min(1.0, tint[k] * 0.82 + (identity_embedding[k] + 1.0) * 0.09)) for k in range(3)]

        group = self._region_group(region_type)
        if group == "face_head":
            tint = [min(1.0, tint[0] + 0.04), min(1.0, tint[1] + 0.02), tint[2] * 0.98]
        elif group == "garment":
            tint = [tint[0] * 0.94, min(1.0, tint[1] * 1.04), min(1.0, tint[2] * 1.05)]
        elif group == "torso":
            tint = [min(1.0, tint[0] * 1.02), tint[1] * 0.99, tint[2] * 0.97]
        elif group == "limbs":
            tint = [min(1.0, tint[0] * 1.01), tint[1] * 0.98, tint[2] * 0.96]

        visibility_boost = 0.2 if phase == "revealing" else (0.09 if phase == "occluding" else 0.14)
        if mode.startswith("garment_"):
            visibility_boost += 0.04
        if mode == "pose_exposure":
            visibility_boost += 0.02

        out = zeros(h, w, c)
        for y in range(h):
            for x in range(w):
                nx = x / max(1, w - 1)
                ny = y / max(1, h - 1)
                center_pull = max(0.0, 1.0 - (abs(nx - 0.5) + abs(ny - 0.5)))
                vertical_bias = 0.6 + 0.4 * ny if group in {"torso", "garment"} else 0.5 + 0.5 * (1.0 - ny)
                spatial = 0.45 * center_pull + 0.55 * vertical_bias
                for k in range(c):
                    std_term = float(hint_std[k]) * (0.11 - 0.06 * spatial)
                    local = roi[y][x][k] * (1.0 - visibility_boost) + tint[k] * visibility_boost
                    out[y][x][k] = max(0.0, min(1.0, local + std_term))
        return out, group

    def execute_known_hidden_reveal(self, rendered: list[list[list[float]]], memory: VideoMemory, region: RegionRef) -> tuple[list[list[list[float]]], float, list[str], dict[str, object]]:
        debug: list[str] = []
        trace: dict[str, object] = {"strategy": "known_hidden_reveal"}
        slot = memory.hidden_region_slots.get(region.region_id)
        if slot and slot.candidate_patch_ids:
            hidden_id = slot.candidate_patch_ids[0]
            hidden_patch = memory.patch_cache.get(hidden_id)
            if hidden_patch:
                out = self._blend_memory_patch(rendered, hidden_patch, min(0.8, 0.3 + slot.confidence * 0.5))
                debug.extend(["strategy=known_hidden_reveal", f"hidden_candidate={hidden_id}", f"hidden_slot={slot.hidden_type}:{slot.last_transition}"])
                trace["hidden_lifecycle_transition"] = slot.last_transition
                trace["hidden_lifecycle_reason"] = slot.last_transition_reason
                return out, min(0.99, 0.65 + slot.confidence * 0.3), debug, trace
        debug.append("strategy=known_hidden_reveal")
        trace["fallback_reason"] = "missing_hidden_patch"
        return rendered, 0.55, debug, trace

    def execute_unknown_hidden_synthesis(
        self,
        rendered: list[list[list[float]]],
        roi: list[list[list[float]]],
        scene_graph: SceneGraph,
        delta: GraphDelta,
        memory: VideoMemory,
        region: RegionRef,
    ) -> tuple[list[list[list[float]]], float, list[str], dict[str, object]]:
        entity, region_type = parse_region_id(region.region_id)
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
                "region_transition_mode": delta.region_transition_mode.get(region_type, ""),
            },
        )
        identity = memory.identity_memory.get(entity)
        synth_hint = route["candidates"][0].descriptor if route["candidates"] else query_desc
        synth_context = self._synthesis_context(delta, region_type, route)
        rendered, synth_group = self._unknown_hidden_synthesis(
            rendered,
            region_type,
            identity.embedding if identity else None,
            synth_hint,
            query_desc,
            synth_context,
        )
        debug = ["strategy=unknown_hidden_synthesis", f"retrieval_hint={route['strategy_hint']}", f"retrieval_debug={route['explanation']}"]
        trace = {
            "strategy": "unknown_hidden_synthesis",
            "retrieval_explanation_summary": route["explanation"],
            "synthesis_mode": f"{synth_group}:{synth_context.get('region_transition_mode', 'stable')}",
        }
        return rendered, 0.4, debug, trace

    def execute_identity_face_update(self, rendered: list[list[list[float]]], roi: list[list[list[float]]], delta: GraphDelta, region_type: str) -> tuple[list[list[list[float]]], float, list[str], dict[str, object]]:
        mode = delta.region_transition_mode.get(region_type, "expression_refine")
        mc = mean_color(roi)
        h, w, c = shape(rendered)
        for y in range(h):
            for x in range(w):
                gain = 0.04 if mode == "expression_refine" else 0.025
                rendered[y][x][0] = max(0.0, min(1.0, rendered[y][x][0] * (1.0 - gain) + (mc[0] + gain) * gain))
        return rendered, 0.56, ["strategy=identity_face_update"], {"strategy": "identity_face_update", "synthesis_mode": mode}

    def execute_garment_update(self, rendered: list[list[list[float]]], delta: GraphDelta, region_type: str) -> tuple[list[list[list[float]]], float, list[str], dict[str, object]]:
        mode = delta.region_transition_mode.get(region_type, delta.state_after.get("garment_phase", "worn"))
        h, w, c = shape(rendered)
        gain = 0.03 if str(mode).startswith("garment_") else 0.015
        for y in range(h):
            for x in range(w):
                rendered[y][x][1] = max(0.0, min(1.0, rendered[y][x][1] + gain))
        return rendered, 0.58, ["strategy=garment_update"], {"strategy": "garment_update", "synthesis_mode": str(mode)}

    def execute_pose_transform(self, rendered: list[list[list[float]]], delta: GraphDelta, region_type: str) -> tuple[list[list[list[float]]], float, list[str], dict[str, object]]:
        pose_phase = delta.state_after.get("pose_phase", "stable")
        mode = delta.region_transition_mode.get(region_type, "pose_exposure")
        h, w, _ = shape(rendered)
        factor = 0.015 if pose_phase in {"lowering", "transitioning"} else 0.01
        for y in range(h):
            row_factor = factor * (y / max(1, h - 1))
            for x in range(w):
                rendered[y][x][2] = max(0.0, min(1.0, rendered[y][x][2] + row_factor))
        return rendered, 0.54, ["strategy=pose_transform"], {"strategy": "pose_transform", "synthesis_mode": mode}

    def execute_memory_blend(self, rendered: list[list[list[float]]], roi: list[list[list[float]]], scene_graph: SceneGraph, delta: GraphDelta, memory: VideoMemory, region: RegionRef) -> tuple[list[list[list[float]]], float, list[str], dict[str, object]]:
        entity, region_type = parse_region_id(region.region_id)
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
                "region_transition_mode": delta.region_transition_mode.get(region_type, ""),
            },
        )
        candidates = route["candidates"]
        if candidates:
            top = candidates[0]
            mem_patch = memory.patch_cache.get(top.patch_id)
            if mem_patch:
                mem_weight = min(0.75, 0.25 + top.confidence * 0.5)
                rendered = self._blend_memory_patch(rendered, mem_patch, mem_weight)
                return rendered, min(0.98, 0.6 + top.confidence * 0.35), [f"strategy={route['strategy_hint']}", f"memory_patch={top.patch_id}", f"retrieval_source={route['explanation']}"], {
                    "strategy": "memory_blend",
                    "retrieval_explanation_summary": route["explanation"],
                }
        return rendered, 0.5, ["strategy=memory_blend", "fallback_reason=no_viable_candidate"], {"strategy": "memory_blend", "fallback_reason": "no_viable_candidate"}

    def execute_fallback(self, rendered: list[list[list[float]]], roi: list[list[list[float]]]) -> tuple[list[list[list[float]]], float, list[str], dict[str, object]]:
        h, w, ch = shape(roi)
        mc = mean_color(roi)
        for y in range(h):
            for x in range(w):
                for k in range(ch):
                    rendered[y][x][k] = max(0.0, min(1.0, rendered[y][x][k] * 0.95 + mc[k] * 0.05))
        return rendered, 0.45, ["strategy=deterministic_refine_fallback", "fallback_reason=no_viable_retrieval"], {"strategy": "fallback", "fallback_reason": "no_viable_retrieval"}

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
        _, region_type = parse_region_id(region.region_id)
        debug_trace: list[str] = [f"reason={region.reason}", f"region={region.region_id}"]
        rendered = [[px[:] for px in row] for row in roi]

        strategy = self._pick_strategy(scene_graph, delta, region, memory)
        debug_trace.append(f"strategy_selected={strategy.value}")
        debug_trace.append(f"state={delta.state_before.get('pose_phase', 'na')}->{delta.state_after.get('pose_phase', 'na')}")
        debug_trace.append(f"visibility_hint={delta.predicted_visibility_changes.get(region_type, 'none')}")

        exec_trace: dict[str, object] = {
            "selected_render_strategy": strategy.value,
            "state_contract": {
                "pose_phase": delta.state_after.get("pose_phase", "na"),
                "garment_phase": delta.state_after.get("garment_phase", "na"),
                "visibility_phase": delta.state_after.get("visibility_phase", "na"),
                "expression_state": delta.state_after.get("expression_state", "na"),
                "support_contact_state": delta.state_after.get("support_contact_state", "na"),
                "region_transition_mode": delta.region_transition_mode.get(region_type, "none"),
            },
        }

        if strategy == self.RenderStrategy.KNOWN_HIDDEN_REVEAL:
            rendered, confidence, logs, trace = self.execute_known_hidden_reveal(rendered, memory, region)
        elif strategy == self.RenderStrategy.UNKNOWN_HIDDEN_SYNTHESIS:
            rendered, confidence, logs, trace = self.execute_unknown_hidden_synthesis(rendered, roi, scene_graph, delta, memory, region)
        elif strategy == self.RenderStrategy.IDENTITY_FACE_UPDATE:
            rendered, confidence, logs, trace = self.execute_identity_face_update(rendered, roi, delta, region_type)
        elif strategy == self.RenderStrategy.GARMENT_UPDATE:
            rendered, confidence, logs, trace = self.execute_garment_update(rendered, delta, region_type)
        elif strategy == self.RenderStrategy.POSE_TRANSFORM:
            rendered, confidence, logs, trace = self.execute_pose_transform(rendered, delta, region_type)
        elif strategy == self.RenderStrategy.MEMORY_BLEND:
            rendered, confidence, logs, trace = self.execute_memory_blend(rendered, roi, scene_graph, delta, memory, region)
        else:
            rendered, confidence, logs, trace = self.execute_fallback(rendered, roi)

        debug_trace.extend(logs)
        exec_trace.update(trace)

        semantic_tint = 0.02 if ("expression_change" in delta.semantic_reasons or region_type == "face") else 0.01
        for y in range(h):
            for x in range(w):
                rendered[y][x][0] = max(0.0, min(1.0, rendered[y][x][0] + semantic_tint))

        uncertainty = [[1.0 - confidence for _ in range(w)] for _ in range(h)]
        return RenderedPatch(region, rendered, alpha, h, w, ch, uncertainty, confidence, 1, debug_trace, exec_trace)
