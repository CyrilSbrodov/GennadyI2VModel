from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import GraphDelta, RegionRef, SceneGraph, VideoMemory


@dataclass(slots=True)
class RenderedPatch:
    region: RegionRef
    rgb_patch: str
    alpha: float
    uncertainty_map: list[float] = field(default_factory=list)
    confidence: float = 0.0
    z_index: int = 0


class ROISelector:
    def select(self, scene_graph: SceneGraph, delta: GraphDelta) -> list[RegionRef]:
        selected = list(delta.newly_revealed_regions)
        if delta.newly_occluded_regions:
            selected.extend(delta.newly_occluded_regions)

        if delta.visibility_deltas and scene_graph.persons:
            selected.append(
                RegionRef(
                    region_id="visibility_shift",
                    bbox=scene_graph.persons[0].bbox,
                    reason="visibility_change",
                )
            )

        if not selected:
            selected.append(RegionRef(region_id="fallback_roi", bbox=scene_graph.persons[0].bbox, reason="delta_low"))
        return selected


class PatchRenderer:
    def render(
        self,
        current_frame: str,
        scene_graph: SceneGraph,
        delta: GraphDelta,
        memory: VideoMemory,
        region: RegionRef,
        image_tensor: str | None = None,
        crop_tensor: str | None = None,
    ) -> RenderedPatch:
        _ = (scene_graph, delta, memory)
        payload = image_tensor or crop_tensor or current_frame
        return RenderedPatch(
            region=region,
            rgb_patch=f"patch::{region.region_id}::{payload}",
            alpha=0.92,
            uncertainty_map=[0.1, 0.15, 0.08],
            confidence=0.8,
            z_index=1,
        )
