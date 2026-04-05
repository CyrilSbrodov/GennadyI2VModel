from __future__ import annotations

from dataclasses import dataclass

from core.schema import GraphDelta, RegionRef, SceneGraph, VideoMemory


@dataclass(slots=True)
class RenderedPatch:
    region: RegionRef
    rgb_stub: str
    alpha: float
    confidence: float


class ROISelector:
    def select(self, scene_graph: SceneGraph, delta: GraphDelta) -> list[RegionRef]:
        _ = scene_graph
        return delta.newly_revealed_regions


class PatchRenderer:
    def render(
        self,
        current_frame: str,
        scene_graph: SceneGraph,
        delta: GraphDelta,
        memory: VideoMemory,
        region: RegionRef,
    ) -> RenderedPatch:
        _ = (current_frame, scene_graph, delta, memory)
        return RenderedPatch(region=region, rgb_stub=f"patch::{region.region_id}", alpha=0.92, confidence=0.8)
