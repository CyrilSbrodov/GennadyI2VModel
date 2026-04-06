from __future__ import annotations

from dataclasses import dataclass

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


class ROISelector:
    def select(self, scene_graph: SceneGraph, delta: GraphDelta) -> list[RegionRef]:
        selected = list(delta.newly_revealed_regions)
        if delta.newly_occluded_regions:
            selected.extend(delta.newly_occluded_regions)
        if delta.visibility_deltas and scene_graph.persons:
            selected.append(RegionRef(region_id="visibility_shift", bbox=scene_graph.persons[0].bbox, reason="visibility_change"))
        if not selected and scene_graph.persons:
            selected.append(RegionRef(region_id="fallback_roi", bbox=scene_graph.persons[0].bbox, reason="delta_low"))
        return selected


class PatchRenderer:
    def _to_tensor(self, payload: str | list | None, fallback_size: tuple[int, int] = (256, 256)) -> list:
        if isinstance(payload, list):
            return payload
        seed = abs(hash(str(payload))) % 255
        h, w = fallback_size
        base = zeros(h, w, 3)
        for y in range(h):
            for x in range(w):
                base[y][x][0] = (seed % 97) / 96.0
                base[y][x][1] = (seed % 53) / 52.0
                base[y][x][2] = (seed % 29) / 28.0
        return base

    def _bbox_to_pixels(self, bbox: BBox, frame: list) -> tuple[int, int, int, int]:
        h, w, _ = shape(frame)
        x0 = max(0, min(w - 1, int(bbox.x * w)))
        y0 = max(0, min(h - 1, int(bbox.y * h)))
        x1 = max(x0 + 1, min(w, int((bbox.x + bbox.w) * w)))
        y1 = max(y0 + 1, min(h, int((bbox.y + bbox.h) * h)))
        return x0, y0, x1, y1

    def render(
        self,
        current_frame: str | list,
        scene_graph: SceneGraph,
        delta: GraphDelta,
        memory: VideoMemory,
        region: RegionRef,
        image_tensor: str | list | None = None,
        crop_tensor: str | list | None = None,
    ) -> RenderedPatch:
        _ = (scene_graph, delta, memory)
        frame = self._to_tensor(image_tensor if image_tensor is not None else current_frame)
        x0, y0, x1, y1 = self._bbox_to_pixels(region.bbox, frame)
        c = self._to_tensor(crop_tensor) if crop_tensor is not None else crop(frame, x0, y0, x1, y1)
        if not c or not c[0]:
            c = self._to_tensor("empty_crop", (32, 32))

        h, w, ch = shape(c)
        alpha = alpha_radial(h, w)
        mc = mean_color(c)
        rendered = zeros(h, w, ch)
        uncertainty = [[0.08 for _ in range(w)] for _ in range(h)]
        for y in range(h):
            for x in range(w):
                for k in range(ch):
                    rendered[y][x][k] = c[y][x][k] * 0.65 + mc[k] * 0.35
        return RenderedPatch(region, rendered, alpha, h, w, ch, uncertainty, 0.8, 1)
