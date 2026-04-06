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

        # derive ROI from explicit pose/garment deltas
        if delta.pose_deltas and scene_graph.persons:
            pbox = scene_graph.persons[0].bbox
            selected.append(RegionRef(region_id="pose_roi", bbox=BBox(pbox.x, pbox.y, pbox.w, pbox.h * 0.75), reason="pose_change"))
        if delta.expression_deltas and scene_graph.persons:
            pbox = scene_graph.persons[0].bbox
            selected.append(RegionRef(region_id="face_roi", bbox=BBox(pbox.x + pbox.w * 0.3, pbox.y, pbox.w * 0.35, pbox.h * 0.25), reason="expression_change"))
        if delta.garment_deltas and scene_graph.persons:
            pbox = scene_graph.persons[0].bbox
            selected.append(RegionRef(region_id="garment_roi", bbox=BBox(pbox.x + pbox.w * 0.1, pbox.y + pbox.h * 0.2, pbox.w * 0.8, pbox.h * 0.5), reason="garment_change"))

        if not selected and scene_graph.persons:
            selected.append(RegionRef(region_id="fallback_roi", bbox=scene_graph.persons[0].bbox, reason="delta_low"))
        return selected


class PatchRenderer:
    def _bbox_to_pixels(self, bbox: BBox, frame: list) -> tuple[int, int, int, int]:
        h, w, _ = shape(frame)
        x0 = max(0, min(w - 1, int(bbox.x * w)))
        y0 = max(0, min(h - 1, int(bbox.y * h)))
        x1 = max(x0 + 1, min(w, int((bbox.x + bbox.w) * w)))
        y1 = max(y0 + 1, min(h, int((bbox.y + bbox.h) * h)))
        return x0, y0, x1, y1

    def _warped_copy(self, patch: list[list[list[float]]], x_shift: int = 1) -> list[list[list[float]]]:
        h, w, _ = shape(patch)
        out = zeros(h, w, 3)
        for y in range(h):
            for x in range(w):
                src_x = max(0, min(w - 1, x - x_shift))
                out[y][x] = patch[y][src_x][:]
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
        _ = (scene_graph, memory, image_tensor, crop_tensor)
        frame = current_frame
        x0, y0, x1, y1 = self._bbox_to_pixels(region.bbox, frame)
        c = crop(frame, x0, y0, x1, y1)
        if not c or not c[0]:
            c = zeros(32, 32, 3)

        h, w, ch = shape(c)
        alpha = alpha_radial(h, w)
        base = self._warped_copy(c, x_shift=1 if delta.pose_deltas else 0)
        mc = mean_color(c)
        rendered = zeros(h, w, ch)
        uncertainty = [[0.1 for _ in range(w)] for _ in range(h)]

        residual = 0.06 if delta.expression_deltas else 0.03
        garment_tint = 0.05 if delta.garment_deltas else 0.0
        for y in range(h):
            for x in range(w):
                for k in range(ch):
                    refined = base[y][x][k] * (1.0 - residual) + mc[k] * residual
                    if k == 0:
                        refined += garment_tint
                    rendered[y][x][k] = max(0.0, min(1.0, refined))

        return RenderedPatch(region, rendered, alpha, h, w, ch, uncertainty, 0.85, 1)
