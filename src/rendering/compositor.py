from __future__ import annotations

from core.schema import GraphDelta, VideoMemory
from rendering.roi_renderer import RenderedPatch
from utils_tensor import blend, roll_x, shape, zeros


class Compositor:
    def compose(self, current_frame: list, patches: list[RenderedPatch], delta: GraphDelta) -> list:
        _ = delta
        frame = [[px[:] for px in row] for row in current_frame]
        ordered = sorted(patches, key=lambda p: p.z_index)
        h, w, _ = shape(frame)

        for p in ordered:
            x0 = max(0, min(w - 1, int(p.region.bbox.x * w)))
            y0 = max(0, min(h - 1, int(p.region.bbox.y * h)))
            roi_w = min(p.width, w - x0)
            roi_h = min(p.height, h - y0)
            if roi_w <= 0 or roi_h <= 0:
                continue
            dst = [row[x0 : x0 + roi_w] for row in frame[y0 : y0 + roi_h]]
            src = [row[:roi_w] for row in p.rgb_patch[:roi_h]]
            alpha = [row[:roi_w] for row in p.alpha_mask[:roi_h]]
            blended = blend(dst, src, alpha)
            for yy in range(roi_h):
                frame[y0 + yy][x0 : x0 + roi_w] = blended[yy]
        return frame


class TemporalStabilizer:
    def refine(self, previous_frame: list, new_frame: list, memory: VideoMemory, enabled: bool = True) -> list:
        prev = previous_frame
        cur = new_frame
        if not enabled:
            return cur
        flow_strength = min(0.35, 0.08 + 0.015 * len(memory.temporal_history))
        warped = roll_x(prev, shift=1)
        h, w, c = shape(cur)
        out = zeros(h, w, c)
        for y in range(h):
            for x in range(w):
                for k in range(c):
                    out[y][x][k] = cur[y][x][k] * (1.0 - flow_strength) + warped[y][x][k] * flow_strength
        return out
