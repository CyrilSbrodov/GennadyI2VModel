from __future__ import annotations

from core.schema import GraphDelta, RegionRef, VideoMemory
from rendering.roi_renderer import RenderedPatch
from utils_tensor import blend, shape


class Compositor:
    @staticmethod
    def _effective_alpha(patch: RenderedPatch, roi_h: int, roi_w: int) -> list[list[float]]:
        alpha = [[float(v) for v in row[:roi_w]] for row in patch.alpha_mask[:roi_h]]
        mode = str(patch.execution_trace.get("selection", {}).get("selected_render_mode", "")) if isinstance(patch.execution_trace, dict) else ""
        if not patch.uncertainty_map:
            return alpha

        conf = max(0.0, min(1.0, float(patch.confidence)))
        out: list[list[float]] = []
        for y in range(roi_h):
            row: list[float] = []
            for x in range(roi_w):
                unc = float(patch.uncertainty_map[y][x]) if y < len(patch.uncertainty_map) and x < len(patch.uncertainty_map[y]) else 0.0
                edge = 1.0 - min(x, roi_w - 1 - x, y, roi_h - 1 - y) / max(1.0, min(roi_h, roi_w) / 2.0)
                mode_boost = 0.1 if mode == "insert_new" else (0.06 if mode == "reveal" else 0.0)
                attenuation = 1.0 - unc * (0.22 + 0.28 * edge) - (1.0 - conf) * (0.10 + 0.08 * edge) + mode_boost * (1.0 - edge)
                row.append(max(0.0, min(1.0, alpha[y][x] * attenuation)))
            out.append(row)
        return out

    def compose(self, current_frame: list, patches: list[RenderedPatch], delta: GraphDelta) -> list:
        _ = delta
        frame = [[px[:] for px in row] for row in current_frame]
        ordered = sorted(
            patches,
            key=lambda p: (
                p.z_index,
                int(p.execution_trace.get("layer_priority", 0)) if isinstance(p.execution_trace, dict) else 0,
                1 if isinstance(p.execution_trace, dict) and p.execution_trace.get("selection", {}).get("selected_render_mode") == "insert_new" else 0,
                1 if isinstance(p.execution_trace, dict) and p.execution_trace.get("selection", {}).get("selected_render_mode") == "reveal" else 0,
                float(p.confidence),
            ),
        )
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
            alpha = self._effective_alpha(p, roi_h, roi_w)
            blended = blend(dst, src, alpha)
            for yy in range(roi_h):
                frame[y0 + yy][x0 : x0 + roi_w] = blended[yy]
        return frame


class TemporalStabilizer:
    def _bbox_to_pixels(self, bbox, frame: list) -> tuple[int, int, int, int]:
        h, w, _ = shape(frame)
        x0 = max(0, min(w - 1, int(bbox.x * w)))
        y0 = max(0, min(h - 1, int(bbox.y * h)))
        x1 = max(x0 + 1, min(w, int((bbox.x + bbox.w) * w)))
        y1 = max(y0 + 1, min(h, int((bbox.y + bbox.h) * h)))
        return x0, y0, x1, y1

    def refine(
        self,
        previous_frame: list,
        new_frame: list,
        memory: VideoMemory,
        enabled: bool = True,
        updated_regions: list[RegionRef] | None = None,
        region_confidence: float = 0.7,
    ) -> list:
        _ = memory
        prev = previous_frame
        cur = new_frame
        if not enabled:
            return cur
        out = [[px[:] for px in row] for row in cur]
        if not updated_regions:
            return out

        h, w, c = shape(cur)
        temporal_weight = min(0.85, 0.2 + 0.5 * region_confidence)
        for region in updated_regions:
            x0, y0, x1, y1 = self._bbox_to_pixels(region.bbox, cur)
            for y in range(max(0, y0), min(h, y1)):
                for x in range(max(0, x0), min(w, x1)):
                    for k in range(c):
                        out[y][x][k] = cur[y][x][k] * (1.0 - temporal_weight) + prev[y][x][k] * temporal_weight
        return out
