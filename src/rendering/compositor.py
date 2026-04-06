from __future__ import annotations

from core.schema import GraphDelta, VideoMemory
from rendering.roi_renderer import RenderedPatch


class Compositor:
    def compose(self, current_frame: str, patches: list[RenderedPatch], delta: GraphDelta) -> str:
        ordered = sorted(patches, key=lambda p: p.z_index)
        layers = "|".join(f"{p.region.region_id}:a{p.alpha:.2f}" for p in ordered)
        edge_hint = "edge-aware"
        return (
            f"{current_frame}|compose[{layers}]|{edge_hint}|"
            f"d={len(delta.pose_deltas)+len(delta.garment_deltas)+len(delta.expression_deltas)}"
        )


class TemporalStabilizer:
    def refine(self, previous_frame: str, new_frame: str, memory: VideoMemory, enabled: bool = True) -> str:
        _ = memory
        if not enabled:
            return new_frame
        return f"{new_frame}|flow_smooth({previous_frame[-16:]})"
