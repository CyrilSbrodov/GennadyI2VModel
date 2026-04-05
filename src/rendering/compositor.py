from __future__ import annotations

from core.schema import GraphDelta, VideoMemory
from rendering.roi_renderer import RenderedPatch


class Compositor:
    def compose(self, current_frame: str, patches: list[RenderedPatch], delta: GraphDelta) -> str:
        patch_ids = ",".join(p.region.region_id for p in patches)
        return f"{current_frame}|compose[{patch_ids}]|d={len(delta.pose_deltas)+len(delta.garment_deltas)+len(delta.expression_deltas)}"


class TemporalStabilizer:
    def refine(self, previous_frame: str, new_frame: str, memory: VideoMemory) -> str:
        _ = (previous_frame, memory)
        return f"{new_frame}|stable"
