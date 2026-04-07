from __future__ import annotations

from core.schema import GraphDelta, SceneGraph, VideoMemory
from learned.interfaces import PatchSynthesisModel, PatchSynthesisOutput, PatchSynthesisRequest
from rendering.roi_renderer import PatchRenderer


class BaselinePatchSynthesisModel(PatchSynthesisModel):
    """Deterministic patch generation backend behind learned-ready interface."""

    def __init__(self, renderer: PatchRenderer | None = None) -> None:
        self.renderer = renderer or PatchRenderer()

    def synthesize_patch(self, request: PatchSynthesisRequest) -> PatchSynthesisOutput:
        delta = request.transition_context.get("graph_delta")
        memory = request.transition_context.get("video_memory")
        if not isinstance(delta, GraphDelta) or not isinstance(memory, VideoMemory):
            raise ValueError("transition_context must include graph_delta and video_memory for baseline synthesis")

        rendered = self.renderer.render(
            current_frame=request.current_frame,
            scene_graph=request.scene_state,
            delta=delta,
            memory=memory,
            region=request.region,
        )
        return PatchSynthesisOutput(
            region=request.region,
            rgb_patch=rendered.rgb_patch,
            alpha_mask=rendered.alpha_mask,
            height=rendered.height,
            width=rendered.width,
            channels=rendered.channels,
            confidence=rendered.confidence,
            z_index=rendered.z_index,
            debug_trace=rendered.debug_trace,
            execution_trace=rendered.execution_trace,
            uncertainty_map=rendered.uncertainty_map,
            metadata={
                "debug_trace": rendered.debug_trace,
                "execution_trace": rendered.execution_trace,
                "retrieval_summary": request.retrieval_summary,
            },
        )
