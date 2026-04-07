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
        graph_signal = len(request.graph_encoding.graph_embedding) if request.graph_encoding else 0
        identity_signal = len(request.identity_embedding)
        used_channels = [name for name, payload in request.memory_channels.items() if payload]
        confidence = max(0.0, min(1.0, rendered.confidence + min(0.2, 0.01 * graph_signal + 0.01 * identity_signal)))
        exec_trace = dict(rendered.execution_trace)
        exec_trace.update(
            {
                "learned_ready_usage": {
                    "graph_encoding_used": bool(graph_signal),
                    "identity_embedding_used": bool(identity_signal),
                    "memory_channels_used": used_channels,
                    "ignored_fields": [name for name, payload in request.memory_channels.items() if not payload],
                },
                "selected_render_strategy": exec_trace.get("selected_render_strategy", "graph_identity_guided"),
            }
        )
        return PatchSynthesisOutput(
            region=request.region,
            rgb_patch=rendered.rgb_patch,
            alpha_mask=rendered.alpha_mask,
            height=rendered.height,
            width=rendered.width,
            channels=rendered.channels,
            confidence=confidence,
            z_index=rendered.z_index,
            debug_trace=rendered.debug_trace,
            execution_trace=exec_trace,
            uncertainty_map=rendered.uncertainty_map,
            metadata={
                "debug_trace": rendered.debug_trace,
                "execution_trace": exec_trace,
                "retrieval_summary": request.retrieval_summary,
                "learned_ready_usage": exec_trace.get("learned_ready_usage", {}),
            },
        )
