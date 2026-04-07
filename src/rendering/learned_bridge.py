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
        hidden_signal = request.memory_channels.get("hidden_regions", {})
        retrieval_profile = str(request.retrieval_summary.get("profile", "")) if isinstance(request.retrieval_summary, dict) else ""
        blend_weight = min(0.3, 0.04 * graph_signal + 0.015 * identity_signal)
        if isinstance(hidden_signal, dict) and hidden_signal:
            blend_weight = min(0.45, blend_weight + 0.12)
        if retrieval_profile == "rich":
            blend_weight = min(0.5, blend_weight + 0.08)
        if blend_weight > 0.0 and rendered.rgb_patch:
            for y, row in enumerate(rendered.rgb_patch):
                for x, px in enumerate(row):
                    id_mod = (request.identity_embedding[(x + y) % max(1, len(request.identity_embedding))] + 1.0) * 0.5 if request.identity_embedding else 0.5
                    px[0] = max(0.0, min(1.0, px[0] * (1.0 - blend_weight) + id_mod * blend_weight))
                    px[1] = max(0.0, min(1.0, px[1] * (1.0 - blend_weight * 0.6) + id_mod * blend_weight * 0.3))
        used_channels = [name for name, payload in request.memory_channels.items() if payload]
        confidence = max(0.0, min(1.0, rendered.confidence + min(0.2, 0.01 * graph_signal + 0.01 * identity_signal)))
        exec_trace = dict(rendered.execution_trace)
        selected_mode = "identity_graph_refine" if blend_weight >= 0.2 else "deterministic"
        if retrieval_profile == "rich" and selected_mode != "identity_graph_refine":
            selected_mode = "retrieval_pref"
        exec_trace.update(
            {
                "learned_ready_usage": {
                    "graph_encoding_used": bool(graph_signal),
                    "identity_embedding_used": bool(identity_signal),
                    "memory_channels_used": used_channels,
                    "ignored_fields": [name for name, payload in request.memory_channels.items() if not payload],
                },
                "selected_render_strategy": exec_trace.get("selected_render_strategy", "graph_identity_guided"),
                "blend_weight": blend_weight,
                "retrieval_preference": retrieval_profile or "default",
                "patch_refinement_strength": round(blend_weight * (1.2 if hidden_signal else 1.0), 3),
                "synthesis_mode": selected_mode,
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
