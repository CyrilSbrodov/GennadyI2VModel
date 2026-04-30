from __future__ import annotations

from core.schema import GraphDelta, VideoMemory
from learned.interfaces import PatchSynthesisModel, PatchSynthesisOutput, PatchSynthesisRequest
from rendering.roi_renderer import PatchRenderer
from rendering.trainable_patch_renderer import (
    RendererInferenceError,
    RendererInputError,
    TrainableLocalPatchModel,
    build_patch_batch,
    output_from_prediction,
    summarize_patch_batch,
    summarize_request_contract,
    _extract_roi,
)


class LegacyDeterministicPatchSynthesisModel(PatchSynthesisModel):
    def __init__(self, renderer: PatchRenderer | None = None) -> None:
        self.renderer = renderer or PatchRenderer()

    def synthesize_patch(self, request: PatchSynthesisRequest) -> PatchSynthesisOutput:
        delta = request.transition_context.get("graph_delta")
        memory = request.transition_context.get("video_memory")
        if not isinstance(delta, GraphDelta) or not isinstance(memory, VideoMemory):
            raise RendererInputError("transition_context must include graph_delta and video_memory")
        rendered = self.renderer.render(
            request.current_frame,
            request.scene_state,
            delta,
            memory,
            request.region,
            transition_context=request.transition_context,
        )
        trace = dict(rendered.execution_trace)
        trace.update({"renderer_path": "legacy_fallback", "synthesis_mode": "legacy_fallback"})
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
            execution_trace=trace,
            uncertainty_map=rendered.uncertainty_map,
            metadata={"renderer_path": "legacy_fallback"},
        )


class TrainablePatchSynthesisModel(PatchSynthesisModel):
    def __init__(self, model: TrainableLocalPatchModel | None = None, fallback: LegacyDeterministicPatchSynthesisModel | None = None, strict_mode: bool = False) -> None:
        self.model = model or TrainableLocalPatchModel()
        self.fallback = fallback or LegacyDeterministicPatchSynthesisModel()
        self.strict_mode = strict_mode

    def synthesize_patch(self, request: PatchSynthesisRequest) -> PatchSynthesisOutput:
        request_contract = summarize_request_contract(request)
        try:
            roi_before = _extract_roi(request.current_frame, request.region)
            batch = build_patch_batch(request, roi_before)
            pred = self.model.infer(batch)
            batch_summary = summarize_patch_batch(batch)
            return output_from_prediction(
                request,
                pred,
                "learned_primary",
                {
                    "fallback_used": False,
                    "strict_mode": self.strict_mode,
                    "roi_shape": list(roi_before.shape),
                    "transition_mode": batch.transition_mode,
                    "profile_role": batch.profile_role,
                    "reveal_like": bool(batch.conditioning_summary.get("reveal_like", False)),
                    "output_provenance": "trainable_local_patch_model",
                    "conditioning_summary": batch_summary,
                    "conditioning": {
                        "has_graph_encoding": bool(request.graph_encoding),
                        "identity_dim": len(request.identity_embedding),
                        "memory_channels": [k for k, v in request.memory_channels.items() if v],
                        "delta_cond_norm": float(batch.delta_cond.mean()),
                        "planner_cond_norm": float(batch.planner_cond.mean()),
                        "graph_cond_norm": float(batch.graph_cond.mean()),
                        "memory_cond_norm": float(batch.memory_cond.mean()),
                        "appearance_cond_norm": float(batch.appearance_cond.mean()),
                        "mode_cond_norm": float(batch.mode_cond.mean()) if batch.mode_cond is not None else 0.0,
                        "role_cond_norm": float(batch.role_cond.mean()) if batch.role_cond is not None else 0.0,
                        "preservation_mean": batch_summary["preservation_mean"],
                        "seam_prior_mean": batch_summary["seam_prior_mean"],
                        "uncertainty_target_mean": batch_summary["uncertainty_target_mean"],
                    },
                },
                batch=batch,
            )
        except (RendererInputError, RendererInferenceError, ValueError) as err:
            if self.strict_mode:
                raise
            fb = self.fallback.synthesize_patch(request)
            fb.execution_trace = dict(fb.execution_trace)
            fb.execution_trace.update(
                {
                    "renderer_path": "legacy_fallback",
                    "fallback_reason": type(err).__name__,
                    "fallback_message": str(err),
                    "strict_mode": self.strict_mode,
                    "requested_transition_mode": request_contract["requested_transition_mode"],
                    "requested_profile_role": request_contract["requested_profile_role"],
                    "fallback_provenance": "legacy_deterministic_patch_renderer",
                }
            )
            fb.metadata = dict(fb.metadata)
            fb.metadata.update(
                {
                    "fallback_reason": type(err).__name__,
                    "fallback_message": str(err),
                    "requested_transition_mode": request_contract["requested_transition_mode"],
                    "requested_profile_role": request_contract["requested_profile_role"],
                    "output_provenance": "legacy_deterministic_patch_renderer",
                }
            )
            fb.debug_trace = list(fb.debug_trace) + [
                f"fallback_reason={type(err).__name__}",
                f"requested_mode={request_contract['requested_transition_mode']}",
                f"requested_role={request_contract['requested_profile_role']}",
            ]
            return fb
