from __future__ import annotations

from core.pipeline_contract import ContractValidationError, validate_rendering_context
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
from rendering.renderer_checkpoint_loader import load_renderer_model_from_checkpoint
from rendering.torch_local_patch_generator import TorchBackendUnavailableError, TorchLocalPatchGenerator


class LegacyDeterministicPatchSynthesisModel(PatchSynthesisModel):
    def __init__(self, renderer: PatchRenderer | None = None) -> None:
        self.renderer = renderer or PatchRenderer()

    def synthesize_patch(self, request: PatchSynthesisRequest) -> PatchSynthesisOutput:
        validate_rendering_context(region_id=request.region.region_id, transition_context=request.transition_context, region_metadata=request.region_metadata)
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
    def __init__(self, model: TrainableLocalPatchModel | TorchLocalPatchGenerator | None = None, fallback: LegacyDeterministicPatchSynthesisModel | None = None, strict_mode: bool = False, backend: str = "torch_local") -> None:
        self.backend = backend
        self.fallback = fallback or LegacyDeterministicPatchSynthesisModel()
        self.strict_mode = strict_mode
        self._torch_unavailable_reason = ""
        self._checkpoint_requested = False
        self._checkpoint_loaded = False
        self._checkpoint_path = ""
        self._checkpoint_backend = ""
        self._checkpoint_fallback_used = False
        self._checkpoint_fallback_backend = ""
        self._checkpoint_load_error = ""
        self._checkpoint_metadata: dict[str, object] = {}
        if model is not None:
            self.model = model
        elif backend == "numpy_local":
            self.model = TrainableLocalPatchModel()
        elif backend == "torch_local":
            try:
                self.model = TorchLocalPatchGenerator()
            except (RendererInferenceError, TorchBackendUnavailableError) as err:
                if strict_mode:
                    raise
                self._torch_unavailable_reason = str(err)
                self.model = TrainableLocalPatchModel()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        fallback: LegacyDeterministicPatchSynthesisModel | None = None,
        strict_mode: bool = False,
        strict_checkpoint: bool = True,
    ) -> "TrainablePatchSynthesisModel":
        try:
            loaded_model, backend, metadata = load_renderer_model_from_checkpoint(checkpoint_path)
        except Exception as err:
            message = f"Failed to load renderer patch checkpoint '{checkpoint_path}': {err}"
            if strict_checkpoint:
                raise RuntimeError(message) from err
            inst = cls(fallback=fallback, strict_mode=strict_mode, backend="numpy_local")
            inst._checkpoint_requested = True
            inst._checkpoint_loaded = False
            inst._checkpoint_path = checkpoint_path
            inst._checkpoint_backend = ""
            inst._checkpoint_fallback_used = True
            inst._checkpoint_fallback_backend = "numpy_local"
            inst._checkpoint_load_error = message
            return inst

        inst = cls(model=loaded_model, fallback=fallback, strict_mode=strict_mode, backend=backend)
        inst._checkpoint_requested = True
        inst._checkpoint_loaded = True
        inst._checkpoint_path = checkpoint_path
        inst._checkpoint_backend = backend
        inst._checkpoint_fallback_used = False
        inst._checkpoint_fallback_backend = ""
        inst._checkpoint_load_error = ""
        inst._checkpoint_metadata = dict(metadata)
        return inst

    def checkpoint_status(self) -> dict[str, object]:
        metadata = self._checkpoint_metadata if isinstance(self._checkpoint_metadata, dict) else {}
        return {
            "checkpoint_requested": self._checkpoint_requested,
            "checkpoint_loaded": self._checkpoint_loaded,
            "checkpoint_path": self._checkpoint_path,
            "checkpoint_backend": self._checkpoint_backend,
            "checkpoint_fallback_used": self._checkpoint_fallback_used,
            "checkpoint_fallback_backend": self._checkpoint_fallback_backend,
            "checkpoint_load_error": self._checkpoint_load_error,
            "checkpoint_contract_version": str(metadata.get("checkpoint_contract_version", "")) if self._checkpoint_loaded else "",
            "checkpoint_model_family": str(metadata.get("model_family", "")) if self._checkpoint_loaded else "",
            "checkpoint_runtime_loadable": bool(metadata.get("runtime_loadable", False)) if self._checkpoint_loaded else False,
            "checkpoint_global_cond_dim": int(metadata.get("global_cond_dim", 0) or 0) if self._checkpoint_loaded else 0,
        }

    def synthesize_patch(self, request: PatchSynthesisRequest) -> PatchSynthesisOutput:
        route_context = validate_rendering_context(region_id=request.region.region_id, transition_context=request.transition_context, region_metadata=request.region_metadata)
        request_contract = summarize_request_contract(request)
        try:
            roi_before = _extract_roi(request.current_frame, request.region)
            batch = build_patch_batch(request, roi_before)
            pred = self.model.infer(batch)
            batch_summary = summarize_patch_batch(batch)
            torch_used = isinstance(self.model, TorchLocalPatchGenerator) if isinstance(TorchLocalPatchGenerator, type) else False
            fallback_used = self.backend == "torch_local" and (not torch_used) and bool(self._torch_unavailable_reason)
            renderer_path = "torch_local_patch_generator" if torch_used else "learned_primary"
            return output_from_prediction(
                request,
                pred,
                renderer_path,
                {
                    "fallback_used": fallback_used,
                    "fallback_reason": "torch_unavailable" if fallback_used else "",
                    "fallback_message": self._torch_unavailable_reason if fallback_used else "",
                    **self.checkpoint_status(),
                    "strict_mode": self.strict_mode,
                    "roi_shape": list(roi_before.shape),
                    "transition_mode": batch.transition_mode,
                    "profile_role": batch.profile_role,
                    "reveal_like": bool(batch.conditioning_summary.get("reveal_like", False)),
                    "output_provenance": "torch_local_patch_generator" if torch_used else "trainable_local_patch_model",
                    "torch_backend_used": torch_used,
                    "model_family": "local_conv_conditioned_patch_generator" if torch_used else "numpy_linear_patch_generator",
                    "conditioning_summary": batch_summary,
                    "region_route_decision": route_context,
                    "routing_region_id": request.region.region_id,
                    "canonical_region_id": route_context["canonical_region_id"],
                    "render_mode": route_context["render_mode"],
                    "material_provenance": route_context["material_provenance"],
                    "source_provenance": route_context["source_provenance"],
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
        except ContractValidationError:
            raise
        except (RendererInputError, RendererInferenceError, ValueError) as err:
            if self.strict_mode:
                raise
            fb = self.fallback.synthesize_patch(request)
            fb.execution_trace = dict(fb.execution_trace)
            fb.execution_trace.update(
                {
                    "renderer_path": "legacy_fallback",
                    "fallback_used": True,
                    "fallback_reason": type(err).__name__,
                    "fallback_message": str(err),
                    **self.checkpoint_status(),
                    "strict_mode": self.strict_mode,
                    "torch_backend_used": False,
                    "model_family": "legacy_deterministic_patch_renderer",
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
