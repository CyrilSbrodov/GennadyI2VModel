from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.schema import BBox, RegionRef, SceneGraph
from learned.factory import BackendConfig, LearnedBackendFactory
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from rendering.learned_roi_renderer import (
    LightweightROIRendererBackend,
    RendererObservedPairDataset,
    ROIFeatureBuilder,
    TinyROIRenderer,
    _extract_region,
    load_lightweight_roi_checkpoint,
    roi_request_to_dict,
    train_lightweight_roi_renderer,
)
from rendering.roi_renderer_contract import ROIRenderMode, ROIRenderValidationError, build_roi_render_request
from runtime.region_routing_contract import RegionRoutingContract, RegionRoutingDecision, RegionRoutingDecisionType


def _meta(**overrides):
    base = {
        "region_id": "p1:torso",
        "canonical_region_id": "torso",
        "route_decision_type": "route_pose_update",
        "render_mode": "pose_local_update",
        "memory_family": "body",
        "memory_authority": "none",
        "output_authority": "generated",
        "identity_locked": False,
        "weak_memory_candidate": False,
        "source_delta_type": "pose_delta",
        "source_reveal_decision_type": "none",
        "action_type": "turn",
        "phase_id": "phase_0",
        "target_provenance": "observed",
    }
    base.update(overrides)
    return base


def _routing_decision(region="torso", decision_type=RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value, *, weak=False, authority="none"):
    identity = region in {"face", "head", "hair", "scalp"}
    return RegionRoutingDecision(
        region_id=f"p1:{region}", entity_id="p1", canonical_region_id=region,
        decision_type=decision_type, source="test", source_delta_type="pose_delta", source_reveal_decision_type="none",
        action_type="turn", phase_id="phase_0", action_order=0, phase_order=0,
        route_allowed=True, render_candidate_allowed=True, blocked=False, block_reason=None,
        diagnostic_only=False, identity_locked=identity, allowed_to_modify_identity=False,
        requires_identity_memory=False, requires_appearance_memory=False, requires_reveal_memory="reveal" in decision_type,
        memory_reference_kind="identity_reference" if identity else "body_shape_reference",
        memory_authority=authority, memory_family="identity" if identity else "body_shape",
        weak_memory_candidate=weak, preserve_visible=False, occlusion_reasoning_only=False,
        newly_occluded_tracking_only=False, private_or_optional_region=False, roi_required=True,
        roi_source_policy="explicit_roi_required", renderer_strategy_allowed_values=("deterministic",),
        selected_render_strategy="deterministic", renderer_must_preserve_identity=identity,
        renderer_must_not_create_observed_evidence=True, renderer_must_not_create_identity_memory=True,
        validation_reasons=("test",), provenance=("test",), output_reference_authority="weak" if weak else "generated",
        memory_write_allowed=False, identity_memory_write_allowed=False, observed_evidence_claim_allowed=False,
    )


def _roi_request(region="torso", decision_type=RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value, *, weak=False, authority="none"):
    d = _routing_decision(region=region, decision_type=decision_type, weak=weak, authority=authority)
    contract = RegionRoutingContract(decisions=(d,), renderable_region_ids=(d.region_id,))
    return build_roi_render_request(route_decision=d, routing_contract=contract, roi_bbox=(0, 0, 1, 1), current_frame_shape=(8, 8, 3), frame_index=0)


def _manifest(tmp_path: Path, records):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"manifest_version": "renderer_observed_pair_manifest_v3", "samples": records}))
    return p


def test_tiny_roi_renderer_forward_cpu_is_deterministic():
    torch.manual_seed(7)
    model_a = TinyROIRenderer()
    x = torch.rand(1, 16, 8, 8)
    y = model_a(x)
    torch.manual_seed(7)
    model_b = TinyROIRenderer()
    y2 = model_b(x)
    assert y["rgb"].shape == (1, 3, 8, 8)
    assert y["alpha"].shape == (1, 1, 8, 8)
    assert torch.allclose(y["rgb"], y2["rgb"])


def test_feature_builder_preserves_contract_flags_and_rejects_private():
    roi = np.zeros((4, 4, 3), dtype=np.float32)
    f1 = ROIFeatureBuilder().build(roi, _meta(render_mode="pose_local_update", identity_locked=True, weak_memory_candidate=True))
    f2 = ROIFeatureBuilder().build(roi, _meta(render_mode="deform"))
    assert f1.tensor.shape == (16, 4, 4)
    assert f1.metadata["identity_locked"] is True
    assert f1.metadata["weak_memory_candidate"] is True
    assert not torch.allclose(f1.tensor[7:], f2.tensor[7:])
    with pytest.raises(ROIRenderValidationError):
        ROIFeatureBuilder().build(roi, _meta(private_or_optional_region=True))
    with pytest.raises(ROIRenderValidationError):
        ROIFeatureBuilder().build(roi, _meta(output_authority="authoritative"))


def test_roi_request_as_dict_is_flattened_from_real_sprint8_request():
    req = _roi_request(decision_type=RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value, weak=True, authority="weak")
    normalized = roi_request_to_dict(req.as_dict())
    assert normalized["region_id"] == req.region_id
    assert normalized["entity_id"] == req.entity_id
    assert normalized["output_authority"] == "weak"
    assert normalized["identity_locked"] is False
    assert normalized["weak_memory_candidate"] is True
    assert normalized["memory_authority"] == "weak"
    assert normalized["source_memory_authority"] == "weak"
    assert normalized["renderer_must_preserve_identity"] is False
    assert "memory_usage" in normalized and "identity_policy" in normalized


def test_weak_identity_reveal_rejected_for_flat_and_nested_forms():
    flat = _meta(canonical_region_id="face", region_id="p1:face", route_decision_type="route_reveal_weak_memory", render_mode="weak_memory_refine", identity_locked=True, weak_memory_candidate=True, output_authority="weak")
    with pytest.raises(ROIRenderValidationError, match="weak_identity_reveal_forbidden"):
        ROIFeatureBuilder().build(np.zeros((4, 4, 3), dtype=np.float32), flat)
    nested = _roi_request(region="face", decision_type=RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value).as_dict()
    nested["route_decision_type"] = "route_reveal_weak_memory"
    nested["render_mode"] = ROIRenderMode.WEAK_MEMORY_REFINE.value
    nested["memory_usage"]["weak_memory_candidate"] = True
    nested["memory_usage"]["output_reference_authority"] = "weak"
    with pytest.raises(ROIRenderValidationError, match="weak_identity_reveal_forbidden"):
        LightweightROIRendererBackend(enabled=True).synthesize_patch(_request(nested))


def test_extract_region_supports_normalized_and_absolute_bbox():
    frame = np.zeros((8, 8, 3), dtype=np.float32)
    normalized = _extract_region(frame, RegionRef("p1:torso", BBox(0.25, 0.25, 0.5, 0.5), "normalized"))
    absolute = _extract_region(frame, RegionRef("p1:torso", BBox(2, 2, 4, 4), "absolute"))
    assert normalized.shape == (4, 4, 3)
    assert absolute.shape == (4, 4, 3)


def test_dataset_skips_unsafe_and_generated_targets(tmp_path):
    valid = _meta(roi_before=[[[0, 0, 0]]], roi_after=[[[1, 1, 1]]])
    private = _meta(roi_before=[[[0, 0, 0]]], roi_after=[[[1, 1, 1]]], private_or_optional_region=True)
    generated = _meta(roi_before=[[[0, 0, 0]]], roi_after=[[[1, 1, 1]]], target_role="generated_runtime_output")
    unknown = _meta(roi_before=[[[0, 0, 0]]], roi_after=[[[1, 1, 1]]], route_decision_type="block_unknown_defer")
    ds = RendererObservedPairDataset(str(_manifest(tmp_path, [valid, private, generated, unknown])), roi_size=4)
    assert len(ds) == 1
    assert ds.skipped_reasons["private_blocked"] == 1
    assert ds.skipped_reasons["generated_target_forbidden"] == 1
    assert ds.skipped_reasons["unknown_blocked"] == 1
    sample = ds[0]
    assert sample["x"].shape == (16, 4, 4)
    assert sample["rgb"].shape == (3, 4, 4)


def test_training_loop_saves_contract_safe_checkpoint(tmp_path):
    rec = _meta(roi_before=np.zeros((4, 4, 3)).tolist(), roi_after=np.ones((4, 4, 3)).tolist(), identity_locked=True, weak_memory_candidate=True, output_authority="weak")
    ckpt = tmp_path / "roi.pt"
    metrics = train_lightweight_roi_renderer(str(_manifest(tmp_path, [rec])), str(ckpt), epochs=1, batch_size=1, roi_size=4, seed=3, device="cpu", strict=True)
    assert ckpt.exists()
    assert metrics["sample_count"] == 1
    assert "train_loss" in metrics and metrics["identity_locked_count"] == 1 and metrics["weak_memory_count"] == 1
    loaded, payload = load_lightweight_roi_checkpoint(str(ckpt))
    assert isinstance(loaded, TinyROIRenderer)
    assert payload["model_state_dict"]
    assert payload["contract_version"] == "lightweight_roi_renderer_v1"
    assert "roi_before" not in payload and "roi_after" not in payload


def _request(meta):
    return PatchSynthesisRequest(
        region=RegionRef("p1:torso", BBox(0, 0, 4, 4), "test"),
        scene_state=SceneGraph(frame_index=0),
        memory_summary={},
        transition_context={"roi_renderer_contract": meta},
        retrieval_summary={},
        current_frame=np.zeros((4, 4, 3), dtype=np.float32).tolist(),
    )


def test_inference_backend_loads_checkpoint_and_keeps_weak_output(tmp_path):
    rec = _meta(roi_before=np.zeros((4, 4, 3)).tolist(), roi_after=np.ones((4, 4, 3)).tolist(), output_authority="weak", weak_memory_candidate=True, route_decision_type="route_reveal_weak_memory", render_mode="weak_memory_refine")
    ckpt = tmp_path / "roi.pt"
    train_lightweight_roi_renderer(str(_manifest(tmp_path, [rec])), str(ckpt), roi_size=4, strict=True)
    backend = LightweightROIRendererBackend(str(ckpt))
    out = backend.synthesize_patch(_request(_roi_request(decision_type=RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value, weak=True, authority="weak").as_dict()))
    assert out.height == 4 and out.width == 4
    assert out.execution_trace["renderer_contract_validated"] is True
    assert out.execution_trace["lightweight_roi_renderer_used"] is True
    assert out.metadata["output_authority"] == "weak"
    assert all(v is False for v in out.execution_trace["forbidden_operation_flags"].values())


class _SafeFallback:
    def synthesize_patch(self, request: PatchSynthesisRequest) -> PatchSynthesisOutput:
        return PatchSynthesisOutput(
            region=request.region,
            rgb_patch=np.zeros((4, 4, 3), dtype=np.float32).tolist(),
            alpha_mask=np.ones((4, 4), dtype=np.float32).tolist(),
            height=4,
            width=4,
            channels=3,
            confidence=0.5,
            execution_trace={"fallback_source": "test"},
            metadata={},
        )


def test_disabled_backend_fallback_is_wrapped_with_safe_roi_output():
    backend = LightweightROIRendererBackend(enabled=False, fallback=_SafeFallback())
    out = backend.synthesize_patch(_request(_roi_request().as_dict()))
    assert out.execution_trace["fallback_used"] is True
    assert out.execution_trace["renderer_contract_validated"] is True
    assert "roi_renderer_output" in out.execution_trace
    assert out.metadata["output_authority"] in {"generated", "generated_from_memory", "weak"}
    assert all(v is False for v in out.execution_trace["forbidden_operation_flags"].values())


def test_backend_rejects_unsafe_routes_and_factory_supports_optional_backend():
    backend = LightweightROIRendererBackend(enabled=True)
    with pytest.raises(ROIRenderValidationError):
        backend.synthesize_patch(_request(_meta(route_decision_type="block_private_region", private_or_optional_region=True)))
    with pytest.raises(ROIRenderValidationError, match="weak_identity_reveal_forbidden"):
        backend.synthesize_patch(_request(_meta(canonical_region_id="face", route_decision_type="route_reveal_weak_memory", render_mode="weak_memory_refine", identity_locked=True, weak_memory_candidate=True, output_authority="weak")))
    bundle = LearnedBackendFactory(BackendConfig(patch_backend="lightweight_roi", patch_checkpoint_path="")).build()
    assert isinstance(bundle.patch_backend, LightweightROIRendererBackend)
