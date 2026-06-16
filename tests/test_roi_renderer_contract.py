from __future__ import annotations

import json
from dataclasses import replace

import pytest

from runtime.region_routing_contract import RegionRoutingContract, RegionRoutingDecision, RegionRoutingDecisionType
from rendering.roi_renderer_contract import (
    ROIRenderMode,
    ROIRenderOutput,
    ROIRenderValidationError,
    ROIRendererContract,
    build_roi_render_request,
    validate_roi_render_output,
    wrap_roi_render_output,
)


def _decision(region="torso", decision_type=RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value, *, blocked=False, diagnostic=False, render=True, route=True, roi=True, private=False, weak=False, authority="reusable"):
    identity = region in {"face", "head", "hair", "scalp"}
    return RegionRoutingDecision(
        region_id=f"p1:{region}", entity_id="p1", canonical_region_id=region,
        decision_type=decision_type, source="test", source_delta_type="pose_delta", source_reveal_decision_type=None,
        action_type="turn", phase_id="phase", action_order=0, phase_order=0,
        route_allowed=route, render_candidate_allowed=render, blocked=blocked, block_reason="blocked" if blocked else None,
        diagnostic_only=diagnostic, identity_locked=identity, allowed_to_modify_identity=False,
        requires_identity_memory=identity and decision_type == RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value,
        requires_appearance_memory=False, requires_reveal_memory="reveal" in decision_type,
        memory_reference_kind="identity_reference" if identity else "body_shape_reference",
        memory_authority=authority, memory_family="identity" if identity else "body_shape",
        weak_memory_candidate=weak, preserve_visible=decision_type == RegionRoutingDecisionType.ROUTE_PRESERVE_VISIBLE.value,
        occlusion_reasoning_only=decision_type == RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value,
        newly_occluded_tracking_only=decision_type == RegionRoutingDecisionType.ROUTE_NEWLY_OCCLUDED_TRACKING_ONLY.value,
        private_or_optional_region=private, roi_required=roi, roi_source_policy="explicit_roi_required",
        renderer_strategy_allowed_values=("deterministic",), selected_render_strategy="deterministic",
        renderer_must_preserve_identity=identity, renderer_must_not_create_observed_evidence=True,
        renderer_must_not_create_identity_memory=True, validation_reasons=("test",), provenance=("test",),
        output_reference_authority="weak" if weak else "generated", memory_write_allowed=False, identity_memory_write_allowed=False,
        observed_evidence_claim_allowed=False,
    )


def _contract(d):
    return RegionRoutingContract(decisions=(d,), renderable_region_ids=(() if d.blocked or d.diagnostic_only or not d.render_candidate_allowed or not d.roi_required or not d.route_allowed else (d.region_id,)))


def _request(d):
    return build_roi_render_request(route_decision=d, routing_contract=_contract(d), roi_bbox=(0, 0, 1, 1), current_frame_shape=(8, 8, 3), frame_index=0)


def test_schema_serializes_and_modes_exist():
    d = _decision()
    req = _request(d)
    out = wrap_roi_render_output(request=req, patch_ref="p", patch_shape=(2, 2, 3), alpha_mask_ref="a")
    contract = ROIRendererContract(requests=(req,), outputs=(out,), renderable_region_ids=(d.region_id,))
    assert ROIRenderMode.PRESERVE.value == "preserve"
    assert ROIRenderMode.WEAK_MEMORY_REFINE.value == "weak_memory_refine"
    assert isinstance(json.dumps(contract.as_dict()), str)
    assert ROIRenderValidationError


@pytest.mark.parametrize("route,mode", [
    (RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value, ROIRenderMode.POSE_LOCAL_UPDATE.value),
    (RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value, ROIRenderMode.EXPRESSION_LOCAL_UPDATE.value),
    (RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value, ROIRenderMode.REVEAL_MEMORY.value),
    (RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value, ROIRenderMode.WEAK_MEMORY_REFINE.value),
    (RegionRoutingDecisionType.ROUTE_PRESERVE_VISIBLE.value, ROIRenderMode.PRESERVE.value),
])
def test_request_builder_route_mode_mapping(route, mode):
    d = _decision(decision_type=route, weak=route.endswith("weak_memory"), authority="reusable")
    req = _request(d)
    assert req.render_mode == mode


@pytest.mark.parametrize("kwargs", [
    {"route": False}, {"blocked": True}, {"diagnostic": True}, {"render": False}, {"roi": False},
])
def test_region_routing_enforcement(kwargs):
    d = _decision(**kwargs)
    with pytest.raises(ROIRenderValidationError):
        _request(d)


def test_region_not_in_renderable_allowlist_fails():
    d = _decision()
    with pytest.raises(ROIRenderValidationError):
        build_roi_render_request(route_decision=d, routing_contract=RegionRoutingContract(decisions=(d,), renderable_region_ids=()), roi_bbox=(0,0,1,1), current_frame_shape=(1,1,3))


@pytest.mark.parametrize("region", ["face", "head", "hair", "scalp"])
def test_identity_regions_are_locked_and_outputs_do_not_modify_identity(region):
    req = _request(_decision(region=region, decision_type=RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value))
    assert req.identity_policy.identity_locked is True
    assert req.identity_policy.allowed_to_modify_identity is False
    assert req.identity_policy.renderer_must_preserve_identity is True
    out = wrap_roi_render_output(request=req, patch_ref="p", patch_shape=(1, 1, 3))
    assert out.identity_modified is False
    assert out.identity_memory_created is False


def test_weak_identity_reveal_rejected():
    d = _decision(region="face", decision_type=RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value, weak=True)
    with pytest.raises(ROIRenderValidationError):
        _request(d)


def test_weak_memory_stays_weak_and_non_authoritative():
    req = _request(_decision(region="torso", decision_type=RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value, weak=True))
    out = wrap_roi_render_output(request=req, patch_ref="p", patch_shape=(1, 1, 3))
    assert req.render_mode == ROIRenderMode.WEAK_MEMORY_REFINE.value
    assert out.output_authority == "weak"
    assert out.memory_write_allowed is False
    assert out.observed_evidence_created is False
    assert out.identity_memory_created is False
    with pytest.raises(ROIRenderValidationError):
        validate_roi_render_output(replace(out, output_authority="authoritative"), request=req)


@pytest.mark.parametrize("region", ["external_genital_region", "male_external_genital_region", "male_pelvic_region", "female_pelvic_region", "breast_region"])
def test_private_optional_regions_fail_before_patch_or_mask(region):
    d = _decision(region=region, private=True)
    with pytest.raises(ROIRenderValidationError):
        _request(d)


@pytest.mark.parametrize("route", [
    RegionRoutingDecisionType.BLOCK_UNKNOWN_DEFER.value,
    RegionRoutingDecisionType.BLOCK_NO_SAFE_MEMORY.value,
    RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value,
    RegionRoutingDecisionType.ROUTE_NEWLY_OCCLUDED_TRACKING_ONLY.value,
    RegionRoutingDecisionType.BLOCK_UNSUPPORTED_REGION.value,
])
def test_unknown_diagnostic_and_tracking_only_routes_blocked(route):
    d = _decision(decision_type=route, blocked=route.startswith("block_"), diagnostic=route.startswith("route_"), render=False, roi=False)
    with pytest.raises(ROIRenderValidationError):
        _request(d)


@pytest.mark.parametrize("field", [
    "observed_evidence_created", "identity_memory_created", "memory_write_allowed", "scene_graph_mutated",
    "private_region_rendered", "hidden_anatomy_generated", "clothing_removed", "identity_modified",
])
def test_output_forbidden_claims_fail(field):
    req = _request(_decision())
    out = wrap_roi_render_output(request=req, patch_ref="p", patch_shape=(1, 1, 3))
    with pytest.raises(ROIRenderValidationError):
        validate_roi_render_output(replace(out, **{field: True}), request=req)


def test_invalid_output_mode_for_route_fails():
    req = _request(_decision(decision_type=RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value))
    out = ROIRenderOutput(req.region_id, req.canonical_region_id, ROIRenderMode.POSE_LOCAL_UPDATE.value, req.route_decision_type, "p", (1,1,3))
    with pytest.raises(ROIRenderValidationError):
        validate_roi_render_output(out, request=req)


def test_request_builder_uses_renderable_decision_when_diagnostic_is_first():
    routeable = _decision(region="face", decision_type=RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value)
    diagnostic = replace(
        routeable,
        decision_type=RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value,
        route_allowed=False,
        render_candidate_allowed=False,
        diagnostic_only=True,
        roi_required=False,
    )
    contract = RegionRoutingContract(decisions=(diagnostic, routeable), renderable_region_ids=(routeable.region_id,))

    selected = contract.renderable_decision_for_region_id("p1:face")
    assert selected is routeable
    request = build_roi_render_request(route_decision=selected, routing_contract=contract, roi_bbox=(0, 0, 1, 1), current_frame_shape=(8, 8, 3))
    assert request.route_decision_type == RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value
    with pytest.raises(ROIRenderValidationError):
        build_roi_render_request(route_decision=diagnostic, routing_contract=contract, roi_bbox=(0, 0, 1, 1), current_frame_shape=(8, 8, 3))


def test_diagnostic_only_region_has_no_renderable_decision_and_no_request():
    diagnostic = _decision(decision_type=RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value, diagnostic=True, render=False, route=False, roi=False)
    contract = RegionRoutingContract(decisions=(diagnostic,), renderable_region_ids=())
    assert contract.renderable_decision_for_region_id(diagnostic.region_id) is None
    with pytest.raises(ROIRenderValidationError):
        build_roi_render_request(route_decision=diagnostic, routing_contract=contract, roi_bbox=(0, 0, 1, 1), current_frame_shape=(8, 8, 3))


def test_authoritative_identity_memory_source_does_not_make_renderer_output_authoritative():
    d = _decision(region="face", decision_type=RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value, authority="authoritative")
    req = _request(d)
    assert req.memory_usage.memory_authority == "authoritative"
    assert req.memory_usage.source_memory_authority == "authoritative"
    assert req.memory_usage.output_reference_authority != "authoritative"
    out = wrap_roi_render_output(request=req, patch_ref="p", patch_shape=(1, 1, 3))
    assert out.output_authority == "generated_from_memory"
    with pytest.raises(ROIRenderValidationError):
        validate_roi_render_output(replace(out, output_authority="authoritative"), request=req)


@pytest.mark.parametrize("authority", ["authoritative", "reusable", "observed"])
def test_output_validation_rejects_authoritative_reusable_or_observed_authority(authority):
    req = _request(_decision())
    out = wrap_roi_render_output(request=req, patch_ref="p", patch_shape=(1, 1, 3))
    with pytest.raises(ROIRenderValidationError):
        validate_roi_render_output(replace(out, output_authority=authority), request=req)
