from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.schema import BBox, PersonNode, SceneGraph, VideoMemory
from dynamics.graph_delta_contract import GraphDeltaContract, GraphDeltaStep, RegionDelta, RoutingCandidate
from memory.memory_policy import MemoryAuthority, MemoryFamily, MemoryMaterialProvenance
from reveal.reveal_contract import OcclusionLifecycleState, RevealContract, RevealDecision, RevealDecisionType, RevealMemoryEvidence, RevealRoutingCandidate
from runtime.region_routing_contract import (
    RegionRoutingBlockReason,
    RegionRoutingContract,
    RegionRoutingDecision,
    RegionRoutingDecisionType,
    RegionRoutingValidationError,
    build_region_routing_handoff,
    validate_region_routing_contract,
    validate_region_routing_decision,
)


def _scene() -> SceneGraph:
    return SceneGraph(frame_index=0, persons=[PersonNode(person_id="p1", track_id=None, bbox=BBox(0.1, 0.1, 0.8, 0.9), mask_ref=None)])


def _delta(region: str, delta_type: str = "visibility_delta", *, private: bool = False, render: bool = True) -> RegionDelta:
    return RegionDelta(
        entity_id="p1",
        canonical_region_id=region,
        region_id=f"p1:{region}",
        delta_type=delta_type,
        action_type="test_action",
        phase_id="action_0:phase_0:test",
        phase_type="test",
        region_role="primary",
        expected_motion_role="test",
        confidence=0.8,
        provenance="test",
        source_planner_trace=("test",),
        identity_locked=region in {"face", "head", "hair", "scalp"},
        protected_region=region in {"face", "head", "hair", "scalp"},
        requires_routing=True,
        requires_rendering_candidate=render,
        reveal_may_be_required=delta_type in {"visibility_delta", "occlusion_delta"},
        occlusion_reasoning_required=delta_type == "occlusion_delta",
        secondary_motion_required=False,
        private_or_optional_region=private,
        allowed_to_modify_geometry=True,
        allowed_to_modify_visibility=True,
        allowed_to_modify_identity=False,
        validation_reasons=("test",),
    )


def _graph(region: str = "torso", delta_type: str = "visibility_delta", *, private: bool = False, render: bool = True) -> GraphDeltaContract:
    d = _delta(region, delta_type, private=private, render=render)
    step = GraphDeltaStep(
        step_id="step_0_0",
        action_order=0,
        phase_order=0,
        action_type=d.action_type,
        phase_id=d.phase_id,
        phase_type=d.phase_type,
        region_deltas=(d,),
    )
    candidate = RoutingCandidate(
        region_id=d.region_id,
        canonical_region_id=d.canonical_region_id,
        reason="test_requires_routing",
        delta_type=str(d.delta_type),
        action_type=str(d.action_type),
        phase_id=d.phase_id,
        requires_rendering_candidate=d.requires_rendering_candidate,
    )
    return GraphDeltaContract(steps=(step,), routing_candidates=(candidate,))



def _graph_multi(*items: tuple[str, str]) -> GraphDeltaContract:
    deltas = tuple(_delta(region, delta_type) for region, delta_type in items)
    step = GraphDeltaStep(
        step_id="step_0_0",
        action_order=0,
        phase_order=0,
        action_type="test_action",
        phase_id="action_0:phase_0:test",
        phase_type="test",
        region_deltas=deltas,
    )
    candidates = tuple(
        RoutingCandidate(
            region_id=d.region_id,
            canonical_region_id=d.canonical_region_id,
            reason="test_requires_routing",
            delta_type=str(d.delta_type),
            action_type=str(d.action_type),
            phase_id=d.phase_id,
            requires_rendering_candidate=d.requires_rendering_candidate,
        )
        for d in deltas
    )
    return GraphDeltaContract(steps=(step,), routing_candidates=candidates)


def _combine_reveals(*contracts: RevealContract) -> RevealContract:
    decisions = tuple(decision for contract in contracts for decision in contract.decisions)
    candidates = tuple(candidate for contract in contracts for candidate in contract.routing_candidates)
    return RevealContract(decisions=decisions, routing_candidates=candidates)

def _evidence(*, authority: str = MemoryAuthority.REUSABLE.value, family: str = MemoryFamily.BODY_SHAPE.value, weak: bool = False, material: str = MemoryMaterialProvenance.OBSERVED_PARSER.value, observed_directly: bool = True) -> RevealMemoryEvidence:
    return RevealMemoryEvidence(
        memory_family=family,
        reference_kind="identity_reference" if family == MemoryFamily.IDENTITY.value else "body_shape_reference",
        authority=authority,
        material_provenance=material,
        memory_support_level="weak" if weak else "strong",
        evidence_score=0.8,
        confidence=0.8,
        observed_directly=observed_directly,
        generated=material == MemoryMaterialProvenance.GENERATED.value,
        inferred=False,
        record_id="mem",
        mask_evidence_type="parser_mask",
        supports_observed_reveal=authority == MemoryAuthority.AUTHORITATIVE.value or not weak,
        supports_weak_reveal=weak,
    )


def _reveal(region: str, decision_type: RevealDecisionType, *, delta_type: str = "visibility_delta", private: bool = False, evidence: RevealMemoryEvidence | None = None, render: bool | None = None) -> RevealContract:
    if render is None:
        render = decision_type in {RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY, RevealDecisionType.REVEAL_FROM_WEAK_MEMORY, RevealDecisionType.PRESERVE_VISIBLE}
    lifecycle = {
        RevealDecisionType.REVEAL_UNKNOWN_DEFER: OcclusionLifecycleState.NEWLY_REVEALED_UNKNOWN,
        RevealDecisionType.REVEAL_BLOCKED_PRIVATE: OcclusionLifecycleState.PRIVATE_BLOCKED,
        RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK: OcclusionLifecycleState.IDENTITY_RISK_BLOCKED,
        RevealDecisionType.REVEAL_BLOCKED_UNSUPPORTED_REGION: OcclusionLifecycleState.REVEAL_BLOCKED,
        RevealDecisionType.REVEAL_BLOCKED_NO_EVIDENCE: OcclusionLifecycleState.REVEAL_BLOCKED,
        RevealDecisionType.OCCLUSION_REASONING_REQUIRED: OcclusionLifecycleState.OCCLUDED_UNKNOWN,
        RevealDecisionType.NEWLY_OCCLUDED: OcclusionLifecycleState.NEWLY_OCCLUDED,
        RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY: OcclusionLifecycleState.NEWLY_REVEALED_KNOWN,
        RevealDecisionType.REVEAL_FROM_WEAK_MEMORY: OcclusionLifecycleState.NEWLY_REVEALED_WEAK,
        RevealDecisionType.PRESERVE_VISIBLE: OcclusionLifecycleState.VISIBLE_STABLE,
    }[decision_type]
    decision = RevealDecision(
        entity_id="p1",
        canonical_region_id=region,
        region_id=f"p1:{region}",
        decision_type=decision_type,
        lifecycle_state=lifecycle,
        reason="test_reveal_decision",
        source_delta_type=delta_type,
        action_type="test_action",
        phase_id="action_0:phase_0:test",
        action_order=0,
        phase_order=0,
        identity_locked=region in {"face", "head", "hair", "scalp"},
        allowed_to_modify_identity=False,
        reveal_allowed=render,
        requires_rendering_candidate=render,
        private_or_optional_region=private,
        memory_evidence=evidence or RevealMemoryEvidence(),
        policy_reasons=("test",),
    )
    candidates = ()
    if decision_type in {RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY, RevealDecisionType.REVEAL_FROM_WEAK_MEMORY, RevealDecisionType.PRESERVE_VISIBLE}:
        candidates = (
            RevealRoutingCandidate(
                region_id=decision.region_id,
                canonical_region_id=region,
                reveal_decision_type=decision_type.value,
                reason="test_reveal_route",
                source_delta_type=delta_type,
                action_type=decision.action_type,
                phase_id=decision.phase_id,
                requires_rendering_candidate=True,
                identity_locked=decision.identity_locked,
                reveal_allowed=True,
                memory_reference_kind=decision.memory_evidence.reference_kind,
                memory_authority=decision.memory_evidence.authority,
            ),
        )
    return RevealContract(decisions=(decision,), routing_candidates=candidates)


def _handoff(graph: GraphDeltaContract, reveal: RevealContract) -> RegionRoutingContract:
    return build_region_routing_handoff(scene_graph=_scene(), graph_delta_contract=graph, reveal_contract=reveal, memory=VideoMemory()).region_routing_contract


def test_region_routing_schema_exists_and_serializes() -> None:
    contract = _handoff(_graph("torso", "pose_delta"), RevealContract())
    assert isinstance(contract, RegionRoutingContract)
    assert RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value == "route_pose_update"
    assert RegionRoutingDecisionType.BLOCK_UNKNOWN_DEFER.value == "block_unknown_defer"
    assert RegionRoutingValidationError
    payload = contract.decisions[0].as_dict()
    assert json.loads(json.dumps(payload))["decision_type"] == "route_pose_update"


@pytest.mark.parametrize(
    ("reveal_type", "expected_type", "render"),
    [
        (RevealDecisionType.REVEAL_UNKNOWN_DEFER, RegionRoutingDecisionType.BLOCK_UNKNOWN_DEFER, False),
        (RevealDecisionType.REVEAL_BLOCKED_PRIVATE, RegionRoutingDecisionType.BLOCK_PRIVATE_REGION, False),
        (RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK, RegionRoutingDecisionType.BLOCK_IDENTITY_RISK, False),
        (RevealDecisionType.OCCLUSION_REASONING_REQUIRED, RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY, False),
        (RevealDecisionType.NEWLY_OCCLUDED, RegionRoutingDecisionType.ROUTE_NEWLY_OCCLUDED_TRACKING_ONLY, False),
    ],
)
def test_reveal_decisions_override_dynamics_candidates(reveal_type: RevealDecisionType, expected_type: RegionRoutingDecisionType, render: bool) -> None:
    region = "external_genital_region" if reveal_type == RevealDecisionType.REVEAL_BLOCKED_PRIVATE else "torso"
    contract = _handoff(_graph("torso"), _reveal(region, reveal_type, private=reveal_type == RevealDecisionType.REVEAL_BLOCKED_PRIVATE))
    decision = contract.decisions[0]
    assert decision.decision_type == expected_type.value
    assert decision.render_candidate_allowed is render
    assert decision.region_id not in contract.renderable_region_ids


def test_reveal_observed_and_weak_memory_routes_preserve_constraints() -> None:
    observed = _handoff(_graph("torso"), _reveal("torso", RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY, evidence=_evidence(authority=MemoryAuthority.REUSABLE.value)))
    observed_decision = observed.decisions[0]
    assert observed_decision.decision_type == RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value
    assert observed_decision.requires_reveal_memory is True
    assert observed_decision.memory_reference_kind == "body_shape_reference"

    weak = _handoff(_graph("torso"), _reveal("torso", RevealDecisionType.REVEAL_FROM_WEAK_MEMORY, evidence=_evidence(weak=True, authority=MemoryAuthority.WEAK.value)))
    weak_decision = weak.decisions[0]
    assert weak_decision.decision_type == RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value
    assert weak_decision.weak_memory_candidate is True
    assert weak_decision.output_reference_authority == "weak"
    assert weak_decision.memory_write_allowed is False
    assert weak_decision.identity_memory_write_allowed is False
    assert weak_decision.observed_evidence_claim_allowed is False
    assert weak_decision.renderer_must_not_create_observed_evidence is True
    assert weak_decision.renderer_must_not_create_identity_memory is True


@pytest.mark.parametrize(
    ("delta_type", "expected"),
    [
        ("pose_delta", RegionRoutingDecisionType.ROUTE_POSE_UPDATE),
        ("expression_delta", RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE),
        ("visibility_delta", RegionRoutingDecisionType.ROUTE_VISIBILITY_UPDATE),
        ("interaction_delta", RegionRoutingDecisionType.ROUTE_INTERACTION_UPDATE),
        ("garment_intent_delta", RegionRoutingDecisionType.ROUTE_GARMENT_INTENT),
    ],
)
def test_ordinary_dynamics_delta_routes_when_no_reveal_block(delta_type: str, expected: RegionRoutingDecisionType) -> None:
    region = "upper_garment" if delta_type == "garment_intent_delta" else ("face" if delta_type == "expression_delta" else "torso")
    contract = _handoff(_graph(region, delta_type), RevealContract())
    decision = contract.decisions[0]
    assert decision.decision_type == expected.value
    assert decision.route_allowed is True
    assert decision.render_candidate_allowed is True


def test_occlusion_delta_without_reveal_override_is_diagnostic_only() -> None:
    contract = _handoff(_graph("torso", "occlusion_delta"), RevealContract())
    decision = contract.decisions[0]
    assert decision.decision_type == RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value
    assert decision.route_allowed is False
    assert decision.render_candidate_allowed is False
    assert decision.diagnostic_only is True
    assert decision.roi_required is False
    assert "p1:torso" not in contract.renderable_region_ids


def test_diagnostic_decision_does_not_globally_block_separate_routeable_decision() -> None:
    reveal = _reveal("face", RevealDecisionType.OCCLUSION_REASONING_REQUIRED, delta_type="occlusion_delta", render=False)
    contract = _handoff(_graph_multi(("face", "expression_delta"), ("face", "occlusion_delta")), reveal)
    expression = [decision for decision in contract.decisions if decision.decision_type == RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value]
    diagnostic = [decision for decision in contract.decisions if decision.decision_type == RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value]
    assert expression and diagnostic
    assert "p1:face" in contract.renderable_region_ids
    assert "p1:face" not in contract.blocked_region_ids
    assert "p1:face" in contract.diagnostic_region_ids


def test_diagnostic_only_and_hard_blocked_regions_are_not_renderable() -> None:
    diagnostic = _handoff(_graph("torso", "occlusion_delta"), RevealContract())
    assert diagnostic.renderable_region_ids == ()
    assert diagnostic.diagnostic_region_ids == ("p1:torso",)
    assert diagnostic.blocked_region_ids == ()

    hard_blocked = _handoff(_graph("torso"), _reveal("torso", RevealDecisionType.REVEAL_UNKNOWN_DEFER, render=False))
    assert "p1:torso" in hard_blocked.blocked_region_ids
    assert "p1:torso" not in hard_blocked.renderable_region_ids


def test_multiple_reveal_decisions_resolve_by_severity_not_input_order() -> None:
    observed = _reveal("torso", RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY, evidence=_evidence(authority=MemoryAuthority.REUSABLE.value))
    unknown = _reveal("torso", RevealDecisionType.REVEAL_UNKNOWN_DEFER, render=False)
    for reveal in (_combine_reveals(observed, unknown), _combine_reveals(unknown, observed)):
        contract = _handoff(_graph("torso"), reveal)
        assert "p1:torso" in contract.blocked_region_ids
        assert "p1:torso" not in contract.renderable_region_ids
        assert any(decision.decision_type == RegionRoutingDecisionType.BLOCK_UNKNOWN_DEFER.value for decision in contract.decisions)


def test_contract_validation_requires_allowlist_and_keeps_diagnostics_out_of_blocked_ids() -> None:
    contract = _handoff(_graph_multi(("face", "expression_delta"), ("face", "occlusion_delta")), _reveal("face", RevealDecisionType.OCCLUSION_REASONING_REQUIRED, delta_type="occlusion_delta", render=False))
    validate_region_routing_contract(contract)
    assert "p1:face" in contract.renderable_region_ids
    assert "p1:face" in contract.diagnostic_region_ids
    assert "p1:face" not in contract.blocked_region_ids
    with pytest.raises(RegionRoutingValidationError, match="renderable_without_decision"):
        validate_region_routing_contract(replace(contract, renderable_region_ids=("p1:not_in_decisions",)))
    with pytest.raises(RegionRoutingValidationError, match="blocked_region_ids_mismatch"):
        validate_region_routing_contract(replace(contract, blocked_region_ids=("p1:face",)))


def test_identity_regions_are_locked_and_weak_identity_reveal_is_blocked() -> None:
    for region in ("face", "head", "hair", "scalp"):
        decision = _handoff(_graph(region, "pose_delta"), RevealContract()).decisions[0]
        assert decision.identity_locked is True
        assert decision.allowed_to_modify_identity is False
        assert decision.renderer_must_preserve_identity is True

    observed_face = _handoff(
        _graph("face"),
        _reveal("face", RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY, evidence=_evidence(authority=MemoryAuthority.AUTHORITATIVE.value, family=MemoryFamily.IDENTITY.value)),
    ).decisions[0]
    assert observed_face.decision_type == RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value
    assert observed_face.memory_authority == MemoryAuthority.AUTHORITATIVE.value

    generated_face = _reveal(
        "face",
        RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK,
        evidence=_evidence(authority=MemoryAuthority.WEAK.value, family=MemoryFamily.IDENTITY.value, material=MemoryMaterialProvenance.GENERATED.value, observed_directly=False),
        render=False,
    )
    blocked = _handoff(_graph("face"), generated_face).decisions[0]
    assert blocked.decision_type == RegionRoutingDecisionType.BLOCK_IDENTITY_RISK.value

    weak_face = _handoff(
        _graph("face"),
        _reveal("face", RevealDecisionType.REVEAL_FROM_WEAK_MEMORY, evidence=_evidence(weak=True, authority=MemoryAuthority.WEAK.value, family=MemoryFamily.IDENTITY.value)),
    ).decisions[0]
    assert weak_face.decision_type == RegionRoutingDecisionType.BLOCK_IDENTITY_RISK.value
    assert weak_face.render_candidate_allowed is False


@pytest.mark.parametrize("region", ["external_genital_region", "male_external_genital_region", "female_pelvic_region"])
def test_private_optional_regions_are_blocked_without_roi(region: str) -> None:
    decision = _handoff(_graph("torso"), _reveal(region, RevealDecisionType.REVEAL_BLOCKED_PRIVATE, private=True, render=False)).decisions[0]
    assert decision.route_allowed is False
    assert decision.render_candidate_allowed is False
    assert decision.roi_required is False
    assert decision.block_reason == RegionRoutingBlockReason.PRIVATE_REGION.value


def test_unknown_defer_and_unsupported_regions_block_without_fallback_strategy() -> None:
    unknown = _handoff(_graph("torso"), _reveal("torso", RevealDecisionType.REVEAL_UNKNOWN_DEFER, render=False)).decisions[0]
    assert unknown.decision_type == RegionRoutingDecisionType.BLOCK_UNKNOWN_DEFER.value
    assert unknown.roi_required is False
    assert unknown.selected_render_strategy is None

    unsupported = _handoff(_graph("torso"), _reveal("not_a_region", RevealDecisionType.REVEAL_BLOCKED_UNSUPPORTED_REGION, render=False)).decisions[0]
    assert unsupported.decision_type == RegionRoutingDecisionType.BLOCK_UNSUPPORTED_REGION.value
    assert unsupported.render_candidate_allowed is False


def test_validation_fails_loudly_for_contract_violations() -> None:
    good = _handoff(_graph("torso"), _reveal("torso", RevealDecisionType.REVEAL_UNKNOWN_DEFER, render=False)).decisions[0]
    with pytest.raises(RegionRoutingValidationError, match="blocked_route_allowed"):
        validate_region_routing_decision(replace(good, route_allowed=True))
    with pytest.raises(RegionRoutingValidationError, match="diagnostic_render_candidate"):
        validate_region_routing_decision(replace(good, diagnostic_only=True, render_candidate_allowed=True, route_allowed=False, blocked=False, decision_type=RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value))
    with pytest.raises(RegionRoutingValidationError, match="private_region_route_forbidden"):
        validate_region_routing_decision(replace(good, canonical_region_id="external_genital_region", region_id="p1:external_genital_region", render_candidate_allowed=True, blocked=False, block_reason=None, decision_type=RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value))
    with pytest.raises(RegionRoutingValidationError, match="unknown_defer_render_candidate"):
        validate_region_routing_decision(replace(good, render_candidate_allowed=True, blocked=False, diagnostic_only=False, route_allowed=True, decision_type=RegionRoutingDecisionType.ROUTE_VISIBILITY_UPDATE.value))

    identity = _handoff(_graph("face", "pose_delta"), RevealContract()).decisions[0]
    with pytest.raises(RegionRoutingValidationError, match="identity_modification_forbidden"):
        validate_region_routing_decision(replace(identity, allowed_to_modify_identity=True))
    with pytest.raises(RegionRoutingValidationError, match="weak_identity_reveal_forbidden"):
        validate_region_routing_decision(replace(identity, decision_type=RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value, weak_memory_candidate=True, output_reference_authority="weak"))
    with pytest.raises(RegionRoutingValidationError, match="rendered_pixels_claimed"):
        validate_region_routing_decision(replace(identity, rendered_pixels_generated=True))
    with pytest.raises(RegionRoutingValidationError, match="memory_write_claimed"):
        validate_region_routing_decision(replace(identity, memory_write_performed=True))
    with pytest.raises(RegionRoutingValidationError, match="observed_evidence_claimed"):
        validate_region_routing_decision(replace(identity, observed_evidence_created=True))


def test_renderable_decision_for_region_id_skips_diagnostic_first_decision():
    routeable = next(d for d in build_region_routing_handoff(scene_graph=_scene(), graph_delta_contract=_graph("face", "expression_delta"), reveal_contract=RevealContract(), memory=VideoMemory()).region_routing_contract.decisions if d.region_id == "p1:face")
    diagnostic = replace(
        routeable,
        decision_type=RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value,
        route_allowed=False,
        render_candidate_allowed=False,
        diagnostic_only=True,
        roi_required=False,
    )
    contract = RegionRoutingContract(decisions=(diagnostic, routeable), renderable_region_ids=(routeable.region_id,))

    assert contract.decision_for_region_id("p1:face") is diagnostic
    assert contract.renderable_decision_for_region_id("p1:face") is routeable
    assert contract.renderable_decision_for_region_id("p1:torso") is None
