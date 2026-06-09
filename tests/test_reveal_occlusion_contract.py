from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.schema import BBox, CanonicalRegionMemoryEntry, PersonNode, SceneGraph, VideoMemory
from dynamics.graph_delta_contract import DynamicsTrace, GraphDeltaContract, GraphDeltaStep, RegionDelta, RegionDeltaType, build_dynamics_handoff
from planning.action_plan import ActionPlanner, PlannerIntent
from reveal.reveal_contract import (
    OcclusionLifecycleState,
    RevealContract,
    RevealDecision,
    RevealDecisionType,
    RevealHandoffResult,
    RevealMemoryEvidence,
    RevealValidationError,
    build_reveal_handoff,
    validate_reveal_contract,
    validate_reveal_decision,
)


def _scene(*person_ids: str) -> SceneGraph:
    return SceneGraph(frame_index=0, persons=[PersonNode(person_id=pid, track_id=None, bbox=BBox(0.1, 0.1, 0.6, 0.8), mask_ref=None) for pid in person_ids])


def _plan(text: str):
    return ActionPlanner().plan(PlannerIntent(raw_text=text, strict=False), _scene("p1"))


def _memory_entry(
    region: str,
    *,
    entity_id: str = "p1",
    confidence: float = 0.82,
    evidence_score: float = 0.86,
    observed_directly: bool = True,
    generated: bool = False,
    inferred: bool = False,
    provenance: str = "parser_observed",
    mask_evidence_type: str = "parser_mask",
    memory_kind: str = "appearance",
    visibility_state: str = "visible",
) -> CanonicalRegionMemoryEntry:
    if region == "face":
        provenance = provenance if provenance != "parser_observed" else "face_observed"
        mask_evidence_type = "face_mask" if mask_evidence_type == "parser_mask" else mask_evidence_type
        memory_kind = "identity"
    return CanonicalRegionMemoryEntry(
        record_id=f"{entity_id}:{region}",
        entity_id=entity_id,
        canonical_region=region,
        memory_kind=memory_kind,
        mask_ref=f"mask:{entity_id}:{region}" if observed_directly else None,
        confidence=confidence,
        visibility_state=visibility_state,
        provenance=provenance,
        evidence_score=evidence_score,
        observed_directly=observed_directly,
        generated=generated,
        inferred=inferred,
        reliable_for_reuse=True,
        reliable_as_reference=True,
        suitable_for_reveal=True,
        observation_status="observed" if observed_directly else "inferred",
        mask_evidence_type=mask_evidence_type,
        source_frame_kind="observed_face" if region == "face" else "input_frame",
    )


def _memory(*entries: CanonicalRegionMemoryEntry) -> VideoMemory:
    memory = VideoMemory()
    for entry in entries:
        memory.canonical_region_memory[entry.record_id] = entry
    return memory


def _delta(
    region: str,
    *,
    delta_type: str = "visibility_delta",
    reveal: bool = True,
    private: bool = False,
    identity_locked: bool | None = None,
    occlusion_reasoning_required: bool | None = None,
    validation_reasons: tuple[str, ...] = ("test",),
) -> RegionDelta:
    locked = region in {"face", "head", "hair", "scalp"} if identity_locked is None else identity_locked
    return RegionDelta(
        entity_id="p1",
        canonical_region_id=region,
        region_id=f"p1:{region}",
        delta_type=delta_type,
        action_type="head_turn",
        phase_id="action_0:phase_0:prepare",
        phase_type="prepare",
        region_role="primary",
        expected_motion_role="visibility_reasoning",
        confidence=0.8,
        provenance="test_delta",
        source_planner_trace=("test",),
        identity_locked=locked,
        protected_region=locked,
        requires_routing=True,
        requires_rendering_candidate=True,
        reveal_may_be_required=reveal,
        occlusion_reasoning_required=(delta_type == "occlusion_delta" if occlusion_reasoning_required is None else occlusion_reasoning_required),
        secondary_motion_required=False,
        private_or_optional_region=private,
        allowed_to_modify_geometry=True,
        allowed_to_modify_visibility=True,
        allowed_to_modify_identity=False,
        validation_reasons=validation_reasons,
    )


def _contract(*deltas: RegionDelta) -> GraphDeltaContract:
    step = GraphDeltaStep(step_id="step_0_0", action_order=0, phase_order=0, action_type="head_turn", phase_id="action_0:phase_0:prepare", phase_type="prepare", region_deltas=tuple(deltas))
    return GraphDeltaContract(supported=True, steps=(step,))


def _handoff(region: str, memory: VideoMemory | None, **kwargs) -> RevealHandoffResult:
    return build_reveal_handoff(scene_graph=_scene("p1"), memory=memory, graph_delta_contract=_contract(_delta(region, **kwargs)), strict=False)


def test_reveal_schema_exists_and_serializes_to_json() -> None:
    decision = RevealDecision(
        entity_id="p1",
        canonical_region_id="torso",
        region_id="p1:torso",
        decision_type=RevealDecisionType.REVEAL_UNKNOWN_DEFER,
        lifecycle_state=OcclusionLifecycleState.NEWLY_REVEALED_UNKNOWN,
        reason="test",
        source_delta_type="visibility_delta",
        action_type="head_turn",
        phase_id="phase",
    )
    contract = RevealContract(decisions=(decision,))
    assert RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value == "reveal_from_observed_memory"
    assert RevealDecisionType.REVEAL_BLOCKED_PRIVATE.value == "reveal_blocked_private"
    assert OcclusionLifecycleState.NEWLY_REVEALED_WEAK.value == "newly_revealed_weak"
    assert OcclusionLifecycleState.PRIVATE_BLOCKED.value == "private_blocked"
    assert RevealValidationError("x", "y").code == "x"
    json.dumps(decision.as_dict())
    json.dumps(contract.as_dict())


def test_reveal_consumes_graph_delta_preserves_unsupported_and_multi_action_order() -> None:
    handoff = build_reveal_handoff(scene_graph=_scene("p1"), memory=None, graph_delta_contract=build_dynamics_handoff(_plan("turn head and sit down and smile")).graph_delta_contract)
    decisions = handoff.reveal_contract.decisions
    assert decisions
    assert [d.action_order for d in decisions] == sorted(d.action_order for d in decisions)
    assert any(d.action_type == "head_turn" for d in decisions)
    assert any(d.action_type == "sit_down" for d in decisions)
    assert any(d.action_type == "expression_change" for d in decisions)

    unsupported = GraphDeltaContract(
        supported=False,
        trace=DynamicsTrace(unsupported_planner_code="unsupported_action", unsupported_planner_reasons=("unsupported",), unsupported_planner_fragments=("fly away",)),
    )
    unsupported_reveal = build_reveal_handoff(scene_graph=_scene("p1"), memory=None, graph_delta_contract=unsupported)
    assert unsupported_reveal.supported is False
    assert unsupported_reveal.reveal_contract.supported is False
    assert unsupported_reveal.routing_candidates == ()
    assert unsupported_reveal.trace.unsupported_planner_fragments == ("fly away",)

    partial = build_reveal_handoff(scene_graph=_scene("p1"), memory=None, graph_delta_contract=build_dynamics_handoff(_plan("sit down and снимает пальто and smile")).graph_delta_contract)
    assert any("снимает пальто" in fragment for fragment in partial.trace.unsupported_planner_fragments)


def test_memory_evidence_policy_for_observed_weak_generated_and_missing_memory() -> None:
    observed_identity = _handoff("face", _memory(_memory_entry("face")))
    face_decision = observed_identity.reveal_contract.decisions[0]
    assert face_decision.decision_type == RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value
    assert face_decision.memory_evidence.authority == "authoritative"
    assert face_decision.memory_evidence.supports_observed_reveal is True

    generated_face = _handoff("face", _memory(_memory_entry("face", generated=True, provenance="generated_renderer", mask_evidence_type="face_mask")))
    assert generated_face.reveal_contract.decisions[0].decision_type == RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK.value
    assert generated_face.reveal_contract.decisions[0].memory_evidence.supports_observed_reveal is False

    torso = _handoff("torso", _memory(_memory_entry("torso")))
    assert torso.reveal_contract.decisions[0].decision_type == RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value
    assert torso.reveal_contract.decisions[0].memory_evidence.authority == "reusable"

    garment = _handoff("upper_garment", _memory(_memory_entry("upper_garment", memory_kind="garment")))
    assert garment.reveal_contract.decisions[0].decision_type == RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value
    assert garment.reveal_contract.decisions[0].memory_evidence.reference_kind == "garment_reference"

    weak = _handoff("torso", _memory(_memory_entry("torso", confidence=0.5, evidence_score=0.5)))
    assert weak.reveal_contract.decisions[0].decision_type == RevealDecisionType.REVEAL_FROM_WEAK_MEMORY.value
    assert weak.reveal_contract.decisions[0].memory_evidence.authority != "authoritative"

    for unsafe_entry in (
        _memory_entry("torso", generated=True, provenance="generated_renderer"),
        _memory_entry("torso", inferred=True, observed_directly=False, provenance="inferred_hidden", mask_evidence_type="bbox_projection"),
        _memory_entry("torso", provenance="fallback_patch", mask_evidence_type="parser_mask"),
    ):
        unsafe = _handoff("torso", _memory(unsafe_entry))
        assert unsafe.reveal_contract.decisions[0].decision_type != RevealDecisionType.REVEAL_FROM_WEAK_MEMORY.value
        assert unsafe.routing_candidates == ()

    missing = _handoff("torso", _memory())
    assert missing.reveal_contract.decisions[0].decision_type == RevealDecisionType.REVEAL_UNKNOWN_DEFER.value
    assert missing.reveal_contract.decisions[0].requires_rendering_candidate is False


def test_identity_regions_are_locked_and_never_create_identity_memory_or_embeddings() -> None:
    for region in ("face", "head", "hair", "scalp"):
        handoff = _handoff(region, _memory())
        decision = handoff.reveal_contract.decisions[0]
        assert decision.identity_locked is True
        assert decision.allowed_to_modify_identity is False
        assert decision.decision_type == RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK.value
        assert handoff.reveal_contract.identity_memory_created is False
        assert handoff.reveal_contract.identity_embedding_created is False


def test_private_optional_regions_are_blocked_without_routing_candidates() -> None:
    for region in ("external_genital_region", "male_external_genital_region", "male_pelvic_region", "female_pelvic_region"):
        handoff = _handoff(region, _memory(_memory_entry(region)), private=True)
        decision = handoff.reveal_contract.decisions[0]
        assert decision.decision_type == RevealDecisionType.REVEAL_BLOCKED_PRIVATE.value
        assert decision.reveal_allowed is False
        assert decision.requires_rendering_candidate is False
        assert handoff.reveal_contract.routing_candidates == ()


def test_routing_handoff_candidates_are_metadata_only_and_include_provenance() -> None:
    handoff = _handoff("torso", _memory(_memory_entry("torso")))
    candidate = handoff.routing_candidates[0]
    assert candidate.region_id == "p1:torso"
    assert candidate.reveal_decision_type == RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value
    assert candidate.action_type == "head_turn"
    assert candidate.phase_id == "action_0:phase_0:prepare"
    assert candidate.source_delta_type == "visibility_delta"
    assert candidate.routing_decision_made is False
    assert candidate.render_strategy_selected is False
    assert "render_mode" not in candidate.__dataclass_fields__
    assert handoff.reveal_contract.region_routing_called is False


def test_occlusion_reasoning_and_newly_occluded_semantics_are_distinct() -> None:
    occlusion_reasoning = _handoff("torso", _memory(_memory_entry("torso")), delta_type="occlusion_delta")
    decision = occlusion_reasoning.reveal_contract.decisions[0]
    assert decision.decision_type == RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value
    assert decision.reveal_allowed is False
    assert decision.requires_rendering_candidate is False
    assert occlusion_reasoning.reveal_contract.routing_candidates == ()

    pose_reasoning = _handoff("torso", _memory(), delta_type="pose_delta", reveal=False, occlusion_reasoning_required=True)
    assert pose_reasoning.reveal_contract.decisions[0].decision_type == RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value
    assert pose_reasoning.reveal_contract.decisions[0].decision_type != RevealDecisionType.NEWLY_OCCLUDED.value

    visibility_reasoning = _handoff("torso", _memory(_memory_entry("torso")), delta_type="visibility_delta", occlusion_reasoning_required=True)
    assert visibility_reasoning.reveal_contract.decisions[0].decision_type == RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value
    assert visibility_reasoning.reveal_contract.decisions[0].reveal_allowed is False

    newly_occluded = _handoff("torso", _memory(_memory_entry("torso")), delta_type="occlusion_delta", validation_reasons=("newly_occluded",))
    newly_decision = newly_occluded.reveal_contract.decisions[0]
    assert newly_decision.decision_type == RevealDecisionType.NEWLY_OCCLUDED.value
    assert newly_decision.lifecycle_state == OcclusionLifecycleState.NEWLY_OCCLUDED.value
    assert newly_decision.reveal_allowed is False
    assert newly_decision.requires_rendering_candidate is False
    assert newly_occluded.reveal_contract.reveal_candidates == ()
    assert newly_occluded.reveal_contract.routing_candidates == ()

    stable = _handoff("torso", _memory(), delta_type="pose_delta", reveal=False)
    assert stable.reveal_contract.decisions[0].decision_type == RevealDecisionType.PRESERVE_VISIBLE.value
    assert stable.reveal_contract.decisions[0].lifecycle_state == OcclusionLifecycleState.VISIBLE_STABLE.value


def test_real_turn_head_dynamics_does_not_mark_reasoning_as_newly_occluded() -> None:
    handoff = build_reveal_handoff(scene_graph=_scene("p1"), memory=None, graph_delta_contract=build_dynamics_handoff(_plan("turn head")).graph_delta_contract)
    reasoning = [decision for decision in handoff.reveal_contract.decisions if decision.source_delta_type == "occlusion_delta" or "occlusion_reasoning" in " ".join(decision.policy_reasons)]
    assert reasoning
    assert all(decision.decision_type != RevealDecisionType.NEWLY_OCCLUDED.value for decision in reasoning)
    assert all(decision.reveal_allowed is False for decision in reasoning if decision.decision_type == RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value)


def test_routing_candidates_for_reveal_only_not_occlusion_or_unknown() -> None:
    observed = _handoff("torso", _memory(_memory_entry("torso")))
    assert observed.reveal_contract.decisions[0].decision_type == RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value
    assert observed.routing_candidates

    unknown = _handoff("torso", _memory())
    assert unknown.reveal_contract.decisions[0].decision_type == RevealDecisionType.REVEAL_UNKNOWN_DEFER.value
    assert unknown.routing_candidates == ()

    occluded = _handoff("torso", _memory(_memory_entry("torso")), delta_type="occlusion_delta", validation_reasons=("newly_occluded",))
    assert occluded.reveal_contract.decisions[0].decision_type == RevealDecisionType.NEWLY_OCCLUDED.value
    assert occluded.routing_candidates == ()


def test_strict_validation_fails_loudly_for_reveal_contract_violations() -> None:
    good = _handoff("torso", _memory(_memory_entry("torso"))).reveal_contract.decisions[0]

    with pytest.raises(RevealValidationError, match="unknown_region"):
        validate_reveal_decision(replace(good, canonical_region_id="not_a_region", region_id="p1:not_a_region"))
    with pytest.raises(RevealValidationError, match="private_reveal_forbidden"):
        validate_reveal_decision(replace(good, canonical_region_id="external_genital_region", region_id="p1:external_genital_region", private_or_optional_region=True))
    with pytest.raises(RevealValidationError, match="observed_reveal_without_safe_evidence"):
        validate_reveal_decision(replace(good, memory_evidence=RevealMemoryEvidence()))
    with pytest.raises(RevealValidationError, match="unknown_defer_claims_known_content"):
        validate_reveal_decision(replace(good, decision_type=RevealDecisionType.REVEAL_UNKNOWN_DEFER, requires_rendering_candidate=True))
    with pytest.raises(RevealValidationError, match="forbidden_reveal_operation"):
        validate_reveal_decision(replace(good, rendered_pixels_generated=True))
    with pytest.raises(RevealValidationError, match="forbidden_reveal_operation"):
        validate_reveal_decision(replace(good, observed_evidence_created=True))
    with pytest.raises(RevealValidationError, match="forbidden_reveal_operation"):
        validate_reveal_decision(replace(good, memory_write_performed=True))
    with pytest.raises(RevealValidationError, match="forbidden_reveal_operation"):
        validate_reveal_decision(replace(good, mask_created=True))
    with pytest.raises(RevealValidationError, match="identity_modification_forbidden"):
        validate_reveal_decision(replace(good, identity_locked=True, allowed_to_modify_identity=True))
    with pytest.raises(RevealValidationError, match="unsafe_material_marked_observed"):
        validate_reveal_decision(replace(good, memory_evidence=replace(good.memory_evidence, material_provenance="generated", observed_directly=True)))
    with pytest.raises(RevealValidationError, match="routing_decision_forbidden"):
        validate_reveal_decision(replace(good, routing_decision_made=True))

    weak_good = _handoff("torso", _memory(_memory_entry("torso", confidence=0.5, evidence_score=0.5))).reveal_contract.decisions[0]
    validate_reveal_decision(weak_good)
    with pytest.raises(RevealValidationError, match="weak_reveal_without_reusable_reference"):
        validate_reveal_decision(replace(weak_good, memory_evidence=RevealMemoryEvidence(memory_support_level="weak", supports_weak_reveal=True)))
    with pytest.raises(RevealValidationError, match="weak_reveal_without_reusable_reference"):
        validate_reveal_decision(replace(weak_good, memory_evidence=replace(weak_good.memory_evidence, reference_kind="none")))
    with pytest.raises(RevealValidationError, match="weak_reveal_unsafe_material"):
        validate_reveal_decision(replace(weak_good, memory_evidence=replace(weak_good.memory_evidence, material_provenance="generated", observed_directly=False)))

    unsupported_with_candidates = RevealContract(supported=False, decisions=(good,))
    with pytest.raises(RevealValidationError, match="unsupported_contract_has_candidates"):
        validate_reveal_contract(unsupported_with_candidates)
