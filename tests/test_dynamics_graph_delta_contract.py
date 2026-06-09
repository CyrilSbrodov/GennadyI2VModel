from __future__ import annotations

import json
from dataclasses import replace
import pytest

from core.schema import BBox, PersonNode, SceneGraph
from dynamics.graph_delta_contract import (
    DynamicsValidationError,
    GraphDeltaContract,
    RegionDelta,
    RegionDeltaType,
    SUPPORTED_REGION_DELTA_TYPES,
    build_dynamics_handoff,
    validate_graph_delta_contract,
    validate_region_delta,
)
from planning.action_plan import ActionPlanner, PlannerIntent


def _scene(*person_ids: str) -> SceneGraph:
    return SceneGraph(
        frame_index=0,
        persons=[PersonNode(person_id=pid, track_id=None, bbox=BBox(0.1, 0.1, 0.6, 0.8), mask_ref=None) for pid in person_ids],
    )


def _plan(text: str):
    return ActionPlanner().plan(PlannerIntent(raw_text=text, strict=False), _scene("p1"))


def _deltas(contract: GraphDeltaContract) -> list[RegionDelta]:
    return [delta for step in contract.steps for delta in step.region_deltas]


def _by_region_and_type(contract: GraphDeltaContract) -> set[tuple[str, str]]:
    return {(delta.canonical_region_id, str(delta.delta_type)) for delta in _deltas(contract)}


def test_dynamics_schema_exists_supported_delta_types_and_json_serialization() -> None:
    handoff = build_dynamics_handoff(_plan("turn head"))
    assert isinstance(handoff.graph_delta_contract, GraphDeltaContract)
    assert {
        "pose_delta",
        "expression_delta",
        "visibility_delta",
        "occlusion_delta",
        "interaction_delta",
        "garment_intent_delta",
        "secondary_motion_hint",
    }.issubset(SUPPORTED_REGION_DELTA_TYPES)
    delta = _deltas(handoff.graph_delta_contract)[0]
    assert delta.entity_id == "p1"
    assert delta.region_id.startswith("p1:")
    assert delta.phase_id
    assert delta.action_type == "head_turn"
    json.dumps(handoff.graph_delta_contract.as_dict())


def test_head_turn_expression_sit_and_garment_action_plan_consumption() -> None:
    head = build_dynamics_handoff(_plan("turn head")).graph_delta_contract
    head_types = _by_region_and_type(head)
    for delta_type in ("pose_delta", "visibility_delta", "occlusion_delta"):
        assert {(r, delta_type) for r in {"head", "face", "hair", "scalp", "neck"}}.issubset(head_types)
    assert all(delta.visibility_intent is not None for delta in _deltas(head) if delta.delta_type == "visibility_delta")
    assert all(delta.occlusion_intent is not None for delta in _deltas(head) if delta.delta_type == "occlusion_delta")

    expression = build_dynamics_handoff(_plan("smile")).graph_delta_contract
    expression_types = _by_region_and_type(expression)
    assert {(r, "expression_delta") for r in {"face", "mouth", "lips", "jaw", "left_eye", "right_eye"}}.issubset(expression_types)
    assert {delta_type for _, delta_type in expression_types} == {"expression_delta"}

    sit = build_dynamics_handoff(_plan("sit down")).graph_delta_contract
    sit_types = _by_region_and_type(sit)
    for delta_type in ("pose_delta", "visibility_delta", "occlusion_delta"):
        assert {(r, delta_type) for r in {"pelvis", "hips", "left_thigh", "right_thigh", "left_knee", "right_knee", "left_foot", "right_foot", "torso"}}.issubset(
            sit_types
        )
    assert ("hips", "secondary_motion_hint") in sit_types
    assert any(delta.occlusion_reasoning_required for delta in _deltas(sit))

    garment = build_dynamics_handoff(_plan("adjust garment")).graph_delta_contract
    garment_types = _by_region_and_type(garment)
    assert {(r, "garment_intent_delta") for r in {"upper_garment", "outer_garment", "sleeves"}}.issubset(garment_types)
    assert ("left_hand", "interaction_delta") in garment_types
    assert ("right_hand", "interaction_delta") in garment_types
    assert all(delta.delta_type in {"garment_intent_delta", "interaction_delta", "visibility_delta", "occlusion_delta"} for delta in _deltas(garment))


def test_multi_action_plan_preserves_ordered_delta_steps() -> None:
    contract = build_dynamics_handoff(_plan("turn head and sit down and smile")).graph_delta_contract
    action_orders = [step.action_order for step in contract.steps]
    assert action_orders == sorted(action_orders)
    assert [step.action_type for step in contract.steps if step.phase_order == 0] == ["head_turn", "sit_down", "expression_change"]
    assert any(delta.action_type == "head_turn" for delta in _deltas(contract))
    assert any(delta.action_type == "sit_down" for delta in _deltas(contract))
    assert any(delta.action_type == "expression_change" for delta in _deltas(contract))


def test_identity_locks_are_preserved_and_do_not_allow_identity_modification() -> None:
    head = build_dynamics_handoff(_plan("turn head")).graph_delta_contract
    locked = {delta.canonical_region_id: delta for delta in _deltas(head) if delta.identity_locked}
    assert {"face", "head", "hair", "scalp"}.issubset(locked)
    assert all(delta.allowed_to_modify_identity is False for delta in locked.values())
    assert any(delta.identity_locked and delta.delta_type == "pose_delta" for delta in _deltas(head))
    assert head.memory_write_performed is False
    assert head.observed_evidence_created is False

    expression = build_dynamics_handoff(_plan("smile")).graph_delta_contract
    face_deltas = [delta for delta in _deltas(expression) if delta.canonical_region_id == "face"]
    assert face_deltas
    assert all(delta.identity_locked and not delta.allowed_to_modify_identity for delta in face_deltas)
    assert any(delta.delta_type == "expression_delta" for delta in face_deltas)


def test_private_optional_regions_are_blocked_and_never_render_or_reveal_targets() -> None:
    good = build_dynamics_handoff(_plan("sit down")).graph_delta_contract
    assert all(not delta.private_or_optional_region for delta in _deltas(good))
    assert all(delta.canonical_region_id != "external_genital_region" for delta in _deltas(good))
    assert all(candidate.canonical_region_id != "external_genital_region" for candidate in good.routing_candidates)

    base = _deltas(good)[0]
    private_delta = replace(
        base,
        canonical_region_id="external_genital_region",
        region_id="p1:external_genital_region",
        private_or_optional_region=True,
        requires_rendering_candidate=True,
        reveal_may_be_required=True,
    )
    with pytest.raises(DynamicsValidationError, match="private_auto_target_forbidden"):
        validate_region_delta(private_delta)

    optional_delta = replace(base, canonical_region_id="breast_region", region_id="p1:breast_region")
    with pytest.raises(DynamicsValidationError, match="private_auto_target_forbidden"):
        validate_region_delta(optional_delta)


def test_garment_regions_validate_consistently_and_unknown_garment_like_region_fails() -> None:
    contract = build_dynamics_handoff(_plan("adjust garment")).graph_delta_contract
    garment_deltas = [delta for delta in _deltas(contract) if delta.delta_type == "garment_intent_delta"]
    assert {delta.canonical_region_id for delta in garment_deltas} >= {"upper_garment", "outer_garment", "sleeves"}
    for delta in garment_deltas:
        validate_region_delta(delta)

    unknown = replace(garment_deltas[0], canonical_region_id="cape_garment", region_id="p1:cape_garment")
    with pytest.raises(DynamicsValidationError, match="unknown_region"):
        validate_region_delta(unknown)


def test_unsupported_and_partial_unsupported_planner_input_trace() -> None:
    unsupported_plan = ActionPlanner().plan(PlannerIntent(raw_text="perform an elegant pirouette", strict=False), _scene("p1"))
    unsupported = build_dynamics_handoff(unsupported_plan)
    assert unsupported.supported is False
    assert unsupported.graph_delta_contract.supported is False
    assert unsupported.graph_delta_contract.steps == ()
    assert unsupported.trace.unsupported_planner_code == "unsupported_action"

    partial = build_dynamics_handoff(_plan("Снимает пальто и садится. Улыбается"))
    assert partial.supported is True
    assert any("снимает пальто" in fragment for fragment in partial.trace.unsupported_planner_fragments)
    assert any(delta.action_type == "sit_down" for delta in _deltas(partial.graph_delta_contract))
    assert any(delta.action_type == "expression_change" for delta in _deltas(partial.graph_delta_contract))


def test_strict_validation_fails_loudly_for_contract_violations() -> None:
    contract = build_dynamics_handoff(_plan("turn head and smile")).graph_delta_contract
    first_step = contract.steps[0]
    first_delta = first_step.region_deltas[0]

    with pytest.raises(DynamicsValidationError, match="unknown_delta_type"):
        validate_region_delta(replace(first_delta, delta_type="not_a_delta"))
    with pytest.raises(DynamicsValidationError, match="unknown_region"):
        validate_region_delta(replace(first_delta, canonical_region_id="not_a_region", region_id="p1:not_a_region"))
    with pytest.raises(DynamicsValidationError, match="missing_phase_id"):
        validate_region_delta(replace(first_delta, phase_id=""))
    with pytest.raises(DynamicsValidationError, match="missing_action_type"):
        validate_region_delta(replace(first_delta, action_type=""))
    with pytest.raises(DynamicsValidationError, match="pixel_generation_forbidden"):
        validate_region_delta(replace(first_delta, rendered_pixels_generated=True))
    with pytest.raises(DynamicsValidationError, match="observed_evidence_forbidden"):
        validate_region_delta(replace(first_delta, observed_evidence_created=True))
    with pytest.raises(DynamicsValidationError, match="memory_write_forbidden"):
        validate_region_delta(replace(first_delta, memory_write_performed=True))
    with pytest.raises(DynamicsValidationError, match="region_routing_call_forbidden"):
        validate_region_delta(replace(first_delta, region_routing_called=True))
    with pytest.raises(DynamicsValidationError, match="identity_modification_forbidden"):
        validate_region_delta(replace(first_delta, identity_locked=True, allowed_to_modify_identity=True))

    empty_step = replace(first_step, region_deltas=())
    with pytest.raises(DynamicsValidationError, match="empty_region_deltas"):
        validate_graph_delta_contract(replace(contract, steps=(empty_step,)))

    non_monotonic_step = replace(contract.steps[1], action_order=-1)
    non_monotonic = replace(contract, steps=(contract.steps[0], non_monotonic_step, *contract.steps[2:]))
    with pytest.raises(DynamicsValidationError, match="non_monotonic_action_order"):
        validate_graph_delta_contract(non_monotonic)

    with pytest.raises(DynamicsValidationError, match="forbidden_dynamics_claim"):
        validate_graph_delta_contract(replace(contract, rendered_pixels_generated=True))
