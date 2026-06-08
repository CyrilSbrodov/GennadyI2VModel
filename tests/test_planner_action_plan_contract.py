from __future__ import annotations

from dataclasses import replace

import pytest

from core.schema import BBox, PersonNode, SceneGraph
from planning.action_plan import (
    ActionPlanner,
    ActionPlan,
    ActionType,
    PlannerIntent,
    PlannerValidationError,
    RegionPlanTarget,
    validate_action_plan,
)


def _scene(*person_ids: str) -> SceneGraph:
    return SceneGraph(
        frame_index=0,
        persons=[PersonNode(person_id=pid, track_id=None, bbox=BBox(0.1, 0.1, 0.6, 0.8), mask_ref=None) for pid in person_ids],
    )


def _regions(plan: ActionPlan) -> set[str]:
    return {target.canonical_region_id or target.region_id.rsplit(":", 1)[-1] for action in plan.actions for target in action.region_targets}


def _memory(plan: ActionPlan) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for action in plan.actions:
        for req in action.memory_requirements:
            out.setdefault(req.family, set()).update(req.regions)
    return out


def _action_values(plan: ActionPlan) -> list[ActionType]:
    return [ActionType(action.action_type) for action in plan.actions]


def test_every_supported_action_has_ordered_non_empty_phases_and_trace() -> None:
    planner = ActionPlanner()
    for action_type in ActionType:
        plan = planner.plan(PlannerIntent(action_type=action_type, target_entity_id="person_a"), _scene("person_a"))
        assert plan.supported is True
        assert plan.trace.planner_source == "controlled_phrase_contract"
        action = plan.actions[0]
        assert action.action_type == action_type
        assert action.phases
        previous_end = 0.0
        for phase in action.phases:
            assert 0.0 <= phase.normalized_start < phase.normalized_end <= 1.0
            assert phase.normalized_start >= previous_end
            assert phase.affected_regions
            assert phase.planner_source == "controlled_phrase_contract"
            previous_end = phase.normalized_end


def test_region_targeting_uses_canonical_body_ontology_and_no_private_auto_targets() -> None:
    planner = ActionPlanner()
    assert {"head", "face", "hair", "scalp", "neck"}.issubset(_regions(planner.plan("turn head", _scene("p1"))))
    assert {"face", "left_eye", "right_eye"}.issubset(_regions(planner.plan("look left", _scene("p1"))))
    assert {"left_arm", "left_upper_arm", "left_elbow", "left_forearm", "left_wrist", "left_hand"}.issubset(
        _regions(planner.plan("raise left arm", _scene("p1")))
    )
    assert {"pelvis", "hips", "left_thigh", "right_thigh", "left_knee", "right_knee", "left_foot", "right_foot", "torso"}.issubset(
        _regions(planner.plan("sit down", _scene("p1")))
    )
    assert {"face", "mouth", "lips", "jaw", "left_eye", "right_eye"}.issubset(_regions(planner.plan("smile", _scene("p1"))))
    for prompt in ("turn head", "look left", "raise left arm", "sit down", "stand up", "smile", "adjust garment"):
        assert not ({"female_pelvic_region", "male_pelvic_region", "external_genital_region", "breast_region"} & _regions(planner.plan(prompt, _scene("p1"))))
    sit_plan = planner.plan("sit down", _scene("p1"))
    assert sit_plan.actions[0].dynamics_requirement.secondary_motion_required is True
    assert sit_plan.actions[0].dynamics_requirement.produces_graph_delta is False


def test_limited_intent_parsing_supported_and_unsupported_cases() -> None:
    planner = ActionPlanner()
    assert planner.plan("turn head", _scene("p1")).actions[0].action_type == ActionType.HEAD_TURN
    assert planner.plan("look left", _scene("p1")).actions[0].action_type == ActionType.GAZE_SHIFT
    left_arm = planner.plan("raise left arm", _scene("p1")).actions[0]
    assert left_arm.action_type == ActionType.ARM_RAISE
    assert left_arm.side == "left"
    assert planner.plan("sit down", _scene("p1")).actions[0].action_type == ActionType.SIT_DOWN
    assert planner.plan("stand up", _scene("p1")).actions[0].action_type == ActionType.STAND_UP
    with pytest.raises(PlannerValidationError, match="unsupported_action"):
        planner.plan("perform an elegant pirouette", _scene("p1"))
    with pytest.raises(PlannerValidationError, match="unsupported_action"):
        planner.plan("make some random motion", _scene("p1"))
    unsupported = planner.plan(PlannerIntent(raw_text="perform an elegant pirouette", strict=False), _scene("p1"))
    assert unsupported.supported is False
    assert unsupported.unsupported_code == "unsupported_action"
    random_motion = planner.plan(PlannerIntent(raw_text="make some random motion", strict=False), _scene("p1"))
    assert random_motion.supported is False
    assert random_motion.actions == ()


def test_multi_action_decomposition_preserves_text_order() -> None:
    planner = ActionPlanner()
    sit_smile = planner.plan("sit down and smile", _scene("p1"))
    assert _action_values(sit_smile) == [ActionType.SIT_DOWN, ActionType.EXPRESSION_CHANGE]
    assert [a.trace.reasons[1] for a in sit_smile.actions] == ["action_order:0", "action_order:1"]
    assert all(action.phases for action in sit_smile.actions)

    turn_sit = planner.plan("turn head and sit down", _scene("p1"))
    assert _action_values(turn_sit) == [ActionType.HEAD_TURN, ActionType.SIT_DOWN]

    raise_smile = planner.plan("raise left arm and smile", _scene("p1"))
    assert _action_values(raise_smile) == [ActionType.ARM_RAISE, ActionType.EXPRESSION_CHANGE]
    assert raise_smile.actions[0].side == "left"

    russian = planner.plan("поворачивает голову и улыбается", _scene("p1"))
    assert _action_values(russian) == [ActionType.HEAD_TURN, ActionType.EXPRESSION_CHANGE]


def test_partial_unsupported_action_fragments_are_reported() -> None:
    planner = ActionPlanner()
    with pytest.raises(PlannerValidationError) as exc_info:
        planner.plan("Снимает пальто и садится. Улыбается", _scene("p1"))
    assert exc_info.value.code == "partial_unsupported_action"

    plan = planner.plan(PlannerIntent(raw_text="Снимает пальто и садится. Улыбается", strict=False), _scene("p1"))
    assert plan.supported is True
    assert _action_values(plan) == [ActionType.SIT_DOWN, ActionType.EXPRESSION_CHANGE]
    assert any("снимает пальто" in fragment for fragment in plan.unsupported_fragments)
    assert any(reason.startswith("unsupported_intent_fragment:снимает пальто") for reason in plan.trace.reasons)

    clean = planner.plan(PlannerIntent(raw_text="Садится и улыбается", strict=False), _scene("p1"))
    assert clean.supported is True
    assert _action_values(clean) == [ActionType.SIT_DOWN, ActionType.EXPRESSION_CHANGE]
    assert clean.unsupported_fragments == ()
    assert not any(reason.startswith("unsupported_intent_fragment:") for reason in clean.trace.reasons)


def test_entity_resolution_single_multi_explicit_and_missing() -> None:
    planner = ActionPlanner()
    assert planner.plan("turn head", _scene("only_person")).actions[0].target_entity_id == "only_person"
    with pytest.raises(PlannerValidationError, match="ambiguous_target"):
        planner.plan("turn head", _scene("p1", "p2"))
    explicit = planner.plan(PlannerIntent(raw_text="turn head", target_entity_id="p2"), _scene("p1", "p2"))
    assert explicit.actions[0].target_entity_id == "p2"
    with pytest.raises(PlannerValidationError, match="missing_target"):
        planner.plan(PlannerIntent(raw_text="turn head", target_entity_id="p3"), _scene("p1", "p2"))


def test_plan_validation_fails_loudly_for_contract_violations() -> None:
    planner = ActionPlanner()
    plan = planner.plan("turn head", _scene("p1"))
    action = plan.actions[0]
    with pytest.raises(PlannerValidationError, match="unknown_action"):
        validate_action_plan(replace(plan, actions=(replace(action, action_type="generic_motion"),)))
    bad_region = RegionPlanTarget(region_id="p1:not_a_region", canonical_region_id="not_a_region", entity_id="p1", role="primary")
    with pytest.raises(PlannerValidationError, match="unknown_region"):
        validate_action_plan(replace(plan, actions=(replace(action, region_targets=(bad_region,)),)))
    private_region = RegionPlanTarget(region_id="p1:external_genital_region", canonical_region_id="external_genital_region", entity_id="p1", role="primary")
    with pytest.raises(PlannerValidationError, match="private_auto_target_forbidden"):
        validate_action_plan(replace(plan, actions=(replace(action, region_targets=(private_region,)),)))
    with pytest.raises(PlannerValidationError, match="empty_phases"):
        validate_action_plan(replace(plan, actions=(replace(action, phases=()),)))
    bad_phase = replace(action.phases[1], normalized_start=0.0, normalized_end=0.2)
    with pytest.raises(PlannerValidationError, match="non_monotonic_phase_timing"):
        validate_action_plan(replace(plan, actions=(replace(action, phases=(action.phases[0], bad_phase, action.phases[2])),)))
    with pytest.raises(PlannerValidationError, match="forbidden_planner_output"):
        validate_action_plan(replace(plan, forbidden_outputs={"render_mode": "direct"}))
    with pytest.raises(PlannerValidationError, match="forbidden_planner_output"):
        validate_action_plan(replace(plan, forbidden_outputs={"observed_region_created": True}))


def test_memory_requirements_and_dynamics_handoff_do_not_generate_deltas_or_private_memory() -> None:
    planner = ActionPlanner()
    head = planner.plan("turn head", _scene("p1"))
    assert {"face", "head", "hair", "scalp"}.issubset(_memory(head)["identity"])
    assert set(head.actions[0].dynamics_requirement.identity_lock_regions) >= {"face", "head", "hair", "scalp"}

    garment = planner.plan("adjust garment", _scene("p1"))
    assert "garment" in _memory(garment)
    assert garment.actions[0].dynamics_requirement.reveal_may_be_required is True
    assert garment.actions[0].dynamics_requirement.produces_graph_delta is False

    sit = planner.plan("sit down", _scene("p1"))
    assert "body_shape" in _memory(sit)
    assert "soft_tissue" in _memory(sit)
    assert sit.actions[0].dynamics_requirement.secondary_motion_required is True
    assert all(req.family != "private" for action in sit.actions for req in action.memory_requirements)
    assert sit.scene_graph_mutation_performed is False
    assert sit.actions[0].dynamics_requirement.expected_graph_delta_types
    assert sit.actions[0].dynamics_requirement.required_dynamics_capabilities
