from core.schema import BBox, GarmentNode, PersonNode, SceneGraph
from text.intent_parser import IntentParser


def test_parser_extracts_core_actions() -> None:
    parser = IntentParser()
    plan = parser.parse("Снимает пальто и садится на стул. Улыбается.")
    actions = [a.type for a in plan.actions]
    assert "remove_garment" in actions
    assert "sit_down" in actions
    assert "smile" in actions
    assert any(step.semantic_transition and step.semantic_transition.family == "pose_transition" for step in plan.actions)
    assert any(step.semantic_transition and step.semantic_transition.goal == "seated_pose" for step in plan.actions)
    assert any(step.semantic_transition and step.semantic_transition.lexical_bootstrap_score <= 1.0 for step in plan.actions)


def test_parser_synonyms_and_word_order() -> None:
    parser = IntentParser()
    plan = parser.parse("На стул садится и потом улыбнувшись поворачивает голову")
    actions = [a.type for a in plan.actions]
    assert "sit_down" in actions
    assert "smile" in actions
    assert "turn_head" in actions
    head_step = next(step for step in plan.actions if step.type == "turn_head")
    assert head_step.semantic_transition is not None
    assert "face" in head_step.semantic_transition.target_profile.primary_regions


def test_parser_multiple_actions_and_modifiers() -> None:
    parser = IntentParser()
    plan = parser.parse("Сначала снимает пальто, затем медленно и слегка улыбается, одновременно поднимает руку")

    assert len(plan.actions) >= 3
    assert any(step.start_after for step in plan.actions[1:])
    assert plan.parallel_groups
    assert any(step.modifiers.get("speed") == "slow" for step in plan.actions)
    assert any(float(step.modifiers.get("parser_confidence", 0.0)) > 0 for step in plan.actions)
    assert any(step.semantic_transition and step.semantic_transition.modifiers["speed"] == "slow" for step in plan.actions)


def test_parser_missing_target_handling() -> None:
    parser = IntentParser()
    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.5, 0.8), mask_ref=None)
    person.garments = [GarmentNode(garment_id="g1", garment_type="shirt")]
    scene = SceneGraph(frame_index=0, persons=[person], objects=[])

    plan = parser.parse("Снимает пальто и садится на стул", scene_graph=scene)

    assert "requires:chair" in plan.constraints
    assert "requires:outer_garment" in plan.constraints


def test_parser_emits_phase_aware_semantic_transition_contract() -> None:
    parser = IntentParser()
    plan = parser.parse("Садится на стул и снимает куртку")

    sit_step = next(step for step in plan.actions if step.type == "sit_down")
    garment_step = next(step for step in plan.actions if step.type == "remove_garment")
    assert sit_step.semantic_transition is not None
    assert garment_step.semantic_transition is not None
    assert "legs" in sit_step.semantic_transition.target_profile.primary_regions
    assert "support_zone" in sit_step.semantic_transition.target_profile.context_regions
    assert sit_step.semantic_transition.phase.global_phase in {"prepare", "transition", "contact_or_reveal", "stabilize"}
    assert sit_step.semantic_transition.phase.pose_subphase in {"weight_shift", "lowering", "contact_settle", "seated_stabilization"}
    assert sit_step.semantic_transition.phase.global_phase != sit_step.semantic_transition.phase.pose_subphase
    assert garment_step.semantic_transition.family == "garment_transition"
    assert garment_step.semantic_transition.goal == "outer_layer_removal"


def test_single_canonical_semantic_contract_is_used_in_parser_output() -> None:
    parser = IntentParser()
    plan = parser.parse("Улыбается")
    step = plan.actions[0]
    assert step.semantic_transition is not None
    assert "semantic_transition" in step.modifiers  # compatibility bridge
    assert step.semantic_transition.family == "expression_transition"
