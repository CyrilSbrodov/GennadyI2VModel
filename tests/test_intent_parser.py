from core.schema import BBox, GarmentNode, PersonNode, SceneGraph
from text.intent_parser import IntentParser


def test_parser_extracts_core_actions() -> None:
    parser = IntentParser()
    plan = parser.parse("Снимает пальто и садится на стул. Улыбается.")
    actions = [a.type for a in plan.actions]
    assert "remove_garment" in actions
    assert "sit_down" in actions
    assert "smile" in actions


def test_parser_synonyms_and_word_order() -> None:
    parser = IntentParser()
    plan = parser.parse("На стул садится и потом улыбнувшись поворачивает голову")
    actions = [a.type for a in plan.actions]
    assert "sit_down" in actions
    assert "smile" in actions
    assert "turn_head" in actions


def test_parser_multiple_actions_and_modifiers() -> None:
    parser = IntentParser()
    plan = parser.parse("Сначала снимает пальто, затем медленно и слегка улыбается, одновременно поднимает руку")

    assert len(plan.actions) >= 3
    assert any(step.start_after for step in plan.actions[1:])
    assert plan.parallel_groups
    assert any(step.modifiers.get("speed") == "slow" for step in plan.actions)
    assert any(float(step.modifiers.get("parser_confidence", 0.0)) > 0 for step in plan.actions)


def test_parser_missing_target_handling() -> None:
    parser = IntentParser()
    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.5, 0.8), mask_ref=None)
    person.garments = [GarmentNode(garment_id="g1", garment_type="shirt")]
    scene = SceneGraph(frame_index=0, persons=[person], objects=[])

    plan = parser.parse("Снимает пальто и садится на стул", scene_graph=scene)

    assert "requires:chair" in plan.constraints
    assert "requires:outer_garment" in plan.constraints
