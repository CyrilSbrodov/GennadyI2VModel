from core.schema import BBox, GarmentNode, PersonNode, SceneGraph
from text.intent_parser import IntentParser


def test_parser_extracts_core_actions() -> None:
    parser = IntentParser()
    plan = parser.parse("Снимает пальто и садится на стул. Улыбается.")
    actions = [a.type for a in plan.actions]
    assert "remove_garment" in actions
    assert "sit_down" in actions
    assert "smile" in actions


def test_parser_extracts_sequence_parallel_and_modifiers() -> None:
    parser = IntentParser()
    plan = parser.parse("Сначала снимает пальто, потом медленно улыбается, одновременно поворачивает голову.")

    assert len(plan.actions) >= 2
    assert any(step.start_after for step in plan.actions[1:])
    assert plan.parallel_groups
    assert any(step.modifiers.get("speed") == "slow" for step in plan.actions)


def test_parser_validates_against_scene_graph() -> None:
    parser = IntentParser()
    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.5, 0.8), mask_ref=None)
    person.garments = [GarmentNode(garment_id="g1", garment_type="shirt")]
    scene = SceneGraph(frame_index=0, persons=[person], objects=[])

    plan = parser.parse("Снимает пальто и садится на стул", scene_graph=scene)

    assert "requires:chair" in plan.constraints
    assert "requires:outer_garment" in plan.constraints
