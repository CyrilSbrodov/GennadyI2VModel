from __future__ import annotations

from core.schema import BBox, BodyPartNode, GarmentNode, GraphDelta, PersonNode, RegionRef, SceneGraph
from dynamics.state_update import apply_delta


def _canonical_payload(name: str, visibility: str = "visible") -> dict[str, object]:
    return {
        "canonical_name": name,
        "raw_sources": [name],
        "source_regions": [name],
        "mask_ref": None,
        "confidence": 0.7,
        "visibility_state": visibility,
        "provenance": "canonical_reasoner",
        "attachment_hints": [],
        "ownership_hints": ["person"],
        "coverage_hints": [],
    }


def _scene_with_person(person_id: str = "p1") -> SceneGraph:
    person = PersonNode(
        person_id=person_id,
        track_id=f"t_{person_id}",
        bbox=BBox(0.1, 0.1, 0.6, 0.8),
        mask_ref="mask://person",
        body_parts=[BodyPartNode(part_id=f"{person_id}_left_arm", part_type="left_arm", visibility="visible", confidence=0.8)],
        garments=[GarmentNode(garment_id="coat_1", garment_type="coat", visibility="visible", confidence=0.9)],
        canonical_regions={"torso": _canonical_payload("torso"), "outer_garment": _canonical_payload("outer_garment", visibility="partially_visible")},
        confidence=0.95,
    )
    return SceneGraph(frame_index=0, persons=[person], objects=[])


def test_apply_delta_updates_canonical_region_visibility_from_predicted_visibility_changes() -> None:
    graph = _scene_with_person()
    delta = GraphDelta(
        affected_entities=["p1"],
        predicted_visibility_changes={"torso": "partially_visible"},
        region_transition_mode={"torso": "garment_reveal"},
    )

    updated = apply_delta(graph, delta)
    torso = updated.persons[0].canonical_regions["torso"]

    assert torso["visibility_state"] == "partially_visible"
    assert torso["lifecycle_state"] == "visibility_changed"
    assert torso["last_transition_mode"] == "garment_reveal"
    assert torso["last_update_source"] == "graph_delta.predicted_visibility_changes"


def test_apply_delta_marks_newly_revealed_region_lifecycle() -> None:
    graph = _scene_with_person()
    delta = GraphDelta(
        affected_entities=["p1"],
        newly_revealed_regions=[RegionRef(region_id="p1:inner_garment", bbox=BBox(0.0, 0.0, 0.2, 0.2), reason="reveal")],
    )

    updated = apply_delta(graph, delta)
    region = updated.persons[0].canonical_regions["inner_garment"]

    assert region["visibility_state"] in {"visible", "partially_visible"}
    assert region["lifecycle_state"] == "newly_revealed"
    assert region["last_update_source"] == "graph_delta.newly_revealed_regions"


def test_apply_delta_marks_newly_occluded_region_lifecycle() -> None:
    graph = _scene_with_person()
    delta = GraphDelta(
        affected_entities=["p1"],
        newly_occluded_regions=[RegionRef(region_id="p1:outer_garment", bbox=BBox(0.0, 0.0, 0.2, 0.2), reason="occlusion")],
    )

    updated = apply_delta(graph, delta)
    region = updated.persons[0].canonical_regions["outer_garment"]

    assert region["visibility_state"] in {"hidden", "occluded"}
    assert region["lifecycle_state"] == "newly_occluded"
    assert region["last_update_source"] == "graph_delta.newly_occluded_regions"


def test_apply_delta_updates_body_part_visibility_without_fuzzy_matching() -> None:
    graph = _scene_with_person()
    delta = GraphDelta(affected_entities=["p1"], visibility_deltas={"left_arm": "hidden"})

    updated = apply_delta(graph, delta)

    left_arm = next(part for part in updated.persons[0].body_parts if part.part_type == "left_arm")
    assert left_arm.visibility == "hidden"


def test_apply_delta_updates_garment_visibility_by_type_and_id() -> None:
    graph = _scene_with_person()

    updated_by_id = apply_delta(graph, GraphDelta(affected_entities=["p1"], visibility_deltas={"coat_1": "hidden"}))
    assert updated_by_id.persons[0].garments[0].visibility == "hidden"

    updated_by_type = apply_delta(graph, GraphDelta(affected_entities=["p1"], visibility_deltas={"coat": "partially_visible"}))
    assert updated_by_type.persons[0].garments[0].visibility == "partially_visible"


def test_visibility_delta_garment_type_does_not_create_canonical_region() -> None:
    graph = _scene_with_person()

    updated = apply_delta(graph, GraphDelta(affected_entities=["p1"], visibility_deltas={"coat": "hidden"}))

    assert updated.persons[0].garments[0].visibility == "hidden"
    assert "coat" not in updated.persons[0].canonical_regions


def test_visibility_delta_garment_id_does_not_create_canonical_region() -> None:
    graph = _scene_with_person()

    updated = apply_delta(graph, GraphDelta(affected_entities=["p1"], visibility_deltas={"coat_1": "hidden"}))

    assert updated.persons[0].garments[0].visibility == "hidden"
    assert "coat_1" not in updated.persons[0].canonical_regions


def test_visibility_delta_canonical_region_updates_canonical_payload() -> None:
    graph = _scene_with_person()

    updated = apply_delta(graph, GraphDelta(affected_entities=["p1"], visibility_deltas={"outer_garment": "hidden"}))

    assert updated.persons[0].canonical_regions["outer_garment"]["visibility_state"] == "hidden"
    assert updated.persons[0].canonical_regions["outer_garment"]["last_update_source"] == "graph_delta.visibility_deltas"


def test_apply_delta_does_not_create_person_when_affected_entity_missing() -> None:
    graph = _scene_with_person(person_id="p1")
    graph.persons[0].expression_state.smile_intensity = 0.1
    delta = GraphDelta(affected_entities=["p2"], expression_deltas={"smile_intensity": 0.4})

    updated = apply_delta(graph, delta)

    assert len(updated.persons) == 1
    assert updated.persons[0].person_id == "p1"
    assert updated.persons[0].expression_state.smile_intensity == 0.1


def test_apply_delta_preserves_existing_expression_pose_garment_behavior() -> None:
    graph = _scene_with_person()
    graph.persons[0].expression_state.smile_intensity = 0.2
    delta = GraphDelta(
        expression_deltas={"smile_intensity": 0.3, "mouth_state": "grin"},
        pose_deltas={"torso_pitch": 0.2},
        interaction_deltas={"chair_contact": 0.8},
        garment_deltas={"coat_state": "opening"},
        visibility_deltas={"coat": "partially_visible"},
    )

    updated = apply_delta(graph, delta)
    person = updated.persons[0]

    assert person.expression_state.smile_intensity == 0.5
    assert person.expression_state.label == "grin"
    assert person.pose_state.coarse_pose == "seated"
    assert person.garments[0].garment_state == "opening"
    assert person.garments[0].visibility == "partially_visible"
