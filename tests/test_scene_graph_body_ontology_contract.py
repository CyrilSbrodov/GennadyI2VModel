from __future__ import annotations

from core.body_ontology import CANONICAL_BODY_REGION_ORDER
from core.schema import BBox, ExpressionState, GraphDelta, OrientationState, PoseState, RuntimeSemanticTransition, VideoMemory
from memory.video_memory import MemoryManager
from perception.pipeline import PerceptionOutput, PersonFacts
from rendering.roi_renderer import ROISelector
from representation.graph_builder import SceneGraphBuilder
from runtime.region_routing import CanonicalRegionRouter


def _person() -> PersonFacts:
    return PersonFacts(
        bbox=BBox(0.1, 0.1, 0.7, 0.8),
        bbox_confidence=0.9,
        bbox_source="detector:test",
        mask_ref=None,
        mask_confidence=0.0,
        mask_source="unknown",
        pose=PoseState(),
        pose_confidence=0.0,
        pose_source="unknown",
        expression=ExpressionState(),
        expression_confidence=0.0,
        expression_source="unknown",
        orientation=OrientationState(),
        orientation_confidence=0.0,
        orientation_source="unknown",
        person_id="person_1",
        face_regions=[{"region_type": "hair", "mask_ref": "mask://hair", "confidence": 0.88, "source": "parser:test", "observation_status": "observed", "mask_evidence_type": "parser_mask"}],
        body_parts=[{"part_type": "left_hand", "mask_ref": "mask://left_hand", "confidence": 0.86, "source": "parser:test", "visibility": "visible", "observation_status": "observed", "mask_evidence_type": "parser_mask"}],
    )


def test_scene_graph_body_part_nodes_are_ontology_derived_and_preserve_unknown_optional_regions() -> None:
    graph = SceneGraphBuilder().build(PerceptionOutput(persons=[_person()], frame_size=(640, 480)))
    person = graph.persons[0]
    part_types = [part.part_type for part in person.body_parts]
    assert part_types == list(CANONICAL_BODY_REGION_ORDER)
    assert len(part_types) > 60
    by_type = {part.part_type: part for part in person.body_parts}
    assert by_type["hair"].observation_status == "observed"
    assert by_type["left_hand"].observation_status == "observed"
    assert by_type["left_breast"].observation_status == "unknown"
    assert by_type["left_breast"].applicability == "unknown_applicability"
    assert by_type["male_external_genital_region"].mask_ref is None
    assert by_type["male_external_genital_region"].observation_status == "unknown"
    assert person.canonical_regions["chest"]["canonical_name"] == "chest"
    assert person.canonical_regions["breast_region"]["parent_region"] == "chest"
    assert person.canonical_regions["groin_region"]["parent_region"] == "pelvis"


def test_router_addresses_new_regions_but_blocks_unknown_optional_observed_render_update() -> None:
    graph = SceneGraphBuilder().build(PerceptionOutput(persons=[_person()], frame_size=(640, 480)))
    delta = GraphDelta(affected_entities=["person_1"], affected_regions=["left_breast", "left_hand"], visibility_deltas={"left_breast": "visible", "left_hand": "visible"})
    plan = CanonicalRegionRouter(MemoryManager(), ROISelector()).build_plan(
        scene_graph=graph,
        delta=delta,
        memory=VideoMemory(),
        semantic_transition=RuntimeSemanticTransition(),
    )
    breast = plan.decision_for_region_id("person_1:left_breast")
    hand = plan.decision_for_region_id("person_1:left_hand")
    assert breast is not None
    assert breast.should_render is False
    assert "ontology_state_blocks_observed_render_update" in breast.reasons
    assert hand is not None
    assert hand.canonical_region == "left_hand"


def test_private_region_is_addressable_but_not_auto_rendered() -> None:
    graph = SceneGraphBuilder().build(PerceptionOutput(persons=[_person()], frame_size=(640, 480)))
    delta = GraphDelta(
        affected_entities=["person_1"],
        affected_regions=["male_external_genital_region"],
        visibility_deltas={"male_external_genital_region": "visible"},
    )
    plan = CanonicalRegionRouter(MemoryManager(), ROISelector()).build_plan(
        scene_graph=graph,
        delta=delta,
        memory=VideoMemory(),
        semantic_transition=RuntimeSemanticTransition(),
    )
    decision = plan.decision_for_region_id("person_1:male_external_genital_region")
    assert decision is not None
    assert decision.canonical_region == "male_external_genital_region"
    assert decision.should_render is False
    assert "private_region_rendering_not_enabled" in decision.reasons
