from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.mask_store import DEFAULT_MASK_STORE
from perception.pipeline import ObjectFacts, PerceptionOutput, PersonFacts
from representation.canonical_human_state import CanonicalHumanSceneProcessor
from representation.graph_builder import SceneGraphBuilder


def _mask(payload, prefix: str) -> str:
    return DEFAULT_MASK_STORE.put(payload=payload, confidence=0.9, source="test", prefix=prefix, frame_size=(8, 8))


def test_canonical_normalization_relations_and_visibility_reasoning() -> None:
    torso_mask = _mask([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]], "torso")
    coat_mask = _mask([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], "coat")

    person = PersonFacts(
        bbox=BBox(0.1, 0.1, 0.5, 0.8),
        bbox_confidence=0.9,
        bbox_source="yolo",
        mask_ref=None,
        mask_confidence=0.9,
        mask_source="segformer",
        pose=PoseState(),
        pose_confidence=0.8,
        pose_source="vitpose",
        expression=ExpressionState(),
        expression_confidence=0.8,
        expression_source="emonet",
        orientation=OrientationState(),
        orientation_confidence=0.8,
        orientation_source="emonet",
        garments=[{"type": "coat", "confidence": 0.82, "source": "segformer", "mask_ref": coat_mask}],
        body_parts=[{"part_type": "torso", "mask_ref": torso_mask, "confidence": 0.88, "visibility": "visible", "source": "segformer"}],
    )

    state = CanonicalHumanSceneProcessor().process(person=person, person_id="person_1", frame_size=(8, 8), objects=[])

    assert state.regions["torso"].canonical_name == "torso"
    assert state.regions["outer_garment"].mask_ref == coat_mask
    assert state.regions["torso"].visibility_state == "visible"
    assert state.regions["upper_body"].confidence > 0.2
    assert any(r.relation == "covers" and r.source in {"upper_garment", "outer_garment"} and r.target == "torso" for r in state.relations)
    assert any(r.relation == "part_of" and r.source == "torso" and r.target == "upper_body" for r in state.relations)


def test_visibility_reasoning_hidden_by_object_requires_local_evidence() -> None:
    person_no_local_support = PersonFacts(
        bbox=BBox(0.0, 0.0, 0.35, 0.5),
        bbox_confidence=0.9,
        bbox_source="yolo",
        mask_ref=None,
        mask_confidence=0.0,
        mask_source="fallback",
        pose=PoseState(),
        pose_confidence=0.7,
        pose_source="vitpose",
        expression=ExpressionState(),
        expression_confidence=0.7,
        expression_source="emonet",
        orientation=OrientationState(),
        orientation_confidence=0.7,
        orientation_source="emonet",
        occlusion_hints=["something_is_occluded"],
    )
    far_objects = [ObjectFacts(object_type="table", bbox=BBox(0.8, 0.8, 0.15, 0.15), confidence=0.95, source="yolo")]

    state_without_local_evidence = CanonicalHumanSceneProcessor().process(
        person=person_no_local_support,
        person_id="person_1",
        frame_size=(640, 480),
        objects=far_objects,
    )
    assert not any(r.visibility_state == "hidden_by_object" for r in state_without_local_evidence.regions.values())

    person = PersonFacts(
        bbox=BBox(0.0, 0.0, 0.5, 0.9),
        bbox_confidence=0.9,
        bbox_source="yolo",
        mask_ref=None,
        mask_confidence=0.0,
        mask_source="fallback",
        pose=PoseState(),
        pose_confidence=0.7,
        pose_source="vitpose",
        expression=ExpressionState(),
        expression_confidence=0.7,
        expression_source="emonet",
        orientation=OrientationState(),
        orientation_confidence=0.7,
        orientation_source="emonet",
        occlusion_hints=["torso occluded by table object"],
    )
    perception_objects = [ObjectFacts(object_type="table", bbox=BBox(0.2, 0.4, 0.5, 0.4), confidence=0.95, source="yolo")]

    state = CanonicalHumanSceneProcessor().process(person=person, person_id="person_1", frame_size=(640, 480), objects=perception_objects)
    assert state.regions["torso"].visibility_state == "hidden_by_object"
    assert any(rel.relation == "occludes" and rel.source.startswith("object:") and rel.target == "torso" for rel in state.relations)

    graph = SceneGraphBuilder().build(
        perception=PerceptionOutput(persons=[person], objects=perception_objects, frame_size=(640, 480)),
        frame_index=2,
    )

    assert graph.persons[0].canonical_regions
    assert graph.persons[0].region_relations
    assert any(edge.relation == "occludes" and edge.target.endswith("_torso") for edge in graph.relations)
