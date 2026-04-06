from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.pipeline import ObjectFacts, PerceptionOutput, PersonFacts
from representation.graph_builder import SceneGraphBuilder


def test_graph_builder_infers_relations_visibility_and_occlusion() -> None:
    builder = SceneGraphBuilder()
    perception = PerceptionOutput(
        persons=[
            PersonFacts(
                bbox=BBox(0.1, 0.1, 0.3, 0.6),
                bbox_confidence=0.9,
                bbox_source="yolo",
                mask_ref="mask://1",
                mask_confidence=0.8,
                mask_source="segformer",
                pose=PoseState(),
                pose_confidence=0.84,
                pose_source="vitpose",
                expression=ExpressionState(),
                expression_confidence=0.7,
                expression_source="emonet",
                orientation=OrientationState(),
                orientation_confidence=0.7,
                orientation_source="emonet",
                garments=[{"type": "coat", "confidence": 0.4, "source": "fallback"}],
            )
        ],
        objects=[ObjectFacts(object_type="chair", bbox=BBox(0.4, 0.4, 0.2, 0.4), confidence=0.9, source="yolo")],
        frame_size=(1280, 720),
    )

    graph = builder.build(perception, frame_index=3)

    assert graph.persons[0].source == "yolo"
    assert graph.persons[0].frame_index == 3
    assert graph.persons[0].garments[0].visibility == "hidden"
    relation_types = {edge.relation for edge in graph.relations}
    assert {"part_of", "attached_to", "covers", "near", "occludes"}.issubset(relation_types)
    assert all(0.0 <= edge.confidence <= 1.0 for edge in graph.relations)


def test_graph_builder_resolves_duplicate_relation_conflicts() -> None:
    builder = SceneGraphBuilder()
    perception = PerceptionOutput(
        persons=[
            PersonFacts(
                bbox=BBox(0.2, 0.1, 0.2, 0.5),
                bbox_confidence=0.95,
                bbox_source="yolo",
                mask_ref="mask://2",
                mask_confidence=0.9,
                mask_source="segformer",
                pose=PoseState(),
                pose_confidence=0.9,
                pose_source="vitpose",
                expression=ExpressionState(),
                expression_confidence=0.8,
                expression_source="emonet",
                orientation=OrientationState(),
                orientation_confidence=0.8,
                orientation_source="emonet",
                garments=[
                    {"type": "shirt", "confidence": 0.8, "source": "yolo"},
                    {"type": "shirt", "confidence": 0.2, "source": "fallback"},
                ],
            )
        ],
        frame_size=(640, 480),
    )

    graph = builder.build(perception, frame_index=1)

    attached = [edge for edge in graph.relations if edge.relation == "attached_to"]
    keys = {(edge.source, edge.target) for edge in attached}
    assert len(keys) == len(attached)
