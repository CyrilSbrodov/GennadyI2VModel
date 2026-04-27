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
    assert "torso" in graph.persons[0].canonical_regions
    assert graph.persons[0].canonical_regions["torso"]["canonical_name"] == "torso"
    assert graph.persons[0].garments[0].visibility in {"partially_visible", "unknown_expected_region", "visible", "hidden_by_garment", "hidden_by_object"}
    relation_types = {edge.relation for edge in graph.relations}
    assert {"part_of", "attached_to", "covers", "near"}.issubset(relation_types)
    assert {"supports", "interacts_with"} & relation_types  # object relation from base graph path
    assert "overlaps" not in {edge.relation for edge in graph.relations if edge.provenance == "visibility_reasoner"}
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


def test_graph_builder_exports_overlap_without_interaction_aliasing() -> None:
    builder = SceneGraphBuilder()
    from perception.mask_store import DEFAULT_MASK_STORE

    torso_ref = DEFAULT_MASK_STORE.put([[0, 1, 1], [0, 1, 1], [0, 0, 0]], confidence=0.9, source="test", prefix="test", frame_size=(3, 3))
    coat_ref = DEFAULT_MASK_STORE.put([[0, 1, 1], [0, 1, 1], [0, 0, 1]], confidence=0.9, source="test", prefix="test", frame_size=(3, 3))

    perception = PerceptionOutput(
        persons=[
            PersonFacts(
                bbox=BBox(0.1, 0.1, 0.4, 0.7),
                bbox_confidence=0.9,
                bbox_source="yolo",
                mask_ref=None,
                mask_confidence=0.9,
                mask_source="segformer",
                pose=PoseState(),
                pose_confidence=0.8,
                pose_source="vitpose",
                expression=ExpressionState(),
                expression_confidence=0.7,
                expression_source="emonet",
                orientation=OrientationState(),
                orientation_confidence=0.7,
                orientation_source="emonet",
                garments=[{"type": "coat", "confidence": 0.9, "source": "segformer", "mask_ref": coat_ref}],
                body_parts=[{"part_type": "torso", "mask_ref": torso_ref, "confidence": 0.85, "visibility": "visible", "source": "segformer"}],
            )
        ],
        frame_size=(3, 3),
    )

    graph = builder.build(perception, frame_index=4)
    overlap_edges = [edge for edge in graph.relations if edge.relation == "overlaps"]
    assert overlap_edges
    assert not any(edge.relation == "interacts_with" and edge.provenance == "canonical_relation_reasoner" for edge in graph.relations)


def test_graph_builder_deduplicates_canonical_and_base_relation_paths() -> None:
    builder = SceneGraphBuilder()
    perception = PerceptionOutput(
        persons=[
            PersonFacts(
                bbox=BBox(0.15, 0.1, 0.4, 0.75),
                bbox_confidence=0.92,
                bbox_source="yolo",
                mask_ref=None,
                mask_confidence=0.9,
                mask_source="segformer",
                pose=PoseState(),
                pose_confidence=0.85,
                pose_source="vitpose",
                expression=ExpressionState(),
                expression_confidence=0.75,
                expression_source="emonet",
                orientation=OrientationState(),
                orientation_confidence=0.75,
                orientation_source="emonet",
                garments=[
                    {
                        "type": "shirt",
                        "confidence": 0.88,
                        "source": "segformer",
                        "coverage_targets": ["torso", "left_arm", "right_arm"],
                        "attachment_targets": ["torso"],
                    }
                ],
            )
        ],
        frame_size=(1024, 768),
    )
    graph = builder.build(perception, frame_index=5)

    relation_keys = [(edge.source, edge.relation, edge.target) for edge in graph.relations]
    assert len(relation_keys) == len(set(relation_keys))
    assert graph.persons[0].region_relations  # canonical path produced relations before graph-level dedup
    assert any(edge.provenance in {"segformer", "heuristic"} for edge in graph.relations)
