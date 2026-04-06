from __future__ import annotations

from dataclasses import replace

from core.schema import (
    BodyPartNode,
    GarmentNode,
    GlobalSceneContext,
    PersonNode,
    RelationEdge,
    SceneGraph,
    SceneObjectNode,
)
from perception.pipeline import PerceptionOutput


class LegacySceneGraphAdapter:
    """Keeps runtime orchestrator compatibility with the pre-enrichment format."""

    def to_legacy(self, scene_graph: SceneGraph) -> SceneGraph:
        legacy_relations = [
            replace(
                relation,
                confidence=round(relation.confidence, 4),
                alternatives=[],
                provenance=relation.provenance or "legacy_adapter",
            )
            for relation in scene_graph.relations
        ]
        return replace(scene_graph, relations=legacy_relations)


class SceneGraphBuilder:
    def __init__(self) -> None:
        self._source_reliability = {
            "yolo": 0.9,
            "vitpose": 0.88,
            "segformer": 0.84,
            "emonet": 0.8,
            "bytetrack": 0.82,
            "heuristic": 0.72,
            "fallback": 0.6,
            "unknown": 0.65,
        }
        self._source_priority = {
            "yolo": 5,
            "vitpose": 5,
            "segformer": 4,
            "emonet": 4,
            "bytetrack": 4,
            "heuristic": 3,
            "fallback": 2,
            "unknown": 1,
        }
        self._legacy_adapter = LegacySceneGraphAdapter()

    def build(self, perception: PerceptionOutput, frame_index: int = 0) -> SceneGraph:
        persons: list[PersonNode] = []
        objects: list[SceneObjectNode] = []

        for idx, p in enumerate(perception.persons, start=1):
            person_id = f"person_{idx}"
            body_parts = [
                BodyPartNode(
                    part_id=f"{person_id}_head",
                    part_type="head",
                    confidence=self._calibrate_confidence(0.85, p.pose_source),
                    visibility="visible",
                    source=p.pose_source,
                    frame_index=frame_index,
                ),
                BodyPartNode(
                    part_id=f"{person_id}_torso",
                    part_type="torso",
                    confidence=self._calibrate_confidence(0.86, p.mask_source),
                    visibility="visible",
                    source=p.mask_source,
                    frame_index=frame_index,
                ),
                BodyPartNode(
                    part_id=f"{person_id}_left_arm",
                    part_type="left_arm",
                    confidence=self._calibrate_confidence(0.82, p.pose_source),
                    visibility="visible",
                    source=p.pose_source,
                    frame_index=frame_index,
                ),
                BodyPartNode(
                    part_id=f"{person_id}_right_arm",
                    part_type="right_arm",
                    confidence=self._calibrate_confidence(0.82, p.pose_source),
                    visibility="visible",
                    source=p.pose_source,
                    frame_index=frame_index,
                ),
            ]
            garments: list[GarmentNode] = []
            for g_idx, garment in enumerate(p.garments, start=1):
                garment_id = f"{garment['type']}_{idx}_{g_idx}"
                garments.append(
                    GarmentNode(
                        garment_id=garment_id,
                        garment_type=garment["type"],
                        garment_state=garment.get("state", "worn"),
                        coverage_targets=[f"{person_id}_torso"],
                        attachment_targets=[f"{person_id}_torso"],
                        confidence=self._calibrate_confidence(
                            float(garment.get("confidence", 0.5)),
                            garment.get("source", "unknown"),
                        ),
                        visibility="visible",
                        source=garment.get("source", "unknown"),
                        frame_index=frame_index,
                        alternatives=["occludes", "touches"],
                    )
                )

            persons.append(
                PersonNode(
                    person_id=person_id,
                    track_id=p.track_id or person_id,
                    bbox=p.bbox,
                    mask_ref=p.mask_ref,
                    pose_state=p.pose,
                    expression_state=p.expression,
                    orientation=p.orientation,
                    body_parts=body_parts,
                    garments=garments,
                    confidence=self._calibrate_confidence(p.bbox_confidence, p.bbox_source),
                    source=p.bbox_source,
                    frame_index=frame_index,
                    alternatives=["scene_object"],
                )
            )

        for idx, obj in enumerate(perception.objects, start=1):
            obj_id = f"{obj.object_type}_{idx}"
            objects.append(
                SceneObjectNode(
                    obj_id,
                    obj.object_type,
                    obj.bbox,
                    confidence=self._calibrate_confidence(obj.confidence, obj.source),
                    source=obj.source,
                    frame_index=frame_index,
                    alternatives=["unknown_object"],
                )
            )

        relations = self._infer_relations(persons=persons, objects=objects, frame_index=frame_index)
        self._infer_visibility(persons=persons)
        self._infer_occlusions(persons=persons, objects=objects, relations=relations)
        relations = self._resolve_relation_conflicts(relations)

        scene_graph = SceneGraph(
            frame_index=frame_index,
            persons=persons,
            objects=objects,
            relations=relations,
            global_context=GlobalSceneContext(frame_size=perception.frame_size),
        )
        return self._legacy_adapter.to_legacy(scene_graph)

    def _calibrate_confidence(self, raw_confidence: float, source: str) -> float:
        source_key = source.lower()
        reliability = self._source_reliability.get(source_key, self._source_reliability["unknown"])
        calibrated = max(0.0, min(1.0, raw_confidence * reliability))
        return round(calibrated, 4)

    def _infer_relations(
        self,
        persons: list[PersonNode],
        objects: list[SceneObjectNode],
        frame_index: int,
    ) -> list[RelationEdge]:
        relations: list[RelationEdge] = []
        for person in persons:
            for body_part in person.body_parts:
                relations.append(
                    RelationEdge(
                        source=body_part.part_id,
                        relation="part_of",
                        target=person.person_id,
                        confidence=self._calibrate_confidence(body_part.confidence, body_part.source),
                        provenance="heuristic",
                        frame_index=frame_index,
                    )
                )
            for garment in person.garments:
                torso_target = garment.attachment_targets[0] if garment.attachment_targets else person.person_id
                relations.append(
                    RelationEdge(
                        source=garment.garment_id,
                        relation="attached_to",
                        target=torso_target,
                        confidence=self._calibrate_confidence(0.8, garment.source),
                        provenance=garment.source,
                        frame_index=frame_index,
                        alternatives=["covers", "touches"],
                    )
                )
                for covered in garment.coverage_targets:
                    relations.append(
                        RelationEdge(
                            source=garment.garment_id,
                            relation="covers",
                            target=covered,
                            confidence=self._calibrate_confidence(0.74, garment.source),
                            provenance=garment.source,
                            frame_index=frame_index,
                        )
                    )
        if persons:
            for obj in objects:
                relations.append(
                    RelationEdge(
                        source=persons[0].person_id,
                        relation="near",
                        target=obj.object_id,
                        confidence=self._calibrate_confidence(0.7, "heuristic"),
                        provenance="heuristic",
                        frame_index=frame_index,
                        alternatives=["interacts_with", "touches"],
                    )
                )
        return relations

    def _infer_visibility(self, persons: list[PersonNode]) -> None:
        for person in persons:
            for part in person.body_parts:
                if part.confidence < 0.35:
                    part.visibility = "hidden"
                elif part.confidence < 0.6:
                    part.visibility = "partially_visible"
                else:
                    part.visibility = "visible"
            for garment in person.garments:
                if garment.confidence < 0.3:
                    garment.visibility = "hidden"
                elif garment.confidence < 0.58:
                    garment.visibility = "partially_visible"
                else:
                    garment.visibility = "visible"

    def _infer_occlusions(
        self,
        persons: list[PersonNode],
        objects: list[SceneObjectNode],
        relations: list[RelationEdge],
    ) -> None:
        if not objects:
            return
        for person in persons:
            for garment in person.garments:
                if garment.visibility in {"hidden", "partially_visible"}:
                    blocker = objects[0].object_id
                    relations.append(
                        RelationEdge(
                            source=blocker,
                            relation="occludes",
                            target=garment.garment_id,
                            confidence=self._calibrate_confidence(0.65, "heuristic"),
                            provenance="heuristic",
                            frame_index=garment.frame_index,
                        )
                    )
                    garment.alternatives = [*garment.alternatives, "touches"]

    def _resolve_relation_conflicts(self, relations: list[RelationEdge]) -> list[RelationEdge]:
        winners: dict[tuple[str, str, str], RelationEdge] = {}
        for relation in relations:
            key = (relation.source, relation.relation, relation.target)
            current = winners.get(key)
            if current is None:
                winners[key] = relation
                continue
            if relation.confidence > current.confidence:
                winners[key] = relation
                continue
            if relation.confidence == current.confidence:
                left_priority = self._source_priority.get(relation.provenance.lower(), 0)
                right_priority = self._source_priority.get(current.provenance.lower(), 0)
                if left_priority > right_priority:
                    winners[key] = relation
        return list(winners.values())
