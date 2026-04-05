from __future__ import annotations

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


class SceneGraphBuilder:
    def build(self, perception: PerceptionOutput, frame_index: int = 0) -> SceneGraph:
        persons: list[PersonNode] = []
        objects: list[SceneObjectNode] = []
        relations: list[RelationEdge] = []

        for idx, p in enumerate(perception.persons, start=1):
            person_id = f"person_{idx}"
            body_parts = [
                BodyPartNode(part_id=f"{person_id}_head", part_type="head", confidence=0.85, visibility="visible"),
                BodyPartNode(part_id=f"{person_id}_torso", part_type="torso", confidence=0.86, visibility="visible"),
                BodyPartNode(part_id=f"{person_id}_left_arm", part_type="left_arm", confidence=0.82, visibility="visible"),
                BodyPartNode(part_id=f"{person_id}_right_arm", part_type="right_arm", confidence=0.82, visibility="visible"),
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
                        confidence=float(garment.get("confidence", 0.5)),
                        visibility="visible",
                    )
                )
                relations.append(RelationEdge(garment_id, "attached_to", f"{person_id}_torso", 0.8))

            persons.append(
                PersonNode(
                    person_id=person_id,
                    track_id=person_id,
                    bbox=p.bbox,
                    mask_ref=p.mask_ref,
                    pose_state=p.pose,
                    expression_state=p.expression,
                    orientation=p.orientation,
                    body_parts=body_parts,
                    garments=garments,
                    confidence=0.9,
                )
            )

        for idx, obj in enumerate(perception.objects, start=1):
            obj_id = f"{obj.object_type}_{idx}"
            objects.append(SceneObjectNode(obj_id, obj.object_type, obj.bbox, confidence=obj.confidence))
            if persons:
                relations.append(RelationEdge(persons[0].person_id, "near", obj_id, 0.7))

        return SceneGraph(
            frame_index=frame_index,
            persons=persons,
            objects=objects,
            relations=relations,
            global_context=GlobalSceneContext(frame_size=perception.frame_size),
        )
