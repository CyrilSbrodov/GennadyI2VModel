from __future__ import annotations

from dataclasses import replace

from core.schema import BodyPartNode, GarmentNode, GlobalSceneContext, PersonNode, RelationEdge, SceneGraph, SceneObjectNode
from perception.pipeline import PerceptionOutput
from representation.canonical_human_state import CanonicalHumanSceneProcessor, canonical_relations_to_edges, canonical_state_to_dict


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
            "visibility_reasoner": 4,
            "canonical_relation_reasoner": 4,
        }
        self._canonical = CanonicalHumanSceneProcessor()

    def build(self, perception: PerceptionOutput, frame_index: int = 0) -> SceneGraph:
        persons: list[PersonNode] = []
        objects: list[SceneObjectNode] = []
        canonical_states = []

        for idx, p in enumerate(perception.persons, start=1):
            person_id = f"person_{idx}"
            canonical = self._canonical.process(p, person_id=person_id, frame_size=perception.frame_size, objects=perception.objects)
            canonical_states.append(canonical)
            canonical_regions = canonical.regions

            body_part_order = [
                "head",
                "face",
                "hair",
                "neck",
                "torso",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "pelvis",
                "left_leg",
                "right_leg",
                "upper_body",
                "lower_body",
            ]
            body_parts = [
                BodyPartNode(
                    part_id=f"{person_id}_{part_name}",
                    part_type=part_name,
                    mask_ref=canonical_regions[part_name].mask_ref,
                    confidence=self._calibrate_confidence(canonical_regions[part_name].confidence, canonical_regions[part_name].provenance),
                    visibility=canonical_regions[part_name].visibility_state,
                    source=canonical_regions[part_name].provenance,
                    frame_index=frame_index,
                    alternatives=[
                        f"raw:{','.join(canonical_regions[part_name].source_regions) or 'none'}",
                        f"attachment:{','.join(canonical_regions[part_name].attachment_hints) or 'none'}",
                    ],
                )
                for part_name in body_part_order
            ]

            garments: list[GarmentNode] = []
            garment_regions = ["upper_garment", "lower_garment", "outer_garment", "inner_garment", "accessories"]
            for g_name in garment_regions:
                region = canonical_regions[g_name]
                garments.append(
                    GarmentNode(
                        garment_id=f"{person_id}_{g_name}",
                        garment_type=g_name,
                        mask_ref=region.mask_ref,
                        garment_state="worn" if region.confidence >= 0.25 else "uncertain",
                        coverage_targets=[f"{person_id}_{x}" for x in region.coverage_hints] if region.coverage_hints else [f"{person_id}_torso"],
                        attachment_targets=[f"{person_id}_{x}" for x in region.attachment_hints] if region.attachment_hints else [f"{person_id}_torso"],
                        confidence=self._calibrate_confidence(region.confidence, region.provenance),
                        visibility=region.visibility_state,
                        source=region.provenance,
                        frame_index=frame_index,
                        alternatives=[
                            f"source_regions:{','.join(region.source_regions) or 'none'}",
                            f"ownership:{','.join(region.ownership_hints) or 'person'}",
                        ],
                    )
                )

            canonical_payload = canonical_state_to_dict(canonical)
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
                    alternatives=["scene_object", "canonical_human_state"],
                    canonical_regions=canonical_payload["regions"],
                    region_relations=canonical_payload["relations"],
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
                    alternatives=["unknown_object", "held_object"],
                )
            )

        relations = self._infer_relations(persons=persons, objects=objects, frame_index=frame_index)
        for state in canonical_states:
            relations.extend(
                RelationEdge(
                    source=rel.source,
                    relation=rel.relation,
                    target=rel.target,
                    confidence=rel.confidence,
                    provenance=rel.provenance,
                    frame_index=frame_index,
                    alternatives=["canonical_reasoning"],
                )
                for rel in canonical_relations_to_edges(state, frame_index=frame_index)
            )

        relations = self._resolve_relation_conflicts(relations)

        scene_graph = SceneGraph(
            frame_index=frame_index,
            persons=persons,
            objects=objects,
            relations=relations,
            global_context=GlobalSceneContext(frame_size=perception.frame_size),
        )
        self.validate(scene_graph)
        return scene_graph

    def validate(self, graph: SceneGraph) -> None:
        ids = {p.person_id for p in graph.persons} | {o.object_id for o in graph.objects}
        ids |= {part.part_id for p in graph.persons for part in p.body_parts}
        ids |= {g.garment_id for p in graph.persons for g in p.garments}
        for rel in graph.relations:
            if rel.source not in ids or rel.target not in ids:
                rel.provenance = f"{rel.provenance}:dangling"
                rel.confidence = min(rel.confidence, 0.2)

    def _calibrate_confidence(self, raw_confidence: float, source: str) -> float:
        source_key = source.lower().split(":")[0]
        reliability = self._source_reliability.get(source_key, self._source_reliability["unknown"])
        calibrated = max(0.0, min(1.0, raw_confidence * reliability))
        return round(calibrated, 4)

    def _infer_relations(self, persons: list[PersonNode], objects: list[SceneObjectNode], frame_index: int) -> list[RelationEdge]:
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
                        alternatives=["attached_to"],
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
                        alternatives=["covers", "touches", "held_by"],
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
                            alternatives=["in_front_of", "behind"],
                        )
                    )
        if persons:
            for obj in objects:
                rel = "supports" if obj.object_type in {"chair", "sofa", "bench"} else "interacts_with"
                relations.extend(
                    [
                        RelationEdge(persons[0].person_id, "near", obj.object_id, 0.6, "heuristic", frame_index),
                        RelationEdge(persons[0].person_id, rel, obj.object_id, 0.68, "heuristic", frame_index),
                        RelationEdge(persons[0].person_id, "touches", obj.object_id, 0.52, "heuristic", frame_index),
                    ]
                )
        return relations

    def _resolve_relation_conflicts(self, relations: list[RelationEdge]) -> list[RelationEdge]:
        winners: dict[tuple[str, str, str], RelationEdge] = {}
        priority = {"touches": 4, "interacts_with": 5, "supports": 6, "in_front_of": 3, "behind": 2}
        for relation in relations:
            key = (relation.source, relation.relation, relation.target)
            current = winners.get(key)
            if current is None:
                winners[key] = relation
                continue
            if current.provenance.endswith(":dangling") and not relation.provenance.endswith(":dangling"):
                winners[key] = relation
                continue
            if relation.provenance.endswith(":dangling") and not current.provenance.endswith(":dangling"):
                continue
            if relation.confidence > current.confidence:
                winners[key] = relation
                continue
            if relation.confidence == current.confidence:
                left_priority = self._source_priority.get(relation.provenance.lower().split(":")[0], 0) + priority.get(relation.relation, 0)
                right_priority = self._source_priority.get(current.provenance.lower().split(":")[0], 0) + priority.get(current.relation, 0)
                if left_priority > right_priority:
                    winners[key] = relation
        ordered = sorted(winners.values(), key=lambda r: (r.source, r.relation, r.target))
        return [replace(v, confidence=round(v.confidence, 4)) for v in ordered]
