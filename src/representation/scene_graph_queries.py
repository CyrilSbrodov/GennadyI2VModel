from __future__ import annotations

from core.schema import BodyPartNode, GarmentNode, RelationEdge, SceneGraph


class SceneGraphQueries:
    @staticmethod
    def _person(scene_graph: SceneGraph, person_id: str):
        return next((p for p in scene_graph.persons if p.person_id == person_id), None)

    @classmethod
    def get_body_part(cls, scene_graph: SceneGraph, person_id: str, part_type: str) -> BodyPartNode | None:
        person = cls._person(scene_graph, person_id)
        if person is None:
            return None
        return next((part for part in person.body_parts if part.part_type == part_type), None)

    @classmethod
    def get_garment(cls, scene_graph: SceneGraph, person_id: str, garment_type: str) -> GarmentNode | None:
        person = cls._person(scene_graph, person_id)
        if person is None:
            return None
        return next((garment for garment in person.garments if garment.garment_type == garment_type), None)

    @classmethod
    def get_visible_regions(cls, scene_graph: SceneGraph, person_id: str) -> list[str]:
        person = cls._person(scene_graph, person_id)
        if person is None:
            return []
        regions = [part.part_type for part in person.body_parts if part.visibility in {"visible", "partially_visible"}]
        regions.extend(garment.garment_type for garment in person.garments if garment.visibility in {"visible", "partially_visible"})
        return regions

    @classmethod
    def get_occluded_regions(cls, scene_graph: SceneGraph, person_id: str) -> list[str]:
        person = cls._person(scene_graph, person_id)
        if person is None:
            return []
        regions = [part.part_type for part in person.body_parts if part.visibility == "hidden"]
        regions.extend(garment.garment_type for garment in person.garments if garment.visibility == "hidden")
        covered = [
            relation.target.split("_", 1)[-1]
            for relation in scene_graph.relations
            if relation.relation == "covers" and relation.source.startswith(("coat", "jacket", "hoodie", "shirt"))
        ]
        return sorted(set(regions + covered))

    @staticmethod
    def relations_for_entity(scene_graph: SceneGraph, entity_id: str) -> list[RelationEdge]:
        return [r for r in scene_graph.relations if r.source == entity_id or r.target == entity_id]

    @staticmethod
    def relation_strength(scene_graph: SceneGraph, relation: str, source: str, target: str) -> float:
        scored = [r.confidence for r in scene_graph.relations if r.relation == relation and r.source == source and r.target == target]
        return max(scored) if scored else 0.0
