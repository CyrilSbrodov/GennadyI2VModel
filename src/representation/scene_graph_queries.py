from __future__ import annotations

from core.schema import BodyPartNode, GarmentNode, RelationEdge, SceneGraph


class SceneGraphQueries:
    _GARMENT_HINTS = ("coat", "jacket", "hoodie", "shirt", "dress", "skirt", "pants")

    @staticmethod
    def _person(scene_graph: SceneGraph, person_id: str):
        return next((p for p in scene_graph.persons if p.person_id == person_id), None)

    @classmethod
    def _is_garment_like_entity(cls, entity_id: str, relation: RelationEdge | None = None) -> bool:
        if relation and relation.relation in {"covers", "attached_to"}:
            return True
        return entity_id.startswith(cls._GARMENT_HINTS)

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
            if relation.relation == "covers" and cls._is_garment_like_entity(relation.source, relation)
        ]
        return sorted(set(regions + covered))

    @staticmethod
    def relations_for_entity(scene_graph: SceneGraph, entity_id: str) -> list[RelationEdge]:
        return [r for r in scene_graph.relations if r.source == entity_id or r.target == entity_id]

    @staticmethod
    def relation_strength(scene_graph: SceneGraph, relation: str, source: str, target: str) -> float:
        scored = [r.confidence for r in scene_graph.relations if r.relation == relation and r.source == source and r.target == target]
        return max(scored) if scored else 0.0

    @staticmethod
    def get_covering_entities(scene_graph: SceneGraph, entity_id: str, target_region: str | None = None) -> list[str]:
        matches = [r.source for r in scene_graph.relations if r.relation in {"covers", "occludes"} and r.target.startswith(entity_id)]
        if target_region is not None:
            suffix = f"_{target_region}"
            matches = [src for src in matches if any(r.source == src and r.target.endswith(suffix) for r in scene_graph.relations if r.relation in {"covers", "occludes"})]
        return sorted(set(matches))

    @staticmethod
    def get_supported_by(scene_graph: SceneGraph, entity_id: str) -> list[str]:
        return sorted(set(r.target for r in scene_graph.relations if r.relation == "supports" and r.source == entity_id))

    @classmethod
    def get_attached_garments(cls, scene_graph: SceneGraph, person_id: str) -> list[str]:
        person = cls._person(scene_graph, person_id)
        garments = {g.garment_id for g in person.garments} if person else set()
        for r in scene_graph.relations:
            if r.relation == "attached_to" and (r.target == person_id or r.target.startswith(person_id)):
                if cls._is_garment_like_entity(r.source, r):
                    garments.add(r.source)
        return sorted(garments)

    @staticmethod
    def get_regions_affected_by_relation(scene_graph: SceneGraph, source_entity: str, relation_type: str) -> list[str]:
        affected: list[str] = []
        for relation in scene_graph.relations:
            if relation.source != source_entity or relation.relation != relation_type:
                continue
            target = relation.target.split(":", 1)[-1]
            if "_" in target:
                affected.append(target.split("_", 1)[-1])
            else:
                affected.append(target)
        return sorted(set(affected))

    @staticmethod
    def is_region_likely_revealed(scene_graph: SceneGraph, entity_id: str, region_type: str, predicted_visibility: dict[str, str] | None = None) -> bool:
        vis = (predicted_visibility or {}).get(region_type, "unknown")
        if vis in {"visible", "partially_visible"}:
            return True
        cover_edges = [r for r in scene_graph.relations if r.relation in {"covers", "occludes"} and r.target.startswith(entity_id) and r.target.endswith(f"_{region_type}")]
        if not cover_edges:
            return vis != "hidden"
        avg_cover = sum(r.confidence for r in cover_edges) / max(1, len(cover_edges))
        return avg_cover < 0.45 or vis == "visible"
