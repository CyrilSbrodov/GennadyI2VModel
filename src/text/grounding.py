from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import SceneGraph


@dataclass(slots=True)
class SceneGroundingIndex:
    """Легковесный индекс scene graph для parser-stage grounding."""

    object_types: set[str] = field(default_factory=set)
    object_ids_by_type: dict[str, list[str]] = field(default_factory=dict)
    garment_types: set[str] = field(default_factory=set)
    garment_ids_by_type: dict[str, list[str]] = field(default_factory=dict)
    body_part_types: set[str] = field(default_factory=set)
    has_face: bool = False
    has_usable_arm: bool = False
    has_support_object: bool = False
    has_outer_garment: bool = False


def build_scene_grounding_index(scene_graph: SceneGraph | None) -> SceneGroundingIndex:
    """Строит индекс доступных сущностей для обязательного scene-aware parsing."""

    index = SceneGroundingIndex()
    if scene_graph is None:
        return index

    support_types = {"chair", "stool", "sofa", "bench"}
    outerwear_types = {"coat", "jacket", "hoodie", "outerwear", "upper_garment"}
    arm_parts = {"arm", "left_arm", "right_arm", "upper_arm", "left_upper_arm", "right_upper_arm", "hands", "hand"}
    face_parts = {"face", "head", "face_skin", "mouth", "eyes"}

    for obj in scene_graph.objects:
        typ = obj.object_type
        index.object_types.add(typ)
        index.object_ids_by_type.setdefault(typ, []).append(obj.object_id)

    for person in scene_graph.persons:
        for garment in person.garments:
            gtyp = garment.garment_type
            index.garment_types.add(gtyp)
            index.garment_ids_by_type.setdefault(gtyp, []).append(garment.garment_id)
        for part in person.body_parts:
            ptyp = part.part_type
            index.body_part_types.add(ptyp)

    index.has_support_object = bool(index.object_types.intersection(support_types))
    index.has_outer_garment = bool(index.garment_types.intersection(outerwear_types))
    index.has_usable_arm = bool(index.body_part_types.intersection(arm_parts))
    index.has_face = bool(index.body_part_types.intersection(face_parts))
    return index
