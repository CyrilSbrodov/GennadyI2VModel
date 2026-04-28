from __future__ import annotations

from core.region_ids import parse_region_id
from core.schema import GraphDelta, SceneGraph

_KNOWN_CANONICAL_REGIONS = {
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
    "upper_garment",
    "lower_garment",
    "outer_garment",
    "inner_garment",
    "accessories",
}


def _target_persons(scene_graph: SceneGraph, delta: GraphDelta) -> list:
    if not scene_graph.persons:
        return []
    if delta.affected_entities:
        affected = set(delta.affected_entities)
        return [person for person in scene_graph.persons if person.person_id in affected]
    return [scene_graph.persons[0]]


def _canonical_from_key(person_id: str, key: str) -> str | None:
    if not key:
        return None
    if ":" in key:
        entity_id, canonical = parse_region_id(key)
        if entity_id != person_id:
            return None
        return canonical
    return key


def _ensure_canonical_payload(person, canonical_name: str, fallback_visibility: str) -> dict[str, object]:
    existing = person.canonical_regions.get(canonical_name)
    if existing is None:
        payload: dict[str, object] = {
            "canonical_name": canonical_name,
            "raw_sources": [],
            "source_regions": [canonical_name],
            "mask_ref": None,
            "confidence": 0.5,
            "visibility_state": fallback_visibility,
            "provenance": "graph_delta_state_update",
            "attachment_hints": [],
            "ownership_hints": ["person"],
            "coverage_hints": [],
            "lifecycle_state": "visibility_changed",
            "last_transition_mode": "unknown",
            "last_transition_phase": "single",
            "last_semantic_reasons": [],
            "last_update_source": "graph_delta_state_update",
        }
        person.canonical_regions[canonical_name] = payload  # type: ignore[assignment]
        return payload
    existing.setdefault("lifecycle_state", "visibility_changed")
    existing.setdefault("last_transition_mode", "unknown")
    existing.setdefault("last_transition_phase", "single")
    existing.setdefault("last_semantic_reasons", [])
    existing.setdefault("last_update_source", "graph_delta_state_update")
    return existing


def _should_update_canonical_region(person, key: str) -> bool:
    canonical_name = _canonical_from_key(person.person_id, key)
    if canonical_name is None:
        return False
    if key in person.canonical_regions or canonical_name in person.canonical_regions:
        return True
    if ":" in key:
        entity_id, _ = parse_region_id(key)
        return entity_id == person.person_id
    return canonical_name in _KNOWN_CANONICAL_REGIONS


def _transition_mode_for_region(delta: GraphDelta, person_id: str, canonical_name: str) -> str | None:
    if canonical_name in delta.region_transition_mode:
        return str(delta.region_transition_mode[canonical_name])
    full_region_id = f"{person_id}:{canonical_name}"
    if full_region_id in delta.region_transition_mode:
        return str(delta.region_transition_mode[full_region_id])
    return None


def _update_canonical_region(
    *,
    person,
    delta: GraphDelta,
    key: str,
    visibility: str,
    lifecycle_state: str,
    update_source: str,
) -> None:
    if not _should_update_canonical_region(person, key):
        return
    canonical_name = _canonical_from_key(person.person_id, key)
    if canonical_name is None:
        return
    payload = _ensure_canonical_payload(person, canonical_name, fallback_visibility=visibility)
    payload["visibility_state"] = visibility
    payload["lifecycle_state"] = lifecycle_state
    payload["last_update_source"] = update_source
    payload["last_transition_phase"] = delta.transition_phase
    payload["last_semantic_reasons"] = list(delta.semantic_reasons)
    mode = _transition_mode_for_region(delta, person.person_id, canonical_name)
    if mode is not None:
        payload["last_transition_mode"] = mode


def _apply_body_part_visibility(person, key: str, visibility: str) -> None:
    canonical_name = _canonical_from_key(person.person_id, key)
    if canonical_name is None:
        return
    for body_part in person.body_parts:
        if key == body_part.part_id or key == body_part.part_type or canonical_name == body_part.part_type:
            body_part.visibility = visibility


def _apply_garment_visibility(person, key: str, visibility: str) -> None:
    for garment in person.garments:
        if key == garment.garment_id or key == garment.garment_type:
            garment.visibility = visibility


def apply_delta(scene_graph: SceneGraph, delta: GraphDelta) -> SceneGraph:
    for person in _target_persons(scene_graph, delta):
        if "smile_intensity" in delta.expression_deltas:
            person.expression_state.smile_intensity = min(1.0, max(0.0, person.expression_state.smile_intensity + float(delta.expression_deltas["smile_intensity"])))
            person.expression_state.label = str(delta.expression_deltas.get("mouth_state", "smile"))

        if delta.pose_deltas:
            person.pose_state.coarse_pose = "transition"
            if delta.interaction_deltas.get("chair_contact", 0.0) > 0.5:
                person.pose_state.coarse_pose = "seated"
            elif delta.interaction_deltas.get("chair_contact", 1.0) < 0.3:
                person.pose_state.coarse_pose = "standing"

        for garment in person.garments:
            if garment.garment_type == "coat" and "coat_state" in delta.garment_deltas:
                garment.garment_state = str(delta.garment_deltas["coat_state"])

        for key, visibility in delta.predicted_visibility_changes.items():
            vis = str(visibility)
            _update_canonical_region(
                person=person,
                delta=delta,
                key=key,
                visibility=vis,
                lifecycle_state="visibility_changed",
                update_source="graph_delta.predicted_visibility_changes",
            )
            _apply_body_part_visibility(person, key, vis)

        for key, visibility in delta.visibility_deltas.items():
            vis = str(visibility)
            _update_canonical_region(
                person=person,
                delta=delta,
                key=key,
                visibility=vis,
                lifecycle_state="visibility_changed",
                update_source="graph_delta.visibility_deltas",
            )
            _apply_body_part_visibility(person, key, vis)
            _apply_garment_visibility(person, key, vis)

        for region in delta.newly_revealed_regions:
            canonical_name = _canonical_from_key(person.person_id, region.region_id)
            if canonical_name is None:
                continue
            existing_payload = person.canonical_regions.get(canonical_name)
            existing_vis = str(existing_payload.get("visibility_state", "unknown")) if existing_payload is not None else "unknown"
            reveal_visibility = "partially_visible" if existing_vis == "partially_visible" else "visible"
            _update_canonical_region(
                person=person,
                delta=delta,
                key=region.region_id,
                visibility=reveal_visibility,
                lifecycle_state="newly_revealed",
                update_source="graph_delta.newly_revealed_regions",
            )
            _apply_body_part_visibility(person, region.region_id, reveal_visibility)

        for region in delta.newly_occluded_regions:
            canonical_name = _canonical_from_key(person.person_id, region.region_id)
            if canonical_name is None:
                continue
            existing_payload = person.canonical_regions.get(canonical_name)
            existing_vis = str(existing_payload.get("visibility_state", "unknown")) if existing_payload is not None else "unknown"
            occluded_visibility = "occluded" if existing_vis == "occluded" else "hidden"
            _update_canonical_region(
                person=person,
                delta=delta,
                key=region.region_id,
                visibility=occluded_visibility,
                lifecycle_state="newly_occluded",
                update_source="graph_delta.newly_occluded_regions",
            )
            _apply_body_part_visibility(person, region.region_id, occluded_visibility)

    scene_graph.frame_index += 1
    return scene_graph
