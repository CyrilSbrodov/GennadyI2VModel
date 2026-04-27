from __future__ import annotations

from core.schema import BBox, BodyPartNode, GarmentNode, PersonNode, SceneGraph
from memory.video_memory import MemoryManager


def _canonical_payload(
    name: str,
    *,
    visibility: str,
    confidence: float,
    provenance: str = "canonical_reasoner",
    mask_ref: str | None = None,
    source_regions: list[str] | None = None,
    raw_sources: list[str] | None = None,
) -> dict[str, object]:
    return {
        "canonical_name": name,
        "raw_sources": raw_sources if raw_sources is not None else [name],
        "source_regions": source_regions if source_regions is not None else [name],
        "mask_ref": f"mask://{name}" if mask_ref is None else mask_ref,
        "confidence": confidence,
        "visibility_state": visibility,
        "provenance": provenance,
        "attachment_hints": [],
        "ownership_hints": ["person"],
        "coverage_hints": [],
    }


def _graph(
    *,
    frame_index: int = 0,
    torso_visibility: str = "visible",
    torso_conf: float = 0.9,
    torso_mask_ref: str | None = None,
    torso_provenance: str = "canonical_reasoner",
    outer_visibility: str = "hidden_by_object",
    outer_conf: float = 0.2,
) -> SceneGraph:
    canonical_names = [
        "face",
        "hair",
        "head",
        "neck",
        "torso",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "pelvis",
        "left_leg",
        "right_leg",
        "upper_garment",
        "lower_garment",
        "outer_garment",
        "inner_garment",
        "accessories",
    ]
    canonical_regions = {name: _canonical_payload(name, visibility="unknown_expected_region", confidence=0.1) for name in canonical_names}
    canonical_regions["torso"] = _canonical_payload(
        "torso",
        visibility=torso_visibility,
        confidence=torso_conf,
        provenance=torso_provenance,
        mask_ref=torso_mask_ref,
        source_regions=["aggregate"] if torso_mask_ref is None else ["torso"],
        raw_sources=["reasoning"] if torso_mask_ref is None else ["torso_mask"],
    )
    canonical_regions["face"] = _canonical_payload("face", visibility="visible", confidence=0.88)
    canonical_regions["left_arm"] = _canonical_payload("left_arm", visibility="partially_visible", confidence=0.65)
    canonical_regions["outer_garment"] = _canonical_payload("outer_garment", visibility=outer_visibility, confidence=outer_conf)

    person = PersonNode(
        person_id="person_1",
        track_id="t1",
        bbox=BBox(0.2, 0.1, 0.4, 0.8),
        mask_ref="mask://person",
        body_parts=[
            BodyPartNode(part_id="person_1_torso", part_type="torso", mask_ref="mask://torso", visibility=torso_visibility, confidence=torso_conf, source="canonical_reasoner"),
            BodyPartNode(part_id="person_1_left_arm", part_type="left_arm", mask_ref="mask://left_arm", visibility="partially_visible", confidence=0.7, source="canonical_reasoner"),
            BodyPartNode(part_id="person_1_head", part_type="head", mask_ref="mask://head", visibility="visible", confidence=0.75, source="canonical_reasoner"),
        ],
        garments=[GarmentNode(garment_id="person_1_outer_garment", garment_type="outer_garment", mask_ref="mask://outer", confidence=outer_conf, visibility=outer_visibility, source="canonical_reasoner")],
        confidence=0.9,
        source="yolo",
        frame_index=frame_index,
        canonical_regions=canonical_regions,
    )
    return SceneGraph(frame_index=frame_index, persons=[person], objects=[])


def test_canonical_region_memory_initialized_from_canonical_state() -> None:
    graph = _graph()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)

    torso = mm.get_best_region_memory(memory, "person_1", "torso")
    outer = mm.get_best_region_memory(memory, "person_1", "outer_garment")
    assert torso is not None
    assert outer is not None
    assert torso.observed_directly is True
    assert torso.reliable_for_reuse is True
    assert outer.visibility_state in {"hidden", "hidden_by_object", "unknown_expected_region"}
    assert outer.observed_directly is False
    assert outer.suitable_for_reveal is False


def test_memory_strength_prevents_weaker_overwrite() -> None:
    graph = _graph(torso_visibility="visible", torso_conf=0.92)
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    strong = mm.get_best_region_memory(memory, "person_1", "torso")
    assert strong is not None

    weak_graph = _graph(frame_index=1, torso_visibility="partially_visible", torso_conf=0.2)
    memory = mm.update_from_graph(memory, weak_graph)
    after = mm.get_best_region_memory(memory, "person_1", "torso")
    assert after is not None
    assert after.confidence >= strong.confidence
    assert after.evidence_quality in {"strong", "medium"}


def test_weak_partial_update_stays_inferred_not_direct() -> None:
    graph = _graph(torso_visibility="partially_visible", torso_conf=0.35, torso_mask_ref=None, torso_provenance="heuristic")
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    torso = mm.get_best_region_memory(memory, "person_1", "torso")
    assert torso is not None
    assert torso.observed_directly is False
    assert torso.inferred is True
    assert torso.reliable_for_reuse is False
    assert torso.suitable_for_reveal is False


def test_hidden_and_reveal_lifecycle_transitions() -> None:
    graph = _graph()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)

    mm.apply_visibility_event(memory, {"region_id": "person_1:torso", "entity": "person_1"}, {}, visibility="hidden", transition_reason="occluded")
    hidden = mm.get_best_region_memory(memory, "person_1", "torso")
    assert hidden is not None
    assert hidden.reveal_lifecycle in {"newly_occluded", "currently_hidden"}

    mm.apply_visibility_event(memory, {"region_id": "person_1:torso", "entity": "person_1"}, {}, visibility="revealed", transition_reason="reveal")
    revealed = mm.get_best_region_memory(memory, "person_1", "torso")
    assert revealed is not None
    assert revealed.reveal_lifecycle in {"newly_revealed", "visible"}
    assert revealed.visibility_state == "visible"


def test_refresh_can_promote_hidden_region_to_visible_when_evidence_strong() -> None:
    graph = _graph(torso_visibility="visible", torso_conf=0.88)
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    mm.apply_visibility_event(memory, {"region_id": "person_1:torso", "entity": "person_1"}, {}, visibility="hidden", transition_reason="occluded")

    mm._refresh_canonical_memory_from_descriptor(
        memory=memory,
        entity_id="person_1",
        canonical_region="torso",
        source_frame=1,
        evidence_score=0.82,
        confidence=0.86,
        observed_directly=True,
        generated=False,
    )
    promoted = mm.get_best_region_memory(memory, "person_1", "torso")
    assert promoted is not None
    assert promoted.visibility_state in {"visible", "partially_visible"}
    assert promoted.observed_directly is True


def test_stale_and_generated_memory_becomes_conservative_for_reveal() -> None:
    graph = _graph(torso_visibility="visible", torso_conf=0.9)
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    torso = mm.get_best_region_memory(memory, "person_1", "torso")
    assert torso is not None
    assert torso.reliable_for_reuse is True

    torso.generated = True
    torso.freshness_frames = 12
    mm._refresh_reuse_policy(torso)
    assert torso.reliable_for_reuse is False
    assert torso.suitable_for_reveal is False


def test_canonical_state_to_memory_update_path() -> None:
    graph = _graph(torso_visibility="partially_visible", torso_conf=0.55)
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)

    entries = mm.get_region_memory_entries(memory, entity_id="person_1")
    canonical_regions = {entry.canonical_region for entry in entries}
    required = {
        "face",
        "hair",
        "head",
        "neck",
        "torso",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "pelvis",
        "left_leg",
        "right_leg",
        "upper_garment",
        "lower_garment",
        "outer_garment",
        "inner_garment",
        "accessories",
    }
    assert required.issubset(canonical_regions)


def test_debug_snapshot_contains_visibility_and_evidence_contract() -> None:
    graph = _graph()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    snapshot = mm.debug_canonical_memory(memory, entity_id="person_1")

    assert "regions" in snapshot
    assert snapshot["regions"]
    first = snapshot["regions"][0]
    assert "visibility_state" in first
    assert "evidence_score" in first
    assert "observed_directly" in first
    assert "inferred" in first
    assert "generated" in first
    assert "reliable_for_reuse" in first
    assert "suitable_for_reveal" in first
    assert "freshness_frames" in first
    assert "reveal_lifecycle" in first
    assert "last_transition" in first
