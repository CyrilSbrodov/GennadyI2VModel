from __future__ import annotations

from core.region_ids import make_region_id
from core.schema import BBox, PersonNode, SceneGraph
from memory.video_memory import MemoryManager


CANONICAL_NAMES = [
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


def _canonical_payload(
    canonical_name: str,
    *,
    visibility_state: str = "unknown_expected_region",
    confidence: float = 0.2,
    lifecycle_state: str = "stable",
    last_transition_mode: str = "steady",
    last_transition_phase: str = "stable",
    last_semantic_reasons: list[str] | None = None,
    last_update_source: str = "graph_delta_state_update",
) -> dict[str, object]:
    return {
        "canonical_name": canonical_name,
        "raw_sources": [canonical_name],
        "source_regions": [canonical_name],
        "mask_ref": f"mask://{canonical_name}",
        "confidence": confidence,
        "visibility_state": visibility_state,
        "provenance": "canonical_reasoner",
        "attachment_hints": [],
        "ownership_hints": ["person"],
        "coverage_hints": [],
        "lifecycle_state": lifecycle_state,
        "last_transition_mode": last_transition_mode,
        "last_transition_phase": last_transition_phase,
        "last_semantic_reasons": last_semantic_reasons or [f"{canonical_name}_reason"],
        "last_update_source": last_update_source,
    }


def _person(person_id: str, *, canonical_overrides: dict[str, dict[str, object]] | None = None) -> PersonNode:
    canonical_regions = {name: _canonical_payload(name) for name in CANONICAL_NAMES}
    for key, value in (canonical_overrides or {}).items():
        canonical_regions[key] = value
    return PersonNode(
        person_id=person_id,
        track_id=f"{person_id}_track",
        bbox=BBox(0.1, 0.1, 0.6, 0.8),
        mask_ref=f"mask://{person_id}",
        confidence=0.9,
        source="unit_test",
        canonical_regions=canonical_regions,
    )


def test_update_from_graph_reads_newly_revealed_lifecycle() -> None:
    manager = MemoryManager()
    graph = SceneGraph(
        frame_index=0,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "inner_garment": _canonical_payload(
                        "inner_garment",
                        visibility_state="visible",
                        confidence=0.74,
                        lifecycle_state="newly_revealed",
                        last_transition_mode="garment_open",
                        last_transition_phase="contact_or_reveal",
                        last_semantic_reasons=["zip_opened"],
                    )
                },
            )
        ],
        objects=[],
    )

    memory = manager.update_from_graph(manager.initialize(SceneGraph(frame_index=0, persons=[], objects=[])), graph)
    entry = manager.get_best_region_memory(memory, "p1", "inner_garment")
    assert entry is not None
    assert entry.visibility_state == "visible"
    assert entry.reveal_lifecycle == "newly_revealed"


def test_update_from_graph_reads_newly_occluded_lifecycle() -> None:
    manager = MemoryManager()
    graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "outer_garment": _canonical_payload(
                        "outer_garment",
                        visibility_state="hidden_by_garment",
                        confidence=0.63,
                        lifecycle_state="newly_occluded",
                        last_transition_mode="visibility_occlusion",
                        last_transition_phase="transition",
                        last_semantic_reasons=["outer_layer_occludes"],
                    )
                },
            )
        ],
        objects=[],
    )

    memory = manager.update_from_graph(manager.initialize(SceneGraph(frame_index=0, persons=[], objects=[])), graph)
    entry = manager.get_best_region_memory(memory, "p1", "outer_garment")
    slot = memory.hidden_region_slots.get(make_region_id("p1", "outer_garment"))
    assert entry is not None
    assert entry.reveal_lifecycle == "newly_occluded"
    assert slot is not None
    assert slot.owner_entity == "p1"


def test_update_from_graph_does_not_mark_newly_revealed_as_reliable_without_evidence() -> None:
    manager = MemoryManager()
    graph = SceneGraph(
        frame_index=2,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "inner_garment": _canonical_payload(
                        "inner_garment",
                        visibility_state="visible",
                        confidence=0.5,
                        lifecycle_state="newly_revealed",
                        last_update_source="graph_delta_state_update",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.update_from_graph(manager.initialize(SceneGraph(frame_index=0, persons=[], objects=[])), graph)
    entry = manager.get_best_region_memory(memory, "p1", "inner_garment")
    assert entry is not None
    assert entry.reveal_lifecycle == "newly_revealed"
    assert entry.reliable_for_reuse is False


def test_update_from_graph_preserves_existing_reliable_visible_region() -> None:
    manager = MemoryManager()
    initial_graph = SceneGraph(
        frame_index=0,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "face": _canonical_payload(
                        "face",
                        visibility_state="visible",
                        confidence=0.95,
                        lifecycle_state="stable",
                        last_transition_mode="stable",
                        last_transition_phase="stable",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.initialize(initial_graph)
    before = manager.get_best_region_memory(memory, "p1", "face")
    assert before is not None
    assert before.reliable_for_reuse is True

    stable_graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "face": _canonical_payload(
                        "face",
                        visibility_state="visible",
                        confidence=0.94,
                        lifecycle_state="visibility_changed",
                        last_transition_mode="head_turn",
                        last_transition_phase="stabilize",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.update_from_graph(memory, stable_graph)
    after = manager.get_best_region_memory(memory, "p1", "face")
    assert after is not None
    assert after.visibility_state == "visible"
    assert after.reliable_for_reuse is True


def test_update_from_graph_does_not_cross_update_other_person() -> None:
    manager = MemoryManager()
    initial_graph = SceneGraph(
        frame_index=0,
        persons=[
            _person("p1", canonical_overrides={"inner_garment": _canonical_payload("inner_garment", visibility_state="visible", confidence=0.9)}),
            _person("p2"),
        ],
        objects=[],
    )
    memory = manager.initialize(initial_graph)
    p1_before = manager.get_best_region_memory(memory, "p1", "inner_garment")
    assert p1_before is not None
    p1_before_transition = p1_before.last_transition

    update_graph = SceneGraph(
        frame_index=1,
        persons=[
            _person("p1", canonical_overrides={"inner_garment": _canonical_payload("inner_garment", visibility_state="visible", confidence=0.9, lifecycle_state="stable")}),
            _person(
                "p2",
                canonical_overrides={
                    "inner_garment": _canonical_payload(
                        "inner_garment",
                        visibility_state="visible",
                        confidence=0.72,
                        lifecycle_state="newly_revealed",
                        last_transition_mode="garment_open",
                    )
                },
            ),
        ],
        objects=[],
    )
    memory = manager.update_from_graph(memory, update_graph)
    p1_after = manager.get_best_region_memory(memory, "p1", "inner_garment")
    p2_after = manager.get_best_region_memory(memory, "p2", "inner_garment")
    assert p1_after is not None
    assert p2_after is not None
    assert p1_after.reveal_lifecycle != "newly_revealed"
    assert p1_after.last_transition == p1_before_transition or "newly_revealed" not in p1_after.last_transition
    assert p2_after.reveal_lifecycle == "newly_revealed"
