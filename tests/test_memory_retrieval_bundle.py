from __future__ import annotations

from core.region_ids import make_region_id
from core.schema import BBox, HiddenRegionSlot, PersonNode, SceneGraph
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
    provenance: str = "canonical_reasoner",
    mask_ref: str | None = None,
    last_update_source: str = "graph_delta_state_update",
) -> dict[str, object]:
    return {
        "canonical_name": canonical_name,
        "raw_sources": [canonical_name],
        "source_regions": [canonical_name],
        "mask_ref": mask_ref,
        "confidence": confidence,
        "visibility_state": visibility_state,
        "provenance": provenance,
        "attachment_hints": [],
        "ownership_hints": ["person"],
        "coverage_hints": [],
        "lifecycle_state": lifecycle_state,
        "last_transition_mode": "stable",
        "last_transition_phase": "stable",
        "last_semantic_reasons": [f"{canonical_name}_reason"],
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


def test_bundle_returns_current_reuse_and_identity_reference_for_visible_face() -> None:
    manager = MemoryManager()
    graph = SceneGraph(
        frame_index=0,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "face": _canonical_payload(
                        "face",
                        visibility_state="visible",
                        confidence=0.95,
                        provenance="parser",
                        mask_ref="mask://p1:face",
                        last_update_source="parser",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.initialize(graph)

    bundle = manager.get_region_memory_bundle(memory, "p1", "face")
    assert bundle.current_reuse is not None
    assert bundle.identity_reference is not None
    assert bundle.memory_support_level == "strong"


def test_bundle_occluded_face_has_identity_reference_but_no_current_reuse() -> None:
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
                        provenance="parser",
                        mask_ref="mask://p1:face",
                        last_update_source="parser",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.initialize(initial_graph)
    occluded_graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "face": _canonical_payload(
                        "face",
                        visibility_state="hidden_by_object",
                        confidence=0.32,
                        lifecycle_state="newly_occluded",
                        provenance="graph_delta_state_update",
                        mask_ref=None,
                        last_update_source="graph_delta_state_update",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.update_from_graph(memory, occluded_graph)

    bundle = manager.get_region_memory_bundle(memory, "p1", "face")
    assert bundle.current_reuse is None
    assert bundle.identity_reference is not None
    assert bundle.memory_support_level in {"medium", "strong"}
    assert any("occluded" in reason or "reference" in reason for reason in bundle.retrieval_reasons)


def test_bundle_low_evidence_newly_revealed_face_is_not_strong() -> None:
    manager = MemoryManager()
    graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "face": _canonical_payload(
                        "face",
                        visibility_state="visible",
                        confidence=0.5,
                        lifecycle_state="newly_revealed",
                        provenance="graph_delta_state_update",
                        mask_ref=None,
                        last_update_source="graph_delta_state_update",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.update_from_graph(manager.initialize(SceneGraph(frame_index=0, persons=[], objects=[])), graph)

    bundle = manager.get_region_memory_bundle(memory, "p1", "face")
    assert bundle.current_reuse is None
    assert bundle.identity_reference is None
    assert bundle.memory_support_level in {"weak", "none"}


def test_bundle_visible_outer_garment_returns_garment_reference() -> None:
    manager = MemoryManager()
    graph = SceneGraph(
        frame_index=0,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "outer_garment": _canonical_payload(
                        "outer_garment",
                        visibility_state="visible",
                        confidence=0.92,
                        provenance="parser",
                        mask_ref="mask://p1:outer",
                        last_update_source="parser",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.initialize(graph)
    bundle = manager.get_region_memory_bundle(memory, "p1", "outer_garment")
    assert bundle.garment_reference is not None
    assert bundle.memory_support_level == "strong"


def test_bundle_revealed_history_hidden_slot_not_active_hidden_support() -> None:
    manager = MemoryManager()
    memory = manager.initialize(SceneGraph(frame_index=0, persons=[], objects=[]))
    region_id = make_region_id("p1", "face")
    memory.hidden_region_slots[region_id] = HiddenRegionSlot(
        slot_id=region_id,
        region_type="face",
        owner_entity="p1",
        candidate_patch_ids=["patch::1"],
        hidden_type="revealed_history",
        confidence=0.8,
        evidence_score=0.8,
    )

    bundle = manager.get_region_memory_bundle(memory, "p1", "face")
    assert bundle.hidden_slot is not None
    assert bundle.hidden_slot.hidden_type in {"revealed", "revealed_history"}
    assert bundle.memory_support_level != "strong"
    assert any("revealed_history" in reason for reason in bundle.retrieval_reasons)


def test_bundle_missing_region_returns_empty_none_support() -> None:
    manager = MemoryManager()
    memory = manager.initialize(SceneGraph(frame_index=0, persons=[], objects=[]))
    bundle = manager.get_region_memory_bundle(memory, "unknown", "face")
    assert bundle.current_reuse is None
    assert bundle.identity_reference is None
    assert bundle.appearance_reference is None
    assert bundle.garment_reference is None
    assert bundle.hidden_slot is None
    assert bundle.memory_support_level == "none"

