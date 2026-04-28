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


def test_stable_visible_face_remains_reliable_across_update() -> None:
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

    stable_graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "face": _canonical_payload(
                        "face",
                        visibility_state="visible",
                        confidence=0.93,
                        lifecycle_state="visibility_changed",
                        provenance="parser",
                        mask_ref="mask://p1:face",
                        last_update_source="parser",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.update_from_graph(memory, stable_graph)
    face = manager.get_best_region_memory(memory, "p1", "face")
    assert face is not None
    assert face.visibility_state == "visible"
    assert face.reliable_for_reuse is True


def test_weak_inferred_face_update_does_not_overwrite_strong_face_memory() -> None:
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
                        confidence=0.96,
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
    strong = manager.get_best_region_memory(memory, "p1", "face")
    assert strong is not None

    weak_graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "face": _canonical_payload(
                        "face",
                        visibility_state="visible",
                        confidence=0.4,
                        lifecycle_state="stable",
                        provenance="graph_delta_state_update",
                        mask_ref=None,
                        last_update_source="graph_delta_state_update",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.update_from_graph(memory, weak_graph)
    face = manager.get_best_region_memory(memory, "p1", "face")
    assert face is not None
    assert face.reliable_for_reuse is True
    assert face.evidence_score >= strong.evidence_score - 1e-6
    assert face.confidence >= strong.confidence - 1e-6


def test_newly_revealed_low_evidence_face_not_reliable() -> None:
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
    face = manager.get_best_region_memory(memory, "p1", "face")
    assert face is not None
    assert face.reveal_lifecycle == "newly_revealed"
    assert face.reliable_for_reuse is False


def test_newly_occluded_preserves_previous_face_memory_for_later_reveal() -> None:
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

    face = manager.get_best_region_memory(memory, "p1", "face")
    slot = memory.hidden_region_slots.get(make_region_id("p1", "face"))
    assert face is not None
    assert face.visibility_state in {"visible", "hidden", "hidden_by_object"}
    assert face.confidence >= 0.9
    assert face.evidence_score >= 0.7
    assert slot is not None
    assert "newly_occluded" in slot.last_transition_reason


def test_visible_outer_garment_appearance_memory_survives_stable_update() -> None:
    manager = MemoryManager()
    initial_graph = SceneGraph(
        frame_index=0,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "outer_garment": _canonical_payload(
                        "outer_garment",
                        visibility_state="visible",
                        confidence=0.91,
                        lifecycle_state="stable",
                        provenance="parser",
                        mask_ref="mask://p1:outer",
                        last_update_source="parser",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.initialize(initial_graph)

    stable_graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "outer_garment": _canonical_payload(
                        "outer_garment",
                        visibility_state="partially_visible",
                        confidence=0.88,
                        lifecycle_state="visibility_changed",
                        provenance="parser",
                        mask_ref="mask://p1:outer",
                        last_update_source="parser",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.update_from_graph(memory, stable_graph)

    outer = manager.get_best_region_memory(memory, "p1", "outer_garment")
    assert outer is not None
    assert outer.visibility_state in {"visible", "partially_visible"}
    assert outer.reliable_for_reuse is True


def test_generated_or_inferred_patch_cannot_replace_direct_identity_memory() -> None:
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
    strong = manager.get_best_region_memory(memory, "p1", "face")
    assert strong is not None

    manager._refresh_canonical_memory_from_descriptor(
        memory=memory,
        entity_id="p1",
        canonical_region="face",
        source_frame=1,
        evidence_score=0.38,
        confidence=0.42,
        observed_directly=False,
        generated=True,
    )

    face = manager.get_best_region_memory(memory, "p1", "face")
    assert face is not None
    assert face.generated is False
    assert face.observed_directly is True
    assert face.reliable_for_reuse is True
    assert face.evidence_score >= strong.evidence_score - 1e-6
