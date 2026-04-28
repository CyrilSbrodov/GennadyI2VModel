from __future__ import annotations

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


def test_visible_direct_face_is_reusable_and_identity_reference() -> None:
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
    face = manager.get_best_region_memory(memory, "p1", "face")
    assert face is not None
    assert face.reliable_for_reuse is True
    assert face.reliable_as_reference is True
    assert face.reference_kind == "identity_reference"


def test_occluded_face_not_current_reusable_but_still_identity_reference() -> None:
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
    assert face is not None
    assert face.reliable_for_reuse is False
    assert face.reliable_as_reference is True
    assert face.reference_kind == "identity_reference"


def test_newly_revealed_low_evidence_face_not_reference() -> None:
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
    face = manager.get_best_region_memory(memory, "p1", "face")
    assert face is not None
    assert face.reliable_for_reuse is False
    assert face.reliable_as_reference is False


def test_visible_outer_garment_is_garment_reference() -> None:
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
    outer = manager.get_best_region_memory(memory, "p1", "outer_garment")
    assert outer is not None
    assert outer.reliable_as_reference is True
    assert outer.reference_kind == "garment_reference"


def test_generated_face_candidate_cannot_clear_identity_reference() -> None:
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
    before = manager.get_best_region_memory(memory, "p1", "face")
    assert before is not None

    generated_graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "face": _canonical_payload(
                        "face",
                        visibility_state="visible",
                        confidence=0.4,
                        provenance="generated",
                        mask_ref=None,
                        last_update_source="graph_delta_state_update",
                    )
                },
            )
        ],
        objects=[],
    )
    memory = manager.update_from_graph(memory, generated_graph)
    after = manager.get_best_region_memory(memory, "p1", "face")
    assert after is not None
    assert after.reliable_as_reference is True
    assert after.reference_kind == "identity_reference"
    assert after.observed_directly is True
    assert after.provenance == "parser"


def test_hidden_non_identity_weak_region_not_reference() -> None:
    manager = MemoryManager()
    graph = SceneGraph(
        frame_index=1,
        persons=[
            _person(
                "p1",
                canonical_overrides={
                    "left_arm": _canonical_payload(
                        "left_arm",
                        visibility_state="hidden",
                        confidence=0.3,
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
    memory = manager.update_from_graph(manager.initialize(SceneGraph(frame_index=0, persons=[], objects=[])), graph)
    left_arm = manager.get_best_region_memory(memory, "p1", "left_arm")
    assert left_arm is not None
    assert left_arm.reliable_for_reuse is False
    assert left_arm.reliable_as_reference is False or left_arm.reference_kind == "none"
