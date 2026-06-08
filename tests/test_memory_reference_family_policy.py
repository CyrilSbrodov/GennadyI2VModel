from __future__ import annotations

from core.region_ids import make_region_id
from core.schema import BBox, CanonicalRegionMemoryEntry, PersonNode, SceneGraph, MemoryEntry, VideoMemory
from memory.video_memory import MemoryManager


def _entry(
    region: str,
    *,
    entity_id: str = "p1",
    observed_directly: bool = True,
    generated: bool = False,
    inferred: bool = False,
    evidence_score: float = 0.75,
    confidence: float = 0.75,
    visibility_state: str = "visible",
    reveal_lifecycle: str = "currently_visible",
) -> CanonicalRegionMemoryEntry:
    return CanonicalRegionMemoryEntry(
        record_id=make_region_id(entity_id, region),
        entity_id=entity_id,
        canonical_region=region,
        memory_kind="body",
        mask_ref=f"mask://{entity_id}:{region}" if observed_directly else None,
        region_ref=make_region_id(entity_id, region),
        confidence=confidence,
        visibility_state=visibility_state,
        provenance="generated" if generated else "parser",
        source_frame=0,
        evidence_score=evidence_score,
        evidence_quality="strong" if evidence_score >= 0.72 and observed_directly else ("medium" if observed_directly else "inferred"),
        observed_directly=observed_directly,
        inferred=inferred,
        generated=generated,
        freshness_frames=0,
        last_observed_frame=0 if observed_directly else None,
        reveal_lifecycle=reveal_lifecycle,
        observation_status="observed" if observed_directly else ("generated" if generated else "inferred"),
        mask_evidence_type="parser_mask" if observed_directly else "missing",
        parser_support_level="direct",
        source_frame_kind="generated_runtime_frame" if generated else ("observed_input_frame" if observed_directly else "unknown"),
    )


def _memory_with(entry: CanonicalRegionMemoryEntry) -> VideoMemory:
    memory = VideoMemory()
    memory.canonical_region_memory[entry.record_id] = entry
    return memory


def test_torso_becomes_body_shape_reference_not_identity_reference() -> None:
    manager = MemoryManager()
    torso = _entry("torso", observed_directly=True, generated=False, inferred=False, evidence_score=0.75, confidence=0.75)
    manager._refresh_reuse_policy(torso)

    bundle = manager.get_region_memory_bundle(_memory_with(torso), "p1", "torso")

    assert torso.reliable_as_reference is True
    assert torso.reference_kind == "body_shape_reference"
    assert bundle.has_identity_reference is False
    assert bundle.has_body_shape_reference is True
    assert "body_shape_reference_available" in bundle.retrieval_reasons
    assert "body_shape_reference_observed_strong" in bundle.retrieval_reasons
    assert "identity_reference_available" not in bundle.retrieval_reasons


def test_generated_torso_is_blocked_as_body_shape_reference() -> None:
    manager = MemoryManager()
    torso = _entry("torso", observed_directly=False, generated=True, inferred=True, evidence_score=0.9, confidence=0.9)
    manager._refresh_reuse_policy(torso)

    bundle = manager.get_region_memory_bundle(_memory_with(torso), "p1", "torso")

    assert torso.reliable_as_reference is False
    assert torso.reference_kind == "body_shape_reference"
    assert bundle.has_body_shape_reference is False
    assert "body_shape_reference_blocked_generated" in bundle.retrieval_reasons
    assert "body_shape_reference_blocked_inferred" in bundle.retrieval_reasons


def test_hand_and_neck_become_skin_reference_not_identity_reference() -> None:
    manager = MemoryManager()
    for region in ("left_hand", "neck"):
        entry = _entry(region, observed_directly=True, generated=False, inferred=False, evidence_score=0.72, confidence=0.72)
        manager._refresh_reuse_policy(entry)
        bundle = manager.get_region_memory_bundle(_memory_with(entry), "p1", region)

        assert entry.reliable_as_reference is True
        assert entry.reference_kind == "skin_reference"
        assert bundle.has_skin_reference is True
        assert bundle.has_identity_reference is False
        assert "skin_reference_observed_strong" in bundle.retrieval_reasons
        assert "identity_reference_available" not in bundle.retrieval_reasons


def test_left_arm_becomes_body_shape_reference_not_skin_reference() -> None:
    manager = MemoryManager()
    arm = _entry("left_arm", observed_directly=True, generated=False, inferred=False, evidence_score=0.75, confidence=0.75)
    manager._refresh_reuse_policy(arm)

    bundle = manager.get_region_memory_bundle(_memory_with(arm), "p1", "left_arm")

    assert arm.reliable_as_reference is True
    assert arm.reference_kind == "body_shape_reference"
    assert bundle.has_body_shape_reference is True
    assert bundle.has_skin_reference is False
    assert "body_shape_reference_observed_strong" in bundle.retrieval_reasons
    assert "skin_reference_available" not in bundle.retrieval_reasons


def test_outer_garment_remains_garment_reference_not_identity_reference() -> None:
    manager = MemoryManager()
    outer = _entry("outer_garment", observed_directly=True, generated=False, inferred=False, evidence_score=0.78, confidence=0.82)
    manager._refresh_reuse_policy(outer)

    bundle = manager.get_region_memory_bundle(_memory_with(outer), "p1", "outer_garment")

    assert outer.reliable_as_reference is True
    assert outer.reference_kind == "garment_reference"
    assert bundle.has_garment_reference is True
    assert bundle.has_identity_reference is False
    assert "garment_reference_observed_strong" in bundle.retrieval_reasons


def test_identity_embedding_does_not_update_from_non_core_torso_region() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    original = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]
    memory.identity_memory["p1"] = MemoryEntry(entity_id="p1", entry_type="identity", embedding=list(original), confidence=0.9)
    scene = SceneGraph(
        frame_index=1,
        persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None, confidence=0.95)],
    )
    frame = [[[0.2 + (x / 255.0), 0.3, 0.4] for x in range(16)] for _ in range(16)]
    manager._semantic_region_types = lambda person: ["torso"]  # type: ignore[method-assign]

    manager.update_from_frame(memory, frame, scene, transition_context={"frame_source": "camera"})

    assert memory.identity_memory["p1"].embedding == original
    assert make_region_id("p1", "torso") in memory.canonical_region_memory
    assert memory.texture_patches
