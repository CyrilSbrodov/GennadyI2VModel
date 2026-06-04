from __future__ import annotations

import pytest

from core.region_ids import make_region_id
from core.schema import CanonicalRegionMemoryEntry, VideoMemory
from memory.video_memory import MemoryManager


def _entry(
    region: str,
    *,
    entity_id: str = "p1",
    observed_directly: bool = True,
    generated: bool = False,
    inferred: bool = False,
    evidence_score: float = 0.8,
    confidence: float = 0.8,
    visibility_state: str = "visible",
    reveal_lifecycle: str = "currently_visible",
    provenance: str = "parser",
) -> CanonicalRegionMemoryEntry:
    return CanonicalRegionMemoryEntry(
        record_id=make_region_id(entity_id, region),
        entity_id=entity_id,
        canonical_region=region,
        memory_kind="identity" if region in {"face", "head", "hair"} else "garment",
        mask_ref=f"mask://{entity_id}:{region}" if observed_directly else None,
        region_ref=make_region_id(entity_id, region),
        confidence=confidence,
        visibility_state=visibility_state,
        provenance=provenance,
        source_frame=0,
        evidence_score=evidence_score,
        evidence_quality="strong" if evidence_score >= 0.72 and observed_directly else ("medium" if observed_directly else "inferred"),
        observed_directly=observed_directly,
        inferred=inferred,
        generated=generated,
        freshness_frames=0,
        last_observed_frame=0 if observed_directly else None,
        reveal_lifecycle=reveal_lifecycle,
    )


def _memory_with(entry: CanonicalRegionMemoryEntry) -> VideoMemory:
    memory = VideoMemory()
    memory.canonical_region_memory[entry.record_id] = entry
    return memory


def test_observed_face_becomes_strong_identity_reference() -> None:
    manager = MemoryManager()
    face = _entry("face", observed_directly=True, generated=False, inferred=False, evidence_score=0.8, confidence=0.8)
    manager._refresh_reuse_policy(face)
    memory = _memory_with(face)

    bundle = manager.get_region_memory_bundle(memory, "p1", "face")

    assert face.reliable_as_reference is True
    assert face.reference_kind == "identity_reference"
    assert bundle.has_identity_reference is True
    assert "identity_reference_observed_strong" in bundle.retrieval_reasons


def test_generated_or_inferred_face_cannot_become_identity_reference() -> None:
    manager = MemoryManager()
    face = _entry(
        "face",
        observed_directly=False,
        generated=True,
        inferred=True,
        evidence_score=0.9,
        confidence=0.9,
        provenance="generated",
    )
    manager._refresh_reuse_policy(face)
    memory = _memory_with(face)

    bundle = manager.get_region_memory_bundle(memory, "p1", "face")

    assert face.reliable_as_reference is False
    assert face.reference_kind != "identity_reference"
    assert bundle.has_identity_reference is False
    assert "identity_reference_blocked_generated" in bundle.retrieval_reasons
    assert "identity_reference_blocked_inferred" in bundle.retrieval_reasons


def test_low_evidence_newly_revealed_face_is_not_identity_reference() -> None:
    manager = MemoryManager()
    face = _entry(
        "face",
        reveal_lifecycle="newly_revealed",
        evidence_score=0.62,
        confidence=0.82,
    )
    manager._refresh_reuse_policy(face)
    memory = _memory_with(face)

    bundle = manager.get_region_memory_bundle(memory, "p1", "face")

    assert face.reliable_as_reference is False
    assert face.reference_kind != "identity_reference"
    assert bundle.has_identity_reference is False
    assert bundle.memory_support_level != "strong"
    assert "identity_reference_blocked_low_evidence" in bundle.retrieval_reasons


def test_strong_observed_identity_is_not_overwritten_by_generated_weak_refresh() -> None:
    manager = MemoryManager()
    face = _entry("face", observed_directly=True, generated=False, inferred=False, evidence_score=0.86, confidence=0.88)
    manager._refresh_reuse_policy(face)
    memory = _memory_with(face)

    manager._refresh_canonical_memory_from_descriptor(
        memory=memory,
        entity_id="p1",
        canonical_region="face",
        source_frame=1,
        evidence_score=0.3,
        confidence=0.4,
        observed_directly=False,
        generated=True,
    )

    preserved = memory.canonical_region_memory[make_region_id("p1", "face")]
    bundle = manager.get_region_memory_bundle(memory, "p1", "face")
    assert preserved.observed_directly is True
    assert preserved.generated is False
    assert preserved.reliable_as_reference is True
    assert preserved.reference_kind == "identity_reference"
    assert bundle.has_identity_reference is True
    assert "identity_reference_observed_strong" in bundle.retrieval_reasons


def test_non_identity_garment_reference_policy_still_works() -> None:
    manager = MemoryManager()
    outer = _entry(
        "outer_garment",
        observed_directly=True,
        generated=False,
        inferred=False,
        evidence_score=0.78,
        confidence=0.82,
        provenance="parser",
    )
    manager._refresh_reuse_policy(outer)
    memory = _memory_with(outer)

    bundle = manager.get_region_memory_bundle(memory, "p1", "outer_garment")

    assert outer.reliable_as_reference is True
    assert outer.reference_kind == "garment_reference"
    assert bundle.has_garment_reference is True


@pytest.mark.parametrize("region", ["face", "head", "hair"])
@pytest.mark.parametrize("source_flag,provenance", [("generated", "generated"), ("inferred", "parser"), ("fallback", "fallback")])
def test_non_observed_identity_material_cannot_be_authoritative_memory(region: str, source_flag: str, provenance: str) -> None:
    manager = MemoryManager()
    entry = _entry(
        region,
        observed_directly=False,
        generated=(source_flag in {"generated", "fallback"}),
        inferred=(source_flag in {"inferred", "fallback"}),
        evidence_score=0.95,
        confidence=0.95,
        provenance=provenance,
    )
    manager._refresh_reuse_policy(entry)
    memory = _memory_with(entry)

    bundle = manager.get_region_memory_bundle(memory, "p1", region)

    assert entry.reliable_as_reference is False
    assert entry.reference_kind != "identity_reference"
    assert bundle.has_identity_reference is False


@pytest.mark.parametrize("region", ["face", "head", "hair"])
def test_observed_high_confidence_identity_material_can_be_authoritative_when_policy_allows(region: str) -> None:
    manager = MemoryManager()
    entry = _entry(region, observed_directly=True, generated=False, inferred=False, evidence_score=0.86, confidence=0.88)
    manager._refresh_reuse_policy(entry)
    memory = _memory_with(entry)

    bundle = manager.get_region_memory_bundle(memory, "p1", region)

    assert entry.reliable_as_reference is True
    assert entry.reference_kind == "identity_reference"
    assert bundle.has_identity_reference is True
