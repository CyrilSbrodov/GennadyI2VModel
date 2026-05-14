from __future__ import annotations

from core.region_ids import make_region_id
from core.schema import CanonicalRegionMemoryEntry, TexturePatchMemory, VideoMemory
from memory.video_memory import MemoryManager


def _entry(region: str, kind: str, **overrides: object) -> CanonicalRegionMemoryEntry:
    values = dict(
        record_id=f"mem::p1:{region}",
        entity_id="p1",
        canonical_region=region,
        memory_kind="canonical_region",
        confidence=0.86,
        provenance="parser",
        source_frame=0,
        evidence_score=0.82,
        evidence_quality="strong",
        observed_directly=True,
        inferred=False,
        generated=False,
        reliable_for_reuse=True,
        reliable_as_reference=True,
        reference_kind=kind,
        reveal_lifecycle="stable",
    )
    values.update(overrides)
    return CanonicalRegionMemoryEntry(**values)


def test_strong_observed_face_builds_identity_reference_payload() -> None:
    memory = VideoMemory()
    memory.canonical_region_memory[make_region_id("p1", "face")] = _entry("face", "identity_reference")
    memory.texture_patches["patch::p1:face:0"] = TexturePatchMemory(
        patch_id="patch::p1:face:0",
        region_type="face",
        entity_id="p1",
        source_frame=0,
        patch_ref="roi://0,0,8,8",
        confidence=0.8,
        descriptor={"mean": [0.1, 0.2, 0.3], "std": [0.01, 0.02, 0.03]},
        evidence_score=0.78,
    )

    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "face")

    payload = bundle.identity_reference_payload
    assert payload is not None
    assert payload.reference_kind == "identity_reference"
    assert payload.patch_id == "patch::p1:face:0"
    assert payload.patch_ref == "roi://0,0,8,8"
    assert payload.observed_directly is True
    assert payload.generated is False
    assert payload.inferred is False
    assert "identity_reference_payload_available" in bundle.retrieval_reasons
    assert payload in bundle.reference_payloads


def test_generated_face_does_not_build_trusted_reference_payload() -> None:
    memory = VideoMemory()
    memory.canonical_region_memory[make_region_id("p1", "face")] = _entry(
        "face",
        "identity_reference",
        generated=True,
        inferred=True,
        observed_directly=False,
        reliable_as_reference=False,
        provenance="generated",
    )

    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "face")

    assert bundle.identity_reference_payload is None
    assert bundle.reference_payloads == []


def test_strong_torso_builds_body_shape_payload_not_identity_payload() -> None:
    memory = VideoMemory()
    memory.canonical_region_memory[make_region_id("p1", "torso")] = _entry("torso", "body_shape_reference")
    memory.texture_patches["patch::p1:torso:0"] = TexturePatchMemory(
        patch_id="patch::p1:torso:0",
        region_type="torso",
        entity_id="p1",
        source_frame=0,
        patch_ref="roi://0,0,8,8",
        confidence=0.8,
        descriptor={"shape": [0.3, 0.7]},
        evidence_score=0.77,
    )

    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "torso")

    assert bundle.body_shape_reference_payload is not None
    assert bundle.body_shape_reference_payload.reference_kind == "body_shape_reference"
    assert bundle.identity_reference_payload is None


def test_payload_without_patch_cache_is_diagnosed() -> None:
    memory = VideoMemory()
    memory.canonical_region_memory[make_region_id("p1", "torso")] = _entry("torso", "body_shape_reference")

    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "torso")

    payload = bundle.body_shape_reference_payload
    assert payload is not None
    assert payload.patch_id is None
    assert payload.patch_ref is None
    assert "body_shape_reference_payload_without_patch_cache" in bundle.retrieval_reasons
    assert "body_shape_reference_payload_without_patch_cache" in payload.retrieval_reasons


def test_preserve_identity_transition_does_not_bypass_trust_policy() -> None:
    memory = VideoMemory()
    memory.canonical_region_memory[make_region_id("p1", "face")] = _entry(
        "face",
        "none",
        confidence=0.9,
        evidence_score=0.9,
        observed_directly=True,
        generated=False,
        inferred=False,
        reliable_as_reference=False,
        last_transition="preserve_identity_on_occlusion",
    )

    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "face")

    assert bundle.has_identity_reference is False
    assert bundle.identity_reference is None
    assert bundle.identity_reference_payload is None
    assert "identity_reference_available" not in bundle.retrieval_reasons
