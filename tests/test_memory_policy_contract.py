from __future__ import annotations

import pytest

from core.region_ids import make_region_id
from core.schema import CanonicalRegionMemoryEntry, VideoMemory
from memory.memory_policy import (
    MemoryAuthority,
    MemoryFamily,
    MemoryMaterialProvenance,
    assess_memory_candidate,
    classify_memory_family,
    memory_reference_kind,
)
from memory.summaries import AppearanceMemorySummarizer
from memory.video_memory import MemoryManager


_DEFAULT_MASK = object()


def _entry(
    region: str,
    *,
    entity_id: str = "p1",
    observed_directly: bool = True,
    generated: bool = False,
    inferred: bool = False,
    confidence: float = 0.86,
    evidence_score: float = 0.86,
    visibility_state: str = "visible",
    provenance: str = "parser",
    mask_ref: str | None | object = _DEFAULT_MASK,
    reveal_lifecycle: str = "currently_visible",
) -> CanonicalRegionMemoryEntry:
    actual_mask = f"mask://{entity_id}:{region}" if mask_ref is _DEFAULT_MASK else mask_ref
    return CanonicalRegionMemoryEntry(
        record_id=make_region_id(entity_id, region),
        entity_id=entity_id,
        canonical_region=region,
        memory_kind=classify_memory_family(region).value,
        mask_ref=actual_mask,  # type: ignore[arg-type]
        region_ref=make_region_id(entity_id, region),
        confidence=confidence,
        visibility_state=visibility_state,
        provenance=provenance,
        source_frame=0,
        evidence_score=evidence_score,
        evidence_quality="strong" if observed_directly and evidence_score >= 0.72 else ("medium" if observed_directly else "inferred"),
        observed_directly=observed_directly,
        inferred=inferred,
        generated=generated,
        freshness_frames=0,
        last_observed_frame=0 if observed_directly else None,
        reveal_lifecycle=reveal_lifecycle,
        observation_status="observed" if observed_directly else ("generated" if generated else "inferred"),
        mask_evidence_type="parser_mask" if actual_mask and observed_directly else "missing",
        parser_support_level="direct",
        source_frame_kind="generated_runtime_frame" if generated else ("observed_input_frame" if observed_directly else "unknown"),
    )


def _memory_with(*entries: CanonicalRegionMemoryEntry) -> VideoMemory:
    memory = VideoMemory()
    manager = MemoryManager()
    for entry in entries:
        manager._refresh_reuse_policy(entry)
        memory.canonical_region_memory[entry.record_id] = entry
    return memory


@pytest.mark.parametrize("region", ["face", "head", "hair", "scalp"])
def test_identity_family_classification(region: str) -> None:
    assert classify_memory_family(region) == MemoryFamily.IDENTITY
    assert memory_reference_kind(MemoryFamily.IDENTITY) == "identity_reference"


@pytest.mark.parametrize("region", ["neck", "left_hand", "right_hand", "hands"])
def test_skin_family_classification(region: str) -> None:
    assert classify_memory_family(region) == MemoryFamily.SKIN


@pytest.mark.parametrize("region", ["torso", "chest", "abdomen", "pelvis", "left_arm", "right_leg"])
def test_body_shape_family_classification(region: str) -> None:
    assert classify_memory_family(region) == MemoryFamily.BODY_SHAPE


@pytest.mark.parametrize("region", ["left_breast", "right_breast", "buttocks", "hips"])
def test_soft_tissue_family_is_not_identity(region: str) -> None:
    assert classify_memory_family(region) == MemoryFamily.SOFT_TISSUE
    assert memory_reference_kind(MemoryFamily.SOFT_TISSUE) == "body_shape_reference"


@pytest.mark.parametrize("region", ["upper_garment", "lower_garment", "outer_garment", "inner_garment", "garments"])
def test_garment_family_classification(region: str) -> None:
    assert classify_memory_family(region) == MemoryFamily.GARMENT


@pytest.mark.parametrize("region", ["external_genital_region", "male_external_genital_region", "female_pelvic_region", "male_pelvic_region"])
def test_private_family_is_no_reference(region: str) -> None:
    assert classify_memory_family(region) == MemoryFamily.PRIVATE
    assessment = assess_memory_candidate(
        canonical_region=region,
        confidence=0.95,
        evidence_score=0.95,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="parser",
        visibility_state="visible",
        mask_ref="mask://private",
    )
    assert assessment.reference_kind == "none"
    assert assessment.can_seed_identity is False


def test_unknown_region_classifies_unknown() -> None:
    assert classify_memory_family("unknown_region") == MemoryFamily.UNKNOWN


@pytest.mark.parametrize("region", ["face", "hair"])
def test_observed_high_confidence_identity_can_be_authoritative(region: str) -> None:
    entry = _entry(region)
    memory = _memory_with(entry)
    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", region)
    assert entry.authority == "authoritative"
    assert entry.can_seed_identity is True
    assert bundle.has_identity_reference is True


@pytest.mark.parametrize(
    "provenance,generated,inferred,material",
    [
        ("generated", True, True, MemoryMaterialProvenance.GENERATED),
        ("parser", False, True, MemoryMaterialProvenance.INFERRED),
        ("fallback", True, True, MemoryMaterialProvenance.FALLBACK),
        ("training_synthetic", False, False, MemoryMaterialProvenance.SYNTHETIC),
    ],
)
def test_unsafe_identity_material_is_not_authoritative(provenance: str, generated: bool, inferred: bool, material: MemoryMaterialProvenance) -> None:
    assessment = assess_memory_candidate(
        canonical_region="face",
        confidence=0.99,
        evidence_score=0.99,
        observed_directly=not inferred and material != MemoryMaterialProvenance.SYNTHETIC,
        generated=generated,
        inferred=inferred,
        provenance=provenance,
        visibility_state="visible",
        mask_ref="mask://face",
    )
    assert assessment.material_provenance == material
    assert assessment.authority != MemoryAuthority.AUTHORITATIVE
    assert assessment.can_seed_identity is False


def test_hidden_unknown_face_without_mask_is_not_authoritative() -> None:
    entry = _entry("face", observed_directly=False, inferred=True, visibility_state="unknown_expected_region", mask_ref=None, confidence=0.9, evidence_score=0.9)
    memory = _memory_with(entry)
    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "face")
    assert entry.authority in {"diagnostic_only", "rejected"}
    assert bundle.has_identity_reference is False
    assert "hidden_cannot_be_authoritative" in bundle.retrieval_reasons or "material:hidden" in bundle.retrieval_reasons


def test_authoritative_face_not_overwritten_by_generated_or_lower_confidence_observed() -> None:
    manager = MemoryManager()
    strong = _entry("face", confidence=0.95, evidence_score=0.95)
    manager._refresh_reuse_policy(strong)
    memory = _memory_with(strong)
    generated = _entry("face", observed_directly=False, generated=True, inferred=True, provenance="generated", confidence=0.99, evidence_score=0.99, mask_ref="mask://generated")
    generated.source_frame = 1
    manager._refresh_reuse_policy(generated)
    manager._upsert_canonical_memory(memory, generated)
    lower = _entry("face", confidence=0.66, evidence_score=0.71)
    lower.source_frame = 2
    manager._refresh_reuse_policy(lower)
    manager._upsert_canonical_memory(memory, lower)
    preserved = memory.canonical_region_memory[make_region_id("p1", "face")]
    assert preserved.confidence == pytest.approx(0.95)
    assert preserved.generated is False
    assert preserved.authority == "authoritative"


def test_authoritative_hair_not_overwritten_by_fallback() -> None:
    manager = MemoryManager()
    hair = _entry("hair", confidence=0.91, evidence_score=0.91)
    manager._refresh_reuse_policy(hair)
    memory = _memory_with(hair)
    fallback = _entry("hair", observed_directly=False, generated=True, inferred=True, provenance="fallback", confidence=0.99, evidence_score=0.99, mask_ref="mask://fallback")
    fallback.source_frame = 1
    manager._refresh_reuse_policy(fallback)
    manager._upsert_canonical_memory(memory, fallback)
    assert memory.canonical_region_memory[hair.record_id].provenance == "parser"
    assert memory.canonical_region_memory[hair.record_id].authority == "authoritative"


def test_generated_garment_cannot_overwrite_observed_garment_reference() -> None:
    manager = MemoryManager()
    observed = _entry("upper_garment", confidence=0.9, evidence_score=0.9)
    manager._refresh_reuse_policy(observed)
    memory = _memory_with(observed)
    generated = _entry("upper_garment", observed_directly=False, generated=True, inferred=True, provenance="generated", confidence=0.99, evidence_score=0.99, mask_ref="mask://generated")
    generated.source_frame = 1
    manager._refresh_reuse_policy(generated)
    manager._upsert_canonical_memory(memory, generated)
    kept = memory.canonical_region_memory[observed.record_id]
    assert kept.provenance == "parser"
    assert kept.reference_kind == "garment_reference"


def test_body_shape_and_soft_tissue_seed_non_identity_appearance_only_when_observed() -> None:
    manager = MemoryManager()
    torso = _entry("torso")
    soft = _entry("left_breast")
    hidden_soft = _entry("right_breast", observed_directly=False, inferred=True, visibility_state="hidden_by_garment", mask_ref=None)
    memory = _memory_with(torso, soft, hidden_soft)
    assert manager.get_region_memory_bundle(memory, "p1", "torso").has_body_shape_reference is True
    soft_bundle = manager.get_region_memory_bundle(memory, "p1", "left_breast")
    assert soft_bundle.has_body_shape_reference is True
    assert soft_bundle.has_identity_reference is False
    hidden_bundle = manager.get_region_memory_bundle(memory, "p1", "right_breast")
    assert hidden_bundle.has_body_shape_reference is False


def test_private_regions_excluded_from_authoritative_retrieval() -> None:
    private = _entry("male_external_genital_region", confidence=0.99, evidence_score=0.99)
    memory = _memory_with(private)
    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "male_external_genital_region")
    assert private.memory_family == "private"
    assert private.reference_kind == "none"
    assert bundle.has_identity_reference is False
    assert bundle.reference_payloads == []
    assert "private_region_no_reference" in bundle.retrieval_reasons


def test_retrieval_returns_family_specific_bundles_and_missing_trace() -> None:
    face = _entry("face")
    garment = _entry("upper_garment")
    soft = _entry("left_breast")
    private = _entry("male_external_genital_region")
    memory = _memory_with(face, garment, soft, private)
    manager = MemoryManager()
    assert manager.get_region_memory_bundle(memory, "p1", "face").has_identity_reference is True
    assert manager.get_region_memory_bundle(memory, "p1", "upper_garment").has_garment_reference is True
    assert manager.get_region_memory_bundle(memory, "p1", "left_breast").has_body_shape_reference is True
    assert manager.get_region_memory_bundle(memory, "p1", "male_external_genital_region").reference_payloads == []
    missing = manager.get_region_memory_bundle(memory, "p1", "nose")
    assert missing.policy_decision == "missing"
    assert "missing_memory_entry" in missing.policy_reasons


def test_memory_summary_counts_policy_status() -> None:
    authoritative = _entry("face")
    generated = _entry("hair", observed_directly=False, generated=True, inferred=True, provenance="generated", mask_ref="mask://generated")
    private = _entry("male_external_genital_region")
    memory = _memory_with(authoritative, generated, private)
    summary = AppearanceMemorySummarizer().summarize(memory).as_dict()
    assert summary["policy_counts"]["authoritative_identity_entries"] == 1
    assert summary["policy_counts"]["generated_inferred_fallback_entries"] == 1
    assert summary["policy_counts"]["private_no_reference_entries"] == 1
    assert summary["canonical_regions"][make_region_id("p1", "male_external_genital_region")]["memory_family"] == "private"


def test_learned_bridge_does_not_synthesize_missing_identity_embedding() -> None:
    from representation.learned_bridge import BaselineIdentityAppearanceEncoder

    assert BaselineIdentityAppearanceEncoder().encode_identity({"identity": {}}, "missing") == []


def _canonical_payload(
    canonical_name: str,
    *,
    visibility_state: str = "visible",
    confidence: float = 0.95,
    provenance: str = "parser",
    mask_ref: str | None = None,
    applicability: str = "applicable",
    observation_status: str = "observed",
    mask_evidence_type: str = "parser_mask",
    parser_support_level: str = "direct",
    source_frame_kind: str = "observed_input_frame",
    generated: bool = False,
) -> dict[str, object]:
    return {
        "canonical_name": canonical_name,
        "raw_sources": [canonical_name],
        "source_regions": [canonical_name],
        "mask_ref": mask_ref,
        "confidence": confidence,
        "visibility_state": visibility_state,
        "provenance": provenance,
        "applicability": applicability,
        "observation_status": observation_status,
        "mask_evidence_type": mask_evidence_type,
        "parser_support_level": parser_support_level,
        "source_frame_kind": source_frame_kind,
        "generated": generated,
        "last_update_source": provenance,
    }


def _scene_with_canonical(canonical_regions: dict[str, dict[str, object]]):
    from core.schema import BBox, PersonNode, SceneGraph

    return SceneGraph(
        frame_index=0,
        persons=[
            PersonNode(
                person_id="p1",
                track_id="t1",
                bbox=BBox(0.1, 0.1, 0.8, 0.8),
                mask_ref="mask://p1",
                confidence=0.95,
                canonical_regions=canonical_regions,
            )
        ],
    )


def test_ears_remain_identity_and_neck_is_skin_not_authoritative_identity() -> None:
    assert classify_memory_family("left_ear") == MemoryFamily.IDENTITY
    assert classify_memory_family("right_ear") == MemoryFamily.IDENTITY
    assert classify_memory_family("neck") == MemoryFamily.SKIN

    neck = _entry("neck", confidence=0.95, evidence_score=0.95, mask_ref="mask://neck")
    memory = _memory_with(neck)
    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "neck")
    assert neck.memory_family == "skin"
    assert neck.authority != "authoritative"
    assert bundle.has_identity_reference is False
    assert bundle.has_skin_reference is True


def test_unknown_applicability_without_mask_does_not_create_reusable_memory_from_canonical_path() -> None:
    scene = _scene_with_canonical(
        {
            "face": _canonical_payload(
                "face",
                mask_ref=None,
                applicability="unknown_applicability",
                observation_status="unknown",
                mask_evidence_type="missing",
                confidence=0.95,
            )
        }
    )
    memory = MemoryManager().initialize_from_scene(scene)
    assert make_region_id("p1", "face") not in memory.canonical_region_memory


def test_unsupported_by_current_parser_without_mask_is_rejected_by_policy() -> None:
    assessment = assess_memory_candidate(
        canonical_region="face",
        confidence=0.95,
        evidence_score=0.95,
        observed_directly=False,
        generated=False,
        inferred=True,
        provenance="parser",
        visibility_state="visible",
        mask_ref=None,
        applicability="unsupported_by_current_parser",
        observation_status="unknown",
        mask_evidence_type="missing",
        parser_support_level="unsupported",
    )
    assert assessment.authority == MemoryAuthority.REJECTED
    assert assessment.material_provenance == MemoryMaterialProvenance.UNSUPPORTED


def test_observed_parser_mask_can_seed_allowed_memory_but_bbox_projection_cannot_author_identity() -> None:
    torso = assess_memory_candidate(
        canonical_region="torso",
        confidence=0.9,
        evidence_score=0.9,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="parser",
        visibility_state="visible",
        mask_ref="mask://torso",
        observation_status="observed",
        mask_evidence_type="parser_mask",
    )
    assert torso.authority == MemoryAuthority.REUSABLE
    assert torso.can_seed_appearance is True

    face_bbox = assess_memory_candidate(
        canonical_region="face",
        confidence=0.99,
        evidence_score=0.99,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="detector",
        visibility_state="visible",
        mask_ref="bbox://face",
        observation_status="observed",
        mask_evidence_type="bbox_projection",
    )
    assert face_bbox.authority != MemoryAuthority.AUTHORITATIVE
    assert face_bbox.can_seed_identity is False


def test_detector_face_material_is_not_authoritative_unless_face_specific() -> None:
    detector = assess_memory_candidate(
        canonical_region="face",
        confidence=0.99,
        evidence_score=0.99,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="detector",
        visibility_state="visible",
        mask_ref="mask://face",
        observation_status="observed",
        mask_evidence_type="detector_mask",
    )
    assert detector.material_provenance == MemoryMaterialProvenance.OBSERVED_DETECTOR
    assert detector.authority != MemoryAuthority.AUTHORITATIVE
    assert detector.can_seed_identity is False

    parser_face = assess_memory_candidate(
        canonical_region="face",
        confidence=0.9,
        evidence_score=0.9,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="parser",
        visibility_state="visible",
        mask_ref="mask://face",
        observation_status="observed",
        mask_evidence_type="parser_mask",
    )
    assert parser_face.authority == MemoryAuthority.AUTHORITATIVE

    face_specific = assess_memory_candidate(
        canonical_region="face",
        confidence=0.9,
        evidence_score=0.9,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="detector",
        visibility_state="visible",
        mask_ref="mask://face",
        observation_status="observed",
        mask_evidence_type="face_detector_mask",
    )
    assert face_specific.authority == MemoryAuthority.AUTHORITATIVE


def test_canonical_observed_high_confidence_face_becomes_authoritative_identity_memory() -> None:
    scene = _scene_with_canonical({"face": _canonical_payload("face", mask_ref="mask://p1:face")})
    memory = MemoryManager().initialize_from_scene(scene)
    entry = memory.canonical_region_memory[make_region_id("p1", "face")]
    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "face")
    assert entry.authority == "authoritative"
    assert entry.observation_status == "observed"
    assert entry.mask_evidence_type == "parser_mask"
    assert bundle.has_identity_reference is True


def test_canonical_generated_face_does_not_become_identity_reference() -> None:
    scene = _scene_with_canonical(
        {
            "face": _canonical_payload(
                "face",
                mask_ref="mask://generated-face",
                provenance="generated",
                observation_status="generated",
                source_frame_kind="generated_runtime_frame",
                generated=True,
            )
        }
    )
    memory = MemoryManager().initialize_from_scene(scene)
    entry = memory.canonical_region_memory[make_region_id("p1", "face")]
    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "face")
    assert entry.material_provenance == "generated"
    assert entry.authority != "authoritative"
    assert bundle.has_identity_reference is False


def test_canonical_hidden_unknown_face_without_mask_is_not_authoritative() -> None:
    scene = _scene_with_canonical(
        {
            "face": _canonical_payload(
                "face",
                visibility_state="unknown_expected_region",
                mask_ref=None,
                observation_status="unknown",
                mask_evidence_type="missing",
                confidence=0.95,
            )
        }
    )
    memory = MemoryManager().initialize_from_scene(scene)
    entry = memory.canonical_region_memory[make_region_id("p1", "face")]
    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "face")
    assert entry.authority in {"diagnostic_only", "rejected"}
    assert bundle.has_identity_reference is False


def test_canonical_private_observed_region_stays_no_reference_in_retrieval() -> None:
    scene = _scene_with_canonical(
        {
            "male_external_genital_region": _canonical_payload(
                "male_external_genital_region",
                mask_ref="mask://private",
                parser_support_level="unsupported",
                source_frame_kind="observed_input_frame",
            )
        }
    )
    memory = MemoryManager().initialize_from_scene(scene)
    entry = memory.canonical_region_memory[make_region_id("p1", "male_external_genital_region")]
    bundle = MemoryManager().get_region_memory_bundle(memory, "p1", "male_external_genital_region")
    assert entry.memory_family == "private"
    assert entry.reference_kind == "none"
    assert bundle.reference_payloads == []
    assert bundle.has_identity_reference is False


@pytest.mark.parametrize("mask_evidence_type", ["missing", "unknown"])
def test_identity_candidate_with_missing_or_unknown_mask_evidence_is_not_authoritative(mask_evidence_type: str) -> None:
    assessment = assess_memory_candidate(
        canonical_region="face",
        confidence=0.99,
        evidence_score=0.99,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="parser",
        visibility_state="visible",
        mask_ref="mask://face",
        observation_status="observed",
        mask_evidence_type=mask_evidence_type,
    )

    assert assessment.authority != MemoryAuthority.AUTHORITATIVE
    assert assessment.can_seed_identity is False
    assert assessment.reference_kind != "identity_reference" or assessment.can_seed_identity is False


def test_identity_candidate_with_parser_mask_remains_authoritative() -> None:
    assessment = assess_memory_candidate(
        canonical_region="face",
        confidence=0.9,
        evidence_score=0.9,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="parser",
        visibility_state="visible",
        mask_ref="mask://face",
        observation_status="observed",
        mask_evidence_type="parser_mask",
    )

    assert assessment.authority == MemoryAuthority.AUTHORITATIVE
    assert assessment.can_seed_identity is True
    assert assessment.reference_kind == "identity_reference"


@pytest.mark.parametrize(
    "mask_evidence_type,expected_authoritative",
    [("detector_mask", False), ("face_detector_mask", True)],
)
def test_detector_identity_mask_requires_explicit_face_specific_evidence(mask_evidence_type: str, expected_authoritative: bool) -> None:
    assessment = assess_memory_candidate(
        canonical_region="face",
        confidence=0.9,
        evidence_score=0.9,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="detector",
        visibility_state="visible",
        mask_ref="mask://face",
        observation_status="observed",
        mask_evidence_type=mask_evidence_type,
    )

    if expected_authoritative:
        assert assessment.authority == MemoryAuthority.AUTHORITATIVE
        assert assessment.can_seed_identity is True
    else:
        assert assessment.authority != MemoryAuthority.AUTHORITATIVE
        assert assessment.can_seed_identity is False
