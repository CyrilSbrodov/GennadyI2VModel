from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.reference_families import reference_family_for_region


class MemoryFamily(str, Enum):
    IDENTITY = "identity"
    SKIN = "skin"
    BODY_SHAPE = "body_shape"
    SOFT_TISSUE = "soft_tissue"
    GARMENT = "garment"
    ACCESSORY = "accessory"
    PRIVATE = "private"
    UNKNOWN = "unknown"


class MemoryAuthority(str, Enum):
    AUTHORITATIVE = "authoritative"
    REUSABLE = "reusable"
    WEAK = "weak"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    REJECTED = "rejected"


class MemoryMaterialProvenance(str, Enum):
    OBSERVED_INPUT = "observed_input"
    OBSERVED_PARSER = "observed_parser"
    OBSERVED_DETECTOR = "observed_detector"
    OBSERVED_FACE = "observed_face"
    MEMORY_REUSE = "memory_reuse"
    GENERATED = "generated"
    INFERRED = "inferred"
    FALLBACK = "fallback"
    SYNTHETIC = "synthetic"
    HIDDEN = "hidden"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class MemoryUsePermission:
    can_seed_identity: bool = False
    can_seed_appearance: bool = False
    can_seed_reveal: bool = False
    can_overwrite_authoritative: bool = False


@dataclass(frozen=True, slots=True)
class MemoryCandidateAssessment:
    memory_family: MemoryFamily
    reference_kind: str
    authority: MemoryAuthority
    material_provenance: MemoryMaterialProvenance
    permission: MemoryUsePermission
    policy_decision: str
    policy_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def can_seed_identity(self) -> bool:
        return self.permission.can_seed_identity

    @property
    def can_seed_appearance(self) -> bool:
        return self.permission.can_seed_appearance

    @property
    def can_seed_reveal(self) -> bool:
        return self.permission.can_seed_reveal

    @property
    def can_overwrite_authoritative(self) -> bool:
        return self.permission.can_overwrite_authoritative


IDENTITY_MEMORY_REGIONS = frozenset(
    {
        "face",
        "head",
        "hair",
        "scalp",
        "forehead",
        "left_eye",
        "right_eye",
        "nose",
        "mouth",
        "lips",
        "jaw",
        "chin",
    }
)

_NON_AUTHORITATIVE_PROVENANCE = frozenset(
    {
        MemoryMaterialProvenance.GENERATED,
        MemoryMaterialProvenance.INFERRED,
        MemoryMaterialProvenance.FALLBACK,
        MemoryMaterialProvenance.SYNTHETIC,
        MemoryMaterialProvenance.HIDDEN,
        MemoryMaterialProvenance.UNKNOWN,
        MemoryMaterialProvenance.UNSUPPORTED,
    }
)

_OBSERVED_PROVENANCE = frozenset(
    {
        MemoryMaterialProvenance.OBSERVED_INPUT,
        MemoryMaterialProvenance.OBSERVED_PARSER,
        MemoryMaterialProvenance.OBSERVED_DETECTOR,
        MemoryMaterialProvenance.OBSERVED_FACE,
    }
)

DIRECT_OBSERVED_MASK_EVIDENCE_TYPES = frozenset(
    {
        "parser_mask",
        "face_mask",
        "face_parser_mask",
        "face_detector_mask",
        "detector_instance_mask",
        "observed_mask",
        "input_mask",
    }
)


def classify_memory_family(canonical_region: str) -> MemoryFamily:
    region = str(canonical_region or "").strip()
    family = "identity" if region in IDENTITY_MEMORY_REGIONS else reference_family_for_region(region)
    try:
        return MemoryFamily(family)
    except ValueError:
        return MemoryFamily.UNKNOWN


def memory_reference_kind(family: MemoryFamily) -> str:
    if family == MemoryFamily.PRIVATE:
        return "none"
    if family == MemoryFamily.SOFT_TISSUE:
        return "body_shape_reference"
    if family == MemoryFamily.UNKNOWN:
        return "none"
    return {
        MemoryFamily.IDENTITY: "identity_reference",
        MemoryFamily.SKIN: "skin_reference",
        MemoryFamily.BODY_SHAPE: "body_shape_reference",
        MemoryFamily.GARMENT: "garment_reference",
        MemoryFamily.ACCESSORY: "accessory_reference",
    }.get(family, "none")


def material_provenance_from_flags(
    *,
    provenance: str,
    observed_directly: bool,
    generated: bool,
    inferred: bool,
    visibility_state: str,
    mask_ref: str | None,
    applicability: str = "applicable",
    observation_status: str = "unknown",
    mask_evidence_type: str = "missing",
    parser_support_level: str = "unknown",
    source_frame_kind: str = "unknown",
) -> MemoryMaterialProvenance:
    text = " ".join(str(v or "").strip().lower() for v in (provenance, source_frame_kind, observation_status, mask_evidence_type, parser_support_level))
    visibility = str(visibility_state or "").strip().lower()
    applicability = str(applicability or "unknown").strip().lower()
    observation = str(observation_status or "unknown").strip().lower()
    mask_kind = str(mask_evidence_type or "missing").strip().lower()
    parser_support = str(parser_support_level or "unknown").strip().lower()
    if applicability in {"unsupported_by_current_parser", "not_applicable", "unknown_applicability"} and not mask_ref:
        return MemoryMaterialProvenance.UNSUPPORTED
    if parser_support == "unsupported" and not mask_ref:
        return MemoryMaterialProvenance.UNSUPPORTED
    if "fallback" in text or observation == "fallback":
        return MemoryMaterialProvenance.FALLBACK
    if "training_synthetic" in text or "synthetic" in text:
        return MemoryMaterialProvenance.SYNTHETIC
    if generated or observation == "generated" or any(marker in text for marker in ("generated", "renderer", "rendered", "model_output", "compositor")):
        return MemoryMaterialProvenance.GENERATED
    if mask_kind in {"bbox_projection", "bbox"}:
        return MemoryMaterialProvenance.INFERRED
    if visibility in {"hidden", "hidden_by_self", "hidden_by_garment", "hidden_by_object", "out_of_frame", "unknown_expected_region"} and not observed_directly:
        return MemoryMaterialProvenance.HIDDEN
    if inferred or observation == "inferred" or not observed_directly:
        return MemoryMaterialProvenance.INFERRED
    if observation not in {"observed", "single_frame_anchor", "multi_frame_tracked", "tracker_single_frame_observed"} and observation not in {"", "unknown"}:
        return MemoryMaterialProvenance.UNKNOWN
    if "memory" in text:
        return MemoryMaterialProvenance.MEMORY_REUSE
    if "face" in str(mask_evidence_type or "").lower() or "observed_face" in str(source_frame_kind or "").lower() or str(provenance or "").lower().startswith("face"):
        return MemoryMaterialProvenance.OBSERVED_FACE
    if mask_kind == "parser_mask" or any(marker in str(provenance or "").lower() for marker in ("parser", "mask")):
        return MemoryMaterialProvenance.OBSERVED_PARSER
    if any(marker in str(provenance or "").lower() for marker in ("detector", "segformer", "schp", "fashn")):
        return MemoryMaterialProvenance.OBSERVED_DETECTOR
    if any(marker in str(provenance or "").lower() or marker in str(source_frame_kind or "").lower() for marker in ("input", "camera", "frame_observation")):
        return MemoryMaterialProvenance.OBSERVED_INPUT
    return MemoryMaterialProvenance.UNKNOWN


def assess_memory_candidate(
    *,
    canonical_region: str,
    confidence: float,
    evidence_score: float,
    observed_directly: bool,
    generated: bool,
    inferred: bool,
    provenance: str,
    visibility_state: str,
    mask_ref: str | None,
    applicability: str = "applicable",
    observation_status: str = "unknown",
    mask_evidence_type: str = "missing",
    parser_support_level: str = "unknown",
    reveal_lifecycle: str = "unknown",
    source_frame_kind: str = "unknown",
) -> MemoryCandidateAssessment:
    family = classify_memory_family(canonical_region)
    ref_kind = memory_reference_kind(family)
    material = material_provenance_from_flags(
        provenance=provenance,
        observed_directly=observed_directly,
        generated=generated,
        inferred=inferred,
        visibility_state=visibility_state,
        mask_ref=mask_ref,
        applicability=applicability,
        observation_status=observation_status,
        mask_evidence_type=mask_evidence_type,
        parser_support_level=parser_support_level,
        source_frame_kind=source_frame_kind,
    )
    reasons: list[str] = [f"family:{family.value}", f"material:{material.value}"]
    visibility_ok = str(visibility_state) in {"visible", "partially_visible"}
    mask_kind = str(mask_evidence_type or "missing").strip().lower()
    observed_material = material in _OBSERVED_PROVENANCE and observed_directly and bool(mask_ref)
    direct_mask_evidence = bool(mask_ref) and mask_kind in DIRECT_OBSERVED_MASK_EVIDENCE_TYPES
    face_specific_evidence = mask_kind in {"face_mask", "face_parser_mask", "face_detector_mask"} or "observed_face" in str(source_frame_kind or "").lower() or str(provenance or "").lower().startswith("face")
    identity_authoritative_material = material in {MemoryMaterialProvenance.OBSERVED_FACE, MemoryMaterialProvenance.OBSERVED_PARSER, MemoryMaterialProvenance.OBSERVED_INPUT} and direct_mask_evidence
    identity_authoritative_material = identity_authoritative_material or (material == MemoryMaterialProvenance.OBSERVED_DETECTOR and face_specific_evidence and direct_mask_evidence)
    unsafe_material = material in _NON_AUTHORITATIVE_PROVENANCE
    if family == MemoryFamily.UNKNOWN:
        reasons.append("unknown_region_no_reference")
        return MemoryCandidateAssessment(family, "none", MemoryAuthority.REJECTED, material, MemoryUsePermission(), "rejected", tuple(reasons))
    if family == MemoryFamily.PRIVATE:
        reasons.append("private_region_no_reference")
        authority = MemoryAuthority.DIAGNOSTIC_ONLY if observed_material else MemoryAuthority.REJECTED
        return MemoryCandidateAssessment(family, "none", authority, material, MemoryUsePermission(), authority.value, tuple(reasons))
    if unsafe_material:
        reasons.append(f"{material.value}_cannot_be_authoritative")
        authority = MemoryAuthority.DIAGNOSTIC_ONLY if mask_ref or generated else MemoryAuthority.REJECTED
        unsafe_ref_kind = "none" if family == MemoryFamily.IDENTITY else (ref_kind if authority != MemoryAuthority.REJECTED else "none")
        return MemoryCandidateAssessment(
            family,
            unsafe_ref_kind,
            authority,
            material,
            MemoryUsePermission(),
            authority.value,
            tuple(reasons),
        )
    if not visibility_ok or not observed_material:
        reasons.append("not_direct_visible_observed_material")
        return MemoryCandidateAssessment(family, ref_kind, MemoryAuthority.WEAK, material, MemoryUsePermission(), "weak", tuple(reasons))
    identity_strong = confidence >= 0.65 and evidence_score >= 0.70 and identity_authoritative_material and reveal_lifecycle not in {"newly_occluded", "currently_hidden", "expected_unknown"}
    appearance_strong = confidence >= 0.58 and evidence_score >= 0.60 and reveal_lifecycle not in {"newly_occluded", "currently_hidden", "expected_unknown"}
    if family == MemoryFamily.IDENTITY:
        if identity_strong:
            reasons.append("observed_high_confidence_identity_anchor")
            return MemoryCandidateAssessment(
                family,
                ref_kind,
                MemoryAuthority.AUTHORITATIVE,
                material,
                MemoryUsePermission(can_seed_identity=True, can_seed_appearance=True, can_seed_reveal=True, can_overwrite_authoritative=True),
                "authoritative",
                tuple(reasons),
            )
        reasons.append("identity_observed_below_authoritative_threshold")
        authority = MemoryAuthority.REUSABLE if confidence >= 0.55 else MemoryAuthority.WEAK
        return MemoryCandidateAssessment(family, "none", authority, material, MemoryUsePermission(can_seed_appearance=True), authority.value, tuple(reasons))
    if appearance_strong:
        reasons.append("observed_high_confidence_appearance")
        return MemoryCandidateAssessment(
            family,
            ref_kind,
            MemoryAuthority.REUSABLE,
            material,
            MemoryUsePermission(can_seed_appearance=True, can_seed_reveal=True, can_overwrite_authoritative=True),
            "reusable",
            tuple(reasons),
        )
    reasons.append("appearance_observed_below_reusable_threshold")
    return MemoryCandidateAssessment(family, ref_kind, MemoryAuthority.WEAK, material, MemoryUsePermission(can_seed_appearance=True), "weak", tuple(reasons))
