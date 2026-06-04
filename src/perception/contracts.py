from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from perception.mask_store import InMemoryMaskStore

OBSERVED_STATUSES = {"observed"}
NON_OBSERVED_STATUSES = {"inferred", "fallback", "generated", "unknown", "missing"}
OBSERVED_PROVENANCE_MARKERS = ("detector:", "parser:", "face:", "tracker:", "single_frame_observed", "test_fixture:observed")
NON_OBSERVED_PROVENANCE_MARKERS = ("fallback", "inferred", "generated", "unknown", "synthetic", "training_synthetic")
IDENTITY_SENSITIVE_REGIONS = {"face", "head", "hair", "face_skin", "hairline_or_hair_face_boundary"}


@dataclass(slots=True)
class PerceptionContractViolation:
    code: str
    path: str
    message: str


@dataclass(slots=True)
class PerceptionValidationResult:
    violations: list[PerceptionContractViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def raise_for_violations(self) -> None:
        if self.violations:
            details = "; ".join(f"{v.code}@{v.path}: {v.message}" for v in self.violations)
            raise ValueError(f"perception contract violation: {details}")


def _has_observed_provenance(value: str) -> bool:
    lower = str(value or "").lower()
    return any(marker in lower for marker in OBSERVED_PROVENANCE_MARKERS) and not any(marker in lower for marker in NON_OBSERVED_PROVENANCE_MARKERS)


def _mask_exists(mask_store: InMemoryMaskStore | None, snapshot: dict[str, Any], ref: str) -> bool:
    if mask_store is not None and mask_store.get(ref) is not None:
        return True
    return ref in snapshot


def _mask_metadata(mask_store: InMemoryMaskStore | None, snapshot: dict[str, Any], ref: str) -> dict[str, Any]:
    if mask_store is not None:
        stored = mask_store.get(ref)
        if stored is not None:
            return {"source": stored.source, "tags": list(stored.tags), "extra": dict(stored.extra)}
    meta = snapshot.get(ref, {})
    return dict(meta) if isinstance(meta, dict) else {}


def _is_adopted_legacy_mask(mask_store: InMemoryMaskStore | None, snapshot: dict[str, Any], ref: str) -> bool:
    meta = _mask_metadata(mask_store, snapshot, ref)
    tags = meta.get("tags", [])
    extra = meta.get("extra", {})
    source = str(meta.get("source", ""))
    return (
        "adopted_legacy_default_store" in tags
        or source.startswith("legacy_adopted:")
        or (isinstance(extra, dict) and bool(extra.get("adopted_legacy_default_store")))
    )


def _region_items(person: Any) -> Iterable[tuple[str, int, dict[str, Any]]]:
    for idx, part in enumerate(getattr(person, "body_parts", []) or []):
        yield "body_parts", idx, dict(part)
    for idx, region in enumerate(getattr(person, "face_regions", []) or []):
        yield "face_regions", idx, dict(region)


def validate_perception_output(perception: Any, mask_store: InMemoryMaskStore | None = None) -> PerceptionValidationResult:
    """Validate production perception facts without repairing or normalizing them."""

    result = PerceptionValidationResult()
    snapshot = getattr(perception, "mask_store", {}) or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    person_ids: set[str] = set()
    region_ids: set[str] = set()

    for p_idx, person in enumerate(getattr(perception, "persons", []) or []):
        path = f"persons[{p_idx}]"
        person_id = str(getattr(person, "person_id", "") or "")
        if not person_id:
            result.violations.append(PerceptionContractViolation("missing_person_id", path, "person_id is required"))
        elif person_id in person_ids:
            result.violations.append(PerceptionContractViolation("duplicate_person_id", path, f"duplicate person_id {person_id}"))
        person_ids.add(person_id)

        if getattr(person, "bbox", None) is None:
            result.violations.append(PerceptionContractViolation("missing_person_bbox", path, "person bbox is required"))
        if getattr(person, "bbox_confidence", None) is None:
            result.violations.append(PerceptionContractViolation("missing_person_confidence", path, "bbox_confidence is required"))
        if not str(getattr(person, "bbox_source", "") or ""):
            result.violations.append(PerceptionContractViolation("missing_bbox_provenance", path, "bbox_source/provenance is required"))

        identity_status = str(getattr(person, "identity_observation_status", "") or "")
        track_provenance = str(getattr(person, "track_provenance", getattr(person, "track_source", "")) or "")
        if not identity_status:
            result.violations.append(PerceptionContractViolation("missing_identity_status", path, "identity_observation_status is required"))
        if not track_provenance:
            result.violations.append(PerceptionContractViolation("missing_track_provenance", path, "track provenance is required"))
        if identity_status == "multi_frame_tracked" and not getattr(person, "track_id", None):
            result.violations.append(PerceptionContractViolation("missing_track_id", path, "multi-frame identity requires tracker track_id"))
        if identity_status == "single_frame_anchor" and str(getattr(person, "track_id", "") or "").startswith("stable"):
            result.violations.append(PerceptionContractViolation("fake_stable_track", path, "single-frame identity cannot claim stable track_id"))

        person_mask_ref = getattr(person, "mask_ref", None)
        if person_mask_ref == "":
            result.violations.append(PerceptionContractViolation("empty_mask_ref", f"{path}.mask_ref", "mask_ref cannot be empty"))
        elif person_mask_ref and not _mask_exists(mask_store, snapshot, str(person_mask_ref)):
            result.violations.append(PerceptionContractViolation("missing_mask_payload", f"{path}.mask_ref", f"mask_ref {person_mask_ref} is not resolvable"))

        for collection, r_idx, region in _region_items(person):
            r_path = f"{path}.{collection}[{r_idx}]"
            region_name = str(region.get("part_type") or region.get("region_type") or "")
            region_id = str(region.get("region_id") or region.get("canonical_region_id") or f"{person_id}:{collection}:{region_name}")
            if region_id in region_ids:
                result.violations.append(PerceptionContractViolation("duplicate_region_id", r_path, f"duplicate region id {region_id}"))
            region_ids.add(region_id)

            status = str(region.get("observation_status") or region.get("evidence_status") or "unknown")
            provenance = str(region.get("provenance") or region.get("source") or "")
            confidence = region.get("confidence")
            mask_ref = region.get("mask_ref")
            mask_evidence = str(region.get("mask_evidence_type") or "")
            if mask_ref == "":
                result.violations.append(PerceptionContractViolation("empty_mask_ref", f"{r_path}.mask_ref", "mask_ref cannot be empty"))
            elif mask_ref and not _mask_exists(mask_store, snapshot, str(mask_ref)):
                result.violations.append(PerceptionContractViolation("missing_mask_payload", f"{r_path}.mask_ref", f"mask_ref {mask_ref} is not resolvable"))
            if mask_ref and mask_evidence == "parser_mask" and not _mask_exists(mask_store, snapshot, str(mask_ref)):
                result.violations.append(PerceptionContractViolation("unresolvable_parser_mask", r_path, "parser mask evidence must resolve in mask_store"))
            if mask_ref and status == "observed" and _is_adopted_legacy_mask(mask_store, snapshot, str(mask_ref)):
                result.violations.append(PerceptionContractViolation("adopted_legacy_marked_observed", r_path, "adopted DEFAULT_MASK_STORE material cannot be direct production observed evidence"))
            if status == "observed" and not _has_observed_provenance(provenance):
                result.violations.append(PerceptionContractViolation("observed_without_observed_provenance", r_path, "observed region requires detector/parser/face observed provenance"))
            if status == "observed" and any(marker in provenance.lower() for marker in NON_OBSERVED_PROVENANCE_MARKERS):
                result.violations.append(PerceptionContractViolation("non_observed_marked_observed", r_path, "fallback/inferred/generated/synthetic material cannot be observed"))
            if status in NON_OBSERVED_STATUSES and mask_evidence == "parser_mask" and not mask_ref:
                result.violations.append(PerceptionContractViolation("parser_mask_missing_ref", r_path, "parser_mask evidence requires a mask_ref"))
            if confidence is None:
                result.violations.append(PerceptionContractViolation("missing_region_confidence", r_path, "region confidence is required"))
            if region_name in IDENTITY_SENSITIVE_REGIONS and (status == "observed" or mask_ref) and not provenance:
                result.violations.append(PerceptionContractViolation("identity_region_missing_provenance", r_path, "face/head/hair evidence requires provenance"))

    return result
