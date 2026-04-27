from __future__ import annotations

from dataclasses import asdict, dataclass, field

from core.schema import VideoMemory


@dataclass(slots=True)
class IdentitySummary:
    entity_id: str
    embedding: list[float]
    confidence: float


@dataclass(slots=True)
class GarmentSummary:
    garment_id: str
    embedding: list[float]
    confidence: float


@dataclass(slots=True)
class BodyRegionSummary:
    region_id: str
    region_type: str
    visibility: str
    confidence: float
    last_update_frame: int


@dataclass(slots=True)
class HiddenRegionSummary:
    region_id: str
    hidden_type: str
    confidence: float
    candidate_patch_ids: list[str] = field(default_factory=list)
    last_transition: str = "stable"


@dataclass(slots=True)
class CanonicalRegionMemorySummary:
    record_id: str
    canonical_region: str
    memory_kind: str
    visibility_state: str
    confidence: float
    evidence_score: float
    evidence_quality: str
    observed_directly: bool
    inferred: bool
    generated: bool
    reliable_for_reuse: bool
    suitable_for_reveal: bool
    freshness_frames: int
    reveal_lifecycle: str
    source_frame: int
    last_transition: str


@dataclass(slots=True)
class AppearanceMemorySummary:
    identity: dict[str, IdentitySummary] = field(default_factory=dict)
    garments: dict[str, GarmentSummary] = field(default_factory=dict)
    body_regions: dict[str, BodyRegionSummary] = field(default_factory=dict)
    hidden_regions: dict[str, HiddenRegionSummary] = field(default_factory=dict)
    canonical_regions: dict[str, CanonicalRegionMemorySummary] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "identity": {k: asdict(v) for k, v in self.identity.items()},
            "garments": {k: asdict(v) for k, v in self.garments.items()},
            "body_regions": {k: asdict(v) for k, v in self.body_regions.items()},
            "hidden_regions": {k: asdict(v) for k, v in self.hidden_regions.items()},
            "canonical_regions": {k: asdict(v) for k, v in self.canonical_regions.items()},
        }


class AppearanceMemorySummarizer:
    def summarize(self, memory: VideoMemory) -> AppearanceMemorySummary:
        identity = {
            entity_id: IdentitySummary(entity_id=entity_id, embedding=entry.embedding[:], confidence=entry.confidence)
            for entity_id, entry in memory.identity_memory.items()
        }
        garments = {
            garment_id: GarmentSummary(garment_id=garment_id, embedding=entry.embedding[:], confidence=entry.confidence)
            for garment_id, entry in memory.garment_memory.items()
        }
        body_regions = {
            region_id: BodyRegionSummary(
                region_id=region_id,
                region_type=descriptor.region_type,
                visibility=descriptor.visibility,
                confidence=descriptor.confidence,
                last_update_frame=descriptor.last_update_frame,
            )
            for region_id, descriptor in memory.region_descriptors.items()
        }
        hidden_regions = {
            region_id: HiddenRegionSummary(
                region_id=region_id,
                hidden_type=slot.hidden_type,
                confidence=slot.confidence,
                candidate_patch_ids=slot.candidate_patch_ids[:],
                last_transition=slot.last_transition,
            )
            for region_id, slot in memory.hidden_region_slots.items()
        }
        canonical_regions = {
            rec_id: CanonicalRegionMemorySummary(
                record_id=rec.record_id,
                canonical_region=rec.canonical_region,
                memory_kind=rec.memory_kind,
                visibility_state=str(rec.visibility_state),
                confidence=rec.confidence,
                evidence_score=rec.evidence_score,
                evidence_quality=rec.evidence_quality,
                observed_directly=rec.observed_directly,
                inferred=rec.inferred,
                generated=rec.generated,
                reliable_for_reuse=rec.reliable_for_reuse,
                suitable_for_reveal=rec.suitable_for_reveal,
                freshness_frames=rec.freshness_frames,
                reveal_lifecycle=rec.reveal_lifecycle,
                source_frame=rec.source_frame,
                last_transition=rec.last_transition,
            )
            for rec_id, rec in memory.canonical_region_memory.items()
        }
        return AppearanceMemorySummary(
            identity=identity,
            garments=garments,
            body_regions=body_regions,
            hidden_regions=hidden_regions,
            canonical_regions=canonical_regions,
        )
