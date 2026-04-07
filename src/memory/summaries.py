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
class AppearanceMemorySummary:
    identity: dict[str, IdentitySummary] = field(default_factory=dict)
    garments: dict[str, GarmentSummary] = field(default_factory=dict)
    body_regions: dict[str, BodyRegionSummary] = field(default_factory=dict)
    hidden_regions: dict[str, HiddenRegionSummary] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "identity": {k: asdict(v) for k, v in self.identity.items()},
            "garments": {k: asdict(v) for k, v in self.garments.items()},
            "body_regions": {k: asdict(v) for k, v in self.body_regions.items()},
            "hidden_regions": {k: asdict(v) for k, v in self.hidden_regions.items()},
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
        return AppearanceMemorySummary(identity=identity, garments=garments, body_regions=body_regions, hidden_regions=hidden_regions)
