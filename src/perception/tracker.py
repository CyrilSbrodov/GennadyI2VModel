from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from perception.detector import PersonDetection


@dataclass(slots=True)
class TrackPrediction:
    track_id: str
    confidence: float
    source: str


class PersonTracker(Protocol):
    def assign(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, TrackPrediction]:
        ...


class ByteTrackAdapter:
    source_name = "tracker:bytetrack"

    def assign(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, TrackPrediction]:
        stable_seed = abs(hash(image_ref)) % 10
        return {
            p.detection_id: TrackPrediction(track_id=f"track_{stable_seed+idx}", confidence=0.84, source=self.source_name)
            for idx, p in enumerate(persons, start=1)
        }
