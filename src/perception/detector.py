from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.schema import BBox


@dataclass(slots=True)
class PersonDetection:
    detection_id: str
    bbox: BBox
    confidence: float
    source: str


@dataclass(slots=True)
class DetectorOutput:
    persons: list[PersonDetection] = field(default_factory=list)
    frame_size: tuple[int, int] = (1024, 1024)


class Detector(Protocol):
    def detect(self, image_ref: str) -> DetectorOutput:
        ...


class YoloPersonDetectorAdapter:
    """Adapter placeholder for a YOLO-style person detector."""

    source_name = "detector:yolo-person"

    def detect(self, image_ref: str) -> DetectorOutput:
        return DetectorOutput(
            persons=[
                PersonDetection(
                    detection_id=f"det::{image_ref}::person_1",
                    bbox=BBox(0.25, 0.1, 0.5, 0.84),
                    confidence=0.93,
                    source=self.source_name,
                )
            ],
            frame_size=(1024, 1024),
        )
