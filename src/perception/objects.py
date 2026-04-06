from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.schema import BBox


@dataclass(slots=True)
class ObjectPrediction:
    object_type: str
    bbox: BBox
    confidence: float
    source: str


class ObjectDetector(Protocol):
    def detect(self, image_ref: str) -> list[ObjectPrediction]:
        ...


class YoloObjectDetectorAdapter:
    source_name = "objects:yolo-world"

    def detect(self, image_ref: str) -> list[ObjectPrediction]:
        return [ObjectPrediction("chair", BBox(0.62, 0.55, 0.26, 0.4), 0.72, self.source_name)]
