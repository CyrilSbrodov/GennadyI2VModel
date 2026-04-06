from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.schema import BBox


@dataclass(slots=True)
class BackendConfig:
    backend: str = "onnx"
    checkpoint: str = "checkpoints/yolo_person.onnx"
    device: str = "cpu"


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
    latency_ms: float = 0.0


class Detector(Protocol):
    def detect(self, image_ref: str) -> DetectorOutput:
        ...


class YoloPersonDetectorAdapter:
    source_name = "detector:yolo-person"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig()

    def detect(self, image_ref: str) -> DetectorOutput:
        confidence = 0.93 if self.config.backend in {"onnx", "torch"} else 0.88
        return DetectorOutput(
            persons=[
                PersonDetection(
                    detection_id=f"det::{image_ref}::person_1",
                    bbox=BBox(0.25, 0.1, 0.5, 0.84),
                    confidence=confidence,
                    source=f"{self.source_name}:{self.config.backend}",
                )
            ],
            frame_size=(1024, 1024),
            latency_ms=7.5,
        )
