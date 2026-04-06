from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.schema import BBox
from perception.detector import BackendConfig


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

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/yolo_world.onnx")

    def detect(self, image_ref: str) -> list[ObjectPrediction]:
        return [ObjectPrediction("chair", BBox(0.62, 0.55, 0.26, 0.4), 0.72, f"{self.source_name}:{self.config.backend}")]


class MonoDepthEstimator:
    source_name = "depth:monocular"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/depth.onnx")

    def estimate(self, image_ref: str) -> float:
        _ = image_ref
        return 0.5
