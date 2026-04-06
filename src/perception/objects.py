from __future__ import annotations

from dataclasses import dataclass

from core.input_layer import AssetFrame
from typing import Protocol

from core.schema import BBox
from perception.backend import BackendInferenceEngine, frame_to_features
from perception.detector import BackendConfig


@dataclass(slots=True)
class ObjectPrediction:
    object_type: str
    bbox: BBox
    confidence: float
    source: str
    depth_order: float = 0.0


class ObjectDetector(Protocol):
    def detect(self, frame: AssetFrame | list[list[list[float]]] | str) -> list[ObjectPrediction]:
        ...


class YoloObjectDetectorAdapter:
    source_name = "objects:yolo-world"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/yolo_world.torch")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)

    def detect(self, frame: AssetFrame | list[list[list[float]]] | str) -> list[ObjectPrediction]:
        feats = frame_to_features(frame)
        if self.config.backend in {"torch", "onnx"}:
            feats = self.engine.infer(feats)
        return [
            ObjectPrediction(
                "chair",
                BBox(0.45 + 0.2 * feats[0], 0.48 + 0.1 * feats[1], 0.2 + 0.2 * feats[2], 0.2 + 0.25 * feats[3]),
                min(0.99, max(0.2, feats[4])),
                f"{self.source_name}:{self.config.backend}",
                depth_order=1.0 - feats[5],
            )
        ]


class MonoDepthEstimator:
    source_name = "depth:monocular"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/depth.torch")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)

    def estimate(self, frame: AssetFrame | list[list[list[float]]] | str) -> float:
        feats = frame_to_features(frame)
        if self.config.backend in {"torch", "onnx"}:
            feats = self.engine.infer(feats)
        return min(1.0, max(0.0, sum(feats[:4]) / 4))
