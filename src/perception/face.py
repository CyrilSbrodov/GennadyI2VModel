from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.schema import ExpressionState, OrientationState
from perception.detector import BackendConfig, PersonDetection


@dataclass(slots=True)
class FacePrediction:
    expression: ExpressionState
    expression_confidence: float
    orientation: OrientationState
    orientation_confidence: float
    source: str
    face_landmarks: list[tuple[float, float]]


class FaceAnalyzer(Protocol):
    def analyze(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, FacePrediction]:
        ...


class EmoNetFaceAnalyzerAdapter:
    source_name = "face:emonet"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/emonet.onnx")

    def analyze(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, FacePrediction]:
        result: dict[str, FacePrediction] = {}
        for person in persons:
            result[person.detection_id] = FacePrediction(
                expression=ExpressionState(smile_intensity=0.1, label="neutral"),
                expression_confidence=0.74,
                orientation=OrientationState(yaw=0.0, pitch=0.0, roll=0.0),
                orientation_confidence=0.79,
                source=f"{self.source_name}:{self.config.backend}",
                face_landmarks=[(0.48, 0.18), (0.52, 0.18), (0.50, 0.21)],
            )
        return result
