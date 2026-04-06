from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.schema import ExpressionState, OrientationState
from perception.backend import BackendInferenceEngine, image_ref_to_features
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
        self.config = config or BackendConfig(checkpoint="checkpoints/emonet.torch")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)

    def analyze(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, FacePrediction]:
        result: dict[str, FacePrediction] = {}
        feats = image_ref_to_features(image_ref)
        if self.config.backend in {"torch", "onnx"}:
            feats = self.engine.infer(feats)
        for person in persons:
            cx = person.bbox.x + person.bbox.w * 0.5
            cy = person.bbox.y + person.bbox.h * 0.16
            landmarks = [(cx + (i - 34) * 0.0012, cy + ((i % 5) - 2) * 0.0015) for i in range(68)]
            result[person.detection_id] = FacePrediction(
                expression=ExpressionState(smile_intensity=feats[0], label="happy" if feats[0] > 0.55 else "neutral"),
                expression_confidence=min(0.99, max(0.2, feats[1])),
                orientation=OrientationState(yaw=(feats[2] - 0.5) * 20.0, pitch=(feats[3] - 0.5) * 12.0, roll=(feats[4] - 0.5) * 8.0),
                orientation_confidence=min(0.99, max(0.2, feats[5])),
                source=f"{self.source_name}:{self.config.backend}",
                face_landmarks=landmarks,
            )
        return result
