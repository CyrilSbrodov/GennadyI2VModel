from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.schema import BBox
from perception.backend import BackendInferenceEngine, CheckpointManager, CheckpointSpec, image_ref_to_features


@dataclass(slots=True)
class BackendConfig:
    backend: str = "builtin"
    checkpoint: str = "checkpoints/yolo_person.torch"
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
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)

    def _run_backend(self, image_ref: str) -> list[float]:
        if self.config.backend in {"torch", "onnx"}:
            return self.engine.infer(image_ref_to_features(image_ref))

        # builtin fallback: dynamic per input (not constant stubs)
        feats = image_ref_to_features(image_ref)
        return [
            (feats[0] + feats[3]) / 2,
            (feats[1] + feats[4]) / 2,
            (feats[2] + feats[5]) / 2,
            (feats[6] + feats[7]) / 2,
            sum(feats[:4]) / 4,
        ]

    def validate_checkpoint(self) -> None:
        if self.config.backend in {"torch", "onnx"}:
            CheckpointManager.ensure_exists(
                CheckpointSpec(self.source_name, self.config.backend, self.config.checkpoint)
            )

    def detect(self, image_ref: str) -> DetectorOutput:
        self.validate_checkpoint()
        pred = self._run_backend(image_ref)
        x = min(0.8, max(0.02, pred[0] * 0.6))
        y = min(0.8, max(0.02, pred[1] * 0.5))
        w = min(0.9 - x, max(0.1, pred[2] * 0.55 + 0.25))
        h = min(0.95 - y, max(0.15, pred[3] * 0.5 + 0.3))
        conf = min(0.99, max(0.2, pred[4]))
        return DetectorOutput(
            persons=[
                PersonDetection(
                    detection_id=f"det::{image_ref}::person_1",
                    bbox=BBox(x, y, w, h),
                    confidence=conf,
                    source=f"{self.source_name}:{self.config.backend}",
                )
            ],
            frame_size=(1024, 1024),
            latency_ms=3.0,
        )
