from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.input_layer import AssetFrame
from core.schema import BBox
from perception.backend import BackendInferenceEngine, CheckpointManager, CheckpointSpec, frame_to_features
from perception.frame_context import FrameLike
from perception.image_ops import frame_to_numpy_rgb, rgb_to_bgr, xyxy_to_norm_bbox


@dataclass(slots=True)
class BackendConfig:
    backend: str = "builtin"
    checkpoint: str = "checkpoints/yolo_person.torch"
    device: str = "cpu"
    confidence_threshold: float = 0.25


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
    def detect(self, frame: FrameLike) -> DetectorOutput:
        ...


class YoloPersonDetectorAdapter:
    source_name = "detector:yolo-person"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig()
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)
        self._model = None

    def _detect_builtin(self, frame: FrameLike) -> DetectorOutput:
        feats = frame_to_features(frame)
        x = min(0.8, max(0.02, feats[0] * 0.6))
        y = min(0.8, max(0.02, feats[1] * 0.5))
        w = min(0.9 - x, max(0.1, feats[2] * 0.55 + 0.25))
        h = min(0.95 - y, max(0.15, feats[3] * 0.5 + 0.3))
        conf = min(0.99, max(0.2, sum(feats[:4]) / 4))
        return DetectorOutput(
            persons=[
                PersonDetection(
                    detection_id="det::person_1",
                    bbox=BBox(x, y, w, h),
                    confidence=conf,
                    source=f"{self.source_name}:builtin",
                )
            ],
            frame_size=(1024, 1024),
            latency_ms=3.0,
        )

    def _load_ultralytics(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is not installed") from exc
        model_name = self.config.checkpoint or "yolov8n.pt"
        self._model = YOLO(model_name)
        return self._model

    def _detect_ultralytics(self, frame: FrameLike) -> DetectorOutput:
        image = frame_to_numpy_rgb(frame)
        model = self._load_ultralytics()
        results = model.predict(
            source=rgb_to_bgr(image.rgb),
            verbose=False,
            conf=float(self.config.confidence_threshold),
            device=self.config.device,
        )
        persons: list[PersonDetection] = []
        det_id = 1
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls.item())
                if cls != 0:
                    continue
                conf = float(box.conf.item())
                if conf < self.config.confidence_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                persons.append(
                    PersonDetection(
                        detection_id=f"det::person_{det_id}",
                        bbox=xyxy_to_norm_bbox(x1, y1, x2, y2, image.width, image.height),
                        confidence=conf,
                        source=f"{self.source_name}:ultralytics",
                    )
                )
                det_id += 1
        return DetectorOutput(persons=persons, frame_size=(image.width, image.height), latency_ms=0.0)

    def validate_checkpoint(self) -> None:
        if self.config.backend in {"torch", "onnx"}:
            CheckpointManager.ensure_exists(
                CheckpointSpec(self.source_name, self.config.backend, self.config.checkpoint)
            )

    def detect(self, frame: FrameLike) -> DetectorOutput:
        if self.config.backend == "builtin":
            return self._detect_builtin(frame)
        if self.config.backend == "ultralytics":
            return self._detect_ultralytics(frame)

        self.validate_checkpoint()
        pred = self.engine.infer(frame_to_features(frame))
        x = min(0.8, max(0.02, pred[0] * 0.6))
        y = min(0.8, max(0.02, pred[1] * 0.5))
        w = min(0.9 - x, max(0.1, pred[2] * 0.55 + 0.25))
        h = min(0.95 - y, max(0.15, pred[3] * 0.5 + 0.3))
        conf = min(0.99, max(0.2, pred[4]))
        return DetectorOutput(
            persons=[PersonDetection("det::person_1", BBox(x, y, w, h), conf, f"{self.source_name}:{self.config.backend}")],
            frame_size=(1024, 1024),
            latency_ms=3.0,
        )
