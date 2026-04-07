from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.input_layer import AssetFrame
from core.schema import BBox
from perception.backend import BackendInferenceEngine, frame_to_features
from perception.detector import BackendConfig
from perception.image_ops import frame_to_numpy_rgb, rgb_to_bgr, xyxy_to_norm_bbox


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
    _COCO_MAP = {56: "chair", 57: "sofa", 13: "bench", 60: "table", 59: "bed"}

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="yolov8n.pt")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)
        self._model = None

    def _load_ultralytics(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is not installed") from exc
        self._model = YOLO(self.config.checkpoint or "yolov8n.pt")
        return self._model

    def _detect_ultralytics(self, frame: AssetFrame | list[list[list[float]]] | str) -> list[ObjectPrediction]:
        image = frame_to_numpy_rgb(frame)
        model = self._load_ultralytics()
        results = model.predict(source=rgb_to_bgr(image.rgb), verbose=False, conf=float(self.config.confidence_threshold), device=self.config.device)
        predictions: list[ObjectPrediction] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls.item())
                if cls not in self._COCO_MAP:
                    continue
                conf = float(box.conf.item())
                if conf < self.config.confidence_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                norm = xyxy_to_norm_bbox(x1, y1, x2, y2, image.width, image.height)
                predictions.append(
                    ObjectPrediction(
                        object_type=self._COCO_MAP[cls],
                        bbox=norm,
                        confidence=conf,
                        source="objects:yolo-world:ultralytics",
                        depth_order=1.0 - (norm.y + norm.h),
                    )
                )
        return predictions

    def detect(self, frame: AssetFrame | list[list[list[float]]] | str) -> list[ObjectPrediction]:
        if self.config.backend == "ultralytics":
            return self._detect_ultralytics(frame)

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
