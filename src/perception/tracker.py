from __future__ import annotations

from dataclasses import dataclass

from core.input_layer import AssetFrame
from typing import Protocol

from core.schema import BBox
from perception.backend import BackendInferenceEngine, frame_to_features
from perception.detector import BackendConfig, PersonDetection


@dataclass(slots=True)
class TrackPrediction:
    track_id: str
    confidence: float
    source: str


class PersonTracker(Protocol):
    def assign(self, frame: AssetFrame | list[list[list[float]]] | str, persons: list[PersonDetection]) -> dict[str, TrackPrediction]:
        ...


def _iou(a: BBox, b: BBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


class ByteTrackAdapter:
    source_name = "tracker:bytetrack"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/bytetrack.torch")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)
        self._tracks: dict[str, BBox] = {}
        self._next_id = 1

    def assign(self, frame: AssetFrame | list[list[list[float]]] | str, persons: list[PersonDetection]) -> dict[str, TrackPrediction]:
        feats = frame_to_features(frame)
        if self.config.backend in {"torch", "onnx"}:
            feats = self.engine.infer(feats)
        out: dict[str, TrackPrediction] = {}
        for idx, p in enumerate(persons):
            best_tid = None
            best_iou = 0.0
            for tid, prev_bbox in self._tracks.items():
                score = _iou(p.bbox, prev_bbox)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid
            if best_tid is None or best_iou < 0.25:
                best_tid = f"track_{self._next_id}"
                self._next_id += 1
            self._tracks[best_tid] = p.bbox
            conf = min(0.99, max(0.2, 0.4 + best_iou * 0.5 + 0.1 * feats[idx % len(feats)]))
            out[p.detection_id] = TrackPrediction(best_tid, conf, f"{self.source_name}:{self.config.backend}")
        return out
