from __future__ import annotations

from dataclasses import dataclass

from core.input_layer import AssetFrame
from typing import Protocol

from core.schema import Keypoint, PoseState
from perception.backend import BackendInferenceEngine, frame_to_features
from perception.detector import BackendConfig, PersonDetection


@dataclass(slots=True)
class PosePrediction:
    pose: PoseState
    confidence: float
    source: str
    landmarks_2d: list[tuple[float, float]]
    hand_landmarks: dict[str, list[tuple[float, float]]]


class PoseEstimator(Protocol):
    def estimate(self, frame: AssetFrame | list[list[list[float]]] | str, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        ...


class VitPoseAdapter:
    source_name = "pose:vitpose"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/vitpose.torch")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)

    def estimate(self, frame: AssetFrame | list[list[list[float]]] | str, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        result: dict[str, PosePrediction] = {}
        base = frame_to_features(frame)
        for person in persons:
            pred = self.engine.infer(base) if self.config.backend in {"torch", "onnx"} else base
            cy = person.bbox.y + person.bbox.h * 0.12
            cx = person.bbox.x + person.bbox.w * 0.5
            keypoints = [
                Keypoint("nose", cx, cy, 0.9),
                Keypoint("left_shoulder", person.bbox.x + person.bbox.w * 0.28, person.bbox.y + person.bbox.h * 0.35, 0.85),
                Keypoint("right_shoulder", person.bbox.x + person.bbox.w * 0.72, person.bbox.y + person.bbox.h * 0.35, 0.85),
                Keypoint("left_wrist", person.bbox.x + person.bbox.w * (0.2 + 0.2 * pred[0]), person.bbox.y + person.bbox.h * 0.58, 0.8),
                Keypoint("right_wrist", person.bbox.x + person.bbox.w * (0.8 - 0.2 * pred[1]), person.bbox.y + person.bbox.h * 0.58, 0.8),
            ]
            left_hand = [(person.bbox.x + person.bbox.w * (0.2 + 0.03 * i), person.bbox.y + person.bbox.h * (0.55 + 0.02 * i)) for i in range(21)]
            right_hand = [(person.bbox.x + person.bbox.w * (0.75 - 0.03 * i), person.bbox.y + person.bbox.h * (0.55 + 0.02 * i)) for i in range(21)]
            result[person.detection_id] = PosePrediction(
                pose=PoseState(
                    keypoints=keypoints,
                    coarse_pose="standing" if person.bbox.h > person.bbox.w else "leaning",
                    angles={"left_elbow": float(10 + pred[2] * 40), "right_elbow": float(10 + pred[3] * 40)},
                ),
                confidence=min(0.99, max(0.3, 0.55 + pred[4] * 0.4)),
                source=f"{self.source_name}:{self.config.backend}",
                landmarks_2d=[(k.x, k.y) for k in keypoints],
                hand_landmarks={"left": left_hand, "right": right_hand},
            )
        return result
