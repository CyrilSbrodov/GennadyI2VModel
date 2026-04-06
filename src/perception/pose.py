from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.schema import Keypoint, PoseState
from perception.detector import BackendConfig, PersonDetection


@dataclass(slots=True)
class PosePrediction:
    pose: PoseState
    confidence: float
    source: str
    landmarks_2d: list[tuple[float, float]]


class PoseEstimator(Protocol):
    def estimate(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        ...


class VitPoseAdapter:
    source_name = "pose:vitpose"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/vitpose.onnx")

    def estimate(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        result: dict[str, PosePrediction] = {}
        for person in persons:
            result[person.detection_id] = PosePrediction(
                pose=PoseState(
                    keypoints=[
                        Keypoint("nose", 0.5, 0.2, 0.9),
                        Keypoint("left_shoulder", 0.42, 0.35, 0.88),
                        Keypoint("right_shoulder", 0.58, 0.35, 0.88),
                        Keypoint("left_wrist", 0.35, 0.5, 0.81),
                        Keypoint("right_wrist", 0.65, 0.5, 0.81),
                    ],
                    coarse_pose="standing",
                    angles={"left_elbow": 8.0, "right_elbow": 7.0, "left_knee": 2.0, "right_knee": 3.0},
                ),
                confidence=0.9,
                source=f"{self.source_name}:{self.config.backend}",
                landmarks_2d=[(0.5, 0.2), (0.42, 0.35), (0.58, 0.35)],
            )
        return result
