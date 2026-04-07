from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.input_layer import AssetFrame
from core.schema import Keypoint, PoseState
from perception.backend import BackendInferenceEngine, frame_to_features
from perception.detector import BackendConfig, PersonDetection
from perception.frame_context import FrameLike
from perception.image_ops import crop_rgb, frame_to_numpy_rgb


@dataclass(slots=True)
class PosePrediction:
    pose: PoseState
    confidence: float
    source: str
    landmarks_2d: list[tuple[float, float]]
    hand_landmarks: dict[str, list[tuple[float, float]]]


class PoseEstimator(Protocol):
    def estimate(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        ...


class VitPoseAdapter:
    source_name = "pose:vitpose"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/vitpose.torch")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)

    def _builtin_prediction(self, person: PersonDetection, pred: list[float], source: str) -> PosePrediction:
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
        return PosePrediction(
            pose=PoseState(
                keypoints=keypoints,
                coarse_pose="standing" if person.bbox.h > person.bbox.w else "leaning",
                angles={"left_elbow": float(10 + pred[2] * 40), "right_elbow": float(10 + pred[3] * 40)},
            ),
            confidence=min(0.99, max(0.3, 0.55 + pred[4] * 0.4)),
            source=source,
            landmarks_2d=[(k.x, k.y) for k in keypoints],
            hand_landmarks={"left": left_hand, "right": right_hand},
        )

    def _estimate_mediapipe(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mediapipe is not installed") from exc

        rgb = frame_to_numpy_rgb(frame).rgb
        result: dict[str, PosePrediction] = {}
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose_model:
            for person in persons:
                patch = crop_rgb(rgb, person.bbox)
                pose_res = pose_model.process(patch)
                if not pose_res.pose_landmarks:
                    continue
                lms = pose_res.pose_landmarks.landmark
                keypoints: list[Keypoint] = []
                named = {
                    "nose": 0,
                    "left_shoulder": 11,
                    "right_shoulder": 12,
                    "left_wrist": 15,
                    "right_wrist": 16,
                    "left_hip": 23,
                    "right_hip": 24,
                }
                for name, idx in named.items():
                    lm = lms[idx]
                    x = person.bbox.x + person.bbox.w * float(lm.x)
                    y = person.bbox.y + person.bbox.h * float(lm.y)
                    keypoints.append(Keypoint(name, x, y, max(0.0, min(1.0, float(lm.visibility)))))
                conf = sum(k.confidence for k in keypoints) / max(1, len(keypoints))
                result[person.detection_id] = PosePrediction(
                    pose=PoseState(keypoints=keypoints, coarse_pose="standing", angles={}),
                    confidence=conf,
                    source="pose:mediapipe",
                    landmarks_2d=[(k.x, k.y) for k in keypoints],
                    hand_landmarks={},
                )
        return result

    def estimate(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        result: dict[str, PosePrediction] = {}
        if self.config.backend == "mediapipe":
            return self._estimate_mediapipe(frame, persons)

        base = frame_to_features(frame)
        for person in persons:
            pred = self.engine.infer(base) if self.config.backend in {"torch", "onnx"} else base
            result[person.detection_id] = self._builtin_prediction(person, pred, f"{self.source_name}:{self.config.backend}")
        return result
