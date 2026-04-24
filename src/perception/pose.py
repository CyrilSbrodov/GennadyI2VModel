from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.schema import Keypoint, PoseState
from perception.backend import BackendInferenceEngine, frame_to_features
from perception.detector import BackendConfig, PersonDetection
from perception.frame_context import FrameLike
from perception.image_ops import crop_rgb, frame_to_numpy_rgb, rgb_to_bgr


@dataclass(slots=True)
class PosePrediction:
    pose: PoseState
    confidence: float
    source: str
    landmarks_2d: list[tuple[float, float]]
    hand_landmarks: dict[str, list[tuple[float, float]]]
    face_landmarks: list[tuple[float, float]]


class PoseEstimator(Protocol):
    def estimate(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        ...


class VitPoseAdapter:
    source_name = "pose:vitpose"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/vitpose.torch")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)

    def is_builtin_backend(self) -> bool:
        return self.config.backend == "builtin"

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
            face_landmarks=[],
        )

    def estimate(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        result: dict[str, PosePrediction] = {}
        base = frame_to_features(frame)
        for person in persons:
            pred = self.engine.infer(base) if self.config.backend in {"torch", "onnx"} else base
            result[person.detection_id] = self._builtin_prediction(person, pred, f"{self.source_name}:{self.config.backend}")
        return result


class YoloPoseAdapter:
    source_name = "pose:yolo"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(backend="ultralytics", checkpoint="yolov8n-pose.pt")
        self._model = None

    def is_builtin_backend(self) -> bool:
        return self.config.backend == "builtin"

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is not installed") from exc
        self._model = YOLO(self.config.checkpoint or "yolov8n-pose.pt")
        return self._model

    def _to_prediction(self, person: PersonDetection, xs: list[float], ys: list[float], vs: list[float]) -> PosePrediction:
        names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        kps: list[Keypoint] = []
        for idx, name in enumerate(names):
            if idx >= len(xs) or idx >= len(ys):
                continue
            conf = float(vs[idx]) if idx < len(vs) else 1.0
            kps.append(Keypoint(name, max(0.0, min(1.0, xs[idx])), max(0.0, min(1.0, ys[idx])), max(0.0, min(1.0, conf))))
        avg_conf = sum(k.confidence for k in kps) / max(1, len(kps))
        hand_landmarks = {
            "left": [(k.x, k.y) for k in kps if k.name in {"left_wrist", "left_elbow"}],
            "right": [(k.x, k.y) for k in kps if k.name in {"right_wrist", "right_elbow"}],
        }
        face_landmarks = [(k.x, k.y) for k in kps if "eye" in k.name or "ear" in k.name or k.name == "nose"]
        return PosePrediction(
            pose=PoseState(keypoints=kps, coarse_pose="unknown", angles={}),
            confidence=avg_conf,
            source=f"{self.source_name}:ultralytics",
            landmarks_2d=[(k.x, k.y) for k in kps],
            hand_landmarks=hand_landmarks,
            face_landmarks=face_landmarks,
        )

    @staticmethod
    def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = a_area + b_area - inter
        return inter / denom if denom > 0.0 else 0.0

    @staticmethod
    def _person_bbox(person: PersonDetection) -> tuple[float, float, float, float]:
        x1 = max(0.0, min(1.0, person.bbox.x))
        y1 = max(0.0, min(1.0, person.bbox.y))
        x2 = max(x1, min(1.0, person.bbox.x + person.bbox.w))
        y2 = max(y1, min(1.0, person.bbox.y + person.bbox.h))
        return (x1, y1, x2, y2)

    def estimate(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        if self.config.backend == "builtin":
            return VitPoseAdapter(BackendConfig(backend="builtin")).estimate(frame, persons)
        img = frame_to_numpy_rgb(frame)
        model = self._load_model()
        results = model.predict(source=rgb_to_bgr(img.rgb), verbose=False, conf=float(self.config.confidence_threshold), device=self.config.device)
        predictions: dict[str, PosePrediction] = {}
        candidates: list[dict[str, object]] = []
        for res in results:
            keypoints = getattr(res, "keypoints", None)
            boxes = getattr(res, "boxes", None)
            if keypoints is None or boxes is None:
                continue
            xyn = getattr(keypoints, "xyn", None)
            conf = getattr(keypoints, "conf", None)
            for idx, box in enumerate(boxes):
                if int(box.cls.item()) != 0:
                    continue
                bx = box.xyxy[0].tolist()
                px1 = max(0.0, min(1.0, float(bx[0]) / max(1, img.width)))
                py1 = max(0.0, min(1.0, float(bx[1]) / max(1, img.height)))
                px2 = max(0.0, min(1.0, float(bx[2]) / max(1, img.width)))
                py2 = max(0.0, min(1.0, float(bx[3]) / max(1, img.height)))
                if xyn is None or idx >= len(xyn):
                    continue
                xs = [float(v) for v in xyn[idx][:, 0].tolist()]
                ys = [float(v) for v in xyn[idx][:, 1].tolist()]
                vs = [float(v) for v in conf[idx].tolist()] if conf is not None and idx < len(conf) else [1.0 for _ in xs]
                candidates.append(
                    {
                        "bbox": (px1, py1, px2, py2),
                        "center": ((px1 + px2) * 0.5, (py1 + py2) * 0.5),
                        "xs": xs,
                        "ys": ys,
                        "vs": vs,
                    }
                )
        used_candidates: set[int] = set()
        for person in persons:
            person_bbox = self._person_bbox(person)
            best_idx = -1
            best_iou = -1.0
            best_dist = float("inf")
            person_center = ((person_bbox[0] + person_bbox[2]) * 0.5, (person_bbox[1] + person_bbox[3]) * 0.5)
            for idx, cand in enumerate(candidates):
                if idx in used_candidates:
                    continue
                cand_bbox = cand["bbox"]
                if not isinstance(cand_bbox, tuple):
                    continue
                iou = self._bbox_iou(person_bbox, cand_bbox)
                center = cand.get("center", (0.5, 0.5))
                cx, cy = center if isinstance(center, tuple) else (0.5, 0.5)
                dist = ((float(cx) - person_center[0]) ** 2 + (float(cy) - person_center[1]) ** 2) ** 0.5
                if iou > best_iou + 1e-6 or (abs(iou - best_iou) <= 1e-6 and dist < best_dist):
                    best_iou = iou
                    best_dist = dist
                    best_idx = idx
            if best_idx < 0:
                continue
            used_candidates.add(best_idx)
            best = candidates[best_idx]
            xs = best.get("xs")
            ys = best.get("ys")
            vs = best.get("vs")
            if isinstance(xs, list) and isinstance(ys, list) and isinstance(vs, list):
                predictions[person.detection_id] = self._to_prediction(person, xs, ys, vs)
        return predictions


class MediaPipePoseAdapter:
    source_name = "pose:mediapipe"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(backend="mediapipe")
        self._mp = None
        self._pose_model = None

    def _get_pose_model(self):
        if self._pose_model is not None:
            return self._pose_model
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mediapipe is not installed") from exc
        self._mp = mp
        self._pose_model = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
        return self._pose_model

    def is_builtin_backend(self) -> bool:
        return self.config.backend == "builtin"

    def estimate(self, frame: FrameLike, persons: list[PersonDetection]) -> dict[str, PosePrediction]:
        pose_model = self._get_pose_model()
        rgb = frame_to_numpy_rgb(frame).rgb
        result: dict[str, PosePrediction] = {}
        for person in persons:
            patch = crop_rgb(rgb, person.bbox)
            pose_res = pose_model.process(patch)
            if not pose_res.pose_landmarks:
                continue
            lms = pose_res.pose_landmarks.landmark
            named = {
                "nose": 0,
                "left_shoulder": 11,
                "right_shoulder": 12,
                "left_wrist": 15,
                "right_wrist": 16,
                "left_hip": 23,
                "right_hip": 24,
                "left_knee": 25,
                "right_knee": 26,
            }
            keypoints: list[Keypoint] = []
            for name, idx in named.items():
                lm = lms[idx]
                x = person.bbox.x + person.bbox.w * float(lm.x)
                y = person.bbox.y + person.bbox.h * float(lm.y)
                keypoints.append(Keypoint(name, x, y, max(0.0, min(1.0, float(lm.visibility)))))
            face_landmarks = []
            for idx in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
                lm = lms[idx]
                face_landmarks.append((person.bbox.x + person.bbox.w * float(lm.x), person.bbox.y + person.bbox.h * float(lm.y)))
            conf = sum(k.confidence for k in keypoints) / max(1, len(keypoints))
            result[person.detection_id] = PosePrediction(
                pose=PoseState(keypoints=keypoints, coarse_pose="unknown", angles={}),
                confidence=conf,
                source=f"{self.source_name}:runtime",
                landmarks_2d=[(k.x, k.y) for k in keypoints],
                hand_landmarks={},
                face_landmarks=face_landmarks,
            )
        return result
