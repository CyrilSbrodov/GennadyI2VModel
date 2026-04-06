from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.detector import Detector, DetectorOutput, YoloPersonDetectorAdapter
from perception.face import EmoNetFaceAnalyzerAdapter, FaceAnalyzer, FacePrediction
from perception.objects import ObjectDetector, ObjectPrediction, YoloObjectDetectorAdapter
from perception.parser import HumanParser, ParsingPrediction, SegFormerHumanParserAdapter
from perception.pose import PoseEstimator, PosePrediction, VitPoseAdapter
from perception.tracker import ByteTrackAdapter, PersonTracker, TrackPrediction


@dataclass(slots=True)
class PersonFacts:
    bbox: BBox
    bbox_confidence: float
    bbox_source: str
    mask_ref: str | None
    mask_confidence: float
    mask_source: str
    pose: PoseState
    pose_confidence: float
    pose_source: str
    expression: ExpressionState
    expression_confidence: float
    expression_source: str
    orientation: OrientationState
    orientation_confidence: float
    orientation_source: str
    track_id: str | None = None
    track_confidence: float = 0.0
    track_source: str = "fallback"
    garments: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class ObjectFacts:
    object_type: str
    bbox: BBox
    confidence: float
    source: str


@dataclass(slots=True)
class PerceptionOutput:
    persons: list[PersonFacts] = field(default_factory=list)
    objects: list[ObjectFacts] = field(default_factory=list)
    frame_size: tuple[int, int] = (1024, 1024)
    warnings: list[str] = field(default_factory=list)


class PerceptionPipeline:
    """Pluggable perception facade composed from detector/pose/parser/face/objects/tracker modules."""

    def __init__(
        self,
        detector: Detector | None = None,
        pose: PoseEstimator | None = None,
        parser: HumanParser | None = None,
        face: FaceAnalyzer | None = None,
        objects: ObjectDetector | None = None,
        tracker: PersonTracker | None = None,
    ) -> None:
        self.detector = detector or YoloPersonDetectorAdapter()
        self.pose = pose or VitPoseAdapter()
        self.parser = parser or SegFormerHumanParserAdapter()
        self.face = face or EmoNetFaceAnalyzerAdapter()
        self.objects = objects or YoloObjectDetectorAdapter()
        self.tracker = tracker or ByteTrackAdapter()

    def _safe_module_call(self, fn, default, warnings: list[str], module_name: str):
        try:
            return fn()
        except Exception as exc:  # fallback behavior for partially unavailable modules
            warnings.append(f"{module_name}_unavailable:{exc.__class__.__name__}")
            return default

    def analyze(self, image_ref: str) -> PerceptionOutput:
        warnings: list[str] = []
        detection_out = self._safe_module_call(
            lambda: self.detector.detect(image_ref),
            default=DetectorOutput(),
            warnings=warnings,
            module_name="detector",
        )

        pose_predictions: dict[str, PosePrediction] = self._safe_module_call(
            lambda: self.pose.estimate(image_ref, detection_out.persons),
            default={},
            warnings=warnings,
            module_name="pose",
        )
        parsing_predictions: dict[str, ParsingPrediction] = self._safe_module_call(
            lambda: self.parser.parse(image_ref, detection_out.persons),
            default={},
            warnings=warnings,
            module_name="parser",
        )
        face_predictions: dict[str, FacePrediction] = self._safe_module_call(
            lambda: self.face.analyze(image_ref, detection_out.persons),
            default={},
            warnings=warnings,
            module_name="face",
        )
        track_predictions: dict[str, TrackPrediction] = self._safe_module_call(
            lambda: self.tracker.assign(image_ref, detection_out.persons),
            default={},
            warnings=warnings,
            module_name="tracker",
        )
        object_predictions: list[ObjectPrediction] = self._safe_module_call(
            lambda: self.objects.detect(image_ref),
            default=[],
            warnings=warnings,
            module_name="objects",
        )

        persons: list[PersonFacts] = []
        for person in detection_out.persons:
            pose = pose_predictions.get(person.detection_id)
            parsed = parsing_predictions.get(person.detection_id)
            face = face_predictions.get(person.detection_id)
            tracked = track_predictions.get(person.detection_id)

            garments = []
            if parsed:
                garments = [
                    {
                        "type": garment.garment_type,
                        "state": garment.state,
                        "confidence": garment.confidence,
                        "source": garment.source,
                    }
                    for garment in parsed.garments
                ]

            persons.append(
                PersonFacts(
                    bbox=person.bbox,
                    bbox_confidence=person.confidence,
                    bbox_source=person.source,
                    mask_ref=parsed.mask_ref if parsed else None,
                    mask_confidence=parsed.mask_confidence if parsed else 0.0,
                    mask_source=parsed.source if parsed else "fallback",
                    pose=pose.pose if pose else PoseState(),
                    pose_confidence=pose.confidence if pose else 0.0,
                    pose_source=pose.source if pose else "fallback",
                    expression=face.expression if face else ExpressionState(),
                    expression_confidence=face.expression_confidence if face else 0.0,
                    expression_source=face.source if face else "fallback",
                    orientation=face.orientation if face else OrientationState(),
                    orientation_confidence=face.orientation_confidence if face else 0.0,
                    orientation_source=face.source if face else "fallback",
                    track_id=tracked.track_id if tracked else None,
                    track_confidence=tracked.confidence if tracked else 0.0,
                    track_source=tracked.source if tracked else "fallback",
                    garments=garments,
                )
            )

        objects = [
            ObjectFacts(
                object_type=obj.object_type,
                bbox=obj.bbox,
                confidence=obj.confidence,
                source=obj.source,
            )
            for obj in object_predictions
        ]
        return PerceptionOutput(
            persons=persons,
            objects=objects,
            frame_size=detection_out.frame_size,
            warnings=warnings,
        )
