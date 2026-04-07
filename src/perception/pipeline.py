from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.input_layer import AssetFrame
from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.detector import BackendConfig, Detector, DetectorOutput, YoloPersonDetectorAdapter
from perception.face import EmoNetFaceAnalyzerAdapter, FaceAnalyzer, FacePrediction
from perception.objects import MonoDepthEstimator, ObjectDetector, ObjectPrediction, YoloObjectDetectorAdapter
from perception.parser import HumanParser, ParserStackConfig, ParsingPrediction, SegFormerHumanParserAdapter
from perception.pose import PoseEstimator, PosePrediction, VitPoseAdapter
from perception.tracker import ByteTrackAdapter, PersonTracker, TrackPrediction
from utils_tensor import shape


@dataclass(slots=True)
class PerceptionBackendsConfig:
    detector: BackendConfig = field(default_factory=lambda: BackendConfig(backend="builtin", checkpoint="yolov8n.pt"))
    pose: BackendConfig = field(default_factory=lambda: BackendConfig(backend="builtin", checkpoint=""))
    parser: ParserStackConfig | BackendConfig = field(default_factory=ParserStackConfig)
    objects: BackendConfig = field(default_factory=lambda: BackendConfig(backend="builtin", checkpoint="yolov8n.pt"))
    face: BackendConfig = field(default_factory=lambda: BackendConfig(backend="builtin", checkpoint=""))


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
    hand_landmarks: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    face_landmarks: list[tuple[float, float]] = field(default_factory=list)
    depth_order: float = 0.0
    occlusion_hints: list[str] = field(default_factory=list)
    body_parts: list[dict] = field(default_factory=list)
    face_regions: list[dict] = field(default_factory=list)
    garment_masks: dict[str, str] = field(default_factory=dict)
    body_part_masks: dict[str, str] = field(default_factory=dict)
    face_region_masks: dict[str, str] = field(default_factory=dict)
    accessory_masks: dict[str, str] = field(default_factory=dict)
    coverage_hints: dict[str, list[str]] = field(default_factory=dict)
    visibility_hints: dict[str, str] = field(default_factory=dict)
    provenance_by_region: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectFacts:
    object_type: str
    bbox: BBox
    confidence: float
    source: str
    depth_order: float = 0.0


@dataclass(slots=True)
class PerceptionOutput:
    persons: list[PersonFacts] = field(default_factory=list)
    objects: list[ObjectFacts] = field(default_factory=list)
    frame_size: tuple[int, int] = (1024, 1024)
    warnings: list[str] = field(default_factory=list)
    module_confidence: dict[str, float] = field(default_factory=dict)
    module_latency_ms: dict[str, float] = field(default_factory=dict)
    module_fallbacks: dict[str, str] = field(default_factory=dict)
    depth_score: float | None = None


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
        depth: MonoDepthEstimator | None = None,
        backends: PerceptionBackendsConfig | None = None,
    ) -> None:
        cfg = backends or PerceptionBackendsConfig()
        self.detector = detector or YoloPersonDetectorAdapter(cfg.detector)
        self.pose = pose or VitPoseAdapter(cfg.pose)
        self.parser = parser or SegFormerHumanParserAdapter(cfg.parser)
        self.face = face or EmoNetFaceAnalyzerAdapter(cfg.face)
        self.objects = objects or YoloObjectDetectorAdapter(cfg.objects)
        self.tracker = tracker or ByteTrackAdapter()
        self.depth = depth or MonoDepthEstimator()

    def _safe_module_call(self, fn, fallback_fn, warnings: list[str], module_name: str, out: PerceptionOutput,
                          success_mode: str = "native"):
        start = time.perf_counter()
        print(f"[MOD] {module_name}: start")
        try:
            value = fn()
            out.module_fallbacks[module_name] = success_mode
            print(f"[MOD] {module_name}: success ({success_mode}) in {(time.perf_counter() - start):.3f}s")
            return value
        except Exception as exc:
            warnings.append(f"{module_name}_unavailable:{exc}")
            print(f"[MOD] {module_name}: failed in {(time.perf_counter() - start):.3f}s -> {exc}")
            try:
                fb_start = time.perf_counter()
                value = fallback_fn()
                out.module_fallbacks[module_name] = "fallback"
                print(f"[MOD] {module_name}: fallback success in {(time.perf_counter() - fb_start):.3f}s")
                return value
            except Exception as fb_exc:
                warnings.append(f"{module_name}_fallback_failed:{fb_exc}")
                out.module_fallbacks[module_name] = "real_backend_error"
                print(f"[MOD] {module_name}: fallback failed -> {fb_exc}")
                raise
            finally:
                out.module_latency_ms[module_name] = round((time.perf_counter() - start) * 1000.0, 3)

    @staticmethod
    def _backend_fallback_enabled(module: object) -> bool:
        if hasattr(module, "is_builtin_backend") and callable(getattr(module, "is_builtin_backend")):
            return not bool(module.is_builtin_backend())
        config = getattr(module, "config", None)
        backend = getattr(config, "backend", None)
        return isinstance(backend, str) and backend not in {"", "builtin"}

    @staticmethod
    def _module_success_mode(module: object) -> str:
        if hasattr(module, "is_builtin_backend") and callable(getattr(module, "is_builtin_backend")):
            return "builtin" if bool(module.is_builtin_backend()) else "native"
        config = getattr(module, "config", None)
        backend = getattr(config, "backend", None)
        return "builtin" if backend in {"", "builtin"} else "native"

    def analyze(self, frame: AssetFrame | list[list[list[float]]] | str) -> PerceptionOutput:
        out = PerceptionOutput()
        warnings = out.warnings

        default_detector = YoloPersonDetectorAdapter(BackendConfig(backend="builtin"))
        detection_out = self._safe_module_call(
            lambda: self.detector.detect(frame),
            fallback_fn=(lambda: default_detector.detect(frame))
            if self._backend_fallback_enabled(self.detector)
            else (lambda: DetectorOutput()),
            warnings=warnings,
            module_name="detector",
            out=out,
            success_mode=self._module_success_mode(self.detector),
        )

        pose_predictions: dict[str, PosePrediction] = self._safe_module_call(
            lambda: self.pose.estimate(frame, detection_out.persons),
            fallback_fn=(lambda: VitPoseAdapter(BackendConfig(backend="builtin")).estimate(frame, detection_out.persons))
            if self._backend_fallback_enabled(self.pose)
            else (lambda: {}),
            warnings=warnings,
            module_name="pose",
            out=out,
            success_mode=self._module_success_mode(self.pose),
        )
        parsing_predictions: dict[str, ParsingPrediction] = self._safe_module_call(
            lambda: self.parser.parse(frame, detection_out.persons),
            fallback_fn=(lambda: SegFormerHumanParserAdapter(BackendConfig(backend="builtin")).parse(frame, detection_out.persons))
            if self._backend_fallback_enabled(self.parser)
            else (lambda: {}),
            warnings=warnings,
            module_name="parser",
            out=out,
            success_mode=self._module_success_mode(self.parser),
        )
        face_predictions: dict[str, FacePrediction] = self._safe_module_call(
            lambda: self.face.analyze(frame, detection_out.persons),
            fallback_fn=(lambda: EmoNetFaceAnalyzerAdapter(BackendConfig(backend="builtin")).analyze(frame, detection_out.persons))
            if self._backend_fallback_enabled(self.face)
            else (lambda: {}),
            warnings=warnings,
            module_name="face",
            out=out,
            success_mode=self._module_success_mode(self.face),
        )
        track_predictions: dict[str, TrackPrediction] = self._safe_module_call(
            lambda: self.tracker.assign(frame, detection_out.persons),
            fallback_fn=(lambda: ByteTrackAdapter(BackendConfig(backend="builtin")).assign(frame, detection_out.persons))
            if self._backend_fallback_enabled(self.tracker)
            else (lambda: {}),
            warnings=warnings,
            module_name="tracker",
            out=out,
            success_mode=self._module_success_mode(self.tracker),
        )
        object_predictions: list[ObjectPrediction] = self._safe_module_call(
            lambda: self.objects.detect(frame),
            fallback_fn=(lambda: YoloObjectDetectorAdapter(BackendConfig(backend="builtin")).detect(frame))
            if self._backend_fallback_enabled(self.objects)
            else (lambda: []),
            warnings=warnings,
            module_name="objects",
            out=out,
            success_mode=self._module_success_mode(self.objects),
        )
        out.depth_score = self._safe_module_call(
            lambda: self.depth.estimate(frame),
            fallback_fn=(lambda: MonoDepthEstimator(BackendConfig(backend="builtin")).estimate(frame))
            if self._backend_fallback_enabled(self.depth)
            else (lambda: None),
            warnings=warnings,
            module_name="depth",
            out=out,
            success_mode=self._module_success_mode(self.depth),
        )

        persons: list[PersonFacts] = []
        for person in detection_out.persons:
            pose = pose_predictions.get(person.detection_id)
            parsed = parsing_predictions.get(person.detection_id)
            face = face_predictions.get(person.detection_id)
            tracked = track_predictions.get(person.detection_id)
            garments = []
            body_parts = []
            face_regions = []
            if parsed:
                garments = [
                    {
                        "type": g.garment_type,
                        "state": g.state,
                        "confidence": g.confidence,
                        "source": g.source,
                        "mask_ref": g.mask_ref,
                        "coverage_targets": g.coverage_targets,
                        "attachment_targets": g.attachment_targets,
                        "layer_hint": g.layer_hint,
                    }
                    for g in parsed.garments
                ]
                body_parts = [
                    {"part_type": b.part_type, "mask_ref": b.mask_ref, "confidence": b.confidence, "visibility": b.visibility, "source": b.source}
                    for b in parsed.body_parts
                ]
                face_regions = [
                    {"region_type": r.region_type, "mask_ref": r.mask_ref, "confidence": r.confidence, "source": r.source}
                    for r in parsed.face_regions
                ]

            person_depth = 1.0 - (person.bbox.y + person.bbox.h)
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
                    hand_landmarks=pose.hand_landmarks if pose else {},
                    face_landmarks=face.face_landmarks if face else [],
                    depth_order=person_depth,
                    occlusion_hints=(parsed.occlusion_hints if parsed else []),
                    body_parts=body_parts,
                    face_regions=face_regions,
                    garment_masks=(parsed.enriched.garment_masks if parsed else {}),
                    body_part_masks=(parsed.enriched.body_part_masks if parsed else {}),
                    face_region_masks=(parsed.enriched.face_region_masks if parsed else {}),
                    accessory_masks=(parsed.enriched.accessory_masks if parsed else {}),
                    coverage_hints=(parsed.enriched.coverage_hints if parsed else {}),
                    visibility_hints=(parsed.enriched.visibility_hints if parsed else {}),
                    provenance_by_region=(parsed.enriched.provenance_by_region if parsed else {}),
                )
            )

        out.persons = persons
        out.objects = [ObjectFacts(o.object_type, o.bbox, o.confidence, o.source, o.depth_order) for o in object_predictions]

        if isinstance(frame, str):
            out.frame_size = detection_out.frame_size
            out.module_fallbacks["input_mode"] = "string_ref_fallback"
        else:
            tensor = frame.tensor if isinstance(frame, AssetFrame) else frame
            h, w, _ = shape(tensor)
            out.frame_size = (w, h)
            out.module_fallbacks["input_mode"] = "frame_tensor"

        out.module_confidence = {
            "detector": max([p.bbox_confidence for p in out.persons], default=0.0),
            "pose": max([p.pose_confidence for p in out.persons], default=0.0),
            "face": max([p.expression_confidence for p in out.persons], default=0.0),
            "tracker": max([p.track_confidence for p in out.persons], default=0.0),
            "objects": max([o.confidence for o in out.objects], default=0.0),
        }
        return out

    def analyze_video(self, frames: list[AssetFrame | list[list[list[float]]] | str], batch_size: int = 4) -> list[PerceptionOutput]:
        outputs: list[PerceptionOutput] = []
        for start in range(0, len(frames), max(1, batch_size)):
            for frame in frames[start : start + max(1, batch_size)]:
                outputs.append(self.analyze(frame))
        return outputs
