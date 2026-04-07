from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from core.input_layer import AssetFrame
from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.detector import BackendConfig, Detector, DetectorOutput, YoloPersonDetectorAdapter
from perception.frame_context import FrameLike, ensure_frame_context, unwrap_frame
from perception.face import EmoNetFaceAnalyzerAdapter, FaceAnalyzer, FacePrediction
from perception.objects import MonoDepthEstimator, ObjectDetector, ObjectPrediction, YoloObjectDetectorAdapter
from perception.parser import HumanParser, ParserStackConfig, ParsingPrediction, SegFormerHumanParserAdapter
from perception.pose import PoseEstimator, PosePrediction, VitPoseAdapter
from perception.profiling import StageTimer
from perception.tracker import ByteTrackAdapter, PersonTracker, TrackPrediction
from utils_tensor import shape

T = TypeVar("T")


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
    input_mode: str = "unknown"


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
        self._builtin_detector_fallback: Detector | None = None
        self._builtin_pose_fallback: PoseEstimator | None = None
        self._builtin_parser_fallback: HumanParser | None = None
        self._builtin_face_fallback: FaceAnalyzer | None = None
        self._builtin_tracker_fallback: PersonTracker | None = None
        self._builtin_objects_fallback: ObjectDetector | None = None
        self._builtin_depth_fallback: MonoDepthEstimator | None = None

    def _safe_module_call(
        self,
        fn: Callable[[], T],
        *,
        fallback_fn: Callable[[], T] | None,
        default_value: T,
        warnings: list[str],
        module_name: str,
        out: PerceptionOutput,
        success_mode: str = "native",
    ) -> T:
        start = time.perf_counter()
        try:
            value = fn()
            out.module_fallbacks[module_name] = success_mode
            return value
        except Exception as exc:
            warnings.append(f"{module_name}_unavailable:{exc}")
            if fallback_fn is None:
                warnings.append(f"{module_name}_no_fallback")
                out.module_fallbacks[module_name] = "error:no-fallback"
                return default_value
            try:
                value = fallback_fn()
                out.module_fallbacks[module_name] = "fallback"
                return value
            except Exception as fb_exc:
                warnings.append(f"{module_name}_fallback_failed:{fb_exc}")
                out.module_fallbacks[module_name] = "error:fallback-failed"
                return default_value
            finally:
                out.module_latency_ms[module_name] = round((time.perf_counter() - start) * 1000.0, 3)
        finally:
            out.module_latency_ms.setdefault(module_name, round((time.perf_counter() - start) * 1000.0, 3))

    def _get_builtin_detector_fallback(self) -> Detector:
        if self._builtin_detector_fallback is None:
            self._builtin_detector_fallback = YoloPersonDetectorAdapter(BackendConfig(backend="builtin"))
        return self._builtin_detector_fallback

    def _get_builtin_pose_fallback(self) -> PoseEstimator:
        if self._builtin_pose_fallback is None:
            self._builtin_pose_fallback = VitPoseAdapter(BackendConfig(backend="builtin"))
        return self._builtin_pose_fallback

    def _get_builtin_parser_fallback(self) -> HumanParser:
        if self._builtin_parser_fallback is None:
            self._builtin_parser_fallback = SegFormerHumanParserAdapter(BackendConfig(backend="builtin"))
        return self._builtin_parser_fallback

    def _get_builtin_face_fallback(self) -> FaceAnalyzer:
        if self._builtin_face_fallback is None:
            self._builtin_face_fallback = EmoNetFaceAnalyzerAdapter(BackendConfig(backend="builtin"))
        return self._builtin_face_fallback

    def _get_builtin_tracker_fallback(self) -> PersonTracker:
        if self._builtin_tracker_fallback is None:
            self._builtin_tracker_fallback = ByteTrackAdapter(BackendConfig(backend="builtin"))
        return self._builtin_tracker_fallback

    def _get_builtin_objects_fallback(self) -> ObjectDetector:
        if self._builtin_objects_fallback is None:
            self._builtin_objects_fallback = YoloObjectDetectorAdapter(BackendConfig(backend="builtin"))
        return self._builtin_objects_fallback

    def _get_builtin_depth_fallback(self) -> MonoDepthEstimator:
        if self._builtin_depth_fallback is None:
            self._builtin_depth_fallback = MonoDepthEstimator(BackendConfig(backend="builtin"))
        return self._builtin_depth_fallback

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

    def analyze(self, frame: FrameLike) -> PerceptionOutput:
        frame_ctx = ensure_frame_context(frame)
        out = PerceptionOutput()
        warnings = out.warnings

        detection_out = self._safe_module_call(
            lambda: self.detector.detect(frame_ctx),
            fallback_fn=(lambda: self._get_builtin_detector_fallback().detect(frame_ctx)) if self._backend_fallback_enabled(self.detector) else None,
            default_value=DetectorOutput(),
            warnings=warnings,
            module_name="detector",
            out=out,
            success_mode=self._module_success_mode(self.detector),
        )

        pose_predictions: dict[str, PosePrediction] = self._safe_module_call(
            lambda: self.pose.estimate(frame_ctx, detection_out.persons),
            fallback_fn=(lambda: self._get_builtin_pose_fallback().estimate(frame_ctx, detection_out.persons)) if self._backend_fallback_enabled(self.pose) else None,
            default_value={},
            warnings=warnings,
            module_name="pose",
            out=out,
            success_mode=self._module_success_mode(self.pose),
        )
        parsing_predictions: dict[str, ParsingPrediction] = self._safe_module_call(
            lambda: self.parser.parse(frame_ctx, detection_out.persons),
            fallback_fn=(lambda: self._get_builtin_parser_fallback().parse(frame_ctx, detection_out.persons)) if self._backend_fallback_enabled(self.parser) else None,
            default_value={},
            warnings=warnings,
            module_name="parser",
            out=out,
            success_mode=self._module_success_mode(self.parser),
        )
        face_predictions: dict[str, FacePrediction] = self._safe_module_call(
            lambda: self.face.analyze(frame_ctx, detection_out.persons),
            fallback_fn=(lambda: self._get_builtin_face_fallback().analyze(frame_ctx, detection_out.persons)) if self._backend_fallback_enabled(self.face) else None,
            default_value={},
            warnings=warnings,
            module_name="face",
            out=out,
            success_mode=self._module_success_mode(self.face),
        )
        track_predictions: dict[str, TrackPrediction] = self._safe_module_call(
            lambda: self.tracker.assign(frame_ctx, detection_out.persons),
            fallback_fn=(lambda: self._get_builtin_tracker_fallback().assign(frame_ctx, detection_out.persons)) if self._backend_fallback_enabled(self.tracker) else None,
            default_value={},
            warnings=warnings,
            module_name="tracker",
            out=out,
            success_mode=self._module_success_mode(self.tracker),
        )
        object_predictions: list[ObjectPrediction] = self._safe_module_call(
            lambda: self.objects.detect(frame_ctx),
            fallback_fn=(lambda: self._get_builtin_objects_fallback().detect(frame_ctx)) if self._backend_fallback_enabled(self.objects) else None,
            default_value=[],
            warnings=warnings,
            module_name="objects",
            out=out,
            success_mode=self._module_success_mode(self.objects),
        )
        out.depth_score = self._safe_module_call(
            lambda: self.depth.estimate(frame_ctx),
            fallback_fn=(lambda: self._get_builtin_depth_fallback().estimate(frame_ctx)) if self._backend_fallback_enabled(self.depth) else None,
            default_value=None,
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

        raw_frame = unwrap_frame(frame_ctx)
        if isinstance(raw_frame, str):
            out.frame_size = detection_out.frame_size
            out.input_mode = "string_ref_fallback"
        else:
            tensor = raw_frame.tensor if isinstance(raw_frame, AssetFrame) else raw_frame
            h, w, _ = shape(tensor)
            out.frame_size = (w, h)
            out.input_mode = "frame_tensor"
        out.module_fallbacks["input_mode"] = out.input_mode

        out.module_confidence = {
            "detector": max([p.bbox_confidence for p in out.persons], default=0.0),
            "pose": max([p.pose_confidence for p in out.persons], default=0.0),
            "face": max([p.expression_confidence for p in out.persons], default=0.0),
            "tracker": max([p.track_confidence for p in out.persons], default=0.0),
            "objects": max([o.confidence for o in out.objects], default=0.0),
        }
        return out

    def analyze_video(self, frames: list[FrameLike], batch_size: int = 4) -> list[PerceptionOutput]:
        outputs: list[PerceptionOutput] = []
        for start in range(0, len(frames), max(1, batch_size)):
            for frame in frames[start : start + max(1, batch_size)]:
                outputs.append(self.analyze(frame))
        return outputs


class ParserOnlyPipeline:
    """Облегченный runtime путь для parser validation/smoke."""

    def __init__(self, detector: Detector | None = None, parser: HumanParser | None = None, backends: PerceptionBackendsConfig | None = None) -> None:
        cfg = backends or PerceptionBackendsConfig()
        self.detector = detector or YoloPersonDetectorAdapter(cfg.detector)
        self.parser = parser or SegFormerHumanParserAdapter(cfg.parser)
        self._builtin_detector_fallback: Detector | None = None
        self._builtin_parser_fallback: HumanParser | None = None

    def _safe_module_call(
        self,
        fn: Callable[[], T],
        *,
        fallback_fn: Callable[[], T] | None,
        default_value: T,
        warnings: list[str],
        module_name: str,
        out: PerceptionOutput,
        success_mode: str,
    ) -> T:
        start = time.perf_counter()
        try:
            value = fn()
            out.module_fallbacks[module_name] = success_mode
            return value
        except Exception as exc:
            warnings.append(f"{module_name}_unavailable:{exc}")
            if fallback_fn is None:
                warnings.append(f"{module_name}_no_fallback")
                out.module_fallbacks[module_name] = "error:no-fallback"
                return default_value
            try:
                value = fallback_fn()
                out.module_fallbacks[module_name] = "fallback"
                return value
            except Exception as fb_exc:
                warnings.append(f"{module_name}_fallback_failed:{fb_exc}")
                out.module_fallbacks[module_name] = "error:fallback-failed"
                return default_value
            finally:
                out.module_latency_ms[module_name] = round((time.perf_counter() - start) * 1000.0, 3)
        finally:
            out.module_latency_ms.setdefault(module_name, round((time.perf_counter() - start) * 1000.0, 3))

    def _get_builtin_detector_fallback(self) -> Detector:
        if self._builtin_detector_fallback is None:
            self._builtin_detector_fallback = YoloPersonDetectorAdapter(BackendConfig(backend="builtin"))
        return self._builtin_detector_fallback

    def _get_builtin_parser_fallback(self) -> HumanParser:
        if self._builtin_parser_fallback is None:
            self._builtin_parser_fallback = SegFormerHumanParserAdapter(BackendConfig(backend="builtin"))
        return self._builtin_parser_fallback

    def analyze(self, frame: FrameLike, profiler: StageTimer | None = None) -> PerceptionOutput:
        out = PerceptionOutput()
        timer = profiler or StageTimer(enabled=False)
        frame_ctx = ensure_frame_context(frame)
        frame_ctx.put("profiler", timer)

        with timer.track("detector"):
            detection_out = self._safe_module_call(
                lambda: self.detector.detect(frame_ctx),
                fallback_fn=(lambda: self._get_builtin_detector_fallback().detect(frame_ctx))
                if PerceptionPipeline._backend_fallback_enabled(self.detector)
                else None,
                default_value=DetectorOutput(),
                warnings=out.warnings,
                module_name="detector",
                out=out,
                success_mode=PerceptionPipeline._module_success_mode(self.detector),
            )
        with timer.track("parser_total"):
            parsing_predictions = self._safe_module_call(
                lambda: self.parser.parse(frame_ctx, detection_out.persons),
                fallback_fn=(lambda: self._get_builtin_parser_fallback().parse(frame_ctx, detection_out.persons))
                if PerceptionPipeline._backend_fallback_enabled(self.parser)
                else None,
                default_value={},
                warnings=out.warnings,
                module_name="parser",
                out=out,
                success_mode=PerceptionPipeline._module_success_mode(self.parser),
            )

        persons: list[PersonFacts] = []
        for person in detection_out.persons:
            parsed = parsing_predictions.get(person.detection_id)
            persons.append(
                PersonFacts(
                    bbox=person.bbox,
                    bbox_confidence=person.confidence,
                    bbox_source=person.source,
                    mask_ref=parsed.mask_ref if parsed else None,
                    mask_confidence=parsed.mask_confidence if parsed else 0.0,
                    mask_source=parsed.source if parsed else "fallback",
                    pose=PoseState(),
                    pose_confidence=0.0,
                    pose_source="disabled:parser-only",
                    expression=ExpressionState(),
                    expression_confidence=0.0,
                    expression_source="disabled:parser-only",
                    orientation=OrientationState(),
                    orientation_confidence=0.0,
                    orientation_source="disabled:parser-only",
                    track_id=None,
                    track_confidence=0.0,
                    track_source="disabled:parser-only",
                    garments=[{"type": g.garment_type, "state": g.state, "confidence": g.confidence, "source": g.source, "mask_ref": g.mask_ref, "coverage_targets": g.coverage_targets, "attachment_targets": g.attachment_targets, "layer_hint": g.layer_hint} for g in (parsed.garments if parsed else [])],
                    hand_landmarks={},
                    face_landmarks=[],
                    depth_order=1.0 - (person.bbox.y + person.bbox.h),
                    occlusion_hints=(parsed.occlusion_hints if parsed else []),
                    body_parts=[{"part_type": b.part_type, "mask_ref": b.mask_ref, "confidence": b.confidence, "visibility": b.visibility, "source": b.source} for b in (parsed.body_parts if parsed else [])],
                    face_regions=[{"region_type": r.region_type, "mask_ref": r.mask_ref, "confidence": r.confidence, "source": r.source} for r in (parsed.face_regions if parsed else [])],
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
        out.objects = []
        out.depth_score = None
        out.module_fallbacks = {
            "detector": out.module_fallbacks.get("detector", "unknown"),
            "parser": out.module_fallbacks.get("parser", "unknown"),
            "pose": "disabled:parser-only",
            "face": "disabled:parser-only",
            "tracker": "disabled:parser-only",
            "objects": "disabled:parser-only",
            "depth": "disabled:parser-only",
        }
        out.module_confidence = {
            "detector": max([p.bbox_confidence for p in out.persons], default=0.0),
            "pose": 0.0,
            "parser": max([p.mask_confidence for p in out.persons], default=0.0),
            "face": 0.0,
            "tracker": 0.0,
            "objects": 0.0,
            "depth": 0.0,
        }
        out.module_latency_ms.setdefault("pose", 0.0)
        out.module_latency_ms.setdefault("face", 0.0)
        out.module_latency_ms.setdefault("tracker", 0.0)
        out.module_latency_ms.setdefault("objects", 0.0)
        out.module_latency_ms.setdefault("depth", 0.0)

        raw_frame = unwrap_frame(frame_ctx)
        if isinstance(raw_frame, str):
            out.frame_size = detection_out.frame_size
            out.input_mode = "string_ref_fallback"
        else:
            tensor = raw_frame.tensor if isinstance(raw_frame, AssetFrame) else raw_frame
            h, w, _ = shape(tensor)
            out.frame_size = (w, h)
            out.input_mode = "frame_tensor"
        out.module_fallbacks["input_mode"] = out.input_mode
        return out
