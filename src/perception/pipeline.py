from __future__ import annotations

import time
from dataclasses import dataclass, field
import importlib.util
from typing import Any, Callable, TypeVar

from core.input_layer import AssetFrame
from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.detector import BackendConfig, Detector, DetectorOutput, YoloPersonDetectorAdapter
from perception.frame_context import FrameLike, ensure_frame_context, unwrap_frame
from perception.mask_store import DEFAULT_MASK_STORE, InMemoryMaskStore
from perception.face import EmoNetFaceAnalyzerAdapter, FaceAnalyzer, FacePrediction
from perception.objects import MonoDepthEstimator, ObjectDetector, ObjectPrediction, YoloObjectDetectorAdapter
from perception.parser import HumanParser, ParserStackConfig, ParsingPrediction, SegFormerHumanParserAdapter
from perception.pose import MediaPipePoseAdapter, PoseEstimator, PosePrediction, VitPoseAdapter, YoloPoseAdapter
from perception.profiling import StageTimer
from perception.tracker import ByteTrackAdapter, PersonTracker, TrackPrediction
from utils_tensor import shape

T = TypeVar("T")


@dataclass(slots=True)
class PerceptionBackendsConfig:
    detector: BackendConfig = field(default_factory=lambda: BackendConfig(backend="builtin", checkpoint="yolov8n-seg.pt"))
    pose: BackendConfig = field(default_factory=lambda: BackendConfig(backend="builtin", checkpoint=""))
    parser: ParserStackConfig | BackendConfig = field(default_factory=ParserStackConfig)
    objects: BackendConfig = field(default_factory=lambda: BackendConfig(backend="builtin", checkpoint="yolov8n.pt"))
    face: BackendConfig = field(default_factory=lambda: BackendConfig(backend="builtin", checkpoint=""))
    perception_backend: str = "auto"
    parser_model: str = "fashn-ai/fashn-human-parser"
    yolo_seg_model: str = "yolo11n-seg.pt"
    yolo_pose_model: str = "yolo11n-pose.pt"
    perception_device: str = "auto"
    strict_perception: bool = False
    reset_mask_store_per_analyze: bool = False


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
    background_masks: dict[str, str] = field(default_factory=dict)
    coverage_hints: dict[str, list[str]] = field(default_factory=dict)
    visibility_hints: dict[str, str] = field(default_factory=dict)
    provenance_by_region: dict[str, str] = field(default_factory=dict)
    parser_class_names: dict[str, str] = field(default_factory=dict)
    region_mask_refs: dict[str, list[str]] = field(default_factory=dict)
    person_id: str = ""
    bbox_observation_status: str = "observed"
    mask_evidence_type: str = "unknown"
    track_provenance: str = "unknown"
    identity_observation_status: str = "unknown"
    suitable_for_memory_seeding: bool = False


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
    mask_store: dict[str, dict[str, object]] = field(default_factory=dict)
    parser_summary: dict[str, object] = field(default_factory=dict)
    diagnostics: list[dict[str, object]] = field(default_factory=list)
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
        if pose is not None:
            self.pose = pose
        elif cfg.pose.backend in {"ultralytics", "yolo_pose"}:
            self.pose = YoloPoseAdapter(cfg.pose)
        elif cfg.pose.backend == "mediapipe":
            self.pose = MediaPipePoseAdapter(cfg.pose)
        else:
            self.pose = VitPoseAdapter(cfg.pose)
        self.parser = parser or SegFormerHumanParserAdapter(self._parser_stack_config(cfg.parser))
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
        self.strict_perception = cfg.strict_perception
        self.reset_mask_store_per_analyze = cfg.reset_mask_store_per_analyze
        self.mask_store = InMemoryMaskStore()


    def _adopt_referenced_legacy_masks(self, out: PerceptionOutput) -> None:
        """Adopt masks produced by legacy/test modules that still wrote DEFAULT_MASK_STORE.

        Production adapters receive the instance store through frame context. This bridge is
        limited to refs explicitly returned in the current PerceptionOutput, so unrelated
        DEFAULT_MASK_STORE entries do not leak into the pipeline-owned store.
        """

        refs: set[str] = set()
        for person in out.persons:
            for ref in [person.mask_ref, *person.body_part_masks.values(), *person.face_region_masks.values(), *person.garment_masks.values(), *person.accessory_masks.values(), *person.background_masks.values()]:
                if ref:
                    refs.add(ref)
            for collection in (person.body_parts, person.face_regions, person.garments):
                for item in collection:
                    ref = item.get("mask_ref") if isinstance(item, dict) else None
                    if ref:
                        refs.add(str(ref))
            for values in person.region_mask_refs.values():
                refs.update(str(ref) for ref in values if ref)
        for ref in refs:
            if self.mask_store.get(ref) is not None:
                continue
            stored = DEFAULT_MASK_STORE.get(ref)
            if stored is None:
                continue
            extra = dict(stored.extra)
            extra.update({"adopted_legacy_default_store": True, "original_source": stored.source})
            self.mask_store.put(
                stored.payload,
                confidence=stored.confidence,
                source=f"legacy_adopted:{stored.source}",
                prefix="adopted_legacy",
                mask_kind=stored.mask_kind,
                backend=f"legacy_adopted:{stored.backend}",
                roi_bbox=stored.roi_bbox,
                frame_size=stored.frame_size,
                tags=list(stored.tags) + ["adopted_legacy_default_store", "not_production_observed"],
                extra=extra,
                ref=stored.ref,
            )

    @staticmethod
    def _parser_stack_config(config: ParserStackConfig | BackendConfig) -> ParserStackConfig:
        if isinstance(config, BackendConfig):
            return ParserStackConfig.from_legacy_backend_config(config)
        return config

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
            out.diagnostics.append({"module": module_name, "level": "error", "message": str(exc)})
            if self.strict_perception:
                raise RuntimeError(f"strict perception backend failed for {module_name}: {exc}") from exc
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

    def analyze(self, frame: FrameLike, *, reset_mask_store: bool | None = None) -> PerceptionOutput:
        should_reset = self.reset_mask_store_per_analyze if reset_mask_store is None else reset_mask_store
        if should_reset:
            self.mask_store.clear()
        frame_ctx = ensure_frame_context(frame)
        frame_ctx.put("mask_store", self.mask_store)
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
        parser_module_confidence = max((pred.mask_confidence for pred in parsing_predictions.values()), default=0.0)
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
        for idx, person in enumerate(detection_out.persons, start=1):
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
                        "bbox_xyxy": g.bbox_xyxy,
                        "pixel_count": g.pixel_count,
                        "parser_class_name": g.parser_class_name,
                        "class_id": g.class_id,
                    }
                    for g in parsed.garments
                ]
                body_parts = [
                    {"part_type": b.part_type, "mask_ref": b.mask_ref, "confidence": b.confidence, "visibility": b.visibility, "source": b.source, "bbox_xyxy": b.bbox_xyxy, "pixel_count": b.pixel_count, "parser_class_name": b.parser_class_name, "class_id": b.class_id, "observation_status": "observed" if b.mask_ref else "unknown", "provenance": b.source, "mask_evidence_type": "parser_mask" if b.mask_ref else "missing", "suitable_for_memory_seeding": bool(b.mask_ref and b.confidence >= 0.5)}
                    for b in parsed.body_parts
                ]
                face_regions = [
                    {"region_type": r.region_type, "mask_ref": r.mask_ref, "confidence": r.confidence, "source": r.source, "bbox_xyxy": r.bbox_xyxy, "pixel_count": r.pixel_count, "parser_class_name": r.parser_class_name, "class_id": r.class_id, "observation_status": "observed" if r.mask_ref else "unknown", "provenance": r.source, "mask_evidence_type": "parser_mask" if r.mask_ref else "missing", "suitable_for_memory_seeding": bool(r.mask_ref and r.confidence >= 0.5)}
                    for r in parsed.face_regions
                ]

            person_depth = 1.0 - (person.bbox.y + person.bbox.h)
            parser_mask_ref = parsed.mask_ref if (parsed and parsed.mask_ref) else None
            detector_mask_ref = person.mask_ref
            person_mask_ref = parser_mask_ref or detector_mask_ref
            person_mask_source = parsed.source if parser_mask_ref else (person.mask_source or "fallback")
            person_mask_confidence = parsed.mask_confidence if parser_mask_ref else person.mask_confidence
            track_provenance = "single_frame_observed"
            identity_status = "single_frame_anchor"
            track_id = None
            track_confidence = person.confidence
            if getattr(self, "_perception_video_tracking_active", False) and tracked:
                previous_video_track_ids = getattr(self, "_perception_video_seen_track_ids", set())
                current_video_track_ids = getattr(self, "_perception_video_current_track_ids", set())
                track_id = tracked.track_id
                track_confidence = tracked.confidence
                track_provenance = tracked.source
                identity_status = "multi_frame_tracked" if tracked.track_id in previous_video_track_ids else "tracker_single_frame_observed"
                current_video_track_ids.add(tracked.track_id)
                self._perception_video_current_track_ids = current_video_track_ids
            persons.append(
                PersonFacts(
                    bbox=person.bbox,
                    bbox_confidence=person.confidence,
                    bbox_source=person.source,
                    mask_ref=person_mask_ref,
                    mask_confidence=person_mask_confidence,
                    mask_source=person_mask_source,
                    pose=pose.pose if pose else PoseState(),
                    pose_confidence=pose.confidence if pose else 0.0,
                    pose_source=pose.source if pose else "fallback",
                    expression=face.expression if face else ExpressionState(),
                    expression_confidence=face.expression_confidence if face else 0.0,
                    expression_source=face.source if face else "fallback",
                    orientation=face.orientation if face else OrientationState(),
                    orientation_confidence=face.orientation_confidence if face else 0.0,
                    orientation_source=face.source if face else "fallback",
                    track_id=track_id,
                    track_confidence=track_confidence,
                    track_source=track_provenance,
                    garments=garments,
                    hand_landmarks=pose.hand_landmarks if pose else {},
                    face_landmarks=face.face_landmarks if face else (pose.face_landmarks if pose else []),
                    depth_order=person_depth,
                    occlusion_hints=(parsed.occlusion_hints if parsed else []),
                    body_parts=body_parts,
                    face_regions=face_regions,
                    garment_masks=(parsed.enriched.garment_masks if parsed else {}),
                    body_part_masks=(parsed.enriched.body_part_masks if parsed else {}),
                    face_region_masks=(parsed.enriched.face_region_masks if parsed else {}),
                    accessory_masks=(parsed.enriched.accessory_masks if parsed else {}),
                    background_masks=(parsed.enriched.background_masks if parsed else {}),
                    coverage_hints=(parsed.enriched.coverage_hints if parsed else {}),
                    visibility_hints=(parsed.enriched.visibility_hints if parsed else {}),
                    provenance_by_region=(parsed.enriched.provenance_by_region if parsed else {}),
                    parser_class_names=(parsed.enriched.parser_class_names if parsed else {}),
                    region_mask_refs=(parsed.enriched.region_mask_refs if parsed else {}),
                    person_id=f"person_{idx}",
                    bbox_observation_status="observed",
                    mask_evidence_type=("parser_mask" if parser_mask_ref else ("detector_instance_mask" if detector_mask_ref else "missing")),
                    track_provenance=track_provenance,
                    identity_observation_status=identity_status,
                    suitable_for_memory_seeding=bool(person_mask_ref and person_mask_confidence >= 0.5),
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

        self._adopt_referenced_legacy_masks(out)
        out.mask_store = self.mask_store.snapshot_metadata()
        refs_by_canonical: dict[str, list[str]] = {}
        parser_classes: dict[str, str] = {}
        for person in out.persons:
            parser_classes.update(person.parser_class_names)
            for key, refs in person.region_mask_refs.items():
                refs_by_canonical.setdefault(key, []).extend(refs)
        out.parser_summary = {
            "persons": len(out.persons),
            "parser_classes": parser_classes,
            "canonical_regions": sorted(refs_by_canonical.keys()),
            "region_mask_refs": refs_by_canonical,
            "mask_refs": sorted(out.mask_store.keys()),
        }
        out.module_confidence = {
            "detector": max([p.bbox_confidence for p in out.persons], default=0.0),
            "pose": max([p.pose_confidence for p in out.persons], default=0.0),
            "parser": parser_module_confidence,
            "face": max([p.expression_confidence for p in out.persons], default=0.0),
            "tracker": max([p.track_confidence for p in out.persons], default=0.0),
            "objects": max([o.confidence for o in out.objects], default=0.0),
        }
        return out

    def analyze_video(self, frames: list[FrameLike], batch_size: int = 4) -> list[PerceptionOutput]:
        if self.reset_mask_store_per_analyze:
            self.mask_store.clear()
        outputs: list[PerceptionOutput] = []
        self._perception_video_tracking_active = True
        self._perception_video_seen_track_ids: set[str] = set()
        try:
            for start in range(0, len(frames), max(1, batch_size)):
                for frame in frames[start : start + max(1, batch_size)]:
                    self._perception_video_current_track_ids: set[str] = set()
                    outputs.append(self.analyze(frame, reset_mask_store=False))
                    self._perception_video_seen_track_ids.update(self._perception_video_current_track_ids)
        finally:
            self._perception_video_tracking_active = False
            self._perception_video_seen_track_ids = set()
            self._perception_video_current_track_ids = set()
        return outputs


class ParserOnlyPipeline:
    """Облегченный runtime путь для parser validation/smoke."""

    def __init__(self, detector: Detector | None = None, parser: HumanParser | None = None, backends: PerceptionBackendsConfig | None = None) -> None:
        cfg = backends or PerceptionBackendsConfig()
        self.detector = detector or YoloPersonDetectorAdapter(cfg.detector)
        self.parser = parser or SegFormerHumanParserAdapter(PerceptionPipeline._parser_stack_config(cfg.parser))
        self._builtin_detector_fallback: Detector | None = None
        self._builtin_parser_fallback: HumanParser | None = None
        self.strict_perception = cfg.strict_perception
        self.reset_mask_store_per_analyze = cfg.reset_mask_store_per_analyze
        self.mask_store = InMemoryMaskStore()


    def _adopt_referenced_legacy_masks(self, out: PerceptionOutput) -> None:
        """Adopt masks produced by legacy/test modules that still wrote DEFAULT_MASK_STORE.

        Production adapters receive the instance store through frame context. This bridge is
        limited to refs explicitly returned in the current PerceptionOutput, so unrelated
        DEFAULT_MASK_STORE entries do not leak into the pipeline-owned store.
        """

        refs: set[str] = set()
        for person in out.persons:
            for ref in [person.mask_ref, *person.body_part_masks.values(), *person.face_region_masks.values(), *person.garment_masks.values(), *person.accessory_masks.values(), *person.background_masks.values()]:
                if ref:
                    refs.add(ref)
            for collection in (person.body_parts, person.face_regions, person.garments):
                for item in collection:
                    ref = item.get("mask_ref") if isinstance(item, dict) else None
                    if ref:
                        refs.add(str(ref))
            for values in person.region_mask_refs.values():
                refs.update(str(ref) for ref in values if ref)
        for ref in refs:
            if self.mask_store.get(ref) is not None:
                continue
            stored = DEFAULT_MASK_STORE.get(ref)
            if stored is None:
                continue
            extra = dict(stored.extra)
            extra.update({"adopted_legacy_default_store": True, "original_source": stored.source})
            self.mask_store.put(
                stored.payload,
                confidence=stored.confidence,
                source=f"legacy_adopted:{stored.source}",
                prefix="adopted_legacy",
                mask_kind=stored.mask_kind,
                backend=f"legacy_adopted:{stored.backend}",
                roi_bbox=stored.roi_bbox,
                frame_size=stored.frame_size,
                tags=list(stored.tags) + ["adopted_legacy_default_store", "not_production_observed"],
                extra=extra,
                ref=stored.ref,
            )

    @staticmethod
    def _parser_stack_config(config: ParserStackConfig | BackendConfig) -> ParserStackConfig:
        if isinstance(config, BackendConfig):
            return ParserStackConfig.from_legacy_backend_config(config)
        return config

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
            out.diagnostics.append({"module": module_name, "level": "error", "message": str(exc)})
            if self.strict_perception:
                raise RuntimeError(f"strict perception backend failed for {module_name}: {exc}") from exc
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

    def analyze(self, frame: FrameLike, profiler: StageTimer | None = None, *, reset_mask_store: bool | None = None) -> PerceptionOutput:
        should_reset = self.reset_mask_store_per_analyze if reset_mask_store is None else reset_mask_store
        if should_reset:
            self.mask_store.clear()
        out = PerceptionOutput()
        timer = profiler or StageTimer(enabled=False)
        frame_ctx = ensure_frame_context(frame)
        frame_ctx.put("mask_store", self.mask_store)
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
        for idx, person in enumerate(detection_out.persons, start=1):
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
                    track_source="single_frame_observed",
                    garments=[{"type": g.garment_type, "state": g.state, "confidence": g.confidence, "source": g.source, "mask_ref": g.mask_ref, "coverage_targets": g.coverage_targets, "attachment_targets": g.attachment_targets, "layer_hint": g.layer_hint} for g in (parsed.garments if parsed else [])],
                    hand_landmarks={},
                    face_landmarks=[],
                    depth_order=1.0 - (person.bbox.y + person.bbox.h),
                    occlusion_hints=(parsed.occlusion_hints if parsed else []),
                    body_parts=[{"part_type": b.part_type, "mask_ref": b.mask_ref, "confidence": b.confidence, "visibility": b.visibility, "source": b.source, "provenance": b.source, "observation_status": "observed" if b.mask_ref else "unknown", "mask_evidence_type": "parser_mask" if b.mask_ref else "missing", "suitable_for_memory_seeding": bool(b.mask_ref and b.confidence >= 0.5)} for b in (parsed.body_parts if parsed else [])],
                    face_regions=[{"region_type": r.region_type, "mask_ref": r.mask_ref, "confidence": r.confidence, "source": r.source, "provenance": r.source, "observation_status": "observed" if r.mask_ref else "unknown", "mask_evidence_type": "parser_mask" if r.mask_ref else "missing", "suitable_for_memory_seeding": bool(r.mask_ref and r.confidence >= 0.5)} for r in (parsed.face_regions if parsed else [])],
                    garment_masks=(parsed.enriched.garment_masks if parsed else {}),
                    body_part_masks=(parsed.enriched.body_part_masks if parsed else {}),
                    face_region_masks=(parsed.enriched.face_region_masks if parsed else {}),
                    accessory_masks=(parsed.enriched.accessory_masks if parsed else {}),
                    background_masks=(parsed.enriched.background_masks if parsed else {}),
                    coverage_hints=(parsed.enriched.coverage_hints if parsed else {}),
                    visibility_hints=(parsed.enriched.visibility_hints if parsed else {}),
                    provenance_by_region=(parsed.enriched.provenance_by_region if parsed else {}),
                    parser_class_names=(parsed.enriched.parser_class_names if parsed else {}),
                    region_mask_refs=(parsed.enriched.region_mask_refs if parsed else {}),
                    person_id=f"person_{idx}",
                    bbox_observation_status="observed",
                    mask_evidence_type=("parser_mask" if (parsed and parsed.mask_ref) else "missing"),
                    track_provenance="single_frame_observed",
                    identity_observation_status="single_frame_anchor",
                    suitable_for_memory_seeding=bool(parsed and parsed.mask_ref and parsed.mask_confidence >= 0.5),
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
        self._adopt_referenced_legacy_masks(out)
        out.mask_store = self.mask_store.snapshot_metadata()
        refs_by_canonical: dict[str, list[str]] = {}
        parser_classes: dict[str, str] = {}
        for person in out.persons:
            parser_classes.update(person.parser_class_names)
            for key, refs in person.region_mask_refs.items():
                refs_by_canonical.setdefault(key, []).extend(refs)
        out.parser_summary = {
            "persons": len(out.persons),
            "parser_classes": parser_classes,
            "canonical_regions": sorted(refs_by_canonical.keys()),
            "region_mask_refs": refs_by_canonical,
            "mask_refs": sorted(out.mask_store.keys()),
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


def _resolve_perception_device(device: str) -> str:
    if device != "auto":
        return device
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    try:
        import torch  # type: ignore

        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"


def real_human_parsing_config(
    *,
    device: str = "auto",
    parser_model: str = "fashn-ai/fashn-human-parser",
    yolo_seg_model: str = "yolo11n-seg.pt",
    yolo_pose_model: str = "yolo11n-pose.pt",
    strict_perception: bool = False,
    reset_mask_store_per_analyze: bool = False,
) -> PerceptionBackendsConfig:
    runtime_device = _resolve_perception_device(device)
    return PerceptionBackendsConfig(
        detector=BackendConfig(backend="ultralytics", checkpoint=yolo_seg_model, device=runtime_device),
        pose=BackendConfig(backend="ultralytics", checkpoint=yolo_pose_model, device=runtime_device),
        parser=BackendConfig(backend="hf", checkpoint=parser_model, device=runtime_device),
        objects=BackendConfig(backend="builtin"),
        face=BackendConfig(backend="builtin"),
        perception_backend="real_human_parsing",
        parser_model=parser_model,
        yolo_seg_model=yolo_seg_model,
        yolo_pose_model=yolo_pose_model,
        perception_device=runtime_device,
        strict_perception=strict_perception,
        reset_mask_store_per_analyze=reset_mask_store_per_analyze,
    )
