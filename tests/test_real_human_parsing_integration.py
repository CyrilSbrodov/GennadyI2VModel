from __future__ import annotations

import importlib
import sys
import types

from core.schema import BBox, GraphDelta, RegionRef, VideoMemory
from perception.detector import DetectorOutput, PersonDetection
from perception.human_parser_mapping import map_human_parser_class
from perception.mask_store import DEFAULT_MASK_STORE
from perception.parser import (
    BodyPartMaskPrediction,
    EnrichedParsingPayload,
    FaceRegionPrediction,
    GarmentPrediction,
    ParsingPrediction,
)
from perception.pipeline import PerceptionPipeline
from perception.pose import PosePrediction
from rendering.roi_renderer import ROISelector
from representation.graph_builder import SceneGraphBuilder
from runtime.region_metadata import build_region_metadata


def _solid(h: int = 16, w: int = 16):
    return [[[0.2, 0.3, 0.4] for _ in range(w)] for _ in range(h)]


def test_human_parser_mapping_core_classes() -> None:
    assert map_human_parser_class("face").canonical_region_type == "face"
    assert map_human_parser_class("hair").category == "face_hair"
    assert map_human_parser_class("arms").canonical_region_type == "arms"
    assert map_human_parser_class("torso").canonical_region_type == "torso"
    assert map_human_parser_class("dress").garment_type == "dress"
    assert map_human_parser_class("background").category == "background"
    unknown = map_human_parser_class("weird-new-label")
    assert unknown.category == "unknown"
    assert unknown.canonical_region_type == "weird_new_label"


class _FakeDetector:
    def detect(self, frame):
        mask_ref = DEFAULT_MASK_STORE.put(
            [[1] * 8 for _ in range(8)],
            0.91,
            "yolo_person_seg",
            "yolo_person_p1",
            mask_kind="person_mask",
            roi_bbox=(0.1, 0.1, 0.8, 0.8),
            extra={"pixel_count": 64, "bbox_xyxy": (0.1, 0.1, 0.9, 0.9), "model_name": "yolo11n-seg.pt"},
        )
        return DetectorOutput(
            persons=[PersonDetection("p1", BBox(0.1, 0.1, 0.8, 0.8), 0.91, "yolo_person_seg", mask_ref, 0.91, "yolo_person_seg")],
            frame_size=(16, 16),
        )


class _FakePose:
    def estimate(self, frame, persons):
        from core.schema import Keypoint, PoseState

        kps = [Keypoint(f"kp{i}", 0.2 + i * 0.01, 0.3 + i * 0.01, 0.8) for i in range(17)]
        return {p.detection_id: PosePrediction(PoseState(kps, "standing", {}), 0.8, "yolo_pose", [(k.x, k.y) for k in kps], {}, []) for p in persons}


class _FakeParser:
    def parse(self, frame, persons):
        out = {}
        for p in persons:
            enriched = EnrichedParsingPayload()
            body_parts = []
            face_regions = []
            garments = []
            for name in ["face", "hair", "arms", "torso"]:
                ref = DEFAULT_MASK_STORE.put(
                    [[1, 1], [0, 0]],
                    0.88,
                    "hf_human_parser",
                    f"parser_p1_{name}",
                    mask_kind="body_part_mask",
                    roi_bbox=(0.1, 0.1, 0.8, 0.8),
                    extra={"pixel_count": 2, "bbox_xyxy": (0.2, 0.2, 0.5, 0.4)},
                )
                if name in {"face", "hair"}:
                    face_regions.append(FaceRegionPrediction(name, ref, 0.88, "hf_human_parser", parser_class_name=name, pixel_count=2))
                    enriched.face_region_masks[name] = ref
                    enriched.provenance_by_region[f"face:{name}"] = "hf_human_parser"
                else:
                    body_parts.append(BodyPartMaskPrediction(name, ref, 0.88, "visible", "hf_human_parser", parser_class_name=name, pixel_count=2))
                    enriched.body_part_masks[name] = ref
                    enriched.provenance_by_region[f"body:{name}"] = "hf_human_parser"
            dress_ref = DEFAULT_MASK_STORE.put(
                [[1, 1], [1, 0]],
                0.9,
                "hf_human_parser",
                "parser_p1_dress",
                mask_kind="garment_mask",
                roi_bbox=(0.1, 0.1, 0.8, 0.8),
                extra={"pixel_count": 3, "bbox_xyxy": (0.18, 0.35, 0.75, 0.9)},
            )
            garments.append(GarmentPrediction("dress", "visible", 0.9, "hf_human_parser", dress_ref, ["torso_pelvis_upper_legs"], ["torso"], parser_class_name="dress", pixel_count=3))
            enriched.garment_masks["dress"] = dress_ref
            enriched.provenance_by_region["garment:dress"] = "hf_human_parser"
            bg_ref = DEFAULT_MASK_STORE.put([[0, 1], [1, 1]], 0.5, "hf_human_parser", "parser_background", mask_kind="background_mask")
            enriched.background_masks["background"] = bg_ref
            out[p.detection_id] = ParsingPrediction(p.mask_ref, 0.9, "hf_human_parser", garments, [], body_parts, face_regions, enriched)
        return out


class _NoneModule:
    def analyze(self, frame, persons):
        return {}

    def detect(self, frame):
        return []

    def assign(self, frame, persons):
        return {}

    def estimate(self, frame):
        return None


def test_fake_real_pipeline_scene_graph_and_roi_use_parser_masks() -> None:
    pipe = PerceptionPipeline(detector=_FakeDetector(), pose=_FakePose(), parser=_FakeParser(), face=_NoneModule(), objects=_NoneModule(), tracker=_NoneModule(), depth=_NoneModule())
    out = pipe.analyze(_solid())
    assert out.persons[0].mask_ref
    assert len(out.persons[0].pose.keypoints) == 17
    assert out.persons[0].garments[0]["parser_class_name"] == "dress"
    assert out.persons[0].body_part_masks["torso"].startswith("mask://")
    assert out.mask_store

    graph = SceneGraphBuilder().build(out)
    person = graph.persons[0]
    face = next(p for p in person.body_parts if p.part_type == "face")
    dress = next(g for g in person.garments if g.mask_ref)
    assert face.mask_ref is not None and face.source == "hf_human_parser"
    assert "parser_class:face" in face.alternatives
    assert dress.mask_ref is not None and dress.confidence > 0
    assert any(alt.startswith("parser_class:") for alt in dress.alternatives)
    face_metadata = build_region_metadata(
        scene_graph=graph,
        memory=VideoMemory(),
        region=RegionRef(f"{person.person_id}:face", face.bbox if hasattr(face, "bbox") else BBox(0.2, 0.2, 0.3, 0.2), "roi_source=parser_mask_bbox"),
        route_decision=None,
        delta=None,
    )
    assert face_metadata["parser_class_name"] == "face"

    roi = ROISelector().select(graph, GraphDelta(affected_entities=[person.person_id], affected_regions=["face", "dress"]))
    assert any("roi_source=parser_mask_bbox" in r.reason for r in roi)


def test_store_mask_preserves_parser_class_metadata_for_region_bridge() -> None:
    from perception.parser import _store_mask

    DEFAULT_MASK_STORE.clear()
    ref = _store_mask(
        [[1, 0], [1, 1]],
        0.93,
        "parser:fashn",
        "unit_parser_class",
        "body_part_mask",
        "fashn",
        frame_size=(20, 10),
        parser_class_name="left_arm",
        class_id=12,
    )

    assert ref is not None
    stored = DEFAULT_MASK_STORE.get(ref)
    assert stored is not None
    assert stored.extra["parser_class_name"] == "left_arm"
    assert stored.extra["class_id"] == 12
    assert stored.extra["pixel_count"] == 3


def test_lazy_import_does_not_require_heavy_real_dependencies() -> None:
    for name in ("ultralytics", "transformers", "mediapipe"):
        sys.modules.pop(name, None)
    importlib.import_module("perception.pipeline")
    assert "ultralytics" not in sys.modules
    assert "transformers" not in sys.modules
    assert "mediapipe" not in sys.modules


def test_non_strict_missing_backend_diagnoses_and_strict_raises() -> None:
    class Broken:
        config = types.SimpleNamespace(backend="real")

        def is_builtin_backend(self):
            return False

        def detect(self, frame):
            raise RuntimeError("missing model")

    non_strict = PerceptionPipeline(detector=Broken())
    out = non_strict.analyze(_solid())
    assert any("detector_unavailable" in w for w in out.warnings)
    assert out.diagnostics

    from perception.pipeline import PerceptionBackendsConfig

    strict = PerceptionPipeline(detector=Broken(), backends=PerceptionBackendsConfig(strict_perception=True))
    try:
        strict.analyze(_solid())
        assert False, "strict mode must raise"
    except RuntimeError as exc:
        assert "strict perception backend failed" in str(exc)


def test_real_human_parsing_config_auto_device_and_legacy_parser_config_do_not_import_heavy_deps() -> None:
    for name in ("ultralytics", "transformers", "mediapipe"):
        sys.modules.pop(name, None)
    from perception.detector import BackendConfig
    from perception.parser import ParserStackConfig, SegFormerHumanParserAdapter
    from perception.pipeline import real_human_parsing_config

    cfg = real_human_parsing_config(device="auto")
    assert cfg.perception_device in {"cpu", "cuda"}
    assert cfg.detector.backend == "ultralytics"
    assert cfg.detector.checkpoint == "yolo11n-seg.pt"
    assert cfg.pose.backend == "ultralytics"
    assert cfg.pose.checkpoint == "yolo11n-pose.pt"
    assert isinstance(cfg.parser, BackendConfig)

    adapter = SegFormerHumanParserAdapter(cfg.parser)
    assert isinstance(adapter.config, ParserStackConfig)
    assert adapter.config.primary_human_parser.backend == "fashn"
    assert adapter.config.primary_human_parser.variant == "fashn-ai/fashn-human-parser"
    assert adapter.config.structural_body_parser.backend == "builtin"
    assert adapter.config.garment_refinement_parser.backend == "builtin"
    assert adapter.config.face_parser.backend == "builtin"
    assert "ultralytics" not in sys.modules
    assert "transformers" not in sys.modules
    assert "mediapipe" not in sys.modules


def test_default_mask_store_clear_prevents_leakage() -> None:
    DEFAULT_MASK_STORE.clear()
    first = DEFAULT_MASK_STORE.put([[1]], 1.0, "test", "run", extra={"pixel_count": 1, "bbox_xyxy": (0.0, 0.0, 1.0, 1.0)})
    assert first in DEFAULT_MASK_STORE.snapshot_metadata()
    DEFAULT_MASK_STORE.clear()
    assert DEFAULT_MASK_STORE.snapshot_metadata() == {}
    second = DEFAULT_MASK_STORE.put([[1]], 1.0, "test", "run")
    assert second == first


def test_roi_helper_builds_bbox_from_mask_payload_when_metadata_missing() -> None:
    from core.schema import BodyPartNode, PersonNode, PoseState, SceneGraph

    DEFAULT_MASK_STORE.clear()
    ref = DEFAULT_MASK_STORE.put([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], 0.7, "parser:fashn", "payload_bbox", mask_kind="body_part_mask")
    graph = SceneGraph(
        frame_index=0,
        persons=[PersonNode("person_1", "person_1", BBox(0.0, 0.0, 1.0, 1.0), None, PoseState(), body_parts=[BodyPartNode("person_1_face", "face", mask_ref=ref, confidence=0.7, source="parser:fashn")])],
    )
    roi = ROISelector().select(graph, GraphDelta(affected_entities=["person_1"], affected_regions=["face"]))
    assert roi
    assert "roi_source=parser_mask_bbox" in roi[0].reason
    assert 0.20 <= roi[0].bbox.x <= 0.30
    assert 0.25 <= roi[0].bbox.y <= 0.40


def test_debug_overlay_uses_payload_bbox_and_keeps_background_separate(tmp_path) -> None:
    import importlib.util
    from pathlib import Path
    from PIL import Image

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "debug_real_perception_layers.py"
    spec = importlib.util.spec_from_file_location("debug_real_perception_layers", script_path)
    assert spec and spec.loader
    debug_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(debug_mod)
    _bbox_from_mask_payload = debug_mod._bbox_from_mask_payload
    _overlay = debug_mod._overlay

    DEFAULT_MASK_STORE.clear()
    face = DEFAULT_MASK_STORE.put([[0, 1], [0, 0]], 0.9, "parser:fashn", "face", mask_kind="body_part_mask")
    bg = DEFAULT_MASK_STORE.put([[1, 1], [1, 1]], 0.2, "parser:fashn", "background", mask_kind="background_mask")
    assert _bbox_from_mask_payload([[0, 1], [0, 0]]) == (0.5, 0.0, 1.0, 0.5)
    img = Image.new("RGB", (16, 16), (10, 10, 10))
    body = _overlay(img, [face, bg], kinds={"body_part_mask"})
    background = _overlay(img, [face, bg], kinds={"background_mask"})
    assert body.size == img.size
    assert background.size == img.size


def test_fake_parser_duplicate_regions_preserved_in_parser_summary() -> None:
    from perception.parser import AdapterSegmentationOutput, ParserFusionEngine

    DEFAULT_MASK_STORE.clear()
    person = PersonDetection("p1", BBox(0.0, 0.0, 1.0, 1.0), 0.9, "test")
    mask = [[1, 0], [0, 0]]
    fashn = AdapterSegmentationOutput(
        masks={"left_arm": mask, "right_arm": mask, "dress": mask},
        confidences={"left_arm": 0.8, "right_arm": 0.82, "dress": 0.9},
        class_ids={"left_arm": 12, "right_arm": 13, "dress": 4},
    )
    fused = ParserFusionEngine().fuse(person, fashn, AdapterSegmentationOutput(), AdapterSegmentationOutput(), AdapterSegmentationOutput(), frame_size=(2, 2))
    assert fused.enriched.region_mask_refs["body:left_arm"]
    assert fused.enriched.region_mask_refs["body:right_arm"]
    assert fused.body_parts[0].class_id in {12, 13}
    assert fused.garments[0].class_id == 4


def test_reset_mask_store_per_analyze_drops_stale_refs_between_single_image_requests() -> None:
    from perception.pipeline import PerceptionBackendsConfig

    DEFAULT_MASK_STORE.clear()
    stale = DEFAULT_MASK_STORE.put([[1]], 0.1, "test", "stale", mask_kind="body_part_mask")
    pipe = PerceptionPipeline(
        detector=_FakeDetector(),
        pose=_FakePose(),
        parser=_FakeParser(),
        face=_NoneModule(),
        objects=_NoneModule(),
        tracker=_NoneModule(),
        depth=_NoneModule(),
        backends=PerceptionBackendsConfig(reset_mask_store_per_analyze=True),
    )
    first = pipe.analyze(_solid())
    assert stale not in first.mask_store
    stale2 = DEFAULT_MASK_STORE.put([[1]], 0.1, "test", "stale2", mask_kind="body_part_mask")
    second = pipe.analyze(_solid())
    assert stale2 not in second.mask_store
    assert second.mask_store


def test_analyze_video_resets_mask_store_once_for_video_not_per_frame() -> None:
    from perception.pipeline import PerceptionBackendsConfig

    DEFAULT_MASK_STORE.clear()
    stale = DEFAULT_MASK_STORE.put([[1]], 0.1, "test", "stale_video", mask_kind="body_part_mask")
    pipe = PerceptionPipeline(
        detector=_FakeDetector(),
        pose=_FakePose(),
        parser=_FakeParser(),
        face=_NoneModule(),
        objects=_NoneModule(),
        tracker=_NoneModule(),
        depth=_NoneModule(),
        backends=PerceptionBackendsConfig(reset_mask_store_per_analyze=True),
    )
    outputs = pipe.analyze_video([_solid(), _solid()], batch_size=1)
    assert stale not in outputs[0].mask_store
    assert len(outputs[1].mask_store) > len(outputs[0].mask_store)


def test_debug_face_hair_refs_use_parser_summary_not_ref_name_and_background_exclusion() -> None:
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from PIL import Image, ImageChops

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "debug_real_perception_layers.py"
    spec = importlib.util.spec_from_file_location("debug_real_perception_layers", script_path)
    assert spec and spec.loader
    debug_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(debug_mod)

    DEFAULT_MASK_STORE.clear()
    semantic_ref = DEFAULT_MASK_STORE.put([[1, 0], [0, 0]], 0.9, "parser:fashn", "opaque_semantic_ref", mask_kind="face_region_mask")
    bg_ref = DEFAULT_MASK_STORE.put([[1, 1], [1, 1]], 0.5, "parser:fashn", "opaque_background_ref", mask_kind="background_mask")
    output = SimpleNamespace(
        parser_summary={"region_mask_refs": {"face:face": [semantic_ref], "background:background": [bg_ref]}},
        mask_store=DEFAULT_MASK_STORE.snapshot_metadata(),
    )
    refs = debug_mod._refs_from_parser_summary(output, {"face:face"}, fallback_kinds={"face_region_mask"})
    assert refs == [semantic_ref]

    img = Image.new("RGB", (8, 8), (10, 10, 10))
    excluded = debug_mod._overlay(img, [bg_ref], exclude_kinds={"background_mask"})
    background = debug_mod._overlay(img, [bg_ref], kinds={"background_mask"})
    assert ImageChops.difference(img, excluded).getbbox() is None
    assert ImageChops.difference(img, background).getbbox() is not None


def test_mask_stats_projects_crop_local_bbox_to_frame_normalized_xyxy() -> None:
    from perception.parser import _mask_stats

    pixel_count, bbox = _mask_stats([[0, 1], [0, 1]], roi_bbox=(0.25, 0.25, 0.5, 0.5))
    assert pixel_count == 2
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert abs(x1 - 0.5) < 1e-6
    assert abs(x2 - 0.75) < 1e-6
    assert abs(y1 - 0.25) < 1e-6
    assert abs(y2 - 0.75) < 1e-6


def test_common_arms_mask_reaches_left_and_right_arm_roi_from_parser_bbox() -> None:
    DEFAULT_MASK_STORE.clear()
    pipe = PerceptionPipeline(detector=_FakeDetector(), pose=_FakePose(), parser=_FakeParser(), face=_NoneModule(), objects=_NoneModule(), tracker=_NoneModule(), depth=_NoneModule())
    output = pipe.analyze(_solid())
    assert "arms" in output.persons[0].body_part_masks
    graph = SceneGraphBuilder().build(output)
    person = graph.persons[0]
    left_arm = next(part for part in person.body_parts if part.part_type == "left_arm")
    right_arm = next(part for part in person.body_parts if part.part_type == "right_arm")
    assert left_arm.mask_ref == output.persons[0].body_part_masks["arms"]
    assert right_arm.mask_ref == output.persons[0].body_part_masks["arms"]
    rois = ROISelector().select(graph, GraphDelta(affected_entities=[person.person_id], affected_regions=["left_arm", "right_arm"]))
    assert rois
    assert all("roi_source=parser_mask_bbox" in roi.reason for roi in rois)


def test_gennady_engine_accepts_real_perception_config_without_eager_real_backend_imports() -> None:
    for name in ("ultralytics", "transformers", "mediapipe"):
        sys.modules.pop(name, None)
    from perception.pipeline import real_human_parsing_config
    from runtime.orchestrator import GennadyEngine

    cfg = real_human_parsing_config(device="cpu", strict_perception=False)
    engine = GennadyEngine(perception_config=cfg, backend_bundle=types.SimpleNamespace())
    assert engine.perception.strict_perception is False
    assert "ultralytics" not in sys.modules
    assert "transformers" not in sys.modules
    assert "mediapipe" not in sys.modules


def test_roi_helper_projects_payload_bbox_through_stored_roi_when_extra_bbox_missing() -> None:
    from core.schema import BodyPartNode, PersonNode, PoseState, SceneGraph

    DEFAULT_MASK_STORE.clear()
    ref = DEFAULT_MASK_STORE.put(
        [[0, 1], [0, 1]],
        0.8,
        "parser:fashn",
        "payload_roi_bbox",
        mask_kind="body_part_mask",
        roi_bbox=(0.25, 0.25, 0.5, 0.5),
    )
    graph = SceneGraph(
        frame_index=0,
        persons=[
            PersonNode(
                "person_1",
                "person_1",
                BBox(0.25, 0.25, 0.5, 0.5),
                None,
                PoseState(),
                body_parts=[BodyPartNode("person_1_torso", "torso", mask_ref=ref, confidence=0.8, source="parser:fashn")],
            )
        ],
    )
    roi = ROISelector().select(graph, GraphDelta(affected_entities=["person_1"], affected_regions=["torso"]))
    assert roi
    assert "roi_source=parser_mask_bbox" in roi[0].reason
    # The payload occupies the right half of the crop, so frame-space x is 0.5..0.75
    # before ROI context expansion. If treated as frame-space directly it would be 0.5..1.0.
    assert 0.45 <= roi[0].bbox.x <= 0.50
    assert 0.28 <= roi[0].bbox.w <= 0.35


def test_background_masks_are_public_perception_fields() -> None:
    DEFAULT_MASK_STORE.clear()
    pipe = PerceptionPipeline(detector=_FakeDetector(), pose=_FakePose(), parser=_FakeParser(), face=_NoneModule(), objects=_NoneModule(), tracker=_NoneModule(), depth=_NoneModule())
    output = pipe.analyze(_solid())
    assert output.persons[0].background_masks["background"].startswith("mask://")


def test_debug_min_parser_pixels_filters_parser_refs_but_keeps_person_masks() -> None:
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "debug_real_perception_layers.py"
    spec = importlib.util.spec_from_file_location("debug_real_perception_layers", script_path)
    assert spec and spec.loader
    debug_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(debug_mod)

    DEFAULT_MASK_STORE.clear()
    person_ref = DEFAULT_MASK_STORE.put([[1]], 0.9, "yolo_person_seg", "person", mask_kind="person_mask")
    tiny_parser_ref = DEFAULT_MASK_STORE.put([[1]], 0.9, "parser:fashn", "tiny", mask_kind="body_part_mask", extra={"pixel_count": 1})
    large_parser_ref = DEFAULT_MASK_STORE.put([[1, 1], [1, 1]], 0.9, "parser:fashn", "large", mask_kind="body_part_mask", extra={"pixel_count": 4})
    assert debug_mod._filter_refs_by_min_pixels([person_ref, tiny_parser_ref, large_parser_ref], 2) == [person_ref, large_parser_ref]
