from __future__ import annotations

import pytest

from core.pipeline_contract import ContractValidationError
from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.contracts import validate_perception_output
from perception.detector import BackendConfig, DetectorOutput, PersonDetection, YoloPersonDetectorAdapter
from perception.frame_context import PerceptionFrameContext
from perception.mask_store import DEFAULT_MASK_STORE, InMemoryMaskStore, mask_store_from_frame
from perception.parser import (
    BodyPartMaskPrediction,
    FaceRegionPrediction,
    FashnHumanParserAdapter,
    ParserBackendConfig,
    ParserStackConfig,
    ParsingPrediction,
    SegFormerHumanParserAdapter,
)
from perception.pipeline import PerceptionBackendsConfig, PerceptionOutput, PerceptionPipeline, PersonFacts
from perception.tracker import TrackPrediction
from representation.graph_builder import SceneGraphBuilder
from types import SimpleNamespace
from runtime.orchestrator import GennadyEngine


def test_strict_mask_store_from_frame_requires_explicit_store() -> None:
    with pytest.raises(RuntimeError, match="explicit InMemoryMaskStore"):
        mask_store_from_frame(PerceptionFrameContext("frame://missing-store"))

    legacy = mask_store_from_frame(PerceptionFrameContext("frame://legacy"), allow_legacy_default=True)
    assert legacy is DEFAULT_MASK_STORE


def test_production_parser_and_detector_require_frame_context_mask_store() -> None:
    person = PersonDetection("det::person_1", BBox(0.1, 0.1, 0.5, 0.7), 0.8, "detector:test")
    parser = SegFormerHumanParserAdapter(
        ParserStackConfig(primary_human_parser=ParserBackendConfig(backend="fashn")),
        fashn_adapter=FashnHumanParserAdapter(ParserBackendConfig(backend="fashn"), infer_fn=lambda _: {"label_map": [[1]]}),
    )
    with pytest.raises(RuntimeError, match="explicit InMemoryMaskStore"):
        parser.parse(PerceptionFrameContext("frame://parser-no-store"), [person])

    detector = YoloPersonDetectorAdapter(BackendConfig(backend="ultralytics"))
    with pytest.raises(RuntimeError, match="explicit InMemoryMaskStore"):
        detector._detect_ultralytics(PerceptionFrameContext("frame://detector-no-store"))


class OnePersonDetector:
    def detect(self, frame):
        return DetectorOutput(
            persons=[PersonDetection("det::person_1", BBox(0.1, 0.1, 0.5, 0.7), 0.82, "detector:test")],
            frame_size=(64, 64),
        )


class MaskingParser:
    def __init__(self, *, include_face: bool = False) -> None:
        self.include_face = include_face

    def parse(self, frame, persons):
        store = mask_store_from_frame(frame)
        out = {}
        for person in persons:
            ref = store.put([[1, 1], [0, 0]], 0.91, "parser:test", "torso", mask_kind="body_part_mask", backend="test")
            face_regions = []
            if self.include_face:
                face_ref = store.put([[1]], 0.88, "parser:facer", "face", mask_kind="face_region_mask", backend="test")
                hair_ref = store.put([[1]], 0.86, "parser:facer", "hair", mask_kind="face_region_mask", backend="test")
                face_regions = [
                    FaceRegionPrediction("face", face_ref, 0.88, "parser:facer"),
                    FaceRegionPrediction("hair", hair_ref, 0.86, "parser:facer"),
                ]
            out[person.detection_id] = ParsingPrediction(
                ref,
                0.91,
                "parser:test",
                body_parts=[BodyPartMaskPrediction("torso", ref, 0.91, "visible", "parser:test")],
                face_regions=face_regions,
            )
        return out


class LegacyDefaultStoreParser:
    def parse(self, frame, persons):
        out = {}
        for person in persons:
            ref = DEFAULT_MASK_STORE.put([[1]], 0.77, "parser:legacy-default", "legacy_torso", mask_kind="body_part_mask")
            out[person.detection_id] = ParsingPrediction(
                ref,
                0.77,
                "parser:legacy-default",
                body_parts=[BodyPartMaskPrediction("torso", ref, 0.77, "visible", "parser:legacy-default")],
            )
        return out


class NoopPose:
    def estimate(self, frame, persons):
        return {}


class NoopFace:
    def analyze(self, frame, persons):
        return {}


class NoopObjects:
    def detect(self, frame):
        return []


class NoopDepth:
    def estimate(self, frame):
        return None


class StableTracker:
    def assign(self, frame, persons):
        return {p.detection_id: TrackPrediction("track_test_1", 0.77, "tracker:test") for p in persons}


def _pipeline(*, reset: bool = False, include_face: bool = False) -> PerceptionPipeline:
    return PerceptionPipeline(
        detector=OnePersonDetector(),
        parser=MaskingParser(include_face=include_face),
        pose=NoopPose(),
        face=NoopFace(),
        objects=NoopObjects(),
        tracker=StableTracker(),
        depth=NoopDepth(),
        backends=PerceptionBackendsConfig(reset_mask_store_per_analyze=reset),
    )


def _person(*, mask_ref: str | None = None, region: dict | None = None) -> PersonFacts:
    return PersonFacts(
        bbox=BBox(0.1, 0.1, 0.5, 0.7),
        bbox_confidence=0.8,
        bbox_source="detector:test",
        mask_ref=mask_ref,
        mask_confidence=0.8 if mask_ref else 0.0,
        mask_source="parser:test" if mask_ref else "unknown",
        pose=PoseState(),
        pose_confidence=0.0,
        pose_source="unknown",
        expression=ExpressionState(),
        expression_confidence=0.0,
        expression_source="unknown",
        orientation=OrientationState(),
        orientation_confidence=0.0,
        orientation_source="unknown",
        person_id="person_1",
        track_confidence=0.8,
        track_provenance="single_frame_observed",
        track_source="single_frame_observed",
        identity_observation_status="single_frame_anchor",
        body_parts=[region] if region else [],
    )


def test_mask_store_isolation_between_pipeline_instances_and_analyze_resets() -> None:
    p1 = _pipeline(reset=True)
    p2 = _pipeline(reset=True)

    out1 = p1.analyze("frame://a")
    assert out1.mask_store
    assert set(out1.mask_store) == set(p1.mask_store.snapshot_metadata())
    assert not (set(out1.mask_store) & set(p2.mask_store.snapshot_metadata()))

    p1.mask_store.put([[1]], 0.4, "test_fixture", "stale", ref="mask://stale")
    out2 = p1.analyze("frame://b")
    assert "mask://stale" not in out2.mask_store
    assert out2.mask_store


def test_legacy_adoption_imports_only_referenced_masks_and_marks_non_authoritative() -> None:
    DEFAULT_MASK_STORE.clear()
    unrelated = DEFAULT_MASK_STORE.put([[1]], 0.1, "parser:unrelated", "unrelated", mask_kind="body_part_mask")
    pipe = PerceptionPipeline(
        detector=OnePersonDetector(),
        parser=LegacyDefaultStoreParser(),
        pose=NoopPose(),
        face=NoopFace(),
        objects=NoopObjects(),
        tracker=StableTracker(),
        depth=NoopDepth(),
        backends=PerceptionBackendsConfig(reset_mask_store_per_analyze=True),
    )

    out = pipe.analyze("frame://legacy")
    assert unrelated not in out.mask_store
    adopted_refs = set(out.mask_store)
    assert adopted_refs
    adopted_meta = next(iter(out.mask_store.values()))
    assert adopted_meta["source"].startswith("legacy_adopted:")
    assert "adopted_legacy_default_store" in adopted_meta["tags"]
    assert adopted_meta["extra"]["adopted_legacy_default_store"] is True
    assert any(v.code == "adopted_legacy_marked_observed" for v in validate_perception_output(out, pipe.mask_store).violations)


def test_runtime_fails_when_perception_has_no_valid_mask_store(tmp_path) -> None:
    class BadPerception:
        def analyze(self, frame):
            return PerceptionOutput(persons=[])

    img = tmp_path / "ref.ppm"
    img.write_text("P3\n1 1\n255\n0 0 0\n")
    fake_bundle = SimpleNamespace(backend_names={})
    engine = GennadyEngine(backend_bundle=fake_bundle)
    engine.perception = BadPerception()
    with pytest.raises(ContractValidationError, match="mask_store"):
        engine.run([str(img)], "smile")


def test_perception_validation_accepts_valid_parser_mask_and_rejects_fake_observed_paths() -> None:
    store = InMemoryMaskStore()
    ref = store.put([[1]], 0.9, "parser:test", "torso", mask_kind="body_part_mask")
    valid_region = {
        "part_type": "torso",
        "region_id": "person_1:torso",
        "mask_ref": ref,
        "confidence": 0.9,
        "source": "parser:test",
        "provenance": "parser:test",
        "observation_status": "observed",
        "mask_evidence_type": "parser_mask",
    }
    valid = PerceptionOutput(persons=[_person(mask_ref=ref, region=valid_region)], mask_store=store.snapshot_metadata())
    assert validate_perception_output(valid, store).ok

    missing = PerceptionOutput(persons=[_person(region={**valid_region, "mask_ref": "mask://missing"})])
    assert not validate_perception_output(missing, store).ok

    bad_provenance = PerceptionOutput(persons=[_person(region={**valid_region, "provenance": "unknown", "source": "unknown"})], mask_store=store.snapshot_metadata())
    assert any(v.code == "observed_without_observed_provenance" for v in validate_perception_output(bad_provenance, store).violations)

    fallback_observed = PerceptionOutput(persons=[_person(region={**valid_region, "provenance": "fallback:bbox_projection", "source": "fallback:bbox_projection"})], mask_store=store.snapshot_metadata())
    assert any(v.code == "non_observed_marked_observed" for v in validate_perception_output(fallback_observed, store).violations)


def test_validation_rejects_duplicate_person_and_region_ids() -> None:
    region = {"part_type": "torso", "region_id": "dup", "confidence": 0.5, "source": "parser:test", "observation_status": "unknown"}
    out = PerceptionOutput(persons=[_person(region=region), _person(region=region)])
    violations = validate_perception_output(out, InMemoryMaskStore()).violations
    assert any(v.code == "duplicate_person_id" for v in violations)
    assert any(v.code == "duplicate_region_id" for v in violations)


def test_single_image_and_video_tracking_contracts_are_explicit() -> None:
    single = _pipeline(reset=True).analyze("frame://single")
    person = single.persons[0]
    assert person.identity_observation_status == "single_frame_anchor"
    assert person.track_provenance == "single_frame_observed"
    assert person.track_id is None

    video = _pipeline(reset=True).analyze_video(["frame://a", "frame://b"])
    assert [p.persons[0].track_id for p in video] == ["track_test_1", "track_test_1"]
    assert video[0].persons[0].identity_observation_status == "tracker_single_frame_observed"
    assert video[1].persons[0].identity_observation_status == "multi_frame_tracked"


def test_face_head_hair_evidence_is_preserved_and_missing_is_not_faked() -> None:
    observed = _pipeline(reset=True, include_face=True).analyze("frame://face")
    graph = SceneGraphBuilder().build(observed)
    parts = {part.part_type: part for part in graph.persons[0].body_parts}
    assert parts["face"].mask_ref is not None
    assert parts["face"].source == "parser:facer"
    assert parts["face"].observation_status == "observed"
    assert parts["hair"].mask_ref is not None
    assert parts["hair"].mask_evidence_type == "parser_mask"

    missing = _pipeline(reset=True, include_face=False).analyze("frame://noface")
    missing_graph = SceneGraphBuilder().build(missing)
    missing_parts = {part.part_type: part for part in missing_graph.persons[0].body_parts}
    assert missing_parts["face"].mask_ref is None
    assert missing_parts["face"].observation_status != "observed"

    fake_face = {"region_type": "face", "confidence": 0.7, "source": "generated", "provenance": "generated", "observation_status": "observed"}
    bad = PerceptionOutput(persons=[_person()], mask_store={})
    bad.persons[0].face_regions = [fake_face]
    assert any(v.code == "non_observed_marked_observed" for v in validate_perception_output(bad, InMemoryMaskStore()).violations)


def test_graph_builder_preserves_parser_mask_vs_bbox_projection_metadata() -> None:
    store = InMemoryMaskStore()
    ref = store.put([[1]], 0.9, "parser:test", "torso", mask_kind="body_part_mask")
    parser_region = {"part_type": "torso", "mask_ref": ref, "confidence": 0.9, "source": "parser:test", "provenance": "parser:test", "observation_status": "observed", "mask_evidence_type": "parser_mask"}
    bbox_region = {"part_type": "left_arm", "mask_ref": None, "confidence": 0.35, "source": "inferred:bbox_projection", "provenance": "inferred:bbox_projection", "observation_status": "inferred", "mask_evidence_type": "bbox_projection"}
    out = PerceptionOutput(persons=[_person(mask_ref=ref)], mask_store=store.snapshot_metadata())
    out.persons[0].body_parts = [parser_region, bbox_region]
    graph = SceneGraphBuilder().build(out)
    parts = {part.part_type: part for part in graph.persons[0].body_parts}
    assert parts["torso"].mask_evidence_type == "parser_mask"
    assert parts["torso"].source == "parser:test"
    assert parts["left_arm"].mask_evidence_type == "bbox_projection"
    assert parts["left_arm"].observation_status == "inferred"
