from __future__ import annotations

import types

from core.schema import BBox
from perception.detector import BackendConfig, PersonDetection, YoloPersonDetectorAdapter
from perception.mask_store import DEFAULT_MASK_STORE
from perception.parser import (
    EnrichedParsingPayload,
    FashnHumanParserAdapter,
    ParserBackendConfig,
    ParsingPrediction,
)
from perception.pipeline import PerceptionBackendsConfig, PerceptionPipeline
from perception.pose import MediaPipePoseAdapter, PosePrediction, YoloPoseAdapter
from representation.graph_builder import SceneGraphBuilder


def _solid(h: int, w: int, rgb: tuple[float, float, float] = (0.3, 0.4, 0.5)) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


class _Scalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _Vector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _Box:
    def __init__(self, xyxy: list[float], conf: float, cls: int = 0) -> None:
        self.xyxy = [_Vector(xyxy)]
        self.conf = _Scalar(conf)
        self.cls = _Scalar(float(cls))


class _MaskTensor:
    def __init__(self, payload: list[list[float]]) -> None:
        self._payload = payload

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return self._payload

    def __array__(self, dtype=None):
        import numpy as np

        return np.asarray(self._payload, dtype=dtype)


class _FakeDetModel:
    def predict(self, **kwargs):
        result = types.SimpleNamespace(
            boxes=[_Box([2.0, 2.0, 14.0, 14.0], 0.91)],
            masks=types.SimpleNamespace(data=[_MaskTensor([[1.0, 0.0], [0.0, 1.0]])]),
        )
        return [result]


def test_yolo_detector_emits_instance_mask_ref() -> None:
    det = YoloPersonDetectorAdapter(BackendConfig(backend="ultralytics", checkpoint="dummy"))
    det._model = _FakeDetModel()  # noqa: SLF001
    out = det.detect(_solid(16, 16))
    assert out.persons
    assert out.persons[0].mask_ref is not None
    stored = DEFAULT_MASK_STORE.get(out.persons[0].mask_ref or "")
    assert stored is not None
    assert stored.mask_kind == "person_mask"


class _FakePoseXYN:
    def __init__(self, coords: list[tuple[float, float]]) -> None:
        self._coords = coords

    def __getitem__(self, key):
        if isinstance(key, tuple):
            _, axis = key
            return _Vector([c[axis] for c in self._coords])
        return self


class _FakePoseConf:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakePoseModel:
    def predict(self, **kwargs):
        kpts = [(0.5, 0.1)] * 17
        result = types.SimpleNamespace(
            boxes=[_Box([3.0, 1.0, 10.0, 14.0], 0.95)],
            keypoints=types.SimpleNamespace(
                xyn=[_FakePoseXYN(kpts)],
                conf=[_FakePoseConf([0.9] * 17)],
            ),
        )
        return [result]


def test_yolo_pose_adapter_maps_keypoints() -> None:
    pose = YoloPoseAdapter(BackendConfig(backend="ultralytics", checkpoint="dummy"))
    pose._model = _FakePoseModel()  # noqa: SLF001
    persons = [PersonDetection("det::person_1", BBox(0.1, 0.1, 0.6, 0.7), 0.9, "det")]
    out = pose.estimate(_solid(20, 20), persons)
    assert "det::person_1" in out
    assert len(out["det::person_1"].pose.keypoints) >= 5
    assert out["det::person_1"].source.startswith("pose:yolo")


def test_yolo_pose_matching_uses_iou_not_only_x_center() -> None:
    pose = YoloPoseAdapter(BackendConfig(backend="ultralytics", checkpoint="dummy"))

    class _PoseModelTwo:
        def predict(self, **kwargs):
            upper = [(0.50, 0.22)] * 17
            lower = [(0.50, 0.78)] * 17
            return [
                types.SimpleNamespace(
                    boxes=[_Box([2.0, 2.0, 18.0, 10.0], 0.96), _Box([2.0, 10.0, 18.0, 18.0], 0.95)],
                    keypoints=types.SimpleNamespace(
                        xyn=[_FakePoseXYN(upper), _FakePoseXYN(lower)],
                        conf=[_FakePoseConf([0.9] * 17), _FakePoseConf([0.9] * 17)],
                    ),
                )
            ]

    pose._model = _PoseModelTwo()  # noqa: SLF001
    persons = [
        PersonDetection("p_top", BBox(0.1, 0.05, 0.8, 0.4), 0.9, "det"),
        PersonDetection("p_bottom", BBox(0.1, 0.55, 0.8, 0.4), 0.9, "det"),
    ]
    out = pose.estimate(_solid(20, 20), persons)
    top_nose = next(k for k in out["p_top"].pose.keypoints if k.name == "nose")
    bottom_nose = next(k for k in out["p_bottom"].pose.keypoints if k.name == "nose")
    assert top_nose.y < bottom_nose.y


def test_mediapipe_pose_is_separate_backend_path() -> None:
    pose = MediaPipePoseAdapter(BackendConfig(backend="mediapipe"))
    assert pose.source_name == "pose:mediapipe"


def test_fashn_label_map_mapping_and_pipeline_fusion_graph_compat() -> None:
    parser = FashnHumanParserAdapter(
        ParserBackendConfig(backend="fashn"),
        infer_fn=lambda _: {"label_map": [[1, 3, 12], [16, 6, 2], [13, 15, 10]]},
    )
    patch_out = parser.parse_patch(_solid(3, 3))
    assert {"face", "top", "torso", "arms", "pants", "hands", "feet", "scarf"}.issubset(set(patch_out.masks))

    cfg = PerceptionBackendsConfig(
        detector=BackendConfig(backend="builtin"),
        pose=BackendConfig(backend="builtin"),
    )
    out = PerceptionPipeline(backends=cfg).analyze(_solid(48, 48))
    assert out.persons
    graph = SceneGraphBuilder().build(out, frame_index=1)
    assert graph.persons
    assert out.module_confidence.get("parser", 0.0) >= 0.0


class _OnePersonDetector:
    def __init__(self, mask_ref: str | None = None) -> None:
        self._mask_ref = mask_ref

    def detect(self, frame):
        return types.SimpleNamespace(
            persons=[
                PersonDetection(
                    "p1",
                    BBox(0.1, 0.1, 0.8, 0.8),
                    0.9,
                    "detector:test",
                    mask_ref=self._mask_ref,
                    mask_confidence=0.77 if self._mask_ref else 0.0,
                    mask_source="detector:test:seg" if self._mask_ref else "",
                )
            ],
            frame_size=(32, 32),
            latency_ms=0.0,
        )


class _NoFace:
    def analyze(self, frame, persons):
        return {}


class _PoseWithFace:
    def estimate(self, frame, persons):
        return {
            p.detection_id: PosePrediction(
                pose=types.SimpleNamespace(keypoints=[], coarse_pose="unknown", angles={}),
                confidence=0.8,
                source="pose:test",
                landmarks_2d=[],
                hand_landmarks={},
                face_landmarks=[(0.2, 0.3), (0.3, 0.3)],
            )
            for p in persons
        }


class _ParserNoMask:
    def parse(self, frame, persons):
        return {
            p.detection_id: ParsingPrediction(
                mask_ref=None,
                mask_confidence=0.85,
                source="parser:test",
                garments=[],
                body_parts=[],
                face_regions=[],
                enriched=EnrichedParsingPayload(),
            )
            for p in persons
        }


class _ParserWithMask:
    def __init__(self, mask_ref: str) -> None:
        self._mask_ref = mask_ref

    def parse(self, frame, persons):
        return {
            p.detection_id: ParsingPrediction(
                mask_ref=self._mask_ref,
                mask_confidence=0.93,
                source="parser:test",
                garments=[],
                body_parts=[],
                face_regions=[],
                enriched=EnrichedParsingPayload(person_mask_ref=self._mask_ref),
            )
            for p in persons
        }


def test_pipeline_uses_detector_mask_when_parser_has_no_person_mask() -> None:
    detector_mask = DEFAULT_MASK_STORE.put([[1, 1], [1, 1]], 0.7, "det", "unit", mask_kind="person_mask")
    pipe = PerceptionPipeline(detector=_OnePersonDetector(detector_mask), parser=_ParserNoMask())
    out = pipe.analyze(_solid(16, 16))
    assert out.persons[0].mask_ref == detector_mask
    assert out.persons[0].mask_source == "detector:test:seg"


def test_pipeline_prefers_parser_mask_over_detector_mask_when_available() -> None:
    detector_mask = DEFAULT_MASK_STORE.put([[1, 1], [1, 1]], 0.7, "det", "unit", mask_kind="person_mask")
    parser_mask = DEFAULT_MASK_STORE.put([[1, 0], [1, 0]], 0.95, "parser", "unit", mask_kind="person_mask")
    pipe = PerceptionPipeline(detector=_OnePersonDetector(detector_mask), parser=_ParserWithMask(parser_mask))
    out = pipe.analyze(_solid(16, 16))
    assert out.persons[0].mask_ref == parser_mask
    assert out.persons[0].mask_source == "parser:test"


def test_pipeline_face_landmarks_fallback_to_pose_when_face_backend_empty() -> None:
    pipe = PerceptionPipeline(detector=_OnePersonDetector(), parser=_ParserNoMask(), face=_NoFace(), pose=_PoseWithFace())
    out = pipe.analyze(_solid(16, 16))
    assert out.persons[0].face_landmarks == [(0.2, 0.3), (0.3, 0.3)]


def test_parser_module_confidence_not_sourced_from_detector_mask_fallback() -> None:
    detector_mask = DEFAULT_MASK_STORE.put([[1, 1], [1, 1]], 0.7, "det", "unit", mask_kind="person_mask")
    pipe = PerceptionPipeline(detector=_OnePersonDetector(detector_mask), parser=_ParserNoMask())
    out = pipe.analyze(_solid(16, 16))
    assert out.module_confidence["parser"] == 0.85
    assert out.persons[0].mask_confidence == 0.77
