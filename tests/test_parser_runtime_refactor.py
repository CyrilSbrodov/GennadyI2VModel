from __future__ import annotations

import numpy as np

from core.input_layer import AssetFrame, InputAssetLayer
from core.schema import BBox
from perception.detector import DetectorOutput, PersonDetection
from perception.frame_context import ensure_frame_context
from perception.parser import ParsingPrediction
from perception.pipeline import ParserOnlyPipeline, PerceptionPipeline
from perception.profiling import StageTimer


class DummyDetector:
    def detect(self, frame):
        return DetectorOutput(
            persons=[PersonDetection("p1", BBox(0.1, 0.1, 0.6, 0.7), 0.9, "det:test")],
            frame_size=(32, 24),
        )


class DummyParser:
    def __init__(self) -> None:
        self.last_runtime_formats = {"p1": {"fashn": "builtin"}}

    def parse(self, frame, persons):
        return {
            "p1": ParsingPrediction(
                mask_ref="mask::1",
                mask_confidence=0.8,
                source="parser:test",
            )
        }

    def is_builtin_backend(self) -> bool:
        return True


class FailingDetector:
    def __init__(self, backend: str = "builtin") -> None:
        from perception.detector import BackendConfig

        self.config = BackendConfig(backend=backend)

    def detect(self, frame):
        raise RuntimeError("detector boom")


class FailingParser:
    def __init__(self, backend: str = "builtin") -> None:
        from perception.detector import BackendConfig

        self.config = BackendConfig(backend=backend)
        self.last_runtime_formats = {}

    def parse(self, frame, persons):
        raise RuntimeError("parser boom")

    def is_builtin_backend(self) -> bool:
        return self.config.backend == "builtin"


def test_frame_features_cached_single_compute(monkeypatch):
    frame = AssetFrame(frame_id="f0", tensor=np.full((16, 16, 3), 128, dtype=np.uint8), width=16, height=16)
    ctx = ensure_frame_context(frame)

    calls = {"n": 0}

    def wrapped(orig):
        def inner(arg):
            if not hasattr(arg, "frame"):
                calls["n"] += 1
            return orig(arg)

        return inner

    import perception.backend as backend
    import perception.detector as detector
    import perception.face as face
    import perception.objects as objects
    import perception.pose as pose

    monkeypatch.setattr(backend, "frame_to_features", wrapped(backend.frame_to_features))
    monkeypatch.setattr(detector, "frame_to_features", backend.frame_to_features)
    monkeypatch.setattr(face, "frame_to_features", backend.frame_to_features)
    monkeypatch.setattr(objects, "frame_to_features", backend.frame_to_features)
    monkeypatch.setattr(pose, "frame_to_features", backend.frame_to_features)

    pipe = PerceptionPipeline()
    _ = pipe.analyze(ctx)
    assert calls["n"] == 1


def test_input_asset_layer_produces_numpy_uint8_image(tmp_path):
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (32, 24), color=(120, 30, 10)).save(image_path)

    req = InputAssetLayer().build_request(images=[str(image_path)], text="t", quality_profile="debug")
    frame = req.unified_asset.frames[0]
    assert isinstance(frame.tensor, np.ndarray)
    assert frame.tensor.dtype == np.uint8


def test_parser_only_pipeline_disables_heavy_modules_and_collects_timing():
    timer = StageTimer(enabled=True)
    out = ParserOnlyPipeline(detector=DummyDetector(), parser=DummyParser()).analyze(
        AssetFrame(frame_id="f1", tensor=np.zeros((24, 32, 3), dtype=np.uint8), width=32, height=24),
        profiler=timer,
    )
    assert out.persons and out.module_fallbacks["pose"].startswith("disabled")
    assert "detector" in timer.summary()
    assert "parser" in out.module_latency_ms
    assert out.input_mode == "frame_tensor"


def test_legacy_full_pipeline_still_runs_with_string_input():
    out = PerceptionPipeline().analyze("frame://legacy")
    assert out.module_fallbacks["input_mode"] == "string_ref_fallback"


def test_pipeline_accepts_numpy_frame_directly():
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    out = PerceptionPipeline().analyze(frame)
    assert out.input_mode == "frame_tensor"
    assert out.frame_size == (32, 24)


def test_parser_only_pipeline_reports_no_fallback_contract():
    out = ParserOnlyPipeline(detector=DummyDetector(), parser=FailingParser(backend="builtin")).analyze(
        AssetFrame(frame_id="f2", tensor=np.zeros((24, 32, 3), dtype=np.uint8), width=32, height=24)
    )
    assert out.module_fallbacks["parser"] == "error:no-fallback"
    assert any("parser_no_fallback" in w for w in out.warnings)
    assert out.persons[0].mask_ref is None


def test_full_pipeline_reuses_builtin_fallback_adapter_instances():
    pipe = PerceptionPipeline(detector=FailingDetector(backend="ultralytics"))
    frame = AssetFrame(frame_id="f3", tensor=np.zeros((24, 32, 3), dtype=np.uint8), width=32, height=24)
    out1 = pipe.analyze(frame)
    fallback_id = id(pipe._builtin_detector_fallback)
    out2 = pipe.analyze(frame)
    assert out1.module_fallbacks["detector"] == "fallback"
    assert out2.module_fallbacks["detector"] == "fallback"
    assert pipe._builtin_detector_fallback is not None and id(pipe._builtin_detector_fallback) == fallback_id


def test_stage_timer_merge_uses_raw_entries():
    t1 = StageTimer(enabled=True)
    t1.add("parser", 0.01)
    t2 = StageTimer(enabled=True)
    t2.add("parser", 0.02)
    t2.add("detector", 0.03)
    t1.merge(t2)
    summary = t1.summary()
    assert summary["parser"]["count"] == 2.0
    assert summary["detector"]["count"] == 1.0
