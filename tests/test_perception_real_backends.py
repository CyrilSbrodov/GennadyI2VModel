from perception.detector import BackendConfig, YoloPersonDetectorAdapter
from perception.pipeline import PerceptionBackendsConfig, PerceptionPipeline


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def test_builtin_mode_still_returns_person_and_object() -> None:
    pipe = PerceptionPipeline(backends=PerceptionBackendsConfig())
    out = pipe.analyze(_solid(32, 32, (0.2, 0.3, 0.4)))
    assert out.persons
    assert out.objects


def test_detector_real_backend_falls_back_when_unavailable() -> None:
    detector = YoloPersonDetectorAdapter(BackendConfig(backend="ultralytics", checkpoint="missing.pt"))
    pipe = PerceptionPipeline(detector=detector)
    out = pipe.analyze(_solid(32, 32, (0.6, 0.3, 0.2)))
    assert out.persons
    assert out.module_fallbacks["detector"] == "fallback"


def test_real_mode_config_degrades_safely_without_weights() -> None:
    cfg = PerceptionBackendsConfig(
        detector=BackendConfig(backend="ultralytics", checkpoint="missing.pt"),
        pose=BackendConfig(backend="mediapipe"),
        parser=BackendConfig(backend="hf", checkpoint="not-a-model"),
        objects=BackendConfig(backend="ultralytics", checkpoint="missing.pt"),
    )
    out = PerceptionPipeline(backends=cfg).analyze(_solid(48, 48, (0.4, 0.4, 0.4)))
    assert out.persons
    assert out.objects
    assert any("unavailable" in w for w in out.warnings)
