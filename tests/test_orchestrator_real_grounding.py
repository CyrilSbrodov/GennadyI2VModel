from pathlib import Path

from runtime.orchestrator import GennadyEngine


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_pipeline_uses_input_image_pixels(tmp_path: Path) -> None:
    img = tmp_path / "red.ppm"
    _write_ppm(img, 32, 32, (255, 0, 0))
    engine = GennadyEngine()

    artifacts = engine.run([str(img)], "улыбается", quality_profile="debug")
    first = artifacts.frames[0][0][0]

    assert first[0] > 0.9
    assert first[1] < 0.1
    assert first[2] < 0.1


def test_synthetic_seed_mode_not_default(tmp_path: Path) -> None:
    img = tmp_path / "gray.ppm"
    _write_ppm(img, 24, 20, (100, 100, 100))
    engine = GennadyEngine()

    artifacts = engine.run([str(img)], "улыбается")
    assert artifacts.debug["input_metadata"]["source_mode"] == "image_grounded"


def test_output_frame_shape_matches_input_profile(tmp_path: Path) -> None:
    img = tmp_path / "blue.ppm"
    _write_ppm(img, 80, 40, (0, 0, 255))
    engine = GennadyEngine()

    artifacts = engine.run([str(img)], "поворачивает голову", quality_profile="debug")
    h = len(artifacts.frames[0])
    w = len(artifacts.frames[0][0])
    assert artifacts.debug["input_metadata"]["normalized_size"] == (w, h)
