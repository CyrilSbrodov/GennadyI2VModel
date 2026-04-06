from pathlib import Path

from runtime.orchestrator import GennadyEngine


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_engine_frames_are_tensor_like_and_video_export_exists(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    _write_ppm(img, 32, 32, (30, 80, 220))

    engine = GennadyEngine()
    artifacts = engine.run([str(img)], "Снимает пальто и садится на стул. Улыбается.", quality_profile="debug")

    assert isinstance(artifacts.frames[0], list)
    assert isinstance(artifacts.frames[-1], list)
    assert artifacts.frames[0] != artifacts.frames[-1]

    video_path = Path(str(artifacts.debug["video_export"]))
    assert video_path.exists()
    assert video_path.suffix == ".mp4"
