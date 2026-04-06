from pathlib import Path

from runtime.orchestrator import GennadyEngine


def test_engine_frames_are_tensor_like_and_video_export_exists() -> None:
    engine = GennadyEngine()
    artifacts = engine.run(["ref_0001.png"], "Снимает пальто и садится на стул. Улыбается.", quality_profile="debug")

    assert isinstance(artifacts.frames[0], list)
    assert isinstance(artifacts.frames[-1], list)
    assert artifacts.frames[0] != artifacts.frames[-1]

    video_path = Path(str(artifacts.debug["video_export"]))
    assert video_path.exists()
    assert video_path.suffix == ".mp4"
