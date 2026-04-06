from pathlib import Path

from runtime.orchestrator import GennadyEngine


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_engine_runs_transition_loop(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    _write_ppm(img, 32, 32, (120, 90, 40))

    engine = GennadyEngine()
    artifacts = engine.run([str(img)], "Снимает пальто и садится на стул. Улыбается.")

    assert len(artifacts.frames) > 2
    assert artifacts.state_plan.steps[0].labels == ["initial_state"]
    assert artifacts.debug["overlay_log"]
