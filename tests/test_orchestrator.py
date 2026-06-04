from pathlib import Path

from core.pipeline_contract import CANONICAL_STAGE_NAMES, validate_runtime_trace
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


def test_engine_runtime_trace_uses_canonical_pipeline_order(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    _write_ppm(img, 32, 32, (120, 90, 40))

    artifacts = GennadyEngine().run([str(img)], "Улыбается.")
    stages = [entry["stage"] for entry in artifacts.debug["runtime_trace"]]

    validate_runtime_trace(artifacts.debug["runtime_trace"])
    for mandatory_stage in CANONICAL_STAGE_NAMES:
        assert mandatory_stage in stages
    assert min(i for i, stage in enumerate(stages) if stage == "region_routing") < min(i for i, stage in enumerate(stages) if stage == "rendering")
    for step in artifacts.debug["step_execution"]:
        for patch in step["patch"]:
            trace = patch["execution_trace"]
            assert trace["region_route_decision"]["region_id"] == patch["region_id"]
            assert trace["region_route_decision"]["decision"] != "unknown"
