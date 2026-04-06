from pathlib import Path

from core.schema import ActionPlan, ActionStep, SceneGraph
from perception.pipeline import PerceptionPipeline
from planning.transition_engine import TransitionPlanner
from runtime.orchestrator import GennadyEngine


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_perception_analyze_video_keeps_track_ids_and_metrics() -> None:
    pipe = PerceptionPipeline()
    outputs = pipe.analyze_video(["frame://1", "frame://2"])

    assert len(outputs) == 2
    assert outputs[0].persons and outputs[1].persons
    assert outputs[0].persons[0].track_id == outputs[1].persons[0].track_id
    assert "detector" in outputs[0].module_confidence
    assert "pose" in outputs[0].module_latency_ms


def test_planner_sit_down_without_chair_policy_modes() -> None:
    planner = TransitionPlanner()
    plan = ActionPlan(actions=[ActionStep(type="sit_down", priority=1)])

    inserted = planner.expand(SceneGraph(frame_index=0), plan, policy="insert")
    skipped = planner.expand(SceneGraph(frame_index=0), plan, policy="use_existing")
    failed = planner.expand(SceneGraph(frame_index=0), plan, policy="fail")

    assert inserted.diagnostics.inserted_objects
    assert skipped.diagnostics.skipped_actions == ["sit_down"]
    assert failed.diagnostics.constraint_warnings


def test_profile_smoke_comparison(tmp_path: Path) -> None:
    engine = GennadyEngine()
    img = tmp_path / "a.ppm"
    _write_ppm(img, 100, 60, (10, 200, 150))
    out_light = engine.run([str(img)], "улыбается", quality_profile="lightweight")
    out_quality = engine.run([str(img)], "улыбается", quality_profile="quality")

    light_shape = (len(out_light.frames[0]), len(out_light.frames[0][0]))
    quality_shape = (len(out_quality.frames[0]), len(out_quality.frames[0][0]))
    assert light_shape != quality_shape
