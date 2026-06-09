from pathlib import Path

from core.pipeline_contract import CANONICAL_STAGE_NAMES, validate_runtime_trace
from learned.factory import BackendConfig
from runtime.orchestrator import GennadyEngine


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_engine_runs_transition_loop(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    _write_ppm(img, 32, 32, (120, 90, 40))

    engine = GennadyEngine(backend_config=BackendConfig(dynamics_backend="baseline"))
    artifacts = engine.run([str(img)], "Снимает пальто и садится на стул. Улыбается.")

    assert len(artifacts.frames) > 2
    assert artifacts.state_plan.steps[0].labels == ["initial_state"]
    planner_action_plan = artifacts.debug["planner_action_plan"]
    assert planner_action_plan["planner_contract_version"] == "planner_action_plan_v1"
    assert planner_action_plan["supported"] is True
    assert [action["action_type"] for action in planner_action_plan["actions"]] == ["sit_down", "expression_change"]
    assert any("снимает пальто" in fragment for fragment in planner_action_plan["unsupported_fragments"])
    assert any("planner_action_plan_partial_unsupported:снимает пальто" in item for item in artifacts.debug["learned_ready"]["fallbacks"])
    dynamics_contract = artifacts.debug["dynamics_graph_delta_contract"]
    assert dynamics_contract["contract_version"] == "graph_delta_contract_v1"
    assert dynamics_contract["supported"] is True
    assert any("снимает пальто" in fragment for fragment in dynamics_contract["trace"]["unsupported_planner_fragments"])
    assert dynamics_contract["routing_candidates"]
    assert dynamics_contract["region_routing_called"] is False
    assert dynamics_contract["rendered_pixels_generated"] is False
    assert dynamics_contract["memory_write_performed"] is False
    assert dynamics_contract["observed_evidence_created"] is False
    reveal_contract = artifacts.debug["reveal_occlusion_contract"]
    assert reveal_contract["contract_version"] == "reveal_occlusion_contract_v1"
    assert reveal_contract["supported"] is True
    assert any("снимает пальто" in fragment for fragment in reveal_contract["trace"]["unsupported_planner_fragments"])
    assert reveal_contract["region_routing_called"] is False
    assert reveal_contract["rendered_pixels_generated"] is False
    assert reveal_contract["memory_write_performed"] is False
    assert reveal_contract["observed_evidence_created"] is False
    assert artifacts.debug["overlay_log"]


def test_engine_runtime_trace_uses_canonical_pipeline_order(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    _write_ppm(img, 32, 32, (120, 90, 40))

    artifacts = GennadyEngine(backend_config=BackendConfig(dynamics_backend="baseline")).run([str(img)], "Садится и улыбается.")
    stages = [entry["stage"] for entry in artifacts.debug["runtime_trace"]]

    planner_action_plan = artifacts.debug["planner_action_plan"]
    assert planner_action_plan["supported"] is True
    assert [action["action_type"] for action in planner_action_plan["actions"]] == ["sit_down", "expression_change"]
    validate_runtime_trace(artifacts.debug["runtime_trace"])
    for mandatory_stage in CANONICAL_STAGE_NAMES:
        assert mandatory_stage in stages
    assert "dynamics_graph_delta_contract" in artifacts.debug
    assert "reveal_occlusion_contract" in artifacts.debug
    assert min(i for i, stage in enumerate(stages) if stage == "planning") < min(i for i, stage in enumerate(stages) if stage == "dynamics")
    assert min(i for i, stage in enumerate(stages) if stage == "dynamics") < min(i for i, stage in enumerate(stages) if stage == "reveal")
    assert min(i for i, stage in enumerate(stages) if stage == "reveal") < min(i for i, stage in enumerate(stages) if stage == "region_routing")
    assert min(i for i, stage in enumerate(stages) if stage == "region_routing") < min(i for i, stage in enumerate(stages) if stage == "rendering")
    for step in artifacts.debug["step_execution"]:
        for patch in step["patch"]:
            trace = patch["execution_trace"]
            assert trace["region_route_decision"]["region_id"] == patch["region_id"]
            assert trace["region_route_decision"]["decision"] != "unknown"
