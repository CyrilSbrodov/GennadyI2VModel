from runtime.orchestrator import GennadyEngine


def test_engine_runs_transition_loop() -> None:
    engine = GennadyEngine()
    artifacts = engine.run(["ref_0001.png"], "Снимает пальто и садится на стул. Улыбается.")

    assert len(artifacts.frames) > 2
    assert artifacts.state_plan.steps[0].labels == ["initial_state"]
    assert artifacts.debug["overlay_log"]
