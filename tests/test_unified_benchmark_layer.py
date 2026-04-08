from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmark import DEFAULT_SCENARIOS, run_benchmark_suite, run_stage_eval, write_report


def _write_ppm(path: Path, w: int = 28, h: int = 28) -> None:
    pixels = []
    for y in range(h):
        for x in range(w):
            pixels.append(f"{(x * 9) % 255} {(y * 11) % 255} {((x + y) * 7) % 255}")
    path.write_text(f"P3\n{w} {h}\n255\n" + "\n".join(pixels) + "\n", encoding="utf-8")


def test_stage_eval_returns_structured_health_payload(tmp_path: Path) -> None:
    img = tmp_path / "seed.ppm"
    _write_ppm(img)

    payload = run_stage_eval(str(img), "Снимает куртку и улыбается.", backend_mode="learned_primary")

    assert payload["mode"] == "stage"
    assert payload["stage_health"]["dynamics"]["transition_health_proxy"] >= 0.0
    assert "contract_failure_count" in payload["summary"]


def test_benchmark_suite_runs_scenarios_and_produces_comparison(tmp_path: Path) -> None:
    img = tmp_path / "seed.ppm"
    _write_ppm(img)

    report = run_benchmark_suite(backend_modes=["learned_primary", "legacy"], image_path=str(img))

    assert report["kind"] == "single_image_pipeline_benchmark"
    assert len(report["scenario_catalog"]) >= 5
    assert "learned_primary" in report["runs"]
    assert "legacy" in report["runs"]
    assert report["comparison"]["modes"] == ["learned_primary", "legacy"]

    lp = report["runs"]["learned_primary"]
    assert len(lp["scenarios"]) == len(DEFAULT_SCENARIOS)
    first = lp["scenarios"][0]
    assert "stage_health" in first
    assert "metrics" in first
    assert "pipeline_health_proxy" in first["metrics"]


def test_report_writer_persists_regression_oriented_json(tmp_path: Path) -> None:
    report = {
        "kind": "single_image_pipeline_benchmark",
        "runs": {"learned_primary": {"summary": {"scenario_success_ratio": 0.5}, "scenarios": []}},
        "regression_summary": {"warnings": {"learned_primary": ["low_scenario_success"]}, "failing_scenarios": {"learned_primary": []}},
    }
    out = tmp_path / "benchmark.json"
    saved = write_report(report, str(out))

    loaded = json.loads(Path(saved).read_text(encoding="utf-8"))
    assert loaded["regression_summary"]["warnings"]["learned_primary"] == ["low_scenario_success"]
