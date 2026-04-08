from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmark import DEFAULT_SCENARIOS, run_benchmark_suite, run_stage_eval, write_report
from evaluation.dataset import default_benchmark_manifest_path, load_benchmark_dataset


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


def test_curated_dataset_manifest_loads_with_real_assets() -> None:
    dataset, diagnostics = load_benchmark_dataset(default_benchmark_manifest_path())
    assert dataset.dataset_id == "single_image_curated_v1"
    assert diagnostics.valid_records >= 6
    assert diagnostics.missing_assets == 0
    assert any(r.scenario_id == "mixed_multi_step" for r in dataset.records)


def test_benchmark_suite_runs_curated_manifest_by_default() -> None:
    report = run_benchmark_suite(backend_modes=["learned_primary", "legacy"])

    assert report["kind"] == "single_image_pipeline_benchmark"
    assert report["dataset"]["used_curated_pack"] is True
    assert report["dataset"]["selected_record_count"] >= 6
    assert "learned_primary" in report["runs"]
    assert "legacy" in report["runs"]
    assert report["comparison"]["modes"] == ["learned_primary", "legacy"]

    lp = report["runs"]["learned_primary"]
    assert len(lp["scenarios"]) == report["dataset"]["selected_record_count"]
    assert lp["scenario_family_coverage"]["torso_garment_reveal"] >= 2
    assert len(lp["assets"]) >= 5


def test_benchmark_suite_supports_filters_and_asset_level_summary() -> None:
    report = run_benchmark_suite(
        backend_modes=["learned_primary"],
        scenario_filter=["torso_garment_reveal"],
        asset_filter=["torso_outerwear_front_ref"],
    )
    run = report["runs"]["learned_primary"]
    assert report["dataset"]["selected_scenario_ids"] == ["torso_garment_reveal"]
    assert report["dataset"]["selected_asset_ids"] == ["torso_outerwear_front_ref"]
    assert run["summary"]["scenario_count"] == 1
    assert "torso_outerwear_front_ref" in run["assets"]


def test_benchmark_suite_runs_scenarios_and_produces_comparison_with_explicit_image(tmp_path: Path) -> None:
    img = tmp_path / "seed.ppm"
    _write_ppm(img)

    report = run_benchmark_suite(backend_modes=["learned_primary", "legacy"], image_path=str(img))

    assert report["kind"] == "single_image_pipeline_benchmark"
    assert report["dataset"]["used_curated_pack"] is False
    assert len(report["scenario_catalog"]) == len(DEFAULT_SCENARIOS)


def test_manifest_loader_reports_missing_assets(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "manifest.json"
    missing_manifest.write_text(
        json.dumps(
            {
                "dataset_id": "broken",
                "version": "1",
                "records": [
                    {
                        "record_id": "broken_1",
                        "asset_id": "missing",
                        "asset_path": "images/not_found.ppm",
                        "scenario_id": "face_expression_change",
                        "canonical_prompt": "x",
                        "action_family": "expression_change",
                        "transition_family": "face_emotion",
                        "expected_region_families": ["face"],
                        "expected_runtime_conditions": {"max_fallbacks": 1},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    try:
        load_benchmark_dataset(missing_manifest)
    except ValueError as exc:
        assert "No valid benchmark records loaded" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing asset manifest")


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
