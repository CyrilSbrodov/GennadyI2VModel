from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from evaluation.dataset import (
    BenchmarkRecord,
    dataset_diagnostics_to_dict,
    load_benchmark_dataset,
)
from learned.factory import BackendConfig
from runtime.orchestrator import GennadyEngine


@dataclass(slots=True)
class ScenarioSpec:
    scenario_id: str
    prompt: str
    action_family: str
    transition_family: str
    expected_region_families: list[str]
    expected_runtime_conditions: dict[str, object]


DEFAULT_SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec(
        scenario_id="face_expression_change",
        prompt="Смотрит в камеру, постепенно улыбается и смягчает взгляд.",
        action_family="expression_change",
        transition_family="face_emotion",
        expected_region_families=["face"],
        expected_runtime_conditions={"require_contract_validity": True, "max_fallbacks": 4},
    ),
    ScenarioSpec(
        scenario_id="torso_garment_reveal",
        prompt="Расстегивает куртку и приоткрывает верхнюю одежду.",
        action_family="garment_transition",
        transition_family="torso_reveal",
        expected_region_families=["torso", "outerwear"],
        expected_runtime_conditions={"require_contract_validity": True, "max_fallbacks": 5},
    ),
    ScenarioSpec(
        scenario_id="sleeve_arm_outerwear",
        prompt="Закатывает рукав и освобождает руку от внешнего слоя одежды.",
        action_family="arm_outerwear_transition",
        transition_family="sleeve_change",
        expected_region_families=["arm", "sleeve", "outerwear"],
        expected_runtime_conditions={"require_contract_validity": True, "max_fallbacks": 5},
    ),
    ScenarioSpec(
        scenario_id="posture_sit_stand",
        prompt="Сначала встает, затем плавно садится на стул.",
        action_family="posture_change",
        transition_family="sit_stand",
        expected_region_families=["torso", "legs"],
        expected_runtime_conditions={"require_contract_validity": True, "max_fallbacks": 6},
    ),
    ScenarioSpec(
        scenario_id="mixed_multi_step",
        prompt="Снимает легкую куртку, поворачивается, садится и улыбается.",
        action_family="multi_step_mixed",
        transition_family="mixed",
        expected_region_families=["face", "torso", "arm", "outerwear"],
        expected_runtime_conditions={"require_contract_validity": True, "max_fallbacks": 8},
    ),
]


def _write_seed_ppm(path: Path, w: int = 32, h: int = 32) -> None:
    rows: list[str] = []
    for y in range(h):
        for x in range(w):
            r = (x * 7 + y * 3) % 255
            g = (x * 5 + y * 9) % 255
            b = (x * 11 + y * 2) % 255
            rows.append(f"{r} {g} {b}")
    path.write_text(f"P3\n{w} {h}\n255\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _backend_config_for_mode(mode: str) -> BackendConfig:
    normalized = mode.strip().lower()
    if normalized in {"learned_primary", "learned", "primary"}:
        return BackendConfig(dynamics_backend="learned_primary", patch_backend="learned_primary", temporal_backend="learned_primary")
    if normalized in {"legacy", "fallback", "baseline"}:
        return BackendConfig(dynamics_backend="baseline", patch_backend="baseline", temporal_backend="baseline")
    if normalized in {"hybrid"}:
        return BackendConfig(dynamics_backend="learned_primary", patch_backend="baseline", temporal_backend="trainable_temporal")
    raise ValueError(f"Unsupported backend mode: {mode}")


def _stage_health_from_debug(debug: dict[str, object]) -> dict[str, object]:
    steps = debug.get("step_execution", []) if isinstance(debug, dict) else []
    if not isinstance(steps, list):
        steps = []

    dynamics_conf = []
    dynamics_delta = []
    dynamics_viol = []
    temporal_drift = []
    patch_conf = []
    patch_hidden_cases = 0

    patch_contract_failures = 0
    temporal_contract_failures = 0
    parity_missing_count = 0

    patch_total = 0
    patch_learned_primary = 0
    temporal_learned_primary = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        dynamics = step.get("dynamics", {}) if isinstance(step.get("dynamics", {}), dict) else {}
        dynamics_conf.append(float(dynamics.get("confidence", 0.0)))
        dsum = dynamics.get("diagnostics_summary", {}) if isinstance(dynamics.get("diagnostics_summary", {}), dict) else {}
        dynamics_delta.append(float(dsum.get("delta_magnitude", 0.0) or 0.0))
        dynamics_viol.append(float(dsum.get("violations", 0.0) or 0.0))

        temporal = step.get("temporal", {}) if isinstance(step.get("temporal", {}), dict) else {}
        drift_payload = temporal.get("drift_consistency", {}) if isinstance(temporal.get("drift_consistency", {}), dict) else {}
        temporal_drift.append(float(drift_payload.get("drift_proxy", 1.0) or 1.0))

        temporal_contract = temporal.get("contract_validation", {}) if isinstance(temporal.get("contract_validation", {}), dict) else {}
        temporal_issues = temporal_contract.get("issues", []) if isinstance(temporal_contract.get("issues", []), list) else []
        temporal_contract_failures += len(temporal_issues)
        temporal_learned_primary += 1 if bool(temporal_contract.get("is_learned_primary", False)) else 0

        patches = step.get("patch", []) if isinstance(step.get("patch", []), list) else []
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            patch_total += 1
            patch_conf.append(float(patch.get("confidence", 0.0)))
            hidden = patch.get("hidden_reconstruction", {}) if isinstance(patch.get("hidden_reconstruction", {}), dict) else {}
            if bool(hidden.get("patch_hidden_reconstruction_case", False)):
                patch_hidden_cases += 1

            p_contract = patch.get("contract_validation", {}) if isinstance(patch.get("contract_validation", {}), dict) else {}
            p_issues = p_contract.get("issues", []) if isinstance(p_contract.get("issues", []), list) else []
            patch_contract_failures += len(p_issues)
            patch_learned_primary += 1 if bool(p_contract.get("is_learned_primary", False)) else 0

            parity = patch.get("parity", {}) if isinstance(patch.get("parity", {}), dict) else {}
            miss = parity.get("missing_fields", []) if isinstance(parity.get("missing_fields", []), list) else []
            parity_missing_count += len(miss)

        parity_root = step.get("parity", {}) if isinstance(step.get("parity", {}), dict) else {}
        miss_root = parity_root.get("missing_fields", {}) if isinstance(parity_root.get("missing_fields", {}), dict) else {}
        for v in miss_root.values():
            if isinstance(v, list):
                parity_missing_count += sum(1 for x in v if x)

    step_count = max(1, len(steps))
    return {
        "dynamics": {
            "avg_confidence": round(_safe_mean(dynamics_conf), 6),
            "avg_delta_magnitude": round(_safe_mean(dynamics_delta), 6),
            "avg_constraint_violations": round(_safe_mean(dynamics_viol), 6),
            "transition_health_proxy": round(max(0.0, 1.0 - _safe_mean(dynamics_viol)), 6),
        },
        "patch": {
            "avg_patch_confidence": round(_safe_mean(patch_conf), 6),
            "hidden_reconstruction_case_ratio": round(patch_hidden_cases / max(1, patch_total), 6),
            "patch_contract_failure_count": int(patch_contract_failures),
            "patch_learned_primary_ratio": round(patch_learned_primary / max(1, patch_total), 6),
        },
        "temporal": {
            "avg_temporal_drift_proxy": round(_safe_mean(temporal_drift), 6),
            "avg_temporal_consistency_proxy": round(max(0.0, 1.0 - _safe_mean(temporal_drift)), 6),
            "temporal_contract_failure_count": int(temporal_contract_failures),
            "temporal_learned_primary_ratio": round(temporal_learned_primary / step_count, 6),
        },
        "contracts": {
            "parity_missing_count": int(parity_missing_count),
            "contract_failure_count": int(patch_contract_failures + temporal_contract_failures),
            "contract_valid_ratio": round(1.0 if (patch_contract_failures + temporal_contract_failures) == 0 else 0.0, 6),
        },
    }


def _scenario_metrics(spec: ScenarioSpec, artifacts: object, *, record: BenchmarkRecord | None = None) -> dict[str, object]:
    state_plan = getattr(artifacts, "state_plan", None)
    steps = getattr(state_plan, "steps", []) if state_plan is not None else []
    labels = [label.lower() for s in steps if hasattr(s, "labels") for label in (s.labels or [])]
    joined_labels = " ".join(labels)

    debug = getattr(artifacts, "debug", {}) if isinstance(getattr(artifacts, "debug", {}), dict) else {}
    stage_health = _stage_health_from_debug(debug)

    step_exec = debug.get("step_execution", []) if isinstance(debug.get("step_execution", []), list) else []
    changed_regions: list[str] = []
    for step in step_exec:
        if not isinstance(step, dict):
            continue
        patches = step.get("patch", []) if isinstance(step.get("patch", []), list) else []
        for patch in patches:
            if isinstance(patch, dict):
                rid = str(patch.get("region_id", "")).lower()
                if rid:
                    changed_regions.append(rid)

    expected_regions_hit = 0
    for fam in spec.expected_region_families:
        fam_l = fam.lower()
        if any(fam_l in region for region in changed_regions):
            expected_regions_hit += 1

    expected_region_coverage = expected_regions_hit / max(1, len(spec.expected_region_families))
    action_hit = 1.0 if spec.action_family.split("_")[0] in joined_labels else 0.0

    learned_ready = debug.get("learned_ready", {}) if isinstance(debug.get("learned_ready", {}), dict) else {}
    fallback_count = len(learned_ready.get("fallbacks", [])) if isinstance(learned_ready.get("fallbacks", []), list) else 0

    contract_failures = int(stage_health["contracts"]["contract_failure_count"])
    parity_missing = int(stage_health["contracts"]["parity_missing_count"])
    scenario_success = (
        contract_failures == 0
        and parity_missing == 0
        and fallback_count <= int(spec.expected_runtime_conditions.get("max_fallbacks", 9999))
        and expected_region_coverage > 0.0
    )

    pipeline_health_proxy = round(
        (
            0.25 * float(action_hit)
            + 0.20 * float(expected_region_coverage)
            + 0.20 * float(stage_health["dynamics"]["transition_health_proxy"])
            + 0.20 * float(stage_health["patch"]["avg_patch_confidence"])
            + 0.15 * float(stage_health["temporal"]["avg_temporal_consistency_proxy"])
        ),
        6,
    )

    payload: dict[str, object] = {
        "scenario_id": spec.scenario_id,
        "prompt": spec.prompt,
        "expected": {
            "action_family": spec.action_family,
            "transition_family": spec.transition_family,
            "region_families": spec.expected_region_families,
            "runtime_conditions": spec.expected_runtime_conditions,
        },
        "observed": {
            "step_count": max(0, len(steps) - 1),
            "labels": labels,
            "changed_regions": changed_regions,
            "fallback_count": fallback_count,
            "contract_failure_count": contract_failures,
            "parity_missing_count": parity_missing,
        },
        "metrics": {
            "action_parsing_coverage_proxy": float(action_hit),
            "expected_region_coverage_proxy": round(expected_region_coverage, 6),
            "pipeline_health_proxy": pipeline_health_proxy,
            "fallback_free": 1.0 if fallback_count == 0 else 0.0,
            "contract_validity": 1.0 if contract_failures == 0 else 0.0,
            "scenario_success": 1.0 if scenario_success else 0.0,
        },
        "stage_health": stage_health,
    }
    if record is not None:
        payload["dataset_record"] = {
            "record_id": record.record_id,
            "asset_id": record.asset_id,
            "asset_path": record.asset_path,
            "tags": record.tags,
            "notes": record.notes,
            "weak_expectations": record.weak_expectations,
        }
    return payload


def _run_single_scenario(image_path: str, scenario: ScenarioSpec, backend_mode: str, *, record: BenchmarkRecord | None = None) -> dict[str, object]:
    engine = GennadyEngine(backend_config=_backend_config_for_mode(backend_mode))
    artifacts = engine.run([image_path], scenario.prompt, quality_profile="debug", duration=2.5)
    return _scenario_metrics(scenario, artifacts, record=record)


def run_stage_eval(image_path: str, text: str, backend_mode: str = "learned_primary") -> dict[str, object]:
    engine = GennadyEngine(backend_config=_backend_config_for_mode(backend_mode))
    artifacts = engine.run([image_path], text, quality_profile="debug", duration=2.5)
    debug = artifacts.debug if isinstance(artifacts.debug, dict) else {}
    stage_health = _stage_health_from_debug(debug)
    return {
        "mode": "stage",
        "backend_mode": backend_mode,
        "text": text,
        "summary": {
            "contract_failure_count": stage_health["contracts"]["contract_failure_count"],
            "parity_missing_count": stage_health["contracts"]["parity_missing_count"],
            "fallback_count": len(debug.get("learned_ready", {}).get("fallbacks", [])) if isinstance(debug.get("learned_ready", {}), dict) else 0,
        },
        "stage_health": stage_health,
        "backend_selection": debug.get("learned_ready", {}).get("backend_selection", {}) if isinstance(debug.get("learned_ready", {}), dict) else {},
    }


def _record_to_scenario(record: BenchmarkRecord) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=record.scenario_id,
        prompt=record.canonical_prompt,
        action_family=record.action_family,
        transition_family=record.transition_family,
        expected_region_families=record.expected_region_families,
        expected_runtime_conditions=record.expected_runtime_conditions,
    )


def _synthetic_records_for_image(seed_image: str, scenario_specs: list[ScenarioSpec]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for spec in scenario_specs:
        records.append(
            BenchmarkRecord(
                record_id=f"synthetic_{spec.scenario_id}",
                asset_id="synthetic_seed",
                asset_path=seed_image,
                scenario_id=spec.scenario_id,
                canonical_prompt=spec.prompt,
                action_family=spec.action_family,
                transition_family=spec.transition_family,
                expected_region_families=spec.expected_region_families,
                expected_runtime_conditions=spec.expected_runtime_conditions,
                tags=["synthetic_seed"],
                notes="Generated from single user-provided or internal synthetic image.",
                weak_expectations={},
            )
        )
    return records


def run_benchmark_suite(
    backend_modes: list[str] | None = None,
    scenarios: list[ScenarioSpec] | None = None,
    image_path: str | None = None,
    dataset_manifest: str | None = None,
    scenario_filter: list[str] | None = None,
    asset_filter: list[str] | None = None,
) -> dict[str, object]:
    scenario_specs = scenarios or DEFAULT_SCENARIOS
    modes = backend_modes or ["learned_primary", "legacy"]

    temp_dir: Path | None = None
    dataset_info: dict[str, object]
    dataset_records: list[BenchmarkRecord]

    if image_path:
        seed_image = image_path
        dataset_records = _synthetic_records_for_image(seed_image, scenario_specs)
        dataset_info = {
            "dataset_id": "synthetic_seed",
            "name": "synthetic_seed_from_image",
            "version": "1",
            "manifest_path": None,
            "diagnostics": {
                "total_records": len(dataset_records),
                "valid_records": len(dataset_records),
                "invalid_records": 0,
                "missing_assets": 0,
                "invalid_record_ids": [],
                "missing_asset_record_ids": [],
            },
            "used_curated_pack": False,
            "source": "explicit_image",
        }
    else:
        dataset, diagnostics = load_benchmark_dataset(dataset_manifest)
        dataset_records = dataset.records
        dataset_info = {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "version": dataset.version,
            "manifest_path": dataset.manifest_path,
            "diagnostics": dataset_diagnostics_to_dict(diagnostics),
            "used_curated_pack": True,
            "source": "manifest",
        }

    scenario_allow = {s.strip().lower() for s in scenario_filter or [] if s.strip()}
    asset_allow = {a.strip().lower() for a in asset_filter or [] if a.strip()}
    records = [
        r
        for r in dataset_records
        if (not scenario_allow or r.scenario_id.lower() in scenario_allow)
        and (not asset_allow or r.asset_id.lower() in asset_allow)
    ]
    if not records:
        raise ValueError("Benchmark selection resulted in zero dataset records. Adjust scenario/asset filters.")

    runs: dict[str, object] = {}
    for mode in modes:
        scenario_results = [_run_single_scenario(r.asset_path, _record_to_scenario(r), mode, record=r) for r in records]
        success_ratio = _safe_mean([float(r["metrics"]["scenario_success"]) for r in scenario_results])
        fallback_free_ratio = _safe_mean([float(r["metrics"]["fallback_free"]) for r in scenario_results])
        contract_valid_ratio = _safe_mean([float(r["metrics"]["contract_validity"]) for r in scenario_results])
        pipeline_score = _safe_mean([float(r["metrics"]["pipeline_health_proxy"]) for r in scenario_results])

        stage_scores = {
            "dynamics": _safe_mean([float(r["stage_health"]["dynamics"]["transition_health_proxy"]) for r in scenario_results]),
            "patch": _safe_mean([float(r["stage_health"]["patch"]["avg_patch_confidence"]) for r in scenario_results]),
            "temporal": _safe_mean([float(r["stage_health"]["temporal"]["avg_temporal_consistency_proxy"]) for r in scenario_results]),
        }

        asset_results: dict[str, dict[str, object]] = {}
        scenario_family_coverage: dict[str, int] = {}
        for item in scenario_results:
            rec = item.get("dataset_record", {}) if isinstance(item.get("dataset_record", {}), dict) else {}
            aid = str(rec.get("asset_id", "unknown"))
            sid = str(item.get("scenario_id", "unknown"))
            scenario_family_coverage[sid] = scenario_family_coverage.get(sid, 0) + 1

            slot = asset_results.setdefault(
                aid,
                {
                    "asset_path": rec.get("asset_path", ""),
                    "record_count": 0,
                    "scenario_success_ratio": 0.0,
                    "contract_valid_ratio": 0.0,
                    "fallback_free_ratio": 0.0,
                    "scenarios": [],
                },
            )
            slot["record_count"] = int(slot["record_count"]) + 1
            slot["scenarios"].append(sid)

        for aid, slot in asset_results.items():
            items = [
                item
                for item in scenario_results
                if isinstance(item.get("dataset_record", {}), dict) and str(item.get("dataset_record", {}).get("asset_id", "")) == aid
            ]
            slot["scenario_success_ratio"] = round(_safe_mean([float(i["metrics"]["scenario_success"]) for i in items]), 6)
            slot["contract_valid_ratio"] = round(_safe_mean([float(i["metrics"]["contract_validity"]) for i in items]), 6)
            slot["fallback_free_ratio"] = round(_safe_mean([float(i["metrics"]["fallback_free"]) for i in items]), 6)

        warnings: list[str] = []
        if contract_valid_ratio < 1.0:
            warnings.append("contract_failures_detected")
        if fallback_free_ratio < 0.5:
            warnings.append("high_fallback_usage")
        if success_ratio < 0.6:
            warnings.append("low_scenario_success")

        runs[mode] = {
            "backend_mode": mode,
            "summary": {
                "scenario_count": len(scenario_results),
                "scenario_success_ratio": round(success_ratio, 6),
                "fallback_free_ratio": round(fallback_free_ratio, 6),
                "contract_valid_ratio": round(contract_valid_ratio, 6),
                "pipeline_health_score": round(pipeline_score, 6),
                "stage_health_score": {k: round(v, 6) for k, v in stage_scores.items()},
            },
            "warnings": warnings,
            "scenario_family_coverage": scenario_family_coverage,
            "assets": asset_results,
            "scenarios": scenario_results,
        }

    comparison = {}
    if "learned_primary" in runs and "legacy" in runs:
        lp = runs["learned_primary"]["summary"]
        lg = runs["legacy"]["summary"]
        comparison = {
            "modes": ["learned_primary", "legacy"],
            "delta": {
                "scenario_success_ratio": round(float(lp["scenario_success_ratio"]) - float(lg["scenario_success_ratio"]), 6),
                "fallback_free_ratio": round(float(lp["fallback_free_ratio"]) - float(lg["fallback_free_ratio"]), 6),
                "contract_valid_ratio": round(float(lp["contract_valid_ratio"]) - float(lg["contract_valid_ratio"]), 6),
                "pipeline_health_score": round(float(lp["pipeline_health_score"]) - float(lg["pipeline_health_score"]), 6),
            },
        }

    report = {
        "kind": "single_image_pipeline_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            **dataset_info,
            "selected_record_count": len(records),
            "selected_scenario_ids": sorted({r.scenario_id for r in records}),
            "selected_asset_ids": sorted({r.asset_id for r in records}),
        },
        "scenario_catalog": sorted(
            [
                {
                    "scenario_id": r.scenario_id,
                    "action_family": r.action_family,
                    "transition_family": r.transition_family,
                }
                for r in records
            ],
            key=lambda x: (str(x["scenario_id"]), str(x["action_family"])),
        ),
        "runs": runs,
        "comparison": comparison,
        "regression_summary": {
            "warnings": {
                mode: payload.get("warnings", []) for mode, payload in runs.items()
            },
            "failing_scenarios": {
                mode: [
                    s["scenario_id"]
                    for s in payload.get("scenarios", [])
                    if float(s.get("metrics", {}).get("scenario_success", 0.0)) < 1.0
                ]
                for mode, payload in runs.items()
            },
            "failing_assets": {
                mode: [
                    aid
                    for aid, asset_payload in payload.get("assets", {}).items()
                    if float(asset_payload.get("scenario_success_ratio", 0.0)) < 1.0
                ]
                for mode, payload in runs.items()
            },
        },
    }
    if temp_dir is not None:
        report["temp_assets_dir"] = str(temp_dir)
    return report


def write_report(report: dict[str, object], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
