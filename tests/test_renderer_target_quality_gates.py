from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rendering.target_provenance_policy import classify_target_training_role, target_quality_warning
from rendering.trainable_patch_renderer import TrainableLocalPatchModel
from training.datasets import RendererDataset
from training.renderer_trainer import RendererBatchAdapter, RendererTrainer


def _record(*, target_source: str | None = None, training_target_quality: str | None = None) -> dict[str, object]:
    before = np.zeros((2, 2, 3), dtype=np.float32)
    after = np.full((2, 2, 3), 0.5, dtype=np.float32)
    rec: dict[str, object] = {
        "roi_before": before.tolist(),
        "roi_after": after.tolist(),
        "semantic_family": "sleeve_arm_transition",
        "region_id": "person_0:left_arm",
    }
    if target_source is not None:
        rec["target_source"] = target_source
    if training_target_quality is not None:
        rec["training_target_quality"] = training_target_quality
    return rec


def _write_manifest(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "renderer_patch_manifest.json"
    path.write_text(json.dumps({"manifest_type": "renderer_patch_manifest", "records": records}), encoding="utf-8")
    return path


def test_target_policy_classifies_roles() -> None:
    assert classify_target_training_role("external_or_observed_target") == "supervised_external"
    assert target_quality_warning("external_or_observed_target") is None

    assert classify_target_training_role("self_generated_runtime_target") == "bootstrap_self_generated"
    assert target_quality_warning("self_generated_runtime_target") == "self_generated_runtime_target_is_bootstrap_not_ground_truth"

    assert classify_target_training_role("unknown") == "weak_unknown"
    assert target_quality_warning("unknown") == "unknown_training_target_quality"


def test_dataset_target_role_diagnostics_mixed_manifest(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            _record(target_source="provided_ground_truth_roi", training_target_quality="external_or_observed_target"),
            _record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target"),
            _record(target_source="unknown", training_target_quality="unknown"),
        ],
    )

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)

    assert ds.diagnostics["target_training_role_counts"] == {
        "supervised_external": 1,
        "bootstrap_self_generated": 1,
        "weak_unknown": 1,
    }
    assert ds.diagnostics["supervised_external_ratio"] == pytest.approx(1 / 3)
    assert ds.diagnostics["bootstrap_self_generated_ratio"] == pytest.approx(1 / 3)
    assert ds.diagnostics["weak_unknown_target_ratio"] == pytest.approx(1 / 3)
    assert ds.diagnostics["contains_no_supervised_external_targets"] is False


def test_dataset_no_supervised_external_warns_but_loads_strict(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            _record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target"),
            _record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target"),
        ],
    )

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)

    assert len(ds.samples) == 2
    assert ds.diagnostics["contains_no_supervised_external_targets"] is True
    assert ds.diagnostics["contains_only_bootstrap_targets"] is True
    assert any(w.get("type") == "no_supervised_external_targets" for w in ds.diagnostics["warnings"])


def test_batch_adapter_target_role_is_authoritative_over_extra_summary(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [_record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target")],
    )
    sample = RendererDataset.from_renderer_manifest(str(path), strict=True).samples[0]
    sample["renderer_batch_contract"]["conditioning_summary"] = {
        "target_training_role": "supervised_external",
        "target_quality_warning": None,
    }

    batch = RendererBatchAdapter().adapt(sample)

    assert batch.conditioning_summary["target_training_role"] == "bootstrap_self_generated"
    assert batch.conditioning_summary["target_quality_warning"] == "self_generated_runtime_target_is_bootstrap_not_ground_truth"
    assert batch.conditioning_summary["target_supervision_weight"] == 0.35


def test_evaluate_model_reports_target_role_ratios(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            _record(target_source="provided_ground_truth_roi", training_target_quality="external_or_observed_target"),
            _record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target"),
            _record(target_source="unknown", training_target_quality="unknown"),
        ],
    )
    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    batches = [RendererBatchAdapter().adapt(sample) for sample in ds.samples]

    metrics = RendererTrainer.evaluate_model(TrainableLocalPatchModel(), batches)

    assert metrics["target_supervised_external_ratio"] == pytest.approx(1 / 3)
    assert metrics["target_bootstrap_self_generated_ratio"] == pytest.approx(1 / 3)
    assert metrics["target_weak_unknown_ratio"] == pytest.approx(1 / 3)


def test_old_manifest_defaults_to_weak_unknown_role(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, [_record()])

    sample = RendererDataset.from_renderer_manifest(str(path), strict=True).samples[0]
    batch = RendererBatchAdapter().adapt(sample)

    assert sample["target_training_role"] == "weak_unknown"
    assert batch.conditioning_summary["target_training_role"] == "weak_unknown"
    assert batch.conditioning_summary["target_supervision_weight"] == 0.6
