from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rendering.torch_local_patch_generator import TorchLocalPatchGenerator
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


def test_renderer_dataset_preserves_target_provenance_fields(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [_record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target")],
    )

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)

    assert ds.samples[0]["target_source"] == "runtime_output_patch"
    assert ds.samples[0]["training_target_quality"] == "self_generated_runtime_target"


def test_renderer_batch_adapter_adds_target_policy_summary(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [_record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target")],
    )
    sample = RendererDataset.from_renderer_manifest(str(path), strict=True).samples[0]

    batch = RendererBatchAdapter().adapt(sample)

    assert batch.conditioning_summary["target_source"] == "runtime_output_patch"
    assert batch.conditioning_summary["training_target_quality"] == "self_generated_runtime_target"
    assert batch.conditioning_summary["target_is_self_generated"] is True
    assert batch.conditioning_summary["target_is_external_or_observed"] is False
    assert batch.conditioning_summary["target_supervision_weight"] == 0.35


def test_conditioning_summary_cannot_override_target_provenance_policy(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [_record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target")],
    )
    sample = RendererDataset.from_renderer_manifest(str(path), strict=True).samples[0]
    sample["renderer_batch_contract"]["conditioning_summary"] = {
        "target_supervision_weight": 1.0,
        "training_target_quality": "external_or_observed_target",
        "target_is_self_generated": False,
        "target_is_external_or_observed": True,
    }

    batch = RendererBatchAdapter().adapt(sample)

    assert batch.conditioning_summary["training_target_quality"] == "self_generated_runtime_target"
    assert batch.conditioning_summary["target_supervision_weight"] == 0.35
    assert batch.conditioning_summary["target_is_self_generated"] is True
    assert batch.conditioning_summary["target_is_external_or_observed"] is False


def test_external_target_gets_full_supervision_weight(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [_record(target_source="provided_ground_truth_roi", training_target_quality="external_or_observed_target")],
    )
    sample = RendererDataset.from_renderer_manifest(str(path), strict=True).samples[0]

    batch = RendererBatchAdapter().adapt(sample)

    assert batch.conditioning_summary["target_supervision_weight"] == 1.0
    assert batch.conditioning_summary["target_is_external_or_observed"] is True
    assert batch.conditioning_summary["target_is_self_generated"] is False


def test_old_manifest_unknown_target_quality_defaults_safe(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, [_record()])
    sample = RendererDataset.from_renderer_manifest(str(path), strict=True).samples[0]

    batch = RendererBatchAdapter().adapt(sample)

    assert batch.conditioning_summary["target_source"] == "unknown"
    assert batch.conditioning_summary["training_target_quality"] == "unknown"
    assert batch.conditioning_summary["target_supervision_weight"] == 0.6
    assert batch.conditioning_summary["target_is_self_generated"] is False
    assert batch.conditioning_summary["target_is_external_or_observed"] is False


def test_evaluate_model_reports_target_quality_ratios(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            _record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target"),
            _record(target_source="provided_ground_truth_roi", training_target_quality="external_or_observed_target"),
        ],
    )
    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    batches = [RendererBatchAdapter().adapt(sample) for sample in ds.samples]

    metrics = RendererTrainer.evaluate_model(TrainableLocalPatchModel(), batches)

    assert metrics["target_self_generated_ratio"] == 0.5
    assert metrics["target_external_or_observed_ratio"] == 0.5
    assert metrics["avg_target_supervision_weight"] == pytest.approx(0.675)


def test_self_generated_only_manifest_records_warning(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [_record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target")],
    )

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)

    assert ds.diagnostics["contains_only_self_generated_targets"] is True
    assert ds.diagnostics["warning"] == "renderer manifest contains only self-generated runtime targets; use for bootstrap/eval cautiously"
    assert any(w.get("type") == "self_generated_only_targets" for w in ds.diagnostics["warnings"])


def test_torch_train_step_reports_weighted_total_loss_if_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    path = _write_manifest(
        tmp_path,
        [_record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target")],
    )
    sample = RendererDataset.from_renderer_manifest(str(path), strict=True).samples[0]
    batch = RendererBatchAdapter().adapt(sample)

    metrics = TorchLocalPatchGenerator().train_step(batch)

    assert "weighted_total_loss" in metrics
    assert metrics["target_supervision_weight"] == 0.35
    assert metrics["weighted_total_loss"] <= metrics["total_loss"]
