from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.datasets import RendererDataset
from training.renderer_trainer import RendererTrainer
from training.supervised_renderer_manifest import (
    build_supervised_renderer_record,
    write_supervised_renderer_manifest,
)
from training.types import TrainingConfig


def _roi(value: float) -> list:
    return np.full((4, 4, 3), value, dtype=np.float32).tolist()


def _supervised_record(region_id: str = "p1:face") -> dict[str, object]:
    return build_supervised_renderer_record(
        roi_before=_roi(0.1),
        roi_after=_roi(0.6),
        region_id=region_id,
        semantic_family="face_expression",
        bbox=[0.1, 0.2, 0.3, 0.4],
    )


def _bootstrap_record() -> dict[str, object]:
    return {
        "roi_before": _roi(0.2),
        "roi_after": _roi(0.4),
        "region_id": "p1:left_arm",
        "semantic_family": "sleeve_arm_transition",
        "target_source": "runtime_output_patch",
        "training_target_quality": "self_generated_runtime_target",
        "source": "runtime_renderer_patch",
    }


def _write_supervised(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    return Path(write_supervised_renderer_manifest(records, str(tmp_path / "renderer_supervised_manifest.json")))


def test_build_supervised_renderer_record_has_external_target_provenance() -> None:
    record = _supervised_record()

    assert record["target_source"] == "provided_ground_truth_roi"
    assert record["training_target_quality"] == "external_or_observed_target"
    assert record["target_training_role"] == "supervised_external"
    assert record["training_target_origin"] == "external_observed_pair"
    assert record["semantic_family"] == "face_expression"
    assert "selected_render_strategy" in record
    assert "selected_strategy" not in record
    assert "selected_execution_strategy" not in record


def test_build_supervised_renderer_record_rejects_unknown_semantic_family() -> None:
    with pytest.raises(ValueError, match="unsupported supervised renderer semantic_family"):
        build_supervised_renderer_record(
            roi_before=_roi(0.1),
            roi_after=_roi(0.6),
            region_id="p1:pose",
            semantic_family="pose_transition",
        )


def test_write_supervised_renderer_manifest_metadata(tmp_path: Path) -> None:
    path = _write_supervised(tmp_path, [_supervised_record("p1:face"), _supervised_record("p2:face")])

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["manifest_type"] == "renderer_patch_manifest"
    assert payload["contract_version"] == "renderer_patch_manifest.v1"
    assert payload["contains_external_targets"] is True
    assert payload["contains_self_generated_targets"] is False
    assert payload["target_quality_counts"]["external_or_observed_target"] == 2
    assert payload["record_count"] == 2


def test_write_supervised_renderer_manifest_empty_metadata_is_honest(tmp_path: Path) -> None:
    path = _write_supervised(tmp_path, [])

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["record_count"] == 0
    assert payload["contains_external_targets"] is False
    assert payload["contains_self_generated_targets"] is False
    assert payload["target_quality_counts"]["external_or_observed_target"] == 0


def test_write_supervised_renderer_manifest_rejects_non_supervised_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a supervised renderer record"):
        _write_supervised(tmp_path, [_bootstrap_record()])


def test_renderer_dataset_loads_supervised_manifest_strict(tmp_path: Path) -> None:
    records = [_supervised_record("p1:face"), _supervised_record("p2:face")]
    path = _write_supervised(tmp_path, records)

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    sample = ds.samples[0]
    diagnostics = ds.diagnostics

    assert len(ds) == len(records)
    assert sample["target_source"] == "provided_ground_truth_roi"
    assert sample["training_target_quality"] == "external_or_observed_target"
    assert sample["target_training_role"] == "supervised_external"
    assert sample.get("target_quality_warning") is None
    assert diagnostics["target_quality_counts"]["external_or_observed_target"] == len(records)
    assert diagnostics["target_source_counts"]["provided_ground_truth_roi"] == len(records)
    assert diagnostics["target_training_role_counts"]["supervised_external"] == len(records)
    assert diagnostics["supervised_external_ratio"] == 1.0
    assert diagnostics["contains_external_targets"] is True
    assert diagnostics["contains_no_supervised_external_targets"] is False
    assert diagnostics["contains_only_bootstrap_targets"] is False
    assert diagnostics["warnings"] == []


def test_supervised_only_policy_keeps_external_records(tmp_path: Path) -> None:
    records = [_supervised_record("p1:face"), _supervised_record("p2:face")]
    ds = RendererDataset.from_renderer_manifest(str(_write_supervised(tmp_path, records)), strict=True)

    filtered = ds.filtered_by_target_role_policy("supervised_only")

    assert len(filtered) == len(records)
    assert filtered.diagnostics["filtered_out_sample_count"] == 0
    assert filtered.diagnostics["retained_by_role"]["supervised_external"] == len(records)
    assert not any(w.get("type") == "bootstrap_self_generated_not_ground_truth" for w in filtered.diagnostics["warnings"])


def test_trainer_supervised_only_checkpoint_metadata(tmp_path: Path) -> None:
    manifest_path = _write_supervised(tmp_path, [_supervised_record("p1:face"), _supervised_record("p2:face")])
    config = TrainingConfig(
        learned_dataset_path=str(manifest_path),
        renderer_target_role_policy="supervised_only",
        renderer_backend="numpy_local",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        epochs=1,
    )

    result = RendererTrainer().train(config)
    metadata = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))["renderer_model_metadata"]

    assert metadata["supervised_external_ratio"] == 1.0
    assert metadata["bootstrap_self_generated_ratio"] == 0.0
    assert metadata["weak_unknown_target_ratio"] == 0.0
    assert metadata["contains_no_supervised_external_targets"] is False
    assert metadata["contains_only_bootstrap_targets"] is False
    assert metadata["target_training_role_counts"]["supervised_external"] > 0
    assert metadata["renderer_target_role_policy"] == "supervised_only"
    assert metadata["effective_target_role_policy"] == "supervised_only"


def test_mixed_manifest_supervised_only_filters_bootstrap(tmp_path: Path) -> None:
    path = tmp_path / "mixed_renderer_manifest.json"
    path.write_text(
        json.dumps(
            {
                "manifest_type": "renderer_patch_manifest",
                "contract_version": "renderer_patch_manifest.v1",
                "records": [_supervised_record("p1:face"), _bootstrap_record()],
            }
        ),
        encoding="utf-8",
    )
    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)

    filtered = ds.filtered_by_target_role_policy("supervised_only")

    assert len(filtered) == 1
    assert filtered.samples[0]["target_training_role"] == "supervised_external"
    assert filtered.diagnostics["filtered_out_by_role"]["bootstrap_self_generated"] == 1
    assert filtered.diagnostics["retained_by_role"]["supervised_external"] == 1


def test_supervised_manifest_does_not_emit_legacy_strategy_fields(tmp_path: Path) -> None:
    path = _write_supervised(tmp_path, [_supervised_record()])
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload["records"][0]
    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    contract = ds.samples[0]["renderer_batch_contract"]

    assert "selected_render_strategy" in record
    assert "selected_strategy" not in record
    assert "selected_execution_strategy" not in record
    assert "selected_strategy" not in contract
    assert "selected_execution_strategy" not in contract
