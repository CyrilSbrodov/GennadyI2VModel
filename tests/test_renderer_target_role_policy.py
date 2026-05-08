from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rendering.target_provenance_policy import target_role_allowed_by_policy
from training.datasets import RendererDataset
from training.renderer_trainer import RendererTrainer
from training.types import TrainingConfig


def _record(*, target_source: str = "unknown", training_target_quality: str = "unknown") -> dict[str, object]:
    before = np.zeros((2, 2, 3), dtype=np.float32)
    after = np.full((2, 2, 3), 0.5, dtype=np.float32)
    return {
        "roi_before": before.tolist(),
        "roi_after": after.tolist(),
        "semantic_family": "sleeve_arm_transition",
        "region_id": "person_0:left_arm",
        "target_source": target_source,
        "training_target_quality": training_target_quality,
    }


def _mixed_records() -> list[dict[str, object]]:
    return [
        _record(target_source="provided_ground_truth_roi", training_target_quality="external_or_observed_target"),
        _record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target"),
        _record(target_source="unknown", training_target_quality="unknown"),
    ]


def _write_manifest(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "renderer_patch_manifest.json"
    path.write_text(json.dumps({"manifest_type": "renderer_patch_manifest", "records": records}), encoding="utf-8")
    return path


def _mixed_dataset(tmp_path: Path) -> RendererDataset:
    return RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, _mixed_records())), strict=True)


def test_target_role_allowed_by_policy() -> None:
    roles = ["supervised_external", "bootstrap_self_generated", "weak_unknown"]

    assert [target_role_allowed_by_policy(role, "supervised_only") for role in roles] == [True, False, False]
    assert [target_role_allowed_by_policy(role, "supervised_plus_bootstrap") for role in roles] == [True, True, False]
    assert [target_role_allowed_by_policy(role, "bootstrap_only") for role in roles] == [False, True, False]
    assert [target_role_allowed_by_policy(role, "all_non_unknown") for role in roles] == [True, True, False]
    assert [target_role_allowed_by_policy(role, "all") for role in roles] == [True, True, True]
    assert target_role_allowed_by_policy("unexpected", "all") is True
    assert target_role_allowed_by_policy("unexpected", "all_non_unknown") is False

    with pytest.raises(ValueError):
        target_role_allowed_by_policy("supervised_external", "invalid")


def test_renderer_dataset_filtered_by_supervised_only(tmp_path: Path) -> None:
    ds = _mixed_dataset(tmp_path)

    filtered = ds.filtered_by_target_role_policy("supervised_only")

    assert len(filtered) == 1
    assert len(ds) == 3
    assert filtered.diagnostics["pre_filter_sample_count"] == 3
    assert filtered.diagnostics["post_filter_sample_count"] == 1
    assert filtered.diagnostics["filtered_out_sample_count"] == 2
    assert filtered.diagnostics["filtered_out_by_role"] == {
        "supervised_external": 0,
        "bootstrap_self_generated": 1,
        "weak_unknown": 1,
    }


def test_renderer_dataset_filtered_by_bootstrap_only(tmp_path: Path) -> None:
    ds = _mixed_dataset(tmp_path)

    filtered = ds.filtered_by_target_role_policy("bootstrap_only")

    assert len(filtered) == 1
    assert filtered.diagnostics["retained_by_role"]["bootstrap_self_generated"] == 1
    assert filtered.samples[0]["target_training_role"] == "bootstrap_self_generated"


def test_renderer_dataset_filter_all_non_unknown_excludes_weak_unknown(tmp_path: Path) -> None:
    ds = _mixed_dataset(tmp_path)

    filtered = ds.filtered_by_target_role_policy("all_non_unknown")

    assert len(filtered) == 2
    assert filtered.diagnostics["filtered_out_by_role"]["weak_unknown"] == 1
    assert {sample["target_training_role"] for sample in filtered.samples} == {
        "supervised_external",
        "bootstrap_self_generated",
    }


def test_renderer_trainer_applies_target_role_policy(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, _mixed_records())
    config = TrainingConfig(
        learned_dataset_path=str(manifest_path),
        renderer_target_role_policy="supervised_only",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        epochs=1,
    )
    trainer = RendererTrainer()

    train_ds, val_ds = trainer.build_datasets(config)

    assert len(train_ds) == 1
    assert train_ds.samples[0]["target_training_role"] == "supervised_external"
    assert trainer.dataset_diagnostics["renderer_target_role_policy"] == "supervised_only"
    assert trainer.dataset_diagnostics["train_pre_filter_count"] == 2
    assert trainer.dataset_diagnostics["train_post_filter_count"] == 1
    assert trainer.dataset_diagnostics["val_pre_filter_count"] == 1
    assert trainer.dataset_diagnostics["val_post_filter_count"] == 0
    assert trainer.dataset_diagnostics["train_filtered_out_by_role"]["bootstrap_self_generated"] == 1
    assert val_ds.diagnostics["validation_filtered"] is True


def test_renderer_trainer_supervised_only_empty_dataset_fails_loudly(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [_record(target_source="runtime_output_patch", training_target_quality="self_generated_runtime_target")],
    )
    config = TrainingConfig(
        learned_dataset_path=str(manifest_path),
        renderer_target_role_policy="supervised_only",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        epochs=1,
    )
    trainer = RendererTrainer()

    with pytest.raises(ValueError, match="renderer_target_role_policy='supervised_only' produced empty training dataset"):
        trainer.build_datasets(config)


def test_bootstrap_policy_does_not_call_targets_ground_truth(tmp_path: Path) -> None:
    ds = _mixed_dataset(tmp_path)

    filtered = ds.filtered_by_target_role_policy("bootstrap_only")

    warnings_text = json.dumps(filtered.diagnostics.get("warnings", []))
    assert "not ground truth" in warnings_text or "bootstrap" in warnings_text
    assert len(filtered.samples) == 1
    bootstrap_sample = filtered.samples[0]
    assert bootstrap_sample["target_training_role"] == "bootstrap_self_generated"
    assert all("ground_truth" not in key for key in bootstrap_sample.keys())
    assert all("ground_truth" not in key for key in bootstrap_sample["renderer_batch_contract"].keys())


def _legacy_record() -> dict[str, object]:
    rec = _record()
    rec.pop("target_source", None)
    rec.pop("training_target_quality", None)
    return rec


def test_default_policy_keeps_legacy_renderer_manifest_via_compatibility_fallback(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_legacy_record(), _legacy_record()])
    config = TrainingConfig(
        learned_dataset_path=str(manifest_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        epochs=1,
    )
    trainer = RendererTrainer()

    train_ds, _val_ds = trainer.build_datasets(config)

    assert len(train_ds) == 1
    assert trainer.dataset_diagnostics["target_role_policy_compatibility_fallback"] is True
    assert trainer.dataset_diagnostics["requested_target_role_policy"] == "supervised_plus_bootstrap"
    assert trainer.dataset_diagnostics["effective_target_role_policy"] == "all"
    assert trainer.dataset_diagnostics["compatibility_fallback_reason"] == "legacy_missing_target_provenance"
    assert trainer.dataset_diagnostics["missing_target_provenance_count"] == 2
    assert trainer.dataset_diagnostics["contains_legacy_unknown_target_provenance"] is True


def test_supervised_only_does_not_fallback_for_legacy_unknown(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_legacy_record()])
    config = TrainingConfig(
        learned_dataset_path=str(manifest_path),
        renderer_target_role_policy="supervised_only",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        epochs=1,
    )
    trainer = RendererTrainer()

    with pytest.raises(ValueError, match="renderer_target_role_policy='supervised_only' produced empty training dataset"):
        trainer.build_datasets(config)


def test_video_transition_manifest_samples_get_explicit_weak_unknown_target_role(tmp_path: Path) -> None:
    before = np.zeros((2, 2, 3), dtype=np.float32).tolist()
    after = np.full((2, 2, 3), 0.25, dtype=np.float32).tolist()
    path = tmp_path / "video_transition_manifest.json"
    path.write_text(
        json.dumps(
            {
                "manifest_type": "video_transition_manifest",
                "records": [
                    {
                        "transition_family": "pose_transition",
                        "phase_estimate": "transition",
                        "roi_before": before,
                        "roi_after": after,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ds = RendererDataset.from_video_transition_manifest(str(path), strict=True)

    assert len(ds) == 1
    assert ds.samples[0]["target_source"] == "unknown"
    assert ds.samples[0]["training_target_quality"] == "unknown"
    assert ds.samples[0]["target_training_role"] == "weak_unknown"
    assert ds.samples[0]["target_quality_warning"] == "unknown_training_target_quality"
    assert ds.samples[0]["renderer_batch_contract"]["target_training_role"] == "weak_unknown"
    assert ds.diagnostics["contains_legacy_unknown_target_provenance"] is True
    assert ds.diagnostics["missing_target_provenance_count"] == 1
