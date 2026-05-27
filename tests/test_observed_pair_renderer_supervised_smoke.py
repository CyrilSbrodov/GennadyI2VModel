from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.datasets import RendererDataset
from training.renderer_observed_pairs_builder import build_renderer_manifest_from_observed_pairs
from training.renderer_trainer import RendererTrainer
from training.types import TrainingConfig
from rendering.renderer_checkpoint_loader import load_renderer_model_from_checkpoint


def test_example_observed_pairs_builds_supervised_manifest(tmp_path: Path) -> None:
    out = tmp_path / "renderer_observed.json"
    result = build_renderer_manifest_from_observed_pairs(
        observed_pairs_path="examples/observed_pairs.example.json",
        output_path=str(out),
        strict=True,
    )
    payload = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert payload["contract_version"] == "renderer_patch_manifest_v2"
    rec = payload["records"][0]
    assert rec["target_source"] == "provided_ground_truth_roi"
    assert rec["training_target_quality"] == "external_or_observed_target"
    assert rec["target_training_role"] == "supervised_external"


def test_renderer_supervised_smoke_train_and_reload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "renderer_observed.json"
    build_renderer_manifest_from_observed_pairs(
        observed_pairs_path="examples/observed_pairs.example.json",
        output_path=str(manifest_path),
        strict=True,
    )
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, learned_dataset_path=str(manifest_path), checkpoint_dir=str(tmp_path), renderer_target_role_policy="supervised_only"))
    assert Path(result.checkpoint_path).exists()
    model, backend, metadata = load_renderer_model_from_checkpoint(result.checkpoint_path)
    assert model is not None
    assert backend in {"numpy_local", "torch_local"}
    assert metadata["runtime_loadable"] is True


def test_renderer_dataset_rejects_self_generated_in_strict_supervised_policy(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"manifest_type": "renderer_patch_manifest", "records": [{"roi_before": [[[0,0,0]]], "roi_after": [[[1,1,1]]], "target_source": "self_generated_runtime_target", "training_target_quality": "self_generated_runtime_target"}]}), encoding="utf-8")
    ds = RendererDataset.from_renderer_manifest(str(path), strict=False)
    filtered = ds.filtered_by_target_role_policy("supervised_only")
    assert len(filtered) == 0


def test_trainer_raises_when_supervised_only_has_no_valid_records(tmp_path: Path) -> None:
    path = tmp_path / "all_self_generated.json"
    path.write_text(
        json.dumps(
            {
                "manifest_type": "renderer_patch_manifest",
                "records": [
                    {
                        "roi_before": [[[0, 0, 0]]],
                        "roi_after": [[[1, 1, 1]]],
                        "target_source": "runtime_output_patch",
                        "training_target_quality": "self_generated_runtime_target",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trainer = RendererTrainer()
    with pytest.raises(ValueError):
        trainer.train(
            TrainingConfig(
                epochs=1,
                learned_dataset_path=str(path),
                checkpoint_dir=str(tmp_path / "ckpt"),
                renderer_target_role_policy="supervised_only",
            )
        )


def test_single_sample_supervised_reuses_val_not_synthetic(tmp_path: Path) -> None:
    manifest_path = tmp_path / "renderer_observed.json"
    build_renderer_manifest_from_observed_pairs(
        observed_pairs_path="examples/observed_pairs.example.json",
        output_path=str(manifest_path),
        strict=True,
    )
    trainer = RendererTrainer()
    result = trainer.train(
        TrainingConfig(
            epochs=1,
            learned_dataset_path=str(manifest_path),
            checkpoint_dir=str(tmp_path / "ckpt"),
            renderer_target_role_policy="supervised_only",
        )
    )
    diag = result.train_metrics["dataset_diagnostics"]
    assert diag.get("single_sample_train_val_reuse") is True
    assert "synthetic_val_fallback" not in str(diag.get("source", ""))
    assert "reconstruction_mae" in result.val_metrics
    assert "total_loss" in result.val_metrics
    assert "supervised_record_count" in result.val_metrics
    assert "skipped_record_count" in result.val_metrics


def test_mixed_manifest_supervised_only_skips_self_generated(tmp_path: Path) -> None:
    manifest_path = tmp_path / "mixed.json"
    build = build_renderer_manifest_from_observed_pairs(
        observed_pairs_path="examples/observed_pairs.example.json",
        output_path=str(tmp_path / "obs_built.json"),
        strict=True,
    )
    payload = json.loads(Path(build.manifest_path).read_text(encoding="utf-8"))
    valid = payload["records"][0]
    payload["records"] = [
        valid,
        {
            "roi_before": [[[0, 0, 0]]],
            "roi_after": [[[1, 1, 1]]],
            "target_source": "runtime_output_patch",
            "training_target_quality": "self_generated_runtime_target",
            "region_id": "person_1:face",
            "semantic_family": "face_expression",
        },
    ]
    payload["manifest_type"] = "renderer_patch_manifest"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    trainer = RendererTrainer()
    result = trainer.train(
        TrainingConfig(
            epochs=1,
            learned_dataset_path=str(manifest_path),
            checkpoint_dir=str(tmp_path / "ckpt"),
            renderer_target_role_policy="supervised_only",
        )
    )
    meta = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))["renderer_model_metadata"]
    assert meta["training_source"] == "observed_pair_supervised"
    assert meta["supervised_record_count"] >= 1
    assert meta["target_quality_counts"]["self_generated_runtime_target"] >= 1


def test_renderer_trainer_default_policy_uses_synthetic_bootstrap_without_manifest(tmp_path: Path) -> None:
    trainer = RendererTrainer()
    result = trainer.train(
        TrainingConfig(
            epochs=1,
            train_size=2,
            val_size=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            learned_dataset_path="",
        )
    )
    assert Path(result.checkpoint_path).exists()
    meta = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))["renderer_model_metadata"]
    assert meta["training_source"] == "synthetic_bootstrap"


def test_renderer_trainer_supervised_only_without_manifest_raises(tmp_path: Path) -> None:
    trainer = RendererTrainer()
    with pytest.raises(ValueError, match="requires learned dataset manifest"):
        trainer.train(
            TrainingConfig(
                epochs=1,
                checkpoint_dir=str(tmp_path / "ckpt"),
                learned_dataset_path="",
                renderer_target_role_policy="supervised_only",
            )
        )
