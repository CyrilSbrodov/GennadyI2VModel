from __future__ import annotations

from pathlib import Path

import pytest

from core.pipeline_contract import ContractValidationError
from training.datasets import DynamicsDataset, PerceptionDataset, RendererDataset, RepresentationDataset
from training.orchestrator import canonical_training_stage_name, evaluate_stage, train_pipeline, train_stage
from training.types import StageResult, TrainingConfig


def test_dataset_interfaces_expose_required_fields() -> None:
    perception = PerceptionDataset.synthetic(1)[0]
    representation = RepresentationDataset.synthetic(1)[0]
    dynamics = DynamicsDataset.synthetic(1)[0]
    renderer = RendererDataset.synthetic(1)[0]

    assert "frames" in perception
    assert "frames" in representation and "graphs" in representation
    assert "graphs" in dynamics and "actions" in dynamics and "deltas" in dynamics
    assert "roi_pairs" in renderer


def test_train_stage_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=2, checkpoint_dir=str(tmp_path))
    result = train_stage("perception", config)

    assert result.stage_name == "perception"
    assert "loss" in result.train_metrics
    assert "score" in result.val_metrics
    assert Path(result.checkpoint_path).exists()


def test_train_pipeline_runs_all_requested_stages(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path), stage_order=["perception", "dynamics"])

    results = train_pipeline(config)

    assert [result.stage_name for result in results] == ["perception", "dynamics"]
    assert all(Path(result.checkpoint_path).exists() for result in results)


def test_train_pipeline_normalizes_clear_stage_aliases(tmp_path: Path) -> None:
    config = TrainingConfig(
        epochs=1,
        checkpoint_dir=str(tmp_path),
        stage_order=["dynamics_transition", "text_encoder"],
    )

    results = train_pipeline(config)

    assert [result.stage_name for result in results] == ["dynamics", "intent"]


def test_memory_stage_is_first_class_but_training_not_fake(tmp_path: Path) -> None:
    assert canonical_training_stage_name("stage5_memory") == "memory"
    assert canonical_training_stage_name("memory") == "memory"

    with pytest.raises(NotImplementedError, match="Memory is a canonical first-class stage"):
        train_stage("memory", TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path)))

    assert not (tmp_path / "memory" / "memory.ckpt").exists()
    assert not (tmp_path / "memory.ckpt").exists()


def test_renderer_and_patch_synthesis_share_canonical_rendering_stage(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path), batch_size=1)

    renderer = train_stage("renderer", config)
    patch_synthesis = train_stage("patch_synthesis", config)

    assert renderer.stage_name == "rendering"
    assert patch_synthesis.stage_name == "rendering"
    assert canonical_training_stage_name("renderer") == canonical_training_stage_name("patch_synthesis") == "rendering"


def test_evaluation_dispatch_uses_canonical_training_stage_names() -> None:
    metrics = evaluate_stage(
        StageResult(
            stage_name="rendering",
            train_metrics={"progress": 1.0, "loss": 0.1},
            val_metrics={"score": 0.9, "contract_payload": {"patch_quality": 0.9, "identity_consistency": 0.8}},
            checkpoint_path="none",
        )
    )

    assert "patch_quality" in metrics
    assert canonical_training_stage_name("text_encoder") == "intent"
    assert canonical_training_stage_name("dynamics_transition") == "dynamics"


def test_unknown_and_ambiguous_training_stage_names_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(ContractValidationError):
        train_stage("representation_memory_shortcut", TrainingConfig(checkpoint_dir=str(tmp_path)))
    with pytest.raises(ContractValidationError):
        canonical_training_stage_name("stage8_joint_tuning")
