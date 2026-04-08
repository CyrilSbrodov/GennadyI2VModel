from __future__ import annotations

from pathlib import Path

from training.datasets import DynamicsDataset, PerceptionDataset, RendererDataset, RepresentationDataset
from training.orchestrator import train_pipeline, train_stage
from training.types import TrainingConfig


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


def test_train_pipeline_normalizes_legacy_stage_aliases(tmp_path: Path) -> None:
    config = TrainingConfig(
        epochs=1,
        checkpoint_dir=str(tmp_path),
        stage_order=["stage3_dynamics", "stage6_temporal", "stage7_instruction"],
    )

    results = train_pipeline(config)

    assert [result.stage_name for result in results] == [
        "dynamics_transition",
        "temporal_refinement",
        "text_encoder",
    ]
