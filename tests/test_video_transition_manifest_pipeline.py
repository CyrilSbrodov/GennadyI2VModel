from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.datasets import DynamicsDataset, RendererDataset
from training.dynamics_trainer import DynamicsDatasetAdapter, DynamicsTrainer
from training.renderer_trainer import RendererBatchAdapter, RendererTrainer
from training.types import TrainingConfig
from training.video_transition_manifest import VideoTransitionBuilderConfig, VideoTransitionManifestBuilder, save_video_transition_manifest


def _mini_video_frames() -> list[list[list[list[float]]]]:
    frames: list[list[list[list[float]]]] = []
    base = np.full((32, 32, 3), 0.2, dtype=np.float32)
    for i in range(4):
        f = base.copy()
        x0 = 5 + i * 3
        f[10:22, x0 : x0 + 8, :] = np.array([0.75, 0.35, 0.3], dtype=np.float32)
        frames.append(f.tolist())
    return frames


def _build_manifest(tmp_path: Path) -> Path:
    builder = VideoTransitionManifestBuilder(VideoTransitionBuilderConfig(fps=6, duration=1.0, quality_profile="debug"))
    payload = builder.build_from_frames(_mini_video_frames(), source_id="mini_video_fixture")
    path = tmp_path / "video_transition_manifest.json"
    save_video_transition_manifest(payload, str(path))
    return path


def test_video_transition_manifest_builder_smoke(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["manifest_type"] == "video_transition_manifest"
    assert len(payload["records"]) >= 2


def test_video_transition_manifest_contains_paired_roi_and_delta_targets(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec = payload["records"][0]
    assert rec["roi_before"]
    assert rec["roi_after"]
    assert rec["changed_mask"]
    assert rec["preservation_mask"]
    assert rec["graph_delta_target"]["pose_deltas"]
    assert rec["planner_context"]["phase"]


def test_dynamics_trainer_uses_video_transition_manifest_as_primary(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    trainer = DynamicsTrainer()
    result = trainer.train(TrainingConfig(epochs=1, learned_dataset_path=str(manifest_path), checkpoint_dir=str(tmp_path / "ckpt_dyn")))
    assert trainer.dataset_source == "manifest_video_dynamics_primary"
    assert result.val_metrics["usable_sample_count"] >= 1.0
    assert result.val_metrics["family_coverage_count"] >= 1.0


def test_renderer_trainer_uses_video_transition_manifest_as_primary(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, learned_dataset_path=str(manifest_path), checkpoint_dir=str(tmp_path / "ckpt_rnd")))
    assert trainer.dataset_source == "manifest_video_renderer_primary"
    assert result.val_metrics["usable_sample_count"] >= 1.0


def test_heuristic_targets_are_not_primary_when_manifest_targets_exist(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][0]["heuristic_priors"]["bootstrap_after_prior"] = np.zeros((8, 8, 3), dtype=np.float32).tolist()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    ds = RendererDataset.from_video_transition_manifest(str(manifest_path), strict=True)
    sample = ds[0]
    adapted = RendererBatchAdapter().adapt(sample)
    # primary supervision comes from extracted roi_after, not heuristic priors.
    assert float(np.mean(np.asarray(adapted.roi_after, dtype=np.float32))) > 0.05


def test_runtime_contract_compatible_with_video_transition_manifest_records(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    dyn_ds = DynamicsDataset.from_video_transition_manifest(str(manifest_path), strict=True)
    rnd_ds = RendererDataset.from_video_transition_manifest(str(manifest_path), strict=True)

    dyn_batch = DynamicsDatasetAdapter.sample_to_batch(dyn_ds[0], step_index=1)
    rnd_batch = RendererBatchAdapter().adapt(rnd_ds[0])

    assert len(dyn_batch.inputs.graph_features) > 0
    assert len(dyn_batch.targets.pose) > 0
    assert rnd_batch.roi_before.shape[:2] == rnd_batch.roi_after.shape[:2]
    assert rnd_batch.changed_mask.shape[:2] == rnd_batch.roi_before.shape[:2]
