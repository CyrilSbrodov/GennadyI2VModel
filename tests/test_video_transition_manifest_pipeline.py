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


def test_video_transition_manifest_uses_canonical_phase_contract(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {"prepare", "transition", "contact_or_reveal", "stabilize"}
    for rec in payload["records"]:
        assert rec["phase_estimate"] in allowed
        assert rec["planner_context"]["phase"] in allowed
    assert set(payload["diagnostics"]["phase_coverage"]).issubset(allowed)


def test_video_transition_manifest_uses_canonical_transition_families(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {"pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition"}
    for rec in payload["records"]:
        assert rec["transition_family"] in allowed
        assert rec["runtime_semantic_transition"]["family"] in allowed
        assert "family_confidence" in rec["record_diagnostics"]
    assert set(payload["diagnostics"]["family_coverage"]).issubset(allowed)


def test_video_transition_manifest_extracts_region_aware_rois(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec = payload["records"][0]
    assert isinstance(rec["roi_records"], list)
    assert len(rec["roi_records"]) >= 2
    region_types = {r["region_type"] for r in rec["roi_records"]}
    assert "torso" in region_types
    assert "face" in region_types or "left_arm" in region_types
    assert "person_roi" not in region_types
    assert all("region_type" in rm for rm in rec["roi_manifests"])


def test_video_transition_manifest_emits_target_profile(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec = payload["records"][0]
    target_profile = rec["target_profile"]
    assert "primary_regions" in target_profile
    assert "secondary_regions" in target_profile
    assert "context_regions" in target_profile
    assert len(target_profile["primary_regions"]) >= 1
    assert rec["roi_records"][0]["target_profile"]["primary_regions"] == target_profile["primary_regions"]


def test_dynamics_video_manifest_contains_part_aware_delta_targets(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec = payload["records"][0]
    delta = rec["graph_delta_target"]
    assert "torso_motion" in delta["pose_deltas"]
    assert "arm_motion" in delta["pose_deltas"]
    assert delta["transition_diagnostics"]["part_aware_delta"] is True
    ds = DynamicsDataset.from_video_transition_manifest(str(manifest_path), strict=True)
    assert "target_profile_coverage" in ds.diagnostics
    assert "reveal_occlusion_coverage" in ds.diagnostics


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


def test_renderer_video_manifest_expands_multi_region_records(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = payload["records"][0]
    expected_regions = len(first["roi_records"])
    ds = RendererDataset.from_video_transition_manifest(str(manifest_path), strict=True)
    sample_count_for_first = sum(1 for s in ds.samples if (s.get("graph_transition_contract", {}).get("metadata", {}) or {}).get("record_id") == first["record_id"])
    assert sample_count_for_first == expected_regions
    assert expected_regions >= 2


def test_renderer_primary_video_path_is_not_person_bbox_only(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    ds = RendererDataset.from_video_transition_manifest(str(manifest_path), strict=True)
    assert "person_roi" not in ds.diagnostics["region_coverage"]
    regions = set(ds.diagnostics["region_coverage"].keys())
    assert "torso" in regions
    assert len(regions) >= 2


def test_heuristic_priors_are_auxiliary_not_primary_targets(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][0]["heuristic_priors"]["changed_prior"]["torso"] = np.zeros((8, 8, 1), dtype=np.float32).tolist()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    ds = RendererDataset.from_video_transition_manifest(str(manifest_path), strict=True)
    sample = ds[0]
    adapted = RendererBatchAdapter().adapt(sample)
    # primary supervision comes from extracted roi_after, not heuristic priors.
    assert float(np.mean(np.asarray(adapted.roi_after, dtype=np.float32))) > 0.05
    assert sample["renderer_batch_contract"]["heuristic_priors"]


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
