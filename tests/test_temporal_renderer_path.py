from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rendering.trainable_patch_renderer import PatchBatch, TemporalLocalPatchModel
from training.datasets import RendererDataset
from training.renderer_temporal_eval import evaluate_temporal_renderer
from training.renderer_trainer import RendererBatchAdapter, RendererTrainer
from training.rollout_eval import evaluate_rollout_on_video_manifest, tiny_video_overfit_harness
from training.types import TrainingConfig


def _video_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "video_transition_manifest.json"
    records = []
    for idx in range(4):
        roi_prev = np.full((8, 8, 3), 0.2 + 0.03 * idx, dtype=np.float32)
        roi_before = np.full((8, 8, 3), 0.25 + 0.03 * idx, dtype=np.float32)
        roi_after = roi_before.copy()
        roi_after[2:7, 2:7, :] = [0.75, 0.55, 0.5]
        changed = np.clip(np.mean(np.abs(roi_after - roi_before), axis=2, keepdims=True) * 3.0, 0.0, 1.0)
        records.append(
            {
                "record_id": f"r{idx}",
                "scene_graph_before": {"frame_index": idx, "global_context": {"frame_size": [16, 16], "fps": 16, "source_type": "video"}, "persons": [], "objects": [], "relations": []},
                "scene_graph_after": {"frame_index": idx + 1, "global_context": {"frame_size": [16, 16], "fps": 16, "source_type": "video"}, "persons": [], "objects": [], "relations": []},
                "transition_family": "garment_transition",
                "runtime_semantic_transition": "garment_transition",
                "phase_estimate": "transition",
                "planner_context": {"step_index": idx + 1, "total_steps": 4, "phase": "transition", "target_duration": 1.0},
                "target_profile": {"primary_regions": ["torso"], "secondary_regions": ["face"], "context_regions": ["legs"]},
                "reveal_score": 0.65,
                "occlusion_score": 0.25,
                "graph_delta_target": {"pose_deltas": {"torso_pitch": -0.1}, "interaction_deltas": {"support_contact": 0.4}, "affected_regions": ["torso"], "semantic_reasons": ["open"]},
                "roi_records": [
                    {
                        "region_type": "torso",
                        "roi_prev": roi_prev.tolist(),
                        "roi_before": roi_before.tolist(),
                        "roi_after": roi_after.tolist(),
                        "changed_mask": changed.tolist(),
                        "preservation_mask": np.clip(1.0 - changed, 0.0, 1.0).tolist(),
                        "target_profile": {"primary_regions": ["torso"], "secondary_regions": [], "context_regions": []},
                        "priors": {"heuristic": True},
                    }
                ],
            }
        )
    manifest.write_text(json.dumps({"manifest_type": "video_transition_manifest", "records": records}), encoding="utf-8")
    return manifest


def _batch() -> PatchBatch:
    before = np.full((8, 8, 3), 0.25, dtype=np.float32)
    after = before.copy()
    after[1:7, 1:7, :] = [0.75, 0.45, 0.4]
    changed = np.clip(np.mean(np.abs(after - before), axis=2, keepdims=True) * 3.0, 0.0, 1.0)
    return PatchBatch(
        roi_before=before,
        roi_after=after,
        changed_mask=changed,
        alpha_target=np.clip(0.1 + 0.9 * changed, 0.0, 1.0),
        blend_hint=np.clip(0.2 + 0.8 * changed, 0.0, 1.0),
        semantic_embed=np.array([0.0, 1.0, 0.0, 0.2, 0.8, 0.3], dtype=np.float32),
        delta_cond=np.array([0.2] * 9, dtype=np.float32),
        planner_cond=np.array([0.25] * 8, dtype=np.float32),
        graph_cond=np.array([0.1] * 7, dtype=np.float32),
        memory_cond=np.array([0.2] * 8, dtype=np.float32),
        appearance_cond=np.array([0.25, 0.25, 0.25, 0.01, 0.01, 0.01], dtype=np.float32),
        bbox_cond=np.array([0.2, 0.2, 0.4, 0.4], dtype=np.float32),
        previous_roi=np.full((8, 8, 3), 0.22, dtype=np.float32),
        predicted_family="garment_transition",
        predicted_phase="transition",
        target_profile={"primary_regions": ["torso"], "secondary_regions": [], "context_regions": []},
        reveal_score=0.6,
        occlusion_score=0.2,
        support_contact_score=0.3,
        rollout_weight=1.2,
    )


def test_temporal_renderer_dataset_from_video_manifest_smoke(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    ds = RendererDataset.from_video_transition_manifest(str(manifest), strict=True)
    sample = ds[0]
    assert "temporal_roi_window" in sample
    assert sample["temporal_roi_window"]["roi_t_minus_1"]


def test_temporal_renderer_model_forward_shapes() -> None:
    model = TemporalLocalPatchModel()
    batch = _batch()
    out = model.forward(batch)
    assert np.asarray(out["rgb"]).shape == (8, 8, 3)
    assert np.asarray(out["alpha"]).shape == (8, 8)
    assert np.asarray(out["uncertainty"]).shape == (8, 8)


def test_temporal_renderer_train_step_is_finite() -> None:
    model = TemporalLocalPatchModel()
    losses = model.train_step(_batch(), lr=1e-3)
    assert np.isfinite(losses["total_loss"])
    assert losses["temporal_consistency_loss"] >= 0.0


def test_temporal_renderer_uses_temporal_contract_conditioning() -> None:
    adapter = RendererBatchAdapter()
    sample = {
        "frames": [_batch().roi_before.tolist(), _batch().roi_after.tolist()],
        "roi_pairs": [(_batch().roi_before.tolist(), _batch().roi_after.tolist())],
        "renderer_batch_contract": {},
        "temporal_transition_target": {"family": "garment_transition", "phase": "transition", "target_profile": {"primary_regions": ["torso"], "secondary_regions": [], "context_regions": []}, "reveal_score": 0.7, "occlusion_score": 0.2, "support_contact_score": 0.4},
        "temporal_roi_window": {"roi_t_minus_1": np.full((8, 8, 3), 0.2, dtype=np.float32).tolist()},
    }
    batch = adapter.adapt(sample, temporal_mode=True)
    assert batch.predicted_family == "garment_transition"
    assert batch.previous_roi is not None
    assert batch.reveal_score > 0.0


def test_temporal_renderer_temporal_window_path_is_primary_on_video_manifest(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, learned_dataset_path=str(manifest), checkpoint_dir=str(tmp_path / "ckpt"), renderer_backend="temporal_local_renderer"))
    assert trainer.dataset_source == "manifest_video_temporal_renderer_primary"
    assert result.val_metrics["temporal_window_usage_ratio"] > 0.0


def test_temporal_renderer_eval_metrics_smoke() -> None:
    model = TemporalLocalPatchModel()
    metrics = evaluate_temporal_renderer(model, [_batch()])
    assert "renderer_summary_score" in metrics
    assert 0.0 <= metrics["temporal_consistency_proxy"] <= 1.0


def test_rollout_eval_can_use_temporal_renderer_backend(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = evaluate_rollout_on_video_manifest(dataset_manifest=str(manifest), mode="teacher_forced_rollout", rollout_steps=1, renderer_backend="temporal_local_renderer")
    assert out["renderer_backend"] == "temporal_local_renderer"
    assert out["rollout_frame_reconstruction_proxy"] >= 0.0


def test_tiny_video_overfit_harness_improves_temporal_renderer_proxy(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = tiny_video_overfit_harness(str(manifest), tiny_subset_records=3, epochs=4, rollout_steps=1, renderer_backend="temporal_local_renderer")
    assert out["after"]["rollout_frame_reconstruction_proxy"] >= out["before"]["rollout_frame_reconstruction_proxy"]
