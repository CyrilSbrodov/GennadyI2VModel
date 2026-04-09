from __future__ import annotations

import json
from pathlib import Path

from training.dynamics_trainer import DynamicsDatasetAdapter, DynamicsTrainer
from training.renderer_trainer import RendererBatchAdapter, RendererTrainer
from training.types import TrainingConfig


def _graph(frame_index: int) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "global_context": {"frame_size": [16, 16], "fps": 16, "source_type": "video"},
        "persons": [{"person_id": "p1", "track_id": "t1", "bbox": {"x": 0.2, "y": 0.1, "w": 0.5, "h": 0.8}}],
        "objects": [],
        "relations": [],
    }


def _video_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "video_transition_manifest.json"
    recs = []
    for idx in range(3):
        recs.append(
            {
                "record_id": f"r{idx}",
                "scene_graph_before": _graph(idx),
                "scene_graph_after": _graph(idx + 1),
                "transition_family": "garment_transition" if idx % 2 == 0 else "pose_transition",
                "runtime_semantic_transition": "garment_transition" if idx % 2 == 0 else "pose_transition",
                "phase_estimate": "transition",
                "planner_context": {"step_index": idx + 1, "total_steps": 4, "phase": "transition", "target_duration": 1.2},
                "target_profile": {"primary_regions": ["torso"], "secondary_regions": ["face"], "context_regions": ["legs"]},
                "graph_delta_target": {
                    "pose_deltas": {"torso_pitch": -0.1},
                    "garment_deltas": {"coverage_delta": -0.2},
                    "visibility_deltas": {"revealed_regions_score": 0.6, "torso": "partially_visible"},
                    "interaction_deltas": {"support_contact": 0.4},
                    "affected_entities": ["p1"],
                    "affected_regions": ["torso", "face"],
                    "semantic_reasons": ["open_garment"],
                    "region_transition_mode": {"torso": "revealing"},
                },
                "roi_records": [
                    {
                        "region_type": "torso",
                        "roi_before": [[[0.2, 0.2, 0.2] for _ in range(8)] for _ in range(8)],
                        "roi_after": [[[0.6, 0.5, 0.5] for _ in range(8)] for _ in range(8)],
                        "changed_mask": [[[0.7] for _ in range(8)] for _ in range(8)],
                        "preservation_mask": [[[0.3] for _ in range(8)] for _ in range(8)],
                        "target_profile": {"primary_regions": ["torso"], "secondary_regions": [], "context_regions": []},
                        "priors": {"heuristic": True},
                    }
                ],
            }
        )
    manifest.write_text(json.dumps({"manifest_type": "video_transition_manifest", "records": recs}), encoding="utf-8")
    return manifest


def test_dynamics_trainer_uses_learned_temporal_contract_conditioning(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    trainer = DynamicsTrainer()
    result = trainer.train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), learned_dataset_path=str(manifest), contract_conditioning_mode="learned_contract_only"))
    assert result.val_metrics["learned_contract_usage_ratio"] > 0.0


def test_renderer_trainer_uses_learned_temporal_contract_conditioning(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), learned_dataset_path=str(manifest), contract_conditioning_mode="learned_contract_only"))
    assert result.val_metrics["learned_contract_usage_ratio"] > 0.0


def test_mixed_contract_bootstrap_mode_prefers_learned_contract_when_available(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    trainer = DynamicsTrainer()
    result = trainer.train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), learned_dataset_path=str(manifest), contract_conditioning_mode="mixed_contract_bootstrap"))
    assert result.val_metrics["learned_contract_usage_ratio"] >= result.val_metrics["weak_contract_usage_ratio"]


def test_weak_contract_only_mode_remains_supported(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    trainer = DynamicsTrainer()
    result = trainer.train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), learned_dataset_path=str(manifest), contract_conditioning_mode="weak_contract_only"))
    assert result.val_metrics["learned_contract_usage_ratio"] == 0.0
    assert result.val_metrics["weak_contract_usage_ratio"] > 0.0


def test_temporal_conditioned_dynamics_batch_smoke(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    trainer = DynamicsTrainer()
    train_ds, _ = trainer.build_datasets(TrainingConfig(learned_dataset_path=str(manifest)))
    trainer.contract_conditioning_mode = "mixed_contract_bootstrap"
    batch = DynamicsDatasetAdapter.sample_to_batch(train_ds.samples[0], encoder=trainer.temporal_encoder, conditioning_mode=trainer.contract_conditioning_mode, step_index=1)
    assert batch.temporal_contract_conditioning.predicted_phase in {"prepare", "transition", "contact_or_reveal", "stabilize"}


def test_temporal_conditioned_renderer_batch_smoke(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    trainer = RendererTrainer()
    train_ds, _ = trainer.build_datasets(TrainingConfig(learned_dataset_path=str(manifest)))
    adapter = RendererBatchAdapter(encoder=trainer.temporal_encoder, conditioning_mode="mixed_contract_bootstrap")
    batch = adapter.adapt(train_ds.samples[0])
    assert batch.conditioning_summary["contract_source"] in {"learned_temporal_contract", "weak_manifest_bootstrap"}


def test_temporal_to_downstream_consistency_metrics_smoke(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    dyn = DynamicsTrainer().train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "d"), learned_dataset_path=str(manifest), contract_conditioning_mode="mixed_contract_bootstrap"))
    rnd = RendererTrainer().train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "r"), learned_dataset_path=str(manifest), contract_conditioning_mode="mixed_contract_bootstrap"))
    assert "temporal_to_dynamics_phase_consistency" in dyn.val_metrics
    assert "temporal_to_renderer_mode_consistency" in rnd.val_metrics


def test_nonempty_video_manifest_does_not_fallback_to_synthetic_when_temporal_conditioning_enabled(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    trainer = DynamicsTrainer()
    train_ds, _ = trainer.build_datasets(TrainingConfig(learned_dataset_path=str(manifest), contract_conditioning_mode="mixed_contract_bootstrap"))
    assert len(train_ds) > 0
    assert trainer.dataset_source.startswith("manifest_video_dynamics_primary")
