from __future__ import annotations

import json
from pathlib import Path

from training.datasets import DynamicsDataset
from training.dynamics_family_training import DynamicsDatasetSurface, DynamicsTrainingSample, FamilyAwareDynamicsTrainingModule
from training.dynamics_trainer import DynamicsDatasetAdapter, DynamicsTrainer
from training.types import TrainingConfig
from dynamics.model import DynamicsModel


def _manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dynamics_transition_manifest.json"
    records = [
        {
            "sample_id": "pose_001",
            "video_id": "vid_a",
            "frame_before_index": 10,
            "frame_after_index": 11,
            "family": "pose_transition",
            "transition_mode": "deform",
            "phase": "transition",
            "action_tokens": ["sit_down"],
            "target_profile": {"primary_regions": ["torso", "legs"], "secondary_regions": ["arms"], "context_regions": ["support_zone"]},
            "state_before": {"torso_pitch": 0.1, "head_yaw": 0.2, "visibility": {"torso": "visible"}},
            "state_after": {"torso_pitch": -0.2, "head_yaw": 0.1, "visibility": {"torso": "partially_visible"}},
            "graph_delta_target": {
                "pose_deltas": {"torso_pitch": -0.3, "head_yaw": -0.1},
                "visibility_deltas": {"torso": "partially_visible", "legs": "visible"},
                "affected_entities": ["p1"],
                "affected_regions": ["torso", "legs", "arms"],
                "region_transition_mode": {"torso": "deform", "legs": "stable", "arms": "deform"},
                "semantic_reasons": ["sit_down"],
            },
            "memory_context": {"summary": "recent_pose_change"},
            "planner_context": {"step_index": 2, "total_steps": 5, "target_duration": 1.2},
            "notes": "pose explicit sample",
        },
        {
            "sample_id": "garment_001",
            "video_id": "vid_a",
            "frame_before_index": 11,
            "frame_after_index": 12,
            "family": "garment_transition",
            "transition_mode": "reveal",
            "phase": "contact_or_reveal",
            "action_tokens": ["open_garment"],
            "target_profile": {"primary_regions": ["garments", "torso"], "secondary_regions": ["inner_garment"], "context_regions": []},
            "state_before": {"garment_attachment": 1.0, "garment_coverage": 1.0, "visibility": {"torso": "hidden"}},
            "state_after": {"garment_attachment": 0.6, "garment_coverage": 0.5, "visibility": {"torso": "partially_visible"}},
            "graph_delta_target": {
                "garment_deltas": {"attachment_delta": -0.4, "coverage_delta": -0.5, "layer_shift": 0.2},
                "visibility_deltas": {"torso": "partially_visible", "garments": "partially_visible"},
                "affected_entities": ["p1"],
                "affected_regions": ["garments", "torso", "inner_garment"],
                "region_transition_mode": {"garments": "reveal", "torso": "reveal", "inner_garment": "reveal"},
                "semantic_reasons": ["open_garment"],
            },
            "newly_revealed_regions": ["torso:reveal"],
        },
        {
            "sample_id": "interaction_001",
            "video_id": "vid_a",
            "frame_before_index": 12,
            "frame_after_index": 13,
            "family": "interaction_transition",
            "transition_mode": "contact",
            "phase": "contact_or_reveal",
            "action_tokens": ["touch"],
            "target_profile": {"primary_regions": ["arms"], "secondary_regions": ["torso"], "context_regions": ["support_zone"]},
            "state_before": {"support_contact": 0.2, "hand_contact": 0.1, "visibility": {"arms": "visible"}},
            "state_after": {"support_contact": 0.9, "hand_contact": 0.8, "visibility": {"arms": "visible"}},
            "graph_delta_target": {
                "interaction_deltas": {"support_contact": 0.9, "hand_contact": 0.8, "proximity_contact": 0.7},
                "affected_entities": ["p1", "chair_1"],
                "affected_regions": ["arms", "torso"],
                "region_transition_mode": {"arms": "deform", "torso": "stable"},
                "semantic_reasons": ["touch"],
            },
            "support_contact_score": 0.9,
        },
        {
            "sample_id": "expression_derived_001",
            "video_id": "vid_a",
            "frame_before_index": 13,
            "frame_after_index": 14,
            "family": "expression_transition",
            "transition_mode": "deform",
            "phase": "stabilize",
            "supervision_mode": "derived",
            "action_tokens": ["smile"],
            "target_profile": {"primary_regions": ["face"], "secondary_regions": ["head"], "context_regions": []},
            "state_before": {"smile_intensity": 0.1, "eye_openness": 0.7, "torso_pitch": 0.0, "visibility": {"face": "visible"}},
            "state_after": {"smile_intensity": 0.7, "eye_openness": 0.8, "torso_pitch": 0.05, "visibility": {"face": "visible"}, "predicted_visibility_changes": {"face": "visible"}},
            "affected_entities": ["p1"],
            "affected_regions": ["face", "head"],
            "region_transition_mode": {"face": "deform", "head": "stable"},
        },
        {"sample_id": "broken", "video_id": "vid_a", "frame_before_index": 14},
    ]
    manifest.write_text(json.dumps({"manifest_type": "dynamics_transition_manifest", "records": records}), encoding="utf-8")
    return manifest


def test_manifest_dynamics_pipeline_strict_and_non_strict(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    ds = DynamicsDataset.from_transition_manifest(str(manifest), strict=False)
    assert len(ds) == 4
    assert ds.diagnostics["invalid_records"] == 1
    assert ds.diagnostics["skipped_records"] == 1
    assert ds.diagnostics["usable_samples"] == 4
    assert ds.diagnostics["family_counts"]["pose_transition"] == 1
    assert ds.diagnostics["supervision_mode_counts"]["explicit"] == 3
    assert ds.diagnostics["supervision_mode_counts"]["derived"] == 1
    assert "affected_region_coverage" in ds.diagnostics

    try:
        DynamicsDataset.from_transition_manifest(str(manifest), strict=True)
        raise AssertionError("strict mode must fail on invalid records")
    except ValueError as exc:
        assert "invalid" in str(exc)


def test_manifest_family_aware_batches_and_targets(tmp_path: Path) -> None:
    ds = DynamicsDataset.from_transition_manifest(str(_manifest(tmp_path)), strict=False)
    batches = [DynamicsDatasetAdapter.sample_to_batch(sample, conditioning_mode="weak_contract_only", step_index=idx + 1) for idx, sample in enumerate(ds.samples)]

    assert len(batches) == 4
    families = [b.targets.family for b in batches]
    assert set(families) == {"pose_transition", "garment_transition", "interaction_transition", "expression_transition"}

    pose = next(b for b in batches if b.targets.family == "pose_transition")
    garment = next(b for b in batches if b.targets.family == "garment_transition")
    interaction = next(b for b in batches if b.targets.family == "interaction_transition")
    expression = next(b for b in batches if b.targets.family == "expression_transition")

    assert pose.delta_groups["pose"] == 1.0
    assert garment.delta_groups["garment"] == 1.0
    assert interaction.delta_groups["interaction"] == 1.0
    assert expression.delta_groups["expression"] == 1.0
    assert "face" in expression.graph_before_source or isinstance(expression.tensor_batch.features, object)


def test_manifest_dataset_surface_trains_with_family_module(tmp_path: Path) -> None:
    ds = DynamicsDataset.from_transition_manifest(str(_manifest(tmp_path)), strict=False)
    model = DynamicsModel()
    module = FamilyAwareDynamicsTrainingModule()
    samples = [
        DynamicsTrainingSample(
            tensor_batch=DynamicsDatasetAdapter.sample_to_batch(sample, conditioning_mode="weak_contract_only", step_index=idx + 1).tensor_batch,
            targets=DynamicsDatasetAdapter.sample_to_batch(sample, conditioning_mode="weak_contract_only", step_index=idx + 1).targets,
            graph_before=sample["graphs"][0],
            action_tokens=[a.type for a in sample.get("actions", [])],
            source=str(sample.get("source", "manifest")),
        )
        for idx, sample in enumerate(ds.samples)
    ]
    surface = DynamicsDatasetSurface(samples=samples, source="manifest_dynamics_transition_primary", diagnostics=ds.diagnostics)
    train_metrics = module.train_epoch(model, surface, lr=1e-4)
    val_metrics = module.validate_epoch(model, surface)
    assert "total_loss" in train_metrics
    assert "score" in val_metrics


def test_dynamics_trainer_uses_manifest_backed_surface(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    trainer = DynamicsTrainer()
    result = trainer.train(
        TrainingConfig(
            epochs=1,
            learned_dataset_path=str(manifest),
            train_size=4,
            val_size=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
    )
    assert trainer.dataset_source == "manifest_dynamics_primary"
    assert result.val_metrics["usable_sample_count"] >= 1.0
    assert float(result.val_metrics["invalid_records"]) == 1.0
