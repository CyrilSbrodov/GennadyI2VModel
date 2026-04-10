from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rendering.torch_backends import TorchRendererBackendBundle
from training.renderer_path_trainer import RendererPathBootstrapDatasetBuilder, RendererPathManifestDatasetBuilder, RendererPathTrainer


def _torch_available() -> bool:
    bundle = TorchRendererBackendBundle(device="cpu")
    return bundle.available


def _save_patch(path: Path, arr: np.ndarray) -> str:
    np.save(path, arr.astype(np.float32))
    return str(path)


def _build_manifest(tmp_path: Path) -> Path:
    h, w = 20, 20
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    before = np.stack([0.2 + 0.1 * xx, 0.2 + 0.1 * yy, np.full_like(xx, 0.2)], axis=-1).astype(np.float32)
    after_reveal = np.clip(before + np.stack([0.35 * yy, 0.2 * xx, 0.1 * xx], axis=-1), 0.0, 1.0).astype(np.float32)
    after_existing = np.clip(before + np.stack([0.05 * xx, 0.02 * yy, 0.02 * xx], axis=-1), 0.0, 1.0).astype(np.float32)
    after_insert = np.clip(before + np.stack([0.25 * (1.0 - xx), 0.12 * yy, 0.22 * xx], axis=-1), 0.0, 1.0).astype(np.float32)

    before_path = _save_patch(tmp_path / "before.npy", before)
    after_reveal_path = _save_patch(tmp_path / "after_reveal.npy", after_reveal)
    after_existing_path = _save_patch(tmp_path / "after_existing.npy", after_existing)
    after_insert_path = _save_patch(tmp_path / "after_insert.npy", after_insert)
    explicit_rgb = _save_patch(tmp_path / "explicit_rgb.npy", np.clip(after_reveal[2:18, 2:18], 0.0, 1.0))
    explicit_alpha = _save_patch(tmp_path / "explicit_alpha.npy", np.ones((16, 16, 1), dtype=np.float32) * 0.8)
    explicit_changed = _save_patch(tmp_path / "explicit_changed.npy", np.ones((16, 16, 1), dtype=np.float32) * 0.6)
    explicit_unc = _save_patch(tmp_path / "explicit_unc.npy", np.ones((16, 16, 1), dtype=np.float32) * 0.35)

    records = [
        {
            "sample_id": "existing_001",
            "video_id": "vid_a",
            "frame_before_path": before_path,
            "frame_after_path": after_existing_path,
            "frame_before_index": 10,
            "frame_after_index": 11,
            "path_type": "existing_update",
            "transition_family": "expression_transition",
            "transition_mode": "expression_refine",
            "entity_id": "p1",
            "entity_type": "person",
            "region_id": "p1:face",
            "region_type": "face",
            "lifecycle": "already_existing",
            "hidden_mode": "not_hidden",
            "reveal_type": "none",
            "insertion_type": "none",
            "region_role": "primary",
            "roi_bbox_before": [0.1, 0.1, 0.8, 0.8],
            "roi_bbox_after": [0.1, 0.1, 0.8, 0.8],
            "retrieval_evidence": 0.55,
            "memory_hint_strength": 0.52,
            "appearance_conditioning_strength": 0.42,
            "scene_context_strength": 0.58,
            "pose_role": "neutral",
        },
        {
            "sample_id": "reveal_001",
            "video_id": "vid_a",
            "frame_before_path": before_path,
            "frame_after_path": after_reveal_path,
            "frame_before_index": 15,
            "frame_after_index": 16,
            "path_type": "reveal",
            "transition_family": "visibility_transition",
            "transition_mode": "garment_reveal",
            "entity_id": "p1",
            "entity_type": "person",
            "region_id": "p1:torso",
            "region_type": "torso",
            "lifecycle": "previously_hidden_now_revealed",
            "hidden_mode": "unknown_hidden",
            "reveal_type": "garment_change_reveal",
            "insertion_type": "none",
            "region_role": "primary",
            "roi_bbox_before": [0.1, 0.1, 0.8, 0.8],
            "roi_bbox_after": [0.1, 0.1, 0.8, 0.8],
            "context_bbox": [0.05, 0.05, 0.9, 0.9],
            "bbox_summary": [0.5, 0.5, 0.8, 0.8],
            "retrieval_evidence": 0.72,
            "memory_hint_strength": 0.7,
            "reveal_memory_strength": 0.83,
            "appearance_conditioning_strength": 0.5,
            "scene_context_strength": 0.62,
            "pose_role": "active",
            "supervision": {
                "target_rgb_patch": explicit_rgb,
                "target_alpha_mask": explicit_alpha,
                "changed_mask": explicit_changed,
                "target_uncertainty_proxy": explicit_unc,
            },
        },
        {
            "sample_id": "insert_001",
            "video_id": "vid_b",
            "frame_before_path": before_path,
            "frame_after_path": after_insert_path,
            "frame_before_index": 25,
            "frame_after_index": 26,
            "path_type": "insertion",
            "transition_family": "interaction_transition",
            "transition_mode": "pose_exposure",
            "entity_id": "p2",
            "entity_type": "person",
            "region_id": "p2:body",
            "region_type": "torso",
            "lifecycle": "newly_inserted",
            "hidden_mode": "not_hidden",
            "reveal_type": "none",
            "insertion_type": "new_entity",
            "region_role": "primary",
            "roi_bbox_before": [0.05, 0.05, 0.85, 0.85],
            "roi_bbox_after": [0.05, 0.05, 0.85, 0.85],
            "retrieval_evidence": 0.66,
            "memory_hint_strength": 0.61,
            "insertion_context_strength": 0.82,
            "appearance_conditioning_strength": 0.47,
            "scene_context_strength": 0.63,
            "pose_role": "standing",
        },
        {"sample_id": "broken", "video_id": "vid_b", "path_type": "reveal"},
    ]
    manifest = tmp_path / "renderer_path_manifest.json"
    manifest.write_text(json.dumps({"manifest_type": "renderer_path_manifest", "manifest_version": 1, "records": records}), encoding="utf-8")
    return manifest


def test_manifest_builder_validates_and_reports_diagnostics(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    surface = RendererPathManifestDatasetBuilder().build_surface(str(manifest))

    assert len(surface.samples) == 3
    assert surface.source == "manifest_renderer_path_training_surface"
    assert surface.diagnostics["total_records"] == 4
    assert surface.diagnostics["usable_samples"] == 3
    assert surface.diagnostics["invalid_records"] == 1
    assert surface.diagnostics["samples_per_path"] == {"existing_update": 1, "reveal": 1, "insertion": 1}
    assert surface.diagnostics["supervision_mode_counts"] == {"explicit": 1, "derived": 2}
    assert surface.diagnostics["hidden_mode_counts"]["unknown_hidden"] == 1
    assert surface.diagnostics["reveal_type_counts"]["garment_change_reveal"] == 1
    assert surface.diagnostics["insertion_type_counts"]["new_entity"] == 1
    missing = surface.diagnostics["missing_optional_fields"]
    assert isinstance(missing, dict)
    assert missing["supervision.target_alpha_mask"] >= 2
    assert surface.samples[0].summary["sample_source"] == "manifest"


def test_manifest_builder_strict_mode_raises_on_invalid_record(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    with pytest.raises(ValueError):
        RendererPathTrainer.build_manifest_surface(str(manifest), strict=True)


def test_manifest_builder_preserves_path_specific_semantics(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    surface = RendererPathTrainer.build_manifest_surface(str(manifest), strict=False)
    by_path = {s.path_type: s for s in surface.samples}

    assert by_path["existing_update"].summary["supervision_mode"] == "derived"
    assert by_path["reveal"].summary["supervision_mode"] == "explicit"
    assert by_path["insertion"].summary["supervision_mode"] == "derived"
    assert by_path["reveal"].hidden_mode == "unknown_hidden"
    assert abs(float(by_path["reveal"].target_uncertainty_proxy.mean()) - float(by_path["existing_update"].target_uncertainty_proxy.mean())) > 1e-4
    assert by_path["insertion"].target_alpha.mean() >= by_path["existing_update"].target_alpha.mean()
    assert "reveal_hidden_memory_uncertainty_policy" in by_path["reveal"].summary["diagnostic_notes"]
    assert "insertion_occupancy_alpha_policy" in by_path["insertion"].summary["diagnostic_notes"]


@pytest.mark.skipif(not _torch_available(), reason="torch unavailable")
def test_batch_builder_creates_distinct_path_surfaces() -> None:
    surface = RendererPathBootstrapDatasetBuilder(seed=4).build(per_path=3)
    by_path = {"existing_update": [], "reveal": [], "insertion": []}
    for sample in surface.samples:
        by_path[sample.path_type].append(sample)

    assert len(by_path["existing_update"]) == 3
    assert len(by_path["reveal"]) == 3
    assert len(by_path["insertion"]) == 3

    existing = by_path["existing_update"][0]
    reveal = by_path["reveal"][0]
    insertion = by_path["insertion"][0]
    assert existing.lifecycle == "already_existing"
    assert reveal.lifecycle == "previously_hidden_now_revealed"
    assert insertion.lifecycle == "newly_inserted"
    assert reveal.target_uncertainty_proxy.mean() > existing.target_uncertainty_proxy.mean()
    assert insertion.target_alpha.mean() > existing.target_alpha.mean()


@pytest.mark.skipif(not _torch_available(), reason="torch unavailable")
def test_path_specific_losses_are_computed_separately() -> None:
    bundle = TorchRendererBackendBundle(device="cpu")
    trainer = RendererPathTrainer(bundle=bundle)
    surface = RendererPathBootstrapDatasetBuilder(seed=10).build(per_path=1)

    collected = {}
    for sample in surface.samples:
        total, parts = trainer._compute_path_loss(sample, trainer._forward(sample))
        assert float(total.item()) > 0.0
        collected[sample.path_type] = parts

    assert "preservation_bias_loss" in collected["existing_update"]
    assert "reveal_uncertainty_proxy_loss" in collected["reveal"]
    assert "insertion_silhouette_loss" in collected["insertion"]


@pytest.mark.skipif(not _torch_available(), reason="torch unavailable")
def test_train_validate_cycle_and_metrics_are_path_aware(tmp_path: Path) -> None:
    builder = RendererPathBootstrapDatasetBuilder(seed=12)
    train_surface = builder.build(per_path=4)
    val_surface = builder.build(per_path=2)
    trainer = RendererPathTrainer(bundle=TorchRendererBackendBundle(device="cpu"), learning_rate=1e-3)

    out = trainer.train(
        train_surface=train_surface,
        val_surface=val_surface,
        mode="mixed",
        epochs=1,
        batch_size=3,
        checkpoint_dir=str(tmp_path / "renderer_path_train"),
    )

    assert out["best_score"] >= 0.0
    last_val = out["history"][-1]["val"]
    assert "existing_update.path_score" in last_val
    assert "reveal.path_score" in last_val
    assert "insertion.path_score" in last_val
    assert last_val["existing_update.sample_count"] > 0
    assert last_val["reveal.sample_count"] > 0
    assert last_val["insertion.sample_count"] > 0


@pytest.mark.skipif(not _torch_available(), reason="torch unavailable")
def test_checkpoint_runtime_compatibility_and_usability_policy(tmp_path: Path) -> None:
    bundle = TorchRendererBackendBundle(device="cpu")
    trainer = RendererPathTrainer(bundle=bundle, learning_rate=1e-3)
    train_surface = RendererPathBootstrapDatasetBuilder(seed=3).build(per_path=2)
    val_surface = RendererPathBootstrapDatasetBuilder(seed=8).build(per_path=1)

    result = trainer.train(
        train_surface=train_surface,
        val_surface=val_surface,
        mode="existing_update",
        epochs=1,
        batch_size=2,
        checkpoint_dir=str(tmp_path / "renderer_path_ckpt"),
    )
    ckpt_dir = result["latest_checkpoint_dir"]

    untrained = TorchRendererBackendBundle(device="cpu")
    assert untrained.backend_runtime_status("existing_update")["usable_for_inference"] is False

    loaded = TorchRendererBackendBundle(device="cpu", allow_random_init_for_dev=False)
    trace = loaded.load_checkpoint(ckpt_dir)
    assert trace["loaded"]
    assert loaded.backend_runtime_status("existing_update")["usable_for_inference"] is True
    assert loaded.backend_runtime_status("reveal")["usable_for_inference"] is False


@pytest.mark.skipif(not _torch_available(), reason="torch unavailable")
def test_mixed_mode_training_preserves_path_separation() -> None:
    builder = RendererPathBootstrapDatasetBuilder(seed=99)
    surface = builder.build(per_path=2)
    trainer = RendererPathTrainer(bundle=TorchRendererBackendBundle(device="cpu"), learning_rate=1e-3)

    batches = trainer.batch_builder.build_batches(surface, mode="mixed", batch_size=3, balanced_mixed=True)
    train_metrics = trainer.train_epoch(batches, mode="mixed")

    assert train_metrics["trained_count_existing_update"] > 0
    assert train_metrics["trained_count_reveal"] > 0
    assert train_metrics["trained_count_insertion"] > 0
    assert any(k.startswith("existing_update.") for k in train_metrics)
    assert any(k.startswith("reveal.") for k in train_metrics)
    assert any(k.startswith("insertion.") for k in train_metrics)


@pytest.mark.skipif(not _torch_available(), reason="torch unavailable")
def test_trainer_can_train_on_manifest_surface(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    train_surface = RendererPathTrainer.build_manifest_surface(str(manifest), strict=False)
    val_surface = RendererPathTrainer.build_manifest_surface(str(manifest), strict=False)
    trainer = RendererPathTrainer(bundle=TorchRendererBackendBundle(device="cpu"), learning_rate=1e-3)

    out = trainer.train(
        train_surface=train_surface,
        val_surface=val_surface,
        mode="mixed",
        epochs=1,
        batch_size=2,
        checkpoint_dir=str(tmp_path / "renderer_manifest_train"),
    )
    assert out["dataset_sources"]["train"] == "manifest_renderer_path_training_surface"
    assert out["history"][-1]["val"]["existing_update.sample_count"] > 0
    assert out["history"][-1]["val"]["reveal.sample_count"] > 0
    assert out["history"][-1]["val"]["insertion.sample_count"] > 0
