from __future__ import annotations

from pathlib import Path

import pytest

from rendering.torch_backends import TorchRendererBackendBundle
from training.renderer_path_trainer import RendererPathBootstrapDatasetBuilder, RendererPathTrainer


def _torch_available() -> bool:
    bundle = TorchRendererBackendBundle(device="cpu")
    return bundle.available


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
