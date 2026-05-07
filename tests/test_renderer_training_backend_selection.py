from __future__ import annotations

import json
from pathlib import Path

import pytest

from rendering.trainable_patch_renderer import TrainableLocalPatchModel
from training.renderer_trainer import RendererTrainer
from training.types import TrainingConfig


def test_training_config_accepts_renderer_backend_values() -> None:
    assert TrainingConfig(renderer_backend="numpy_local").renderer_backend == "numpy_local"
    assert TrainingConfig(renderer_backend="torch_local").renderer_backend == "torch_local"
    with pytest.raises(ValueError):
        TrainingConfig(renderer_backend="bad_backend")


def test_renderer_trainer_uses_numpy_backend_by_default(tmp_path: Path) -> None:
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=2, val_size=1, checkpoint_dir=str(tmp_path)))
    assert isinstance(trainer.model, TrainableLocalPatchModel)
    assert result.val_metrics["renderer_backend"] == "numpy_local"
    assert result.val_metrics["torch_backend_used"] == 0.0


def test_renderer_trainer_uses_torch_backend_when_requested(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=2, val_size=1, checkpoint_dir=str(tmp_path), renderer_backend="torch_local"))
    assert trainer.renderer_backend == "torch_local"
    assert result.val_metrics["renderer_backend"] == "torch_local"
    assert result.val_metrics["torch_backend_used"] == 1.0


def test_torch_training_checkpoint_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=2, val_size=1, checkpoint_dir=str(tmp_path), renderer_backend="torch_local"))
    loaded_model, loaded_backend = RendererTrainer.load_model_from_checkpoint(result.checkpoint_path)
    assert loaded_backend == "torch_local"
    payload = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))
    assert payload["renderer_model_metadata"]["renderer_backend"] == "torch_local"
    ds, _ = trainer.build_datasets(TrainingConfig(train_size=1, val_size=1))
    batch = next(trainer._iter_batches(ds))
    out = loaded_model.eval_step(batch)
    assert "total_loss" in out


def test_legacy_numpy_checkpoint_still_loads(tmp_path: Path) -> None:
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=2, val_size=1, checkpoint_dir=str(tmp_path), renderer_backend="numpy_local"))
    payload = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))
    payload.pop("renderer_model_metadata", None)
    Path(result.checkpoint_path).write_text(json.dumps(payload), encoding="utf-8")
    loaded_model, loaded_backend = RendererTrainer.load_model_from_checkpoint(result.checkpoint_path)
    assert isinstance(loaded_model, TrainableLocalPatchModel)
    assert loaded_backend == "numpy_local"


def test_torch_backend_unavailable_training_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        from rendering.torch_local_patch_generator import TorchBackendUnavailableError

        raise TorchBackendUnavailableError("torch unavailable")

    monkeypatch.setattr("training.renderer_trainer.TorchLocalPatchGenerator", _boom)
    trainer = RendererTrainer()
    with pytest.raises(RuntimeError, match="torch is unavailable"):
        trainer.train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path), renderer_backend="torch_local"))


def test_eval_metrics_include_backend_and_memory_bundle_conditioning() -> None:
    trainer = RendererTrainer()
    ds, _ = trainer.build_datasets(TrainingConfig(train_size=1, val_size=1))
    sample = ds.samples[0]
    contract = sample.setdefault("renderer_batch_contract", {})
    contract["conditioning_summary"] = {"memory_bundle_present": True, "memory_support_level": "strong"}
    batch = trainer.adapter.adapt(sample)
    batch.conditioning_summary = dict(batch.conditioning_summary)
    batch.conditioning_summary["memory_bundle_present"] = True
    batch.conditioning_summary["memory_support_level"] = "strong"
    metrics = trainer.evaluate_model(trainer.model, [batch], renderer_backend="numpy_local")
    assert metrics["renderer_backend"] == "numpy_local"
    assert metrics["global_cond_dim"] > 0
    assert metrics["memory_bundle_conditioning_present"] == 1.0
    assert metrics["memory_support_level_strong_ratio"] == 1.0
