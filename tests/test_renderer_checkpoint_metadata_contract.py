from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.schema import BBox, GlobalSceneContext, GraphDelta, PersonNode, RegionRef, SceneGraph
from learned.factory import BackendConfig, LearnedBackendFactory
from learned.interfaces import PatchSynthesisRequest
from memory.video_memory import MemoryManager
from rendering.patch_conditioning_contract import GLOBAL_COND_DIM
from rendering.renderer_checkpoint_loader import load_renderer_model_from_checkpoint
from rendering.trainable_patch_renderer import TrainableLocalPatchModel
from training.renderer_trainer import RendererTrainer
from training.types import TrainingConfig


def _save_numpy_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "renderer_numpy.json"
    TrainableLocalPatchModel().save(str(model_path))
    return model_path


def _write_checkpoint(tmp_path: Path, metadata: dict[str, object] | None = None) -> Path:
    model_path = _save_numpy_model(tmp_path)
    checkpoint_path = tmp_path / "renderer_checkpoint.json"
    payload: dict[str, object] = {
        "model_path": str(model_path),
        "renderer_backend": "numpy_local",
    }
    if metadata is not None:
        payload["renderer_model_metadata"] = metadata
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    return checkpoint_path


def _request() -> PatchSynthesisRequest:
    scene = SceneGraph(
        frame_index=0,
        persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.0, 0.0, 1.0, 1.0), mask_ref=None)],
        global_context=GlobalSceneContext(frame_size=(16, 16), fps=16, source_type="test"),
    )
    region = RegionRef(region_id="p1:face", bbox=BBox(0.25, 0.25, 0.5, 0.5), reason="metadata contract test")
    delta = GraphDelta(
        affected_entities=["p1"],
        affected_regions=["face"],
        region_transition_mode={"face": "expression_refine"},
        transition_phase="motion",
        expression_deltas={"smile_intensity": 0.25},
    )
    return PatchSynthesisRequest(
        region=region,
        scene_state=scene,
        memory_summary={},
        transition_context={
            "graph_delta": delta,
            "video_memory": MemoryManager().initialize(scene),
            "transition_phase": "motion",
            "target_profile": {"primary_regions": ["face"], "secondary_regions": [], "context_regions": []},
        },
        retrieval_summary={"backend": "test", "top_score": 0.5},
        current_frame=[[[0.0, 0.0, 0.0] for _ in range(16)] for _ in range(16)],
        memory_channels={},
        identity_embedding=[],
    )


class _DummyDynamicsBackend:
    def predict_transition(self, request):  # pragma: no cover - factory-only test double
        raise AssertionError("dynamics backend should not be called by renderer metadata contract tests")


def test_renderer_training_checkpoint_contains_runtime_metadata(tmp_path: Path) -> None:
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=2, val_size=1, checkpoint_dir=str(tmp_path), renderer_backend="numpy_local"))

    payload = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))
    metadata = payload["renderer_model_metadata"]

    assert payload["checkpoint_contract_version"] == "renderer_checkpoint.v1"
    assert metadata["checkpoint_contract_version"] == "renderer_checkpoint.v1"
    assert payload["runtime_loadable"] is True
    assert metadata["runtime_loadable"] is True
    assert payload["renderer_backend"] == "numpy_local"
    assert metadata["renderer_backend"] == "numpy_local"
    assert metadata["global_cond_dim"] > 0
    assert "target_training_role_counts" in metadata
    assert metadata["contains_only_bootstrap_targets"] is True


def test_renderer_training_torch_checkpoint_uses_pt_model_path(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    trainer = RendererTrainer()
    result = trainer.train(
        TrainingConfig(
            epochs=1,
            train_size=2,
            val_size=1,
            checkpoint_dir=str(tmp_path),
            renderer_backend="torch_local",
        )
    )

    payload = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))
    metadata = payload["renderer_model_metadata"]

    assert payload["model_path"].endswith(".pt")
    assert metadata["model_path"].endswith(".pt")
    assert metadata["runtime_backend"] == "torch_local"
    assert metadata["renderer_backend"] == "torch_local"
    assert metadata["torch_backend_used"] is True


def test_checkpoint_loader_rejects_global_cond_dim_mismatch(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        {
            "checkpoint_contract_version": "renderer_checkpoint.v1",
            "renderer_backend": "numpy_local",
            "runtime_loadable": True,
            "global_cond_dim": GLOBAL_COND_DIM + 1,
        },
    )

    with pytest.raises(ValueError, match="global_cond_dim"):
        load_renderer_model_from_checkpoint(str(checkpoint_path))


def test_checkpoint_loader_rejects_runtime_unloadable_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        {
            "checkpoint_contract_version": "renderer_checkpoint.v1",
            "renderer_backend": "numpy_local",
            "runtime_loadable": False,
            "global_cond_dim": GLOBAL_COND_DIM,
        },
    )

    with pytest.raises(ValueError, match="runtime_loadable"):
        load_renderer_model_from_checkpoint(str(checkpoint_path))


def test_legacy_checkpoint_without_contract_version_still_loads(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path, metadata=None)

    model, backend, metadata = load_renderer_model_from_checkpoint(str(checkpoint_path))

    assert isinstance(model, TrainableLocalPatchModel)
    assert backend == "numpy_local"
    assert metadata == {}


def test_runtime_trace_exposes_checkpoint_metadata_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LearnedBackendFactory, "_build_dynamics", lambda self, name: _DummyDynamicsBackend())
    checkpoint_path = _write_checkpoint(
        tmp_path,
        {
            "checkpoint_contract_version": "renderer_checkpoint.v1",
            "renderer_backend": "numpy_local",
            "model_family": "numpy_linear_patch_generator",
            "runtime_loadable": True,
            "global_cond_dim": GLOBAL_COND_DIM,
        },
    )

    bundle = LearnedBackendFactory(BackendConfig(patch_checkpoint_path=str(checkpoint_path))).build()
    out = bundle.patch_backend.synthesize_patch(_request())

    assert out.execution_trace["checkpoint_contract_version"] == "renderer_checkpoint.v1"
    assert out.execution_trace["checkpoint_model_family"] == "numpy_linear_patch_generator"
    assert out.execution_trace["checkpoint_runtime_loadable"] is True
    assert out.execution_trace["checkpoint_global_cond_dim"] == GLOBAL_COND_DIM


def test_checkpoint_metadata_does_not_call_bootstrap_ground_truth(tmp_path: Path) -> None:
    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=2, val_size=1, checkpoint_dir=str(tmp_path), renderer_backend="numpy_local"))
    metadata = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))["renderer_model_metadata"]

    metadata_text = json.dumps(metadata, sort_keys=True)
    target_training_role_counts = metadata["target_training_role_counts"]

    assert metadata["contains_only_bootstrap_targets"] is True
    assert target_training_role_counts["bootstrap_self_generated"] > 0
    assert target_training_role_counts["supervised_external"] == 0
    assert metadata["bootstrap_self_generated_ratio"] == 1.0
    assert "bootstrap" in metadata_text
