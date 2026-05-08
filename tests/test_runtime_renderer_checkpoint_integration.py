from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.schema import BBox, GlobalSceneContext, GraphDelta, PersonNode, RegionRef, SceneGraph
from learned.factory import BackendConfig, LearnedBackendFactory
from learned.interfaces import PatchSynthesisRequest
from memory.video_memory import MemoryManager
from rendering.learned_bridge import TrainablePatchSynthesisModel
from rendering.trainable_patch_renderer import TrainableLocalPatchModel
from runtime.cli import backend_config_from_args, build_parser


class _DummyDynamicsBackend:
    def predict_transition(self, request):  # pragma: no cover - factory-only test double
        raise AssertionError("dynamics backend should not be called by renderer checkpoint tests")


def _patch_factory_dynamics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LearnedBackendFactory, "_build_dynamics", lambda self, name: _DummyDynamicsBackend())


def _frame(size: int = 16, value: float = 0.0) -> list[list[list[float]]]:
    return [[[value, value, value] for _ in range(size)] for _ in range(size)]


def _request() -> PatchSynthesisRequest:
    scene = SceneGraph(
        frame_index=0,
        persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.0, 0.0, 1.0, 1.0), mask_ref=None)],
        global_context=GlobalSceneContext(frame_size=(16, 16), fps=16, source_type="test"),
    )
    region = RegionRef(region_id="p1:face", bbox=BBox(0.25, 0.25, 0.5, 0.5), reason="checkpoint test")
    delta = GraphDelta(
        affected_entities=["p1"],
        affected_regions=["face"],
        region_transition_mode={"face": "expression_refine"},
        transition_phase="motion",
        expression_deltas={"smile_intensity": 0.25},
    )
    memory = MemoryManager().initialize(scene)
    return PatchSynthesisRequest(
        region=region,
        scene_state=scene,
        memory_summary={},
        transition_context={
            "graph_delta": delta,
            "video_memory": memory,
            "transition_phase": "motion",
            "target_profile": {"primary_regions": ["face"], "secondary_regions": [], "context_regions": []},
        },
        retrieval_summary={"backend": "test", "top_score": 0.5},
        current_frame=_frame(),
        memory_channels={},
        identity_embedding=[],
    )


def _write_numpy_checkpoint(tmp_path: Path) -> Path:
    model_path = tmp_path / "renderer_numpy.json"
    TrainableLocalPatchModel().save(str(model_path))
    checkpoint_path = tmp_path / "renderer_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "renderer_backend": "numpy_local",
                "renderer_model_metadata": {
                    "renderer_backend": "numpy_local",
                    "model_family": "numpy_linear_patch_generator",
                    "torch_backend_used": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def test_factory_without_checkpoint_preserves_default_patch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factory_dynamics(monkeypatch)
    bundle = LearnedBackendFactory(BackendConfig()).build()

    assert isinstance(bundle.patch_backend, TrainablePatchSynthesisModel)
    assert bundle.backend_names.get("patch_checkpoint_requested") is False


def test_factory_loads_numpy_renderer_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factory_dynamics(monkeypatch)
    checkpoint_path = _write_numpy_checkpoint(tmp_path)

    bundle = LearnedBackendFactory(BackendConfig(patch_checkpoint_path=str(checkpoint_path))).build()
    assert isinstance(bundle.patch_backend, TrainablePatchSynthesisModel)
    assert isinstance(bundle.patch_backend.model, TrainableLocalPatchModel)

    out = bundle.patch_backend.synthesize_patch(_request())

    assert out.execution_trace["checkpoint_requested"] is True
    assert out.execution_trace["checkpoint_loaded"] is True
    assert out.execution_trace["checkpoint_backend"] == "numpy_local"
    assert out.execution_trace["checkpoint_path"] == str(checkpoint_path)


def test_checkpoint_load_failure_is_loud_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factory_dynamics(monkeypatch)
    missing = tmp_path / "missing.json"

    with pytest.raises(RuntimeError, match="renderer patch checkpoint"):
        LearnedBackendFactory(BackendConfig(patch_checkpoint_path=str(missing))).build()


def test_checkpoint_load_failure_can_fallback_when_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_factory_dynamics(monkeypatch)
    missing = tmp_path / "missing.json"

    bundle = LearnedBackendFactory(
        BackendConfig(patch_checkpoint_path=str(missing), patch_strict_checkpoint=False)
    ).build()
    out = bundle.patch_backend.synthesize_patch(_request())

    assert out.execution_trace["checkpoint_requested"] is True
    assert out.execution_trace["checkpoint_loaded"] is False
    assert out.execution_trace["checkpoint_fallback_used"] is True
    assert out.execution_trace["checkpoint_load_error"]


def test_checkpoint_load_failure_falls_back_to_numpy_local_not_fresh_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_factory_dynamics(monkeypatch)
    missing = tmp_path / "missing.json"

    bundle = LearnedBackendFactory(
        BackendConfig(patch_checkpoint_path=str(missing), patch_strict_checkpoint=False)
    ).build()

    assert isinstance(bundle.patch_backend, TrainablePatchSynthesisModel)
    assert bundle.patch_backend.backend == "numpy_local"

    out = bundle.patch_backend.synthesize_patch(_request())

    assert out.execution_trace["checkpoint_requested"] is True
    assert out.execution_trace["checkpoint_loaded"] is False
    assert out.execution_trace["checkpoint_fallback_used"] is True
    assert out.execution_trace["checkpoint_fallback_backend"] == "numpy_local"
    assert out.execution_trace["checkpoint_backend"] == ""


def test_torch_checkpoint_load_path_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    _patch_factory_dynamics(monkeypatch)
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    model_path = tmp_path / "renderer_torch.pt"
    TorchLocalPatchGenerator().save(str(model_path))
    checkpoint_path = tmp_path / "renderer_torch_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "renderer_model_metadata": {
                    "renderer_backend": "torch_local",
                    "model_family": "local_conv_conditioned_patch_generator",
                    "torch_backend_used": True,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = LearnedBackendFactory(BackendConfig(patch_checkpoint_path=str(checkpoint_path))).build()
    out = bundle.patch_backend.synthesize_patch(_request())

    assert out.execution_trace["checkpoint_backend"] == "torch_local"
    assert out.execution_trace["checkpoint_loaded"] is True
    assert out.execution_trace["torch_backend_used"] is True


def test_runtime_cli_accepts_patch_checkpoint_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--patch-checkpoint-path",
            "/tmp/renderer_checkpoint.json",
            "--patch-strict-mode",
            "--allow-patch-checkpoint-fallback",
        ]
    )

    config = backend_config_from_args(args)

    assert config.patch_checkpoint_path == "/tmp/renderer_checkpoint.json"
    assert config.patch_strict_mode is True
    assert config.patch_strict_checkpoint is False
