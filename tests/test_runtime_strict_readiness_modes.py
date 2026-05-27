from __future__ import annotations

from pathlib import Path

import pytest

from core.schema import SceneGraph, VideoMemory
from learned.factory import BackendConfig, LearnedBackendFactory
from learned.interfaces import TemporalRefinementRequest
from rendering.temporal_bridge import TrainableTemporalConsistencyBackend


class _StubDynamics:
    def checkpoint_status(self) -> dict[str, object]:
        return {"checkpoint_status": "checkpoint_loaded", "usable_for_inference": True}


class _StubPatch:
    def checkpoint_status(self) -> dict[str, object]:
        return {"checkpoint_loaded": True}


def _touch(path: Path, payload: str = "{}") -> str:
    path.write_text(payload, encoding="utf-8")
    return str(path)


def test_strict_learned_requires_explicit_dynamics_checkpoint_path_before_default_lookup(tmp_path: Path) -> None:
    patch_ckpt = _touch(tmp_path / "patch.json")
    temporal_ckpt = _touch(tmp_path / "temporal.json")
    with pytest.raises(RuntimeError, match="requires dynamics checkpoint"):
        LearnedBackendFactory(
            BackendConfig(
                runtime_mode="strict_learned",
                dynamics_checkpoint_path="",
                patch_checkpoint_path=patch_ckpt,
                temporal_checkpoint_path=temporal_ckpt,
            )
        ).build()


def test_strict_learned_fails_without_patch_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(LearnedBackendFactory, "_build_dynamics", lambda self, name: _StubDynamics())
    temporal_ckpt = _touch(tmp_path / "temporal.json")
    with pytest.raises(RuntimeError, match="requires patch checkpoint"):
        LearnedBackendFactory(
            BackendConfig(runtime_mode="strict_learned", dynamics_checkpoint_path="/explicit/dynamics.json", patch_checkpoint_path="", temporal_checkpoint_path=temporal_ckpt)
        ).build()


def test_strict_learned_fails_without_temporal_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(LearnedBackendFactory, "_build_dynamics", lambda self, name: _StubDynamics())
    monkeypatch.setattr(LearnedBackendFactory, "_build_patch", lambda self, name: _StubPatch())
    patch_ckpt = _touch(tmp_path / "patch.json")
    with pytest.raises(RuntimeError, match="requires temporal checkpoint"):
        LearnedBackendFactory(
            BackendConfig(runtime_mode="strict_learned", dynamics_checkpoint_path="/explicit/dynamics.json", patch_checkpoint_path=patch_ckpt, temporal_checkpoint_path="")
        ).build()


def test_strict_rejects_legacy_backends() -> None:
    with pytest.raises(RuntimeError, match="forbids legacy dynamics"):
        LearnedBackendFactory(BackendConfig(runtime_mode="strict_learned", dynamics_backend="legacy", dynamics_checkpoint_path="/explicit/dyn.json")).build()


def test_trainable_stub_temporal_bootstrap_checkpoint_status_is_honest() -> None:
    backend = TrainableTemporalConsistencyBackend.from_checkpoint_policy(checkpoint_path="", strict_checkpoint=False, strict_mode=False)
    status = backend.checkpoint_status()
    assert status["temporal_checkpoint_loaded"] is False
    assert status["temporal_runtime_loadable"] is False
    assert status["temporal_model_family"] == "trainable_temporal_bootstrap"


def test_trainable_stub_temporal_bootstrap_refine_metadata_is_honest() -> None:
    backend = TrainableTemporalConsistencyBackend.from_checkpoint_policy(checkpoint_path="", strict_checkpoint=False, strict_mode=False)
    frame = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.2, 0.3, 0.4], [0.7, 0.8, 0.9]]]
    req = TemporalRefinementRequest(
        previous_frame=frame,
        current_composed_frame=frame,
        changed_regions=[],
        scene_state=SceneGraph(frame_index=0),
        memory_state=VideoMemory(),
        memory_channels={},
    )
    out = backend.refine_temporal(req)
    assert out.metadata["temporal_path"] == "trainable_stub"
    assert out.metadata["bootstrap_used"] is True


def test_dynamics_strict_mode_never_uses_legacy_fallback() -> None:
    with pytest.raises(RuntimeError, match="requires dynamics checkpoint"):
        LearnedBackendFactory(BackendConfig(runtime_mode="strict_learned", dynamics_checkpoint_path="")).build()


def test_readme_mentions_runtime_modes_and_explicit_strict_requirements() -> None:
    text = Path("README.md").read_text(encoding="utf-8").lower()
    for token in (
        "runtime readiness modes",
        "debug_stub",
        "trainable_stub",
        "strict_learned",
        "production_eval",
        "explicit checkpoint paths",
        "fallback forbidden",
    ):
        assert token in text
