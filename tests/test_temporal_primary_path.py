from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.schema import BBox, PersonNode, RegionRef, SceneGraph
from learned.factory import BackendConfig, LearnedBackendFactory
from learned.interfaces import TemporalRefinementRequest
from memory.video_memory import MemoryManager
from rendering.temporal_bridge import LegacyBaselineTemporalConsistencyModel, TrainableTemporalConsistencyBackend
from rendering.trainable_temporal_consistency import TrainableTemporalConsistencyModel, build_temporal_batch
from training.temporal_trainer import TemporalBatchAdapter, TemporalTrainer
from training.types import TrainingConfig


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def _request(previous: list | None = None, current: list | None = None) -> TemporalRefinementRequest:
    graph = SceneGraph(frame_index=2, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)])
    memory = MemoryManager().initialize(graph)
    region = RegionRef(region_id="p1:torso", bbox=BBox(0.2, 0.2, 0.5, 0.5), reason="temporal_drift")
    return TemporalRefinementRequest(
        previous_frame=previous if previous is not None else _solid(32, 32, (0.2, 0.2, 0.2)),
        current_composed_frame=current if current is not None else _solid(32, 32, (0.45, 0.3, 0.28)),
        changed_regions=[region],
        scene_state=graph,
        memory_state=memory,
        memory_channels={
            "identity": {"p1": 1.0},
            "body_regions": {"roi_count": 1},
            "hidden_regions": {"drift": 0.07},
            "patch_alpha": {"mean_alpha": 0.53, "edge_alpha": 0.36},
            "patch_confidence": {"mean_confidence": 0.74},
        },
    )


def test_learned_temporal_is_default_primary_backend() -> None:
    bundle = LearnedBackendFactory(BackendConfig()).build()
    assert isinstance(bundle.temporal_backend, TrainableTemporalConsistencyBackend)


def test_temporal_model_has_forward_losses_train_eval_separation() -> None:
    model = TrainableTemporalConsistencyModel()
    req = _request()
    target = _solid(32, 32, (0.4, 0.29, 0.27))
    batch = build_temporal_batch(req, target_frame=target)

    out = model.forward(batch)
    losses = model.compute_losses(batch, out)
    trained = model.train_step(batch, lr=1e-3)
    eval_metrics = model.eval_step(batch)

    assert losses["reconstruction_loss"] > 0.0
    assert losses["flicker_loss"] >= 0.0
    assert trained["seam_temporal_loss"] >= 0.0
    assert eval_metrics["flicker_delta_mae"] >= 0.0


def test_runtime_temporal_path_is_learned_primary_and_uses_context() -> None:
    backend = TrainableTemporalConsistencyBackend()
    out = backend.refine_temporal(_request())
    assert out.metadata["temporal_path"] == "learned_primary"
    cond = out.metadata["learned_ready_usage"]["conditioning"]
    assert cond["changed_ratio"] > 0.0
    assert cond["alpha_hint_mean"] > 0.0
    assert cond["confidence_hint_mean"] > 0.0


def test_fallback_is_explicit_and_legacy_only() -> None:
    backend = TrainableTemporalConsistencyBackend(fallback=LegacyBaselineTemporalConsistencyModel(), strict_mode=False)
    out = backend.refine_temporal(_request(previous=[], current=[]))
    assert out.metadata["temporal_path"] == "legacy_fallback"
    assert out.metadata["fallback_reason"]


def test_temporal_batch_adapter_and_trainer_eval_metrics_are_real(tmp_path: Path) -> None:
    sample = {
        "frames": [_solid(16, 16, (0.2, 0.2, 0.2)), _solid(16, 16, (0.5, 0.3, 0.3)), _solid(16, 16, (0.45, 0.28, 0.28))],
        "temporal_consistency_contract": {
            "previous_frame": _solid(16, 16, (0.2, 0.2, 0.2)),
            "composed_frame": _solid(16, 16, (0.5, 0.3, 0.3)),
            "target_frame": _solid(16, 16, (0.45, 0.28, 0.28)),
            "changed_regions": [{"region_id": "p1:torso", "reason": "temporal_drift", "bbox": {"x": 0.2, "y": 0.2, "w": 0.5, "h": 0.5}}],
            "region_consistency_metadata": {"alpha_mean": 0.58, "confidence_mean": 0.73},
            "scene_transition_context": {"step_index": 1, "transition_phase": "motion"},
            "memory_transition_context": {"drift": 0.05},
        },
    }
    batch = TemporalBatchAdapter().adapt(sample)
    assert batch.transition_cond.shape[0] == 6
    assert batch.memory_cond.shape[0] == 6

    manifest = tmp_path / "temporal_manifest.json"
    records = []
    for i in range(4):
        prev = np.full((10, 10, 3), 0.2, dtype=np.float32)
        cur = prev.copy()
        cur[2:8, 2:8, :] = [0.65, 0.42, 0.38]
        target = cur.copy()
        target[3:7, 3:7, :] = [0.58, 0.4, 0.35]
        records.append(
            {
                "previous_frame": prev.tolist(),
                "current_composed_frame": cur.tolist(),
                "target_frame": target.tolist(),
                "changed_regions": [{"region_id": "p1:torso", "reason": "temporal_drift", "bbox": {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6}}],
                "region_consistency_metadata": {"alpha_mean": 0.5, "confidence_mean": 0.7},
                "memory_transition_context": {"drift": 0.06},
            }
        )
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")

    trainer = TemporalTrainer()
    result = trainer.train(TrainingConfig(epochs=1, learned_dataset_path=str(manifest), checkpoint_dir=str(tmp_path / "ckpt")))
    assert trainer.dataset_source.startswith("manifest_temporal_primary")
    assert result.val_metrics["contract_validity"] > 0.0
    assert result.val_metrics["flicker_delta_mae"] >= 0.0
    assert result.val_metrics["region_consistency_mae"] >= 0.0
