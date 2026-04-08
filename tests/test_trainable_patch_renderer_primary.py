from __future__ import annotations

from pathlib import Path

import numpy as np

from core.schema import BBox, GraphDelta, PersonNode, RegionRef, SceneGraph
from learned.factory import BackendConfig, LearnedBackendFactory
from learned.interfaces import PatchSynthesisRequest
from memory.video_memory import MemoryManager
from rendering.learned_bridge import LegacyDeterministicPatchSynthesisModel, TrainablePatchSynthesisModel
from rendering.trainable_patch_renderer import PatchBatch, TrainableLocalPatchModel, build_patch_batch
from training.datasets import RendererDataset
from training.renderer_trainer import RendererBatchAdapter, RendererTrainer
from training.types import TrainingConfig


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def _request(region_id: str, frame: list | None = None) -> PatchSynthesisRequest:
    graph = SceneGraph(frame_index=2, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)])
    region_type = region_id.split(":", 1)[1]
    region = RegionRef(region_id=region_id, bbox=BBox(0.2, 0.2, 0.4, 0.4), reason="test")
    delta = GraphDelta(
        affected_entities=["p1"],
        affected_regions=[region_type],
        region_transition_mode={region_type: "garment_reveal" if region_type == "torso" else ("pose_exposure" if "arm" in region_type else "expression_shift")},
        transition_phase="motion",
        newly_revealed_regions=[region] if region_type in {"torso", "left_arm"} else [],
        expression_deltas={"smile_intensity": 0.5} if region_type == "face" else {},
    )
    memory = MemoryManager().initialize(graph)
    return PatchSynthesisRequest(
        region=region,
        scene_state=graph,
        memory_summary={"hidden_region_count": 1, "texture_patch_count": 3},
        transition_context={"graph_delta": delta, "video_memory": memory, "transition_phase": "motion", "step_index": 2},
        retrieval_summary={"backend": "learned_primary", "profile": "rich", "top_score": 0.72},
        current_frame=frame if frame is not None else _solid(64, 64, (0.3, 0.32, 0.35)),
        memory_channels={"hidden_regions": {"slot": 1}, "garments": {"coat": 1}, "identity": {"p1": 1}},
        identity_embedding=[0.1] * 12,
    )


def test_learned_renderer_is_default_primary_backend() -> None:
    bundle = LearnedBackendFactory(BackendConfig()).build()
    assert isinstance(bundle.patch_backend, TrainablePatchSynthesisModel)


def test_runtime_batch_contract_is_rich_and_conditioned() -> None:
    req = _request("p1:torso")
    roi = np.asarray(req.current_frame, dtype=np.float32)[12:36, 12:36]
    batch = build_patch_batch(req, roi)
    assert batch.semantic_embed.shape[0] >= 6
    assert batch.delta_cond.shape[0] >= 9
    assert batch.planner_cond.shape[0] >= 8
    assert batch.graph_cond.shape[0] >= 7
    assert batch.memory_cond.shape[0] >= 8
    assert batch.appearance_cond.shape[0] >= 6


def test_trainable_model_forward_losses_train_eval_and_save_load(tmp_path: Path) -> None:
    model = TrainableLocalPatchModel()
    before = np.full((8, 8, 3), 0.25, dtype=np.float32)
    after = before.copy()
    after[2:7, 1:7, :] = [0.65, 0.55, 0.5]
    mask = np.clip(np.mean(np.abs(after - before), axis=2, keepdims=True) * 3.0, 0.0, 1.0)
    batch = PatchBatch(
        roi_before=before,
        roi_after=after,
        changed_mask=mask,
        alpha_target=np.clip(0.15 + 0.8 * mask, 0.0, 1.0),
        blend_hint=np.clip(0.2 + 0.7 * mask, 0.0, 1.0),
        semantic_embed=np.array([1.0, 0.0, 0.0, 0.9, 0.2, 0.2], dtype=np.float32),
        delta_cond=np.array([0.3] * 9, dtype=np.float32),
        planner_cond=np.array([0.2] * 8, dtype=np.float32),
        graph_cond=np.array([0.4] * 7, dtype=np.float32),
        memory_cond=np.array([0.5] * 8, dtype=np.float32),
        appearance_cond=np.array([0.25, 0.24, 0.23, 0.02, 0.02, 0.02], dtype=np.float32),
        bbox_cond=np.array([0.2, 0.2, 0.5, 0.5], dtype=np.float32),
    )
    out = model.forward(batch)
    losses = model.compute_losses(batch, out)
    train_losses = model.train_step(batch, lr=1e-3)
    eval_losses = model.eval_step(batch)
    assert losses["alpha_loss"] > 0.0
    assert losses["uncertainty_calibration_loss"] >= 0.0
    assert train_losses["alpha_blend_consistency_loss"] >= 0.0
    assert "alpha_mae" in eval_losses and "uncertainty_mean" in eval_losses

    ckpt = tmp_path / "renderer_model.json"
    model.save(str(ckpt))
    loaded = TrainableLocalPatchModel.load(str(ckpt))
    assert loaded.forward(batch)["confidence"] >= 0.0


def test_renderer_output_contract_includes_alpha_uncertainty_semantics() -> None:
    backend = TrainablePatchSynthesisModel()
    out = backend.synthesize_patch(_request("p1:face"))
    assert out.execution_trace["renderer_path"] == "learned_primary"
    assert out.execution_trace["alpha_semantics"]
    assert out.execution_trace["uncertainty_semantics"]
    assert out.metadata["blend_hint_mean"] >= 0.0


def test_fallback_diagnostics_are_explicit() -> None:
    backend = TrainablePatchSynthesisModel(strict_mode=False)
    out = backend.synthesize_patch(_request("p1:face", frame=[]))
    assert out.execution_trace["renderer_path"] == "legacy_fallback"
    assert out.execution_trace["fallback_reason"]


def test_primary_path_smoke_for_face_torso_and_sleeve_regions() -> None:
    backend = TrainablePatchSynthesisModel(fallback=LegacyDeterministicPatchSynthesisModel())
    for rid in ("p1:face", "p1:torso", "p1:left_arm"):
        out = backend.synthesize_patch(_request(rid))
        assert out.execution_trace["renderer_path"] == "learned_primary"
        assert out.rgb_patch and out.alpha_mask and out.uncertainty_map


def test_dataset_adapter_and_eval_metrics_are_real_not_hardcoded(tmp_path: Path) -> None:
    ds = RendererDataset.synthetic(6)
    batch = RendererBatchAdapter().adapt(ds[0])
    assert batch.delta_cond.shape[0] == 9
    assert batch.planner_cond.shape[0] == 8
    assert batch.memory_cond.shape[0] == 8

    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=6, val_size=4, checkpoint_dir=str(tmp_path)))
    assert result.val_metrics["contract_validity"] > 0.0
    assert result.val_metrics["reconstruction_mae"] >= 0.0
    assert result.val_metrics["alpha_mae"] >= 0.0
    assert result.val_metrics["uncertainty_calibration_mae"] >= 0.0
    assert result.val_metrics["face_family_score"] >= 0.0
