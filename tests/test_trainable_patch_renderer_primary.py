from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.schema import BBox, GlobalSceneContext, GraphDelta, PersonNode, RegionRef, SceneGraph
from learned.factory import BackendConfig, LearnedBackendFactory
from learned.interfaces import PatchSynthesisRequest
from memory.video_memory import MemoryManager
from rendering.learned_bridge import LegacyDeterministicPatchSynthesisModel, TrainablePatchSynthesisModel
from rendering.compositor import Compositor
from rendering.roi_renderer import RenderedPatch
from rendering.trainable_patch_renderer import PatchBatch, TrainableLocalPatchModel, build_patch_batch
from training.datasets import RendererDataset
from training.renderer_trainer import RendererBatchAdapter, RendererTrainer
from training.types import TrainingConfig


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def _request(
    region_id: str,
    frame: list | None = None,
    *,
    mode: str | None = None,
    role: str = "primary",
    top_score: float = 0.72,
) -> PatchSynthesisRequest:
    graph = SceneGraph(
        frame_index=2,
        persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)],
        global_context=GlobalSceneContext(frame_size=(64, 64), fps=16, source_type="single_image"),
    )
    region_type = region_id.split(":", 1)[1]
    region = RegionRef(region_id=region_id, bbox=BBox(0.2, 0.2, 0.4, 0.4), reason="test")
    resolved_mode = mode or ("garment_reveal" if region_type == "torso" else ("pose_exposure" if "arm" in region_type else "expression_refine"))
    target_profile = {
        "primary_regions": [region_type] if role == "primary" else [],
        "secondary_regions": [region_type] if role == "secondary" else [],
        "context_regions": [region_type] if role == "context" else [],
    }
    delta = GraphDelta(
        affected_entities=["p1"],
        affected_regions=[region_type],
        region_transition_mode={region_type: resolved_mode},
        transition_phase="motion",
        newly_revealed_regions=[region] if region_type in {"torso", "left_arm"} else [],
        expression_deltas={"smile_intensity": 0.5} if region_type == "face" else {},
    )
    memory = MemoryManager().initialize(graph)
    return PatchSynthesisRequest(
        region=region,
        scene_state=graph,
        memory_summary={"hidden_region_count": 1, "texture_patch_count": 3},
        transition_context={
            "graph_delta": delta,
            "video_memory": memory,
            "transition_phase": "motion",
            "step_index": 2,
            "target_profile": target_profile,
            "region_selection_rationale": {
                region_type: {
                    "primary": "primary_goal_region",
                    "secondary": "secondary_influence_region",
                    "context": "context_support_region",
                }[role]
            },
            "semantic_families": ["garment_transition"] if "garment" in resolved_mode else (["expression_transition"] if resolved_mode == "expression_refine" else ["pose_transition"]),
        },
        retrieval_summary={
            "backend": "learned_primary",
            "profile": "rich",
            "top_score": top_score,
            "summary": {"candidate_count": 3},
            "top_score_breakdown": {
                "similarity": 0.09,
                "same_entity_bonus": 0.16,
                "reveal_compatibility": 0.22 if resolved_mode == "garment_reveal" else 0.08,
                "visibility_lifecycle_compatibility": 0.10,
            },
        },
        current_frame=frame if frame is not None else _solid(64, 64, (0.3, 0.32, 0.35)),
        memory_channels={"hidden_regions": {"slot": 1}, "garments": {"coat": 1}, "identity": {"p1": 1}},
        identity_embedding=[0.1] * 12,
    )


def _roi_from_request(request: PatchSynthesisRequest) -> np.ndarray:
    frame = np.asarray(request.current_frame, dtype=np.float32)
    h, w, _ = frame.shape
    x0 = int(request.region.bbox.x * w)
    y0 = int(request.region.bbox.y * h)
    x1 = int((request.region.bbox.x + request.region.bbox.w) * w)
    y1 = int((request.region.bbox.y + request.region.bbox.h) * h)
    return frame[y0:y1, x0:x1]


def test_learned_renderer_is_default_primary_backend() -> None:
    bundle = LearnedBackendFactory(BackendConfig()).build()
    assert isinstance(bundle.patch_backend, TrainablePatchSynthesisModel)


def test_runtime_batch_contract_is_rich_and_conditioned() -> None:
    req = _request("p1:torso")
    roi = _roi_from_request(req)
    batch = build_patch_batch(req, roi)
    assert batch.semantic_embed.shape[0] >= 6
    assert batch.delta_cond.shape[0] >= 9
    assert batch.planner_cond.shape[0] >= 8
    assert batch.graph_cond.shape[0] >= 7
    assert batch.memory_cond.shape[0] >= 8
    assert batch.appearance_cond.shape[0] >= 6
    assert batch.mode_cond is not None and batch.role_cond is not None
    assert batch.preservation_mask is not None and batch.seam_prior is not None
    assert batch.transition_mode == "garment_reveal"
    assert batch.profile_role == "primary"


def test_region_modes_drive_distinct_conditioning_contracts() -> None:
    reveal_req = _request("p1:torso", mode="garment_reveal")
    pose_req = _request("p1:left_arm", mode="pose_exposure")
    expr_req = _request("p1:face", mode="expression_refine")
    reveal = build_patch_batch(reveal_req, _roi_from_request(reveal_req))
    pose = build_patch_batch(pose_req, _roi_from_request(pose_req))
    expr = build_patch_batch(expr_req, _roi_from_request(expr_req))

    assert reveal.transition_mode == "garment_reveal"
    assert pose.transition_mode == "pose_exposure"
    assert expr.transition_mode == "expression_refine"
    assert float(np.mean(reveal.uncertainty_target)) > float(np.mean(expr.uncertainty_target))
    assert float(np.mean(expr.changed_mask)) < float(np.mean(reveal.changed_mask))
    assert float(np.mean(expr.preservation_mask)) > float(np.mean(reveal.preservation_mask))


def test_profile_role_changes_edit_strength_and_preservation() -> None:
    primary_req = _request("p1:torso", mode="garment_surface", role="primary")
    secondary_req = _request("p1:torso", mode="garment_surface", role="secondary")
    context_req = _request("p1:torso", mode="garment_surface", role="context")
    primary = build_patch_batch(primary_req, _roi_from_request(primary_req))
    secondary = build_patch_batch(secondary_req, _roi_from_request(secondary_req))
    context = build_patch_batch(context_req, _roi_from_request(context_req))

    assert primary.profile_role == "primary"
    assert secondary.profile_role == "secondary"
    assert context.profile_role == "context"
    assert float(np.mean(primary.changed_mask)) > float(np.mean(secondary.changed_mask)) > float(np.mean(context.changed_mask))
    assert float(np.mean(primary.alpha_target)) > float(np.mean(secondary.alpha_target)) > float(np.mean(context.alpha_target))
    assert float(np.mean(primary.preservation_mask)) < float(np.mean(secondary.preservation_mask)) < float(np.mean(context.preservation_mask))


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
    assert out.execution_trace["transition_mode"] == "expression_refine"
    assert out.execution_trace["profile_role"] == "primary"


def test_expression_refine_is_local_and_conservative() -> None:
    expr_req = _request("p1:face", mode="expression_refine")
    surface_req = _request("p1:torso", mode="garment_surface")
    expr = build_patch_batch(expr_req, _roi_from_request(expr_req))
    surface = build_patch_batch(surface_req, _roi_from_request(surface_req))

    assert float(np.mean(expr.changed_mask)) < float(np.mean(surface.changed_mask))
    assert float(np.mean(expr.preservation_mask)) > float(np.mean(surface.preservation_mask))
    assert float(np.mean(expr.seam_prior)) > float(np.mean(surface.seam_prior))


def test_fallback_diagnostics_are_explicit() -> None:
    backend = TrainablePatchSynthesisModel(strict_mode=False)
    out = backend.synthesize_patch(_request("p1:face", frame=[], mode="expression_refine", role="context"))
    assert out.execution_trace["renderer_path"] == "legacy_fallback"
    assert out.execution_trace["fallback_reason"]
    assert out.execution_trace["requested_transition_mode"] == "expression_refine"
    assert out.execution_trace["requested_profile_role"] == "context"


def test_primary_path_smoke_for_face_torso_and_sleeve_regions() -> None:
    backend = TrainablePatchSynthesisModel(fallback=LegacyDeterministicPatchSynthesisModel())
    for rid in ("p1:face", "p1:torso", "p1:left_arm"):
        out = backend.synthesize_patch(_request(rid))
        assert out.execution_trace["renderer_path"] == "learned_primary"
        assert out.rgb_patch and out.alpha_mask and out.uncertainty_map


def test_alpha_uncertainty_and_confidence_are_mode_aware() -> None:
    backend = TrainablePatchSynthesisModel()
    reveal_out = backend.synthesize_patch(_request("p1:torso", mode="garment_reveal", top_score=0.35))
    expr_out = backend.synthesize_patch(_request("p1:face", mode="expression_refine", top_score=0.86))

    reveal_unc = float(np.mean(np.asarray(reveal_out.uncertainty_map, dtype=np.float32)))
    expr_unc = float(np.mean(np.asarray(expr_out.uncertainty_map, dtype=np.float32)))
    reveal_alpha = float(np.mean(np.asarray(reveal_out.alpha_mask, dtype=np.float32)))

    assert reveal_unc > expr_unc
    assert reveal_out.confidence < expr_out.confidence
    assert abs(reveal_alpha - float(reveal_out.metadata["changed_mask_mean"])) > 1e-3


def test_stable_context_regions_do_not_drift_aggressively() -> None:
    backend = TrainablePatchSynthesisModel()
    req = _request("p1:torso", mode="stable", role="context")
    out = backend.synthesize_patch(req)
    roi_before = _roi_from_request(req)
    roi_after = np.asarray(out.rgb_patch, dtype=np.float32)
    drift = float(np.mean(np.abs(roi_after - roi_before)))
    alpha_mean = float(np.mean(np.asarray(out.alpha_mask, dtype=np.float32)))

    assert drift < 0.03
    assert alpha_mean < 0.25
    assert out.confidence > 0.0


def test_compositor_uses_uncertainty_to_reduce_seam_strength() -> None:
    compositor = Compositor()
    frame = _solid(8, 8, (0.2, 0.2, 0.2))
    patch = RenderedPatch(
        region=RegionRef("p1:face", BBox(0.0, 0.0, 1.0, 1.0), "test"),
        rgb_patch=_solid(8, 8, (0.9, 0.1, 0.1)),
        alpha_mask=[[0.9 for _ in range(8)] for _ in range(8)],
        height=8,
        width=8,
        channels=3,
        uncertainty_map=[[1.0 for _ in range(8)] for _ in range(8)],
        confidence=0.1,
        z_index=1,
    )
    composed = compositor.compose(frame, [patch], GraphDelta())
    assert float(np.mean(np.asarray(composed, dtype=np.float32))) < 0.4


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


def test_renderer_manifest_dataset_pipeline_and_diagnostics(tmp_path: Path) -> None:
    manifest = tmp_path / "renderer_manifest.json"
    good_before = np.full((10, 10, 3), 0.2, dtype=np.float32).tolist()
    good_after = np.full((10, 10, 3), 0.2, dtype=np.float32)
    good_after[2:8, 2:8, :] = [0.7, 0.35, 0.3]
    payload = {
        "records": [
            {
                "roi_before": good_before,
                "roi_after": good_after.tolist(),
                "region_id": "p1:face",
                "semantic_family": "face_expression",
                "delta_cond": [0.1] * 9,
                "planner_cond": [0.2] * 8,
                "graph_cond": [0.3] * 7,
                "memory_cond": [0.4] * 8,
                "appearance_cond": [0.2, 0.2, 0.2, 0.01, 0.01, 0.01],
                "bbox_cond": [0.2, 0.2, 0.4, 0.4],
            },
            {
                "roi_before": good_before,
                "roi_after": good_after.tolist(),
                "region_id": "p1:torso",
                "semantic_family": "torso_reveal",
            },
            {"roi_before": [[1, 2], [3, 4]], "region_id": "p1:left_arm"},
        ]
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    ds = RendererDataset.from_renderer_manifest(str(manifest), strict=False)
    assert len(ds) == 2
    assert ds.diagnostics["invalid_records"] == 1
    assert ds.diagnostics["family_counts"]["face_expression"] == 1
    assert ds.diagnostics["family_counts"]["torso_reveal"] == 1


def test_renderer_trainer_uses_manifest_as_primary_source(tmp_path: Path) -> None:
    manifest = tmp_path / "renderer_manifest.json"
    records = []
    for family, rid in [("face_expression", "p1:face"), ("torso_reveal", "p1:torso"), ("sleeve_arm_transition", "p1:left_arm"), ("face_expression", "p1:face")]:
        before = np.full((8, 8, 3), 0.25, dtype=np.float32)
        after = before.copy()
        after[2:6, 2:6, :] = [0.65, 0.45, 0.4]
        records.append({"roi_before": before.tolist(), "roi_after": after.tolist(), "region_id": rid, "semantic_family": family})
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")

    trainer = RendererTrainer()
    result = trainer.train(TrainingConfig(epochs=1, learned_dataset_path=str(manifest), checkpoint_dir=str(tmp_path / "ckpt")))
    assert trainer.dataset_source.startswith("manifest_paired_roi_primary")
    assert result.val_metrics["usable_sample_count"] >= 1.0
    assert result.val_metrics["invalid_records"] == 0.0
