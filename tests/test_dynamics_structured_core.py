from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core.schema import BBox, BodyPartNode, ExpressionState, GarmentNode, OrientationState, PersonNode, PoseState, SceneGraph
from dynamics.model import (
    DYNAMICS_CHECKPOINT_FORMAT,
    DYNAMICS_CHECKPOINT_VERSION,
    SCENE_REGION_KEYS,
    DynamicsModel,
    DynamicsTargets,
    decode_prediction,
    featurize_runtime,
    targets_from_delta,
    validate_checkpoint_payload,
)
from planning.transition_engine import PlannedState
from training.dynamics_trainer import DynamicsTrainer
from training.types import TrainingConfig


def _scene() -> SceneGraph:
    return SceneGraph(
        frame_index=2,
        persons=[
            PersonNode(
                person_id="p1",
                track_id="t1",
                bbox=BBox(0.25, 0.1, 0.45, 0.78),
                mask_ref=None,
                pose_state=PoseState(coarse_pose="standing", angles={"torso_pitch": 8.0, "head_yaw": 3.5, "arm_raise": 4.0}),
                expression_state=ExpressionState(label="neutral", smile_intensity=0.2, eye_openness=0.82),
                orientation=OrientationState(yaw=5.0, pitch=1.0, roll=0.5),
                body_parts=[
                    BodyPartNode(part_id="bp1", part_type="arms", visibility="visible"),
                    BodyPartNode(part_id="bp2", part_type="torso", visibility="visible"),
                ],
                garments=[GarmentNode(garment_id="g1", garment_type="jacket", garment_state="worn", visibility="visible")],
            )
        ],
    )


def _inputs() -> tuple[SceneGraph, PlannedState, dict[str, float | str]]:
    scene = _scene()
    state = PlannedState(step_index=2, labels=["open_garment", "smile", "intensity=0.7"])
    ctx: dict[str, float | str] = {
        "phase": "contact_or_reveal",
        "step_index": 2.0,
        "total_steps": 5.0,
        "target_duration": 1.5,
        "transition_family": "garment_transition",
    }
    return scene, state, ctx


def test_forward_shapes_and_aux_outputs() -> None:
    model = DynamicsModel()
    scene, state, ctx = _inputs()
    pred = model.forward(featurize_runtime(scene, state, ctx, None), family="garment_transition", phase="contact_or_reveal")
    assert pred.pose.shape == (4,)
    assert pred.garment.shape == (3,)
    assert pred.visibility.shape == (5,)
    assert pred.expression.shape == (3,)
    assert pred.interaction.shape == (3,)
    assert pred.region.shape == (4,)
    assert pred.family_logits is not None and pred.family_logits.shape == (4,)
    assert pred.phase_logits is not None and pred.phase_logits.shape == (4,)
    assert pred.transition_confidence is not None
    assert 0.0 <= float(pred.transition_confidence.detach().cpu().item()) <= 1.0


def test_model_is_nn_module_and_device_consistent_cpu() -> None:
    model = DynamicsModel()
    assert isinstance(model, nn.Module)
    model = model.to(torch.device("cpu"))
    scene, state, ctx = _inputs()
    inp = featurize_runtime(scene, state, ctx, None)
    pred = model.forward(inp, family="pose_transition", phase="transition")
    assert model.device.type == "cpu"
    assert isinstance(pred.pose, torch.Tensor)
    assert tuple(pred.pose.shape) == (4,)


def test_forward_keeps_train_mode_and_predict_restores_mode() -> None:
    model = DynamicsModel(seed=19)
    scene, state, ctx = _inputs()
    inp = featurize_runtime(scene, state, ctx, None)

    model.train()
    assert model.training is True
    _ = model.forward(inp, family="pose_transition", phase="transition")
    assert model.training is True

    _ = model.predict(inp, family="pose_transition", phase="transition")
    assert model.training is True

    model.eval()
    _ = model.predict(inp, family="pose_transition", phase="transition")
    assert model.training is False


def test_forward_respects_autograd_without_forcing_no_grad() -> None:
    model = DynamicsModel(seed=23)
    scene, state, ctx = _inputs()
    inp = featurize_runtime(scene, state, ctx, None)
    model.train()
    with torch.enable_grad():
        pred = model.forward(inp, family="interaction_transition", phase="transition")
    assert isinstance(pred.pose, torch.Tensor)
    assert pred.pose.requires_grad is True
    assert pred.transition_confidence is not None and pred.transition_confidence.requires_grad is True
    assert torch.is_grad_enabled() is True


def test_predict_returns_inference_friendly_numpy_representation() -> None:
    model = DynamicsModel(seed=29)
    scene, state, ctx = _inputs()
    inp = featurize_runtime(scene, state, ctx, None)
    pred = model.predict(inp, family="pose_transition", phase="transition")
    assert isinstance(pred.pose, np.ndarray)
    assert isinstance(pred.transition_confidence, float)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = DynamicsModel()
    scene, state, ctx = _inputs()
    inputs = featurize_runtime(scene, state, ctx, None)
    pred_before = model.predict(inputs, family="pose_transition", phase="transition")

    ckpt = tmp_path / "dynamics.pt"
    model.save(str(ckpt))
    loaded = DynamicsModel.load(str(ckpt))
    pred_after = loaded.predict(inputs, family="pose_transition", phase="transition")

    assert np.allclose(pred_before.pose, pred_after.pose, atol=1e-6)
    assert np.allclose(pred_before.region, pred_after.region, atol=1e-6)


def test_train_step_reduces_loss_on_repeated_batch() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    model = DynamicsModel(seed=42)
    scene, state, ctx = _inputs()
    inputs = featurize_runtime(scene, state, ctx, None)
    targets = DynamicsTargets(
        pose=[-0.2, 0.1, 0.35, -0.1],
        garment=[-0.6, -0.45, 0.22],
        visibility=[0.8, 0.6, 0.5, 0.35, 0.25],
        expression=[0.25, 0.78, 0.52],
        interaction=[0.72, 0.55, 0.38],
        region=[0.82, 0.15, 0.66, 0.71],
        family="garment_transition",
        phase="contact_or_reveal",
    )

    losses = []
    for _ in range(25):
        out = model.train_step(inputs, targets, lr=8e-4)
        losses.append(out["total_loss"])
    assert losses[-1] < losses[0]


def test_family_and_phase_conditioning_change_outputs() -> None:
    model = DynamicsModel(seed=11)
    scene, state, ctx = _inputs()
    inputs = featurize_runtime(scene, state, ctx, None)

    pred_pose = model.forward(inputs, family="pose_transition", phase="transition")
    pred_expr = model.forward(inputs, family="expression_transition", phase="transition")
    pred_phase_a = model.forward(inputs, family="pose_transition", phase="prepare")
    pred_phase_b = model.forward(inputs, family="pose_transition", phase="contact_or_reveal")

    assert float(torch.mean(torch.abs(pred_pose.pose - pred_expr.pose)).detach().cpu().item()) > 1e-4
    assert float(torch.mean(torch.abs(pred_phase_a.region - pred_phase_b.region)).detach().cpu().item()) > 1e-4


def test_decode_prediction_returns_non_empty_valid_delta() -> None:
    model = DynamicsModel()
    scene, state, ctx = _inputs()
    pred = model.forward(featurize_runtime(scene, state, ctx, None), family="garment_transition", phase="contact_or_reveal")
    delta = decode_prediction(pred, scene, phase="contact_or_reveal", semantic_reasons=["open_garment"])
    assert delta.affected_regions
    assert delta.region_transition_mode
    assert delta.transition_diagnostics.get("transition_confidence") is not None


def test_confidence_loss_is_present_and_decode_keeps_confidence_diagnostics() -> None:
    model = DynamicsModel(seed=7)
    scene, state, ctx = _inputs()
    inputs = featurize_runtime(scene, state, ctx, None)
    pred = model.forward(inputs, family="interaction_transition", phase="contact_or_reveal")
    delta = decode_prediction(pred, scene, phase="contact_or_reveal", semantic_reasons=["smile"])
    targets = targets_from_delta(delta, family="interaction_transition")
    targets.phase = "contact_or_reveal"
    losses = model.compute_losses(pred, targets)
    assert "confidence_loss" in losses
    assert "transition_confidence" in delta.transition_diagnostics


def test_decode_region_selection_is_model_driven_not_token_substitution() -> None:
    model = DynamicsModel(seed=5)
    scene, state, ctx = _inputs()
    inputs = featurize_runtime(scene, state, ctx, None)
    pred = model.predict(inputs, family="garment_transition", phase="contact_or_reveal")
    pred = type(pred)(
        pose=pred.pose,
        garment=np.array([-0.9, -0.8, 0.3], dtype=np.float32),
        visibility=np.array([0.2, 0.7, 0.1, 0.1, 0.92], dtype=np.float32),
        expression=np.array([0.05, 0.55, 0.1], dtype=np.float32),
        interaction=np.array([0.2, 0.1, 0.1], dtype=np.float32),
        region=np.array([0.93, 0.08, 0.35, 0.55], dtype=np.float32),
        family="garment_transition",
        phase="contact_or_reveal",
        family_logits=pred.family_logits,
        phase_logits=pred.phase_logits,
        transition_confidence=pred.transition_confidence,
        aux=pred.aux,
    )
    delta = decode_prediction(pred, scene, phase="contact_or_reveal", semantic_reasons=["smile"])
    assert "garments" in delta.affected_regions
    assert "inner_garment" in delta.affected_regions
    assert "face" not in delta.affected_regions


def test_region_presence_uses_scene_regions_not_region_head_keys() -> None:
    model = DynamicsModel(seed=13)
    scene, _, _ = _inputs()
    state = PlannedState(step_index=2, labels=["open_garment"])
    state.semantic_transition = None
    ctx: dict[str, float | str] = {"phase": "transition", "transition_family": "garment_transition"}
    features = featurize_runtime(scene, state, ctx, None)
    assert len(features.target_features) == 37
    region_slice = features.target_features[12 : 12 + len(SCENE_REGION_KEYS)]
    assert len(region_slice) == len(SCENE_REGION_KEYS)
    assert all(v in {0.0, 1.0} for v in region_slice)
    assert isinstance(model, nn.Module)


def test_checkpoint_validation_rejects_incompatible_payload() -> None:
    bad = {
        "format": "wrong",
        "version": DYNAMICS_CHECKPOINT_VERSION,
        "input_dim": 96,
        "d_model": 192,
        "nhead": 6,
        "num_layers": 6,
        "ff_dim": 768,
        "adapter_dim": 64,
        "families": ["pose_transition"],
        "phases": ["transition"],
        "state_dict": {"x": 1},
    }
    ok, reason = validate_checkpoint_payload(bad)
    assert ok is False
    assert reason in {"checkpoint_format_mismatch", "family_config_invalid", "phase_config_invalid", "state_dict_schema_invalid"}


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    record = {
        "sample_id": "pose_01",
        "video_id": "v1",
        "frame_before_index": 1,
        "frame_after_index": 2,
        "family": "pose_transition",
        "phase": "transition",
        "action_tokens": ["sit_down"],
        "target_profile": {"primary_regions": ["torso"], "secondary_regions": ["legs"], "context_regions": []},
        "state_before": {"torso_pitch": 0.0, "visibility": {"torso": "visible"}},
        "state_after": {"torso_pitch": -0.2, "visibility": {"torso": "partially_visible"}},
        "graph_delta_target": {
            "pose_deltas": {"torso_pitch": -0.2, "head_yaw": -0.1},
            "visibility_deltas": {"torso": "partially_visible"},
            "affected_entities": ["p1"],
            "affected_regions": ["torso", "legs"],
            "region_transition_mode": {"torso": "deform", "legs": "stable"},
            "semantic_reasons": ["sit_down"],
        },
        "planner_context": {"step_index": 1, "total_steps": 3, "target_duration": 1.0},
    }
    path.write_text(json.dumps({"manifest_type": "dynamics_transition_manifest", "records": [record, record]}), encoding="utf-8")
    return path


def test_dynamics_trainer_manifest_stage_runs(tmp_path: Path) -> None:
    trainer = DynamicsTrainer()
    result = trainer.train(
        TrainingConfig(
            epochs=1,
            train_size=2,
            val_size=1,
            learned_dataset_path=str(_manifest(tmp_path)),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
    )
    assert trainer.stage_name == "dynamics"
    assert result.val_metrics["usable_sample_count"] >= 1.0


def test_checkpoint_payload_accepts_saved_schema(tmp_path: Path) -> None:
    model = DynamicsModel()
    path = tmp_path / "weights.pt"
    model.save(str(path))
    import torch

    payload = torch.load(path, map_location="cpu")
    ok, reason = validate_checkpoint_payload(payload)
    assert ok, reason
    assert payload["format"] == DYNAMICS_CHECKPOINT_FORMAT
