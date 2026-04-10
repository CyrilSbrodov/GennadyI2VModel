from pathlib import Path

from core.schema import BBox, BodyPartNode, ExpressionState, GarmentNode, OrientationState, PersonNode, PoseState, SceneGraph
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from dynamics.model import DynamicsModel, decode_prediction, featurize_runtime, tensorize_dynamics_inputs, targets_from_delta, validate_checkpoint_payload
from dynamics.runtime_bundle import DynamicsRuntimeBundle
from planning.transition_engine import PlannedState
from training.dynamics_family_training import DynamicsDatasetSurface, DynamicsTrainingSample, FamilyAwareDynamicsTrainingModule


def _scene() -> SceneGraph:
    return SceneGraph(
        frame_index=1,
        persons=[
            PersonNode(
                person_id="p1",
                track_id="t1",
                bbox=BBox(0.2, 0.1, 0.5, 0.8),
                mask_ref=None,
                pose_state=PoseState(coarse_pose="standing", angles={"torso_pitch": 5.0, "head_yaw": 3.0}),
                expression_state=ExpressionState(label="neutral", smile_intensity=0.1, eye_openness=0.8),
                orientation=OrientationState(yaw=10.0, pitch=2.0, roll=0.0),
                body_parts=[BodyPartNode(part_id="bp1", part_type="arms", visibility="visible")],
                garments=[GarmentNode(garment_id="g1", garment_type="coat", garment_state="worn", visibility="visible")],
            )
        ],
    )


def test_runtime_bundle_strict_statuses_missing_and_dev_override(tmp_path: Path) -> None:
    bundle = DynamicsRuntimeBundle(checkpoint_dir=str(tmp_path / "missing"), allow_random_init_for_dev=False)
    status = bundle.load_checkpoint()
    assert status.checkpoint_status == "checkpoint_directory_missing"
    assert status.usable_for_inference is False

    dev = DynamicsRuntimeBundle(checkpoint_dir=str(tmp_path / "missing"), allow_random_init_for_dev=True)
    status_dev = dev.load_checkpoint()
    assert status_dev.checkpoint_status == "checkpoint_directory_missing"
    assert status_dev.usable_for_inference is True


def test_invalid_checkpoint_is_never_usable(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "dynamics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "dynamics_weights.json").write_bytes(b"not_a_torch_checkpoint")
    bundle = DynamicsRuntimeBundle(checkpoint_dir=str(ckpt_dir), allow_random_init_for_dev=False)
    status = bundle.load_checkpoint()
    assert status.checkpoint_status == "checkpoint_invalid"
    assert status.usable_for_inference is False


def test_family_selection_persists_on_fallback() -> None:
    predictor = GraphDeltaPredictor(strict_mode=False)
    delta, _ = predictor.predict(_scene(), PlannedState(step_index=2, labels=["smile", "expression_transition"]))
    d = delta.transition_diagnostics
    assert d["requested_family"] == "expression_transition"
    assert d["selected_family"] == "expression_transition"
    if d["runtime_path"] == "legacy_heuristic_fallback":
        assert d["usable_for_inference"] is False
        assert d.get("fallback_reason")


def test_per_family_losses_are_not_identical() -> None:
    model = DynamicsModel()
    inputs = featurize_runtime(_scene(), PlannedState(step_index=1, labels=["sit_down"]), {"phase": "transition", "transition_family": "pose_transition"}, None)
    pred_pose = model.forward(inputs, family="pose_transition")
    pred_garment = model.forward(inputs, family="garment_transition")
    target_pose = targets_from_delta(decode_prediction(pred_pose, _scene(), "transition", ["sit_down"]))
    target_pose.family = "pose_transition"
    target_garment = targets_from_delta(decode_prediction(pred_garment, _scene(), "transition", ["open_garment"]), family="garment_transition")
    loss_pose = model.compute_losses(pred_pose, target_pose)
    loss_garment = model.compute_losses(pred_garment, target_garment)
    assert loss_pose["total_loss"] != loss_garment["total_loss"]


def test_checkpoint_validation_contract() -> None:
    valid, reason = validate_checkpoint_payload({"format": "bad", "version": -1})
    assert valid is False
    assert reason in {"checkpoint_format_mismatch", "checkpoint_version_mismatch"}


def test_tensor_surface_is_first_class_for_training_module() -> None:
    model = DynamicsModel()
    inputs = featurize_runtime(_scene(), PlannedState(step_index=3, labels=["open_garment"]), {"phase": "contact_or_reveal", "transition_family": "garment_transition"}, None)
    tensor_batch = tensorize_dynamics_inputs(inputs, family="garment_transition", phase="contact_or_reveal")
    pred = model.forward(inputs, family="garment_transition")
    target = targets_from_delta(decode_prediction(pred, _scene(), "transition", ["open_garment"]), family="garment_transition")
    surface = DynamicsDatasetSurface(
        samples=[DynamicsTrainingSample(tensor_batch=tensor_batch, targets=target, graph_before=_scene(), action_tokens=["open_garment"], source="unit")],
        source="bootstrap",
        diagnostics={},
    )
    trainer = FamilyAwareDynamicsTrainingModule()
    tm = trainer.train_epoch(model, surface, lr=1e-4)
    vm = trainer.validate_epoch(model, surface)
    assert "total_loss" in tm
    assert "garment_transition_reveal_occlusion_quality" in vm
