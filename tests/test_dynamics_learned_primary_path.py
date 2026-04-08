from pathlib import Path

from core.schema import BBox, BodyPartNode, ExpressionState, GarmentNode, OrientationState, PersonNode, PoseState, SceneGraph, VideoMemory
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from dynamics.model import DynamicsModel, featurize_runtime, targets_from_delta
from learned.factory import BackendConfig, LearnedBackendFactory
from planning.transition_engine import PlannedState
from training.dynamics_eval import evaluate_dynamics
from training.dynamics_trainer import DynamicsDatasetAdapter, DynamicsTrainer
from training.types import TrainingConfig


def _scene() -> SceneGraph:
    person = PersonNode(
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
    return SceneGraph(frame_index=1, persons=[person])


def test_dynamics_predictor_uses_learned_primary_runtime_path() -> None:
    predictor = GraphDeltaPredictor(strict_mode=True)
    delta, _ = predictor.predict(_scene(), PlannedState(step_index=1, labels=["sit_down", "smile"]))
    assert delta.transition_diagnostics.get("runtime_path") == "learned_primary"
    assert "support_contact" in delta.interaction_deltas
    assert delta.region_transition_mode


def test_fallback_diagnostics_are_explicit_when_model_contract_breaks() -> None:
    predictor = GraphDeltaPredictor(strict_mode=False)
    predictor.model.input_dim = 999  # provoke contract mismatch
    delta, _ = predictor.predict(_scene(), PlannedState(step_index=1, labels=["sit_down"]))
    assert delta.transition_diagnostics.get("runtime_path") == "legacy_heuristic_fallback"
    assert delta.transition_diagnostics.get("fallback_reason") in {"DynamicsModelContractError", "ValueError"}


def test_factory_default_backend_is_learned_graph_delta() -> None:
    bundle = LearnedBackendFactory(BackendConfig()).build()
    assert bundle.backend_names["dynamics_backend"] == "learned_graph_delta"


def test_feature_contract_is_structured_and_conditioned_on_memory() -> None:
    memory = VideoMemory(last_transition_context={"visibility_phase": "mixed"})
    inputs = featurize_runtime(_scene(), PlannedState(step_index=2, labels=["sit_down", "intensity=0.8"]), {"step_index": 2.0, "total_steps": 4.0, "phase": "mid"}, memory)
    assert len(inputs.graph_features) == 27
    assert len(inputs.planner_features) == 8
    assert len(inputs.action_features) == 16
    assert len(inputs.memory_features) == 8
    assert len(inputs.target_features) == 37


def test_model_save_load_and_loss_paths_are_separate(tmp_path: Path) -> None:
    model = DynamicsModel()
    inputs = featurize_runtime(_scene(), PlannedState(step_index=1, labels=["smile"]), {"step_index": 1.0, "phase": "early"}, None)
    pred = model.forward(inputs)
    targets = targets_from_delta(GraphDeltaPredictor()._predict_legacy(_scene(), PlannedState(step_index=1, labels=["smile"]))[0])
    losses_before = model.compute_losses(pred, targets)
    model.train_step(inputs, targets, lr=1e-3)
    losses_after = model.compute_losses(model.forward(inputs), targets)
    save_path = tmp_path / "weights.json"
    model.save(str(save_path))
    loaded = DynamicsModel.load(str(save_path))
    assert losses_before["total_loss"] >= 0.0
    assert losses_after["total_loss"] >= 0.0
    assert loaded.forward(inputs).pose.shape == pred.pose.shape


def test_dataset_adapter_and_trainer_eval_smoke(tmp_path: Path) -> None:
    trainer = DynamicsTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=3, val_size=2, checkpoint_dir=str(tmp_path)))
    assert result.stage_name == "dynamics"
    assert "total" in result.train_metrics
    assert "conditioning_sensitivity" in result.val_metrics

    metrics = evaluate_dynamics(str(tmp_path / "dynamics" / "dynamics_weights.json"), dataset_size=2)
    assert "fallback_free_ratio" in metrics
    assert metrics["contract_valid_ratio"] >= 0.0

    ds = trainer.build_datasets(TrainingConfig(train_size=1, val_size=1))[0]
    batch = DynamicsDatasetAdapter.sample_to_batch(ds[0], step_index=1)
    assert batch.action_tokens
