from pathlib import Path
import json

from core.schema import BBox, BodyPartNode, ExpressionState, GarmentNode, OrientationState, PersonNode, PoseState, SceneGraph, VideoMemory
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from dynamics.model import DynamicsModel, featurize_runtime, targets_from_delta
from learned.factory import BackendConfig, LearnedBackendFactory
from planning.transition_engine import PlannedState
from training.dynamics_eval import evaluate_dynamics
from training.dynamics_trainer import DynamicsDatasetAdapter, DynamicsTrainer
from training.datasets import DynamicsDataset
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


def test_dynamics_predictor_reports_strict_runtime_status() -> None:
    predictor = GraphDeltaPredictor(strict_mode=False)
    delta, _ = predictor.predict(_scene(), PlannedState(step_index=1, labels=["sit_down", "smile"]))
    assert delta.transition_diagnostics.get("requested_family") in {"pose_transition", "interaction_transition", "garment_transition", "expression_transition"}
    assert delta.transition_diagnostics.get("selected_family") in {"pose_transition", "interaction_transition", "garment_transition", "expression_transition"}
    assert delta.transition_diagnostics.get("checkpoint_status") in {"torch_unavailable", "bootstrap_only", "checkpoint_directory_missing", "checkpoint_file_missing", "checkpoint_invalid", "checkpoint_loaded"}
    assert isinstance(delta.transition_diagnostics.get("usable_for_inference"), bool)


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
    inputs = featurize_runtime(_scene(), PlannedState(step_index=2, labels=["sit_down", "intensity=0.8"]), {"step_index": 2.0, "total_steps": 4.0, "phase": "transition"}, memory)
    assert len(inputs.graph_features) == 27
    assert len(inputs.planner_features) == 8
    assert len(inputs.action_features) == 16
    assert len(inputs.memory_features) == 8
    assert len(inputs.target_features) == 37


def test_model_save_load_and_loss_paths_are_separate(tmp_path: Path) -> None:
    model = DynamicsModel()
    inputs = featurize_runtime(_scene(), PlannedState(step_index=1, labels=["smile"]), {"step_index": 1.0, "phase": "prepare"}, None)
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


def test_manifest_backed_dynamics_dataset_and_diagnostics(tmp_path: Path) -> None:
    manifest = tmp_path / "dynamics_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "r1",
                        "scene_graph": {"frame_index": 3, "persons": [{"person_id": "p1", "track_id": "t1", "bbox": {"x": 0.2, "y": 0.1, "w": 0.5, "h": 0.8}}]},
                        "actions": [{"type": "sit_down", "priority": 1, "target_entity": "p1"}],
                        "planner_context": {"step_index": 2, "total_steps": 4, "phase": "transition", "target_duration": 1.4},
                        "target_transition_context": {"target_pose": "sitting"},
                        "memory_context": {"hidden_region_evidence": 0.7},
                        "graph_delta_target": {
                            "pose_deltas": {"torso_pitch": -0.3},
                            "garment_deltas": {"attachment_delta": -0.1},
                            "visibility_deltas": {"torso": "partially_visible"},
                            "expression_deltas": {"smile_intensity": 0.15},
                            "interaction_deltas": {"support_contact": 0.8},
                            "affected_entities": ["p1"],
                            "affected_regions": ["torso", "garments", "face"],
                            "semantic_reasons": ["sit_down", "open_garment"],
                            "region_transition_mode": {"torso": "revealing"},
                        },
                        "tags": ["pose", "garment"],
                    },
                    {"record_id": "r_invalid", "actions": [{"type": "smile", "priority": 1}], "memory_context": "bad_payload"},
                ]
            }
        ),
        encoding="utf-8",
    )
    ds = DynamicsDataset.from_transition_manifest(str(manifest), strict=False)
    assert len(ds) == 1
    assert ds.diagnostics["invalid_records"] == 1
    assert ds.diagnostics["family_counts"]["pose"] == 1
    sample = ds[0]
    batch = DynamicsDatasetAdapter.sample_to_batch(sample, step_index=2)
    assert batch.planner_context["phase"] == "transition"
    assert batch.target_transition_context["target_pose"] == "sitting"
    assert batch.memory_context["hidden_region_evidence"] == 0.7
    assert batch.delta_groups["interaction"] == 1.0


def test_trainer_and_eval_use_manifest_as_primary_when_provided(tmp_path: Path) -> None:
    manifest = tmp_path / "dynamics_manifest.json"
    records = []
    for idx, action in enumerate(("sit_down", "smile", "open_garment", "touch")):
        records.append(
            {
                "record_id": f"r{idx}",
                "scene_graph": {"frame_index": idx, "persons": [{"person_id": "p1", "track_id": "t1", "bbox": {"x": 0.2, "y": 0.1, "w": 0.5, "h": 0.8}}]},
                "labels": [action],
                "planner_context": {"step_index": idx + 1, "total_steps": 5, "phase": "transition"},
                "graph_delta_target": {
                    "pose_deltas": {"torso_pitch": -0.1 * (idx + 1)},
                    "expression_deltas": {"smile_intensity": 0.2 if action == "smile" else 0.0},
                    "interaction_deltas": {"support_contact": 0.7 if action in {"sit_down", "touch"} else 0.0},
                    "visibility_deltas": {"torso": "partially_visible"},
                    "affected_entities": ["p1"],
                    "semantic_reasons": [action],
                },
            }
        )
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    trainer = DynamicsTrainer()
    result = trainer.train(TrainingConfig(epochs=1, learned_dataset_path=str(manifest), checkpoint_dir=str(tmp_path / "ckpt")))
    assert trainer.dataset_source == "manifest_dynamics_primary"
    assert result.val_metrics["usable_sample_count"] >= 1.0
    assert result.val_metrics["pose_group_coverage"] > 0.0
    metrics = evaluate_dynamics(str(tmp_path / "ckpt" / "dynamics" / "dynamics_weights.json"), dataset_size=2, dataset_manifest=str(manifest))
    assert metrics["dataset_source"] == "manifest_dynamics_primary_eval"
    assert float(metrics["invalid_records"]) == 0.0
    assert float(metrics["summary_score"]) >= 0.0


def test_trainer_falls_back_to_synthetic_only_when_manifest_empty(tmp_path: Path) -> None:
    manifest = tmp_path / "empty_dynamics_manifest.json"
    manifest.write_text(json.dumps({"records": []}), encoding="utf-8")
    trainer = DynamicsTrainer()
    train_ds, val_ds = trainer.build_datasets(TrainingConfig(learned_dataset_path=str(manifest), train_size=3, val_size=2))
    assert len(train_ds) == 3
    assert len(val_ds) == 2
    assert trainer.dataset_source == "synthetic_dynamics_bootstrap_fallback_manifest_empty"
