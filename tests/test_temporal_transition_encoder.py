from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dynamics.learned_bridge import LearnedDynamicsTransitionModel
from dynamics.temporal_transition_encoder import FAMILIES, PHASES, REGION_KEYS, TemporalTransitionEncoder, TemporalTransitionTargets
from learned.interfaces import DynamicsTransitionRequest
from planning.transition_engine import PlannedState
from text.encoder_contracts import TextEncodingDiagnostics
from text.learned_bridge import BaselineTextEncoderAdapter
from training.datasets import TemporalTransitionDataset
from training.temporal_transition_eval import evaluate_temporal_transition
from training.temporal_transition_trainer import TemporalTransitionDatasetAdapter, TemporalTransitionTrainer
from training.types import TrainingConfig
from core.schema import BBox, PersonNode, PoseState, ExpressionState, OrientationState, SceneGraph


def _scene_graph_payload(frame_index: int, smile: float = 0.1) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "global_context": {"frame_size": [32, 32], "fps": 8, "source_type": "video"},
        "persons": [
            {
                "person_id": "p1",
                "track_id": "t1",
                "bbox": {"x": 0.2, "y": 0.1, "w": 0.5, "h": 0.8},
                "body_parts": [{"part_id": "bp1", "part_type": "arms", "visibility": "visible"}],
                "garments": [{"garment_id": "g1", "garment_type": "jacket", "visibility": "visible"}],
                "expression_state": {"smile_intensity": smile},
            }
        ],
        "objects": [],
        "relations": [],
    }


def _video_manifest(path: Path, records_count: int = 4) -> None:
    records = []
    for i in range(records_count):
        records.append(
            {
                "record_id": f"r{i}",
                "scene_graph_before": _scene_graph_payload(i, smile=0.1),
                "scene_graph_after": _scene_graph_payload(i + 1, smile=0.25),
                "roi_records": [
                    {"region_type": "torso", "changed_ratio": 0.4},
                    {"region_type": "face", "changed_ratio": 0.25},
                ],
                "graph_delta_target": {
                    "pose_deltas": {"torso_motion": 0.15 + 0.05 * i},
                    "garment_deltas": {"coverage_change": 0.2},
                    "visibility_deltas": {"revealed_regions_score": 0.3, "occluded_regions_score": 0.1},
                    "expression_deltas": {"face_expression_shift": 0.35},
                    "interaction_deltas": {"support_contact": 0.6 if i % 2 == 0 else 0.1, "contact_hint": 0.4},
                },
                "planner_context": {"step_index": i + 1, "total_steps": records_count + 1, "phase": "transition", "target_duration": 1.2},
                "target_profile": {"primary_regions": ["torso"], "secondary_regions": ["face"], "context_regions": ["legs"]},
                "runtime_semantic_transition": FAMILIES[i % len(FAMILIES)],
                "transition_family": FAMILIES[i % len(FAMILIES)],
                "phase_estimate": PHASES[i % len(PHASES)],
                "fallback_flags": {"heuristic_priors_used": False},
            }
        )
    path.write_text(json.dumps({"manifest_type": "video_transition_manifest", "records": records}), encoding="utf-8")


def test_temporal_transition_dataset_from_video_manifest_smoke(tmp_path: Path) -> None:
    manifest = tmp_path / "video_transition_manifest.json"
    _video_manifest(manifest, records_count=3)
    ds = TemporalTransitionDataset.from_video_transition_manifest(str(manifest), strict=True)
    assert len(ds) == 3
    assert ds.diagnostics["source"] == "manifest_video_temporal_transition_primary"


def test_temporal_transition_dataset_emits_structured_targets(tmp_path: Path) -> None:
    manifest = tmp_path / "video_transition_manifest.json"
    _video_manifest(manifest, records_count=2)
    ds = TemporalTransitionDataset.from_video_transition_manifest(str(manifest), strict=True)
    sample = ds[0]
    assert len(sample["temporal_transition_features"]) == 128
    target = sample["temporal_transition_target"]
    assert target["family"] in FAMILIES
    assert target["phase"] in PHASES
    assert "primary_regions" in target["target_profile"]


def test_temporal_transition_model_forward_shapes() -> None:
    model = TemporalTransitionEncoder()
    pred = model.forward([0.1] * 128)
    assert pred.family_logits.shape[0] == len(FAMILIES)
    assert pred.phase_logits.shape[0] == len(PHASES)
    assert pred.target_profile_scores.shape[0] == len(REGION_KEYS)
    assert pred.transition_embedding.shape[0] == model.embed_dim


def test_temporal_transition_model_train_step_is_finite() -> None:
    model = TemporalTransitionEncoder()
    targets = TemporalTransitionTargets(
        family_index=1,
        phase_index=2,
        target_profile_regions=[1.0 if i < 2 else 0.0 for i in range(len(REGION_KEYS))],
        reveal_score=0.4,
        occlusion_score=0.2,
        support_contact_score=0.6,
    )
    losses = model.train_step(np.linspace(0.0, 1.0, num=128), targets, lr=1e-3)
    assert all(np.isfinite(v) for v in losses.values())


def test_temporal_transition_trainer_uses_video_manifest_as_primary(tmp_path: Path) -> None:
    manifest = tmp_path / "video_transition_manifest.json"
    _video_manifest(manifest, records_count=4)
    trainer = TemporalTransitionTrainer()
    result = trainer.train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), learned_dataset_path=str(manifest)))
    assert trainer.dataset_source == "manifest_video_temporal_transition_primary"
    assert result.val_metrics["usable_sample_count"] > 0.0


def test_temporal_transition_eval_reports_family_phase_metrics(tmp_path: Path) -> None:
    manifest = tmp_path / "video_transition_manifest.json"
    _video_manifest(manifest, records_count=4)
    trainer = TemporalTransitionTrainer()
    result = trainer.train(TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), learned_dataset_path=str(manifest)))
    weights = Path(result.checkpoint_path).parent / "temporal_transition_weights.json"
    metrics = evaluate_temporal_transition(str(weights), dataset_manifest=str(manifest))
    assert "family_accuracy" in metrics
    assert "phase_accuracy" in metrics
    assert "family_coverage" in metrics
    assert "phase_coverage" in metrics


def test_temporal_transition_runtime_contract_is_compatible_with_dynamics_renderer() -> None:
    graph = SceneGraph(
        frame_index=1,
        persons=[
            PersonNode(
                person_id="p1",
                track_id="t1",
                bbox=BBox(0.2, 0.1, 0.5, 0.8),
                mask_ref=None,
                pose_state=PoseState(coarse_pose="standing"),
                expression_state=ExpressionState(label="neutral", smile_intensity=0.1),
                orientation=OrientationState(),
            )
        ],
    )
    text_out = BaselineTextEncoderAdapter().encode("sit down")
    request = DynamicsTransitionRequest(graph_state=graph, memory_summary={}, text_action_summary=text_out, memory_channels={}, step_context={"step_index": 1})
    out = LearnedDynamicsTransitionModel(strict_mode=True).predict_transition(request)
    contract = out.metadata.get("temporal_transition_contract", {})
    assert out.delta.region_transition_mode
    assert "predicted_family" in contract
    assert "predicted_phase" in contract
    assert "transition_embedding" in contract


def test_temporal_transition_stage_does_not_fallback_to_synthetic_when_manifest_nonempty(tmp_path: Path) -> None:
    manifest = tmp_path / "video_transition_manifest.json"
    _video_manifest(manifest, records_count=2)
    trainer = TemporalTransitionTrainer()
    train_ds, val_ds = trainer.build_datasets(TrainingConfig(epochs=1, learned_dataset_path=str(manifest), checkpoint_dir=str(tmp_path / "ckpt"), train_size=5, val_size=3))
    assert trainer.dataset_source == "manifest_video_temporal_transition_primary"
    assert train_ds.samples[0]["source"] == "manifest_video_temporal_transition_primary"
    assert len(train_ds) + len(val_ds) == 2
