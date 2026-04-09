from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.schema import BBox, GlobalSceneContext, PersonNode, SceneGraph
from learned.interfaces import DynamicsTransitionRequest
from text.learned_bridge import BaselineTextEncoderAdapter
from dynamics.human_state_transition import HumanStateTransitionModel, HumanStateTransitionTargets
from dynamics.temporal_transition_encoder import TemporalTransitionEncoder
from dynamics.transition_contracts import LearnedHumanStateContract
from training.datasets import HumanStateTransitionDataset
from training.dynamics_trainer import DynamicsDatasetAdapter, DynamicsTrainer
from training.human_state_eval import evaluate_human_state_transition
from training.renderer_trainer import RendererBatchAdapter, RendererTrainer
from training.rollout_eval import evaluate_rollout_modes_on_video_manifest
from dynamics.learned_bridge import LegacyHeuristicDynamicsTransitionModel
from training.human_state_transition_trainer import HumanStateTransitionTrainer
from training.types import TrainingConfig


def _graph(frame_index: int) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "global_context": {"frame_size": [16, 16], "fps": 16, "source_type": "video"},
        "persons": [{"person_id": "p1", "track_id": "t1", "bbox": {"x": 0.2, "y": 0.1, "w": 0.5, "h": 0.8}}],
        "objects": [],
        "relations": [],
    }


def _manifest(tmp_path: Path) -> Path:
    records = []
    for idx in range(3):
        records.append(
            {
                "record_id": f"hs_{idx}",
                "previous_record_id": f"hs_{idx-1}" if idx > 0 else "",
                "scene_graph_before": _graph(idx),
                "scene_graph_after": _graph(idx + 1),
                "transition_family": "garment_transition" if idx % 2 == 0 else "pose_transition",
                "runtime_semantic_transition": "garment_transition" if idx % 2 == 0 else "pose_transition",
                "phase_estimate": "transition",
                "planner_context": {"step_index": idx + 1, "total_steps": 4, "phase": "transition", "target_duration": 1.0},
                "target_profile": {"primary_regions": ["torso"], "secondary_regions": ["face"], "context_regions": ["legs"]},
                "reveal_score": 0.7,
                "occlusion_score": 0.2,
                "support_contact_score": 0.3,
                "graph_delta_target": {
                    "pose_deltas": {"torso_pitch": -0.1},
                    "visibility_deltas": {"revealed_regions_score": 0.7, "occluded_regions_score": 0.2},
                    "interaction_deltas": {"support_contact": 0.3},
                    "affected_entities": ["p1"],
                    "affected_regions": ["torso", "face"],
                    "semantic_reasons": ["open_garment"],
                    "region_transition_mode": {"torso": "revealing"},
                },
                "roi_records": [
                    {
                        "region_type": "torso",
                        "changed_ratio": 0.6,
                        "roi_before": [[[0.2, 0.2, 0.2] for _ in range(8)] for _ in range(8)],
                        "roi_after": [[[0.5, 0.5, 0.5] for _ in range(8)] for _ in range(8)],
                    }
                ],
            }
        )
    path = tmp_path / "video_transition_manifest.json"
    path.write_text(json.dumps({"manifest_type": "video_transition_manifest", "records": records}), encoding="utf-8")
    return path


def test_human_state_transition_dataset_from_video_manifest_smoke(tmp_path: Path) -> None:
    ds = HumanStateTransitionDataset.from_video_transition_manifest(str(_manifest(tmp_path)), strict=True)
    assert ds.samples
    assert "history_availability_ratio" in ds.diagnostics


def test_human_state_transition_model_forward_shapes() -> None:
    model = HumanStateTransitionModel()
    out = model.forward(np.zeros((160,), dtype=np.float64))
    assert out.state_embedding.shape == (24,)
    assert out.region_state_embeddings.shape == (8, 3)
    assert out.visibility_state_scores.shape == (8,)


def test_human_state_transition_train_step_is_finite() -> None:
    model = HumanStateTransitionModel()
    targets = HumanStateTransitionTargets(
        family_index=1,
        phase_index=1,
        region_state_targets=[0.2] * 8,
        visibility_targets=[0.8] * 8,
        reveal_memory_target=0.4,
        support_contact_target=0.3,
    )
    losses = model.train_step(np.zeros((160,), dtype=np.float64), targets, lr=1e-3)
    assert all(np.isfinite(float(v)) for v in losses.values())


def test_human_state_contract_typed_smoke() -> None:
    model = HumanStateTransitionModel()
    contract = model.to_typed_contract(model.forward(np.zeros((160,), dtype=np.float64)))
    restored = LearnedHumanStateContract.from_metadata(contract.to_metadata())
    assert restored is not None
    assert restored.is_learned_primary is True
    assert isinstance(restored.region_state_embeddings, dict)


def test_human_state_contract_is_primary_for_dynamics_conditioning(tmp_path: Path) -> None:
    from training.datasets import DynamicsDataset

    dyn_ds = DynamicsDataset.from_video_transition_manifest(str(_manifest(tmp_path)), strict=True)
    dyn_sample = dyn_ds.samples[0]
    assert isinstance(dyn_sample.get("human_state_transition_features"), list)
    assert isinstance(dyn_sample.get("human_state_transition_target"), dict)
    batch = DynamicsDatasetAdapter.sample_to_batch(
        dyn_sample,
        encoder=TemporalTransitionEncoder(),
        human_encoder=HumanStateTransitionModel(),
        conditioning_mode="learned_contract_only",
        step_index=1,
    )
    assert batch.temporal_contract_conditioning.source == "learned_human_state_contract"


def test_human_state_contract_is_primary_for_renderer_conditioning(tmp_path: Path) -> None:
    from training.datasets import RendererDataset

    rnd_ds = RendererDataset.from_video_transition_manifest(str(_manifest(tmp_path)), strict=True)
    sample = rnd_ds.samples[0]
    assert isinstance(sample.get("human_state_transition_features"), list)
    assert isinstance(sample.get("human_state_transition_target"), dict)
    batch = RendererBatchAdapter(encoder=TemporalTransitionEncoder(), human_encoder=HumanStateTransitionModel(), conditioning_mode="learned_contract_only").adapt(sample)
    assert batch.conditioning_summary.get("contract_source") == "learned_human_state_contract"


def test_human_state_eval_metrics_smoke(tmp_path: Path) -> None:
    out = evaluate_human_state_transition(str(_manifest(tmp_path)))
    assert "human_state_summary_score" in out
    assert "state_transition_consistency" in out


def test_rollout_eval_can_use_human_state_contract_path(tmp_path: Path) -> None:
    out = evaluate_rollout_modes_on_video_manifest(dataset_manifest=str(_manifest(tmp_path)), rollout_steps=1)
    assert "predicted_rollout_with_human_state" in out
    assert "path_comparison" in out
    assert "temporal_plus_human_state" in out["path_comparison"]


def test_rollout_profile_consistency_uses_active_selected_path(tmp_path: Path) -> None:
    out = evaluate_rollout_modes_on_video_manifest(dataset_manifest=str(_manifest(tmp_path)), rollout_steps=1)
    payload = out["predicted_rollout_with_human_state"]["payloads"][0]
    assert payload.get("active_conditioning_source") in {"temporal", "human_state"}
    assert isinstance(payload.get("active_target_profile"), dict)


def test_dynamics_usage_metrics_count_human_state_as_learned(tmp_path: Path) -> None:
    from training.datasets import DynamicsDataset

    trainer = DynamicsTrainer()
    dyn_ds = DynamicsDataset.from_video_transition_manifest(str(_manifest(tmp_path)), strict=True)
    batch = DynamicsDatasetAdapter.sample_to_batch(
        dyn_ds.samples[0],
        encoder=TemporalTransitionEncoder(),
        human_encoder=HumanStateTransitionModel(),
        conditioning_mode="learned_contract_only",
        step_index=1,
    )
    from dynamics.model import DynamicsModel

    metrics = trainer._evaluate(DynamicsModel(), [batch])
    assert metrics["learned_contract_usage_ratio"] == 1.0
    assert metrics["weak_contract_usage_ratio"] == 0.0


def test_renderer_usage_metrics_count_human_state_as_learned(tmp_path: Path) -> None:
    from training.datasets import RendererDataset

    trainer = RendererTrainer()
    rnd_ds = RendererDataset.from_video_transition_manifest(str(_manifest(tmp_path)), strict=True)
    batch = RendererBatchAdapter(encoder=TemporalTransitionEncoder(), human_encoder=HumanStateTransitionModel(), conditioning_mode="learned_contract_only").adapt(rnd_ds.samples[0])
    metrics = RendererTrainer.evaluate_model(trainer.model, [batch])
    assert metrics["learned_contract_usage_ratio"] == 1.0
    assert metrics["weak_contract_usage_ratio"] == 0.0


def test_human_state_eval_uses_saved_weights(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    trainer = HumanStateTransitionTrainer()
    out = trainer.train(
        TrainingConfig(
            epochs=1,
            train_size=2,
            val_size=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            learned_dataset_path=str(manifest),
        )
    )
    payload = json.loads(Path(out.checkpoint_path).read_text(encoding="utf-8"))
    weights_path = str(payload.get("weights_path", ""))
    assert weights_path
    metrics = evaluate_human_state_transition(str(manifest), weights_path=weights_path)
    assert metrics.get("weights_path") == weights_path


def test_legacy_heuristic_dynamics_path_has_no_broken_human_state_method() -> None:
    model = LegacyHeuristicDynamicsTransitionModel(strict_mode=True)
    assert not hasattr(model, "human_state_transition_model")
    req = DynamicsTransitionRequest(
        graph_state=SceneGraph(
            frame_index=0,
            persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.2, 0.2, 0.5, 0.6), mask_ref=None)],
            global_context=GlobalSceneContext(frame_size=(16, 16), fps=16, source_type="video"),
        ),
        memory_summary={},
        text_action_summary=BaselineTextEncoderAdapter().encode("hold"),
        memory_channels={},
        step_context={"step_index": 1},
    )
    out = model.predict_transition(req)
    assert out.metadata.get("backend") == "legacy_heuristic_fallback"
