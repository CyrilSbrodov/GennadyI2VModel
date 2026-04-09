from __future__ import annotations

import json
from pathlib import Path

from training.rollout_eval import evaluate_rollout_modes_on_video_manifest, evaluate_rollout_on_video_manifest, tiny_video_overfit_harness


def _graph(frame_index: int) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "global_context": {"frame_size": [16, 16], "fps": 16, "source_type": "video"},
        "persons": [{"person_id": "p1", "track_id": "t1", "bbox": {"x": 0.2, "y": 0.1, "w": 0.5, "h": 0.8}}],
        "objects": [],
        "relations": [],
    }


def _video_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "video_transition_manifest.json"
    records = []
    for idx in range(4):
        records.append(
            {
                "record_id": f"r{idx}",
                "scene_graph_before": _graph(idx),
                "scene_graph_after": _graph(idx + 1),
                "transition_family": "garment_transition" if idx % 2 == 0 else "pose_transition",
                "runtime_semantic_transition": "garment_transition" if idx % 2 == 0 else "pose_transition",
                "phase_estimate": "transition" if idx < 3 else "stabilize",
                "planner_context": {"step_index": idx + 1, "total_steps": 5, "phase": "transition", "target_duration": 1.2},
                "target_profile": {"primary_regions": ["torso"], "secondary_regions": ["face"], "context_regions": ["legs"]},
                "reveal_score": 0.65,
                "occlusion_score": 0.25,
                "graph_delta_target": {
                    "pose_deltas": {"torso_pitch": -0.1},
                    "garment_deltas": {"coverage_delta": -0.2},
                    "visibility_deltas": {"revealed_regions_score": 0.65, "torso": "partially_visible"},
                    "interaction_deltas": {"support_contact": 0.4},
                    "affected_entities": ["p1"],
                    "affected_regions": ["torso", "face"],
                    "semantic_reasons": ["open_garment"],
                    "region_transition_mode": {"torso": "revealing"},
                },
                "roi_records": [
                    {
                        "region_type": "torso",
                        "roi_before": [[[0.2, 0.2, 0.2] for _ in range(8)] for _ in range(8)],
                        "roi_after": [[[0.65, 0.55, 0.55] for _ in range(8)] for _ in range(8)],
                        "changed_mask": [[[0.7] for _ in range(8)] for _ in range(8)],
                        "preservation_mask": [[[0.3] for _ in range(8)] for _ in range(8)],
                        "target_profile": {"primary_regions": ["torso"], "secondary_regions": [], "context_regions": []},
                        "priors": {"heuristic": True},
                    }
                ],
            }
        )
    manifest.write_text(json.dumps({"manifest_type": "video_transition_manifest", "records": records}), encoding="utf-8")
    return manifest


def test_rollout_eval_smoke_on_video_manifest(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = evaluate_rollout_on_video_manifest(dataset_manifest=str(manifest), mode="teacher_forced_rollout", rollout_steps=2)
    assert out["steps_evaluated"] > 0
    assert out["rollout_frame_reconstruction_proxy"] >= 0.0


def test_teacher_forced_rollout_metrics_exist(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = evaluate_rollout_modes_on_video_manifest(dataset_manifest=str(manifest), rollout_steps=2)
    tf = out["teacher_forced_rollout"]
    assert "rollout_phase_accuracy" in tf
    assert "rollout_renderer_contract_validity" in tf


def test_predicted_rollout_metrics_exist(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = evaluate_rollout_modes_on_video_manifest(dataset_manifest=str(manifest), rollout_steps=2)
    pr = out["predicted_rollout"]
    assert "rollout_family_accuracy" in pr
    assert "rollout_dynamics_contract_validity" in pr


def test_rollout_contract_contains_temporal_dynamics_renderer_outputs(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = evaluate_rollout_on_video_manifest(dataset_manifest=str(manifest), mode="teacher_forced_rollout")
    payload = out["payloads"][0]
    assert "predicted_temporal_contract" in payload
    assert "predicted_graph_delta" in payload
    assert "predicted_rendered_roi" in payload


def test_rollout_target_profile_consistency_metric_smoke(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = evaluate_rollout_on_video_manifest(dataset_manifest=str(manifest), mode="teacher_forced_rollout")
    assert 0.0 <= out["rollout_target_profile_consistency"] <= 1.0


def test_rollout_phase_family_metrics_smoke(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = evaluate_rollout_on_video_manifest(dataset_manifest=str(manifest), mode="predicted_rollout", rollout_steps=2)
    assert 0.0 <= out["rollout_phase_accuracy"] <= 1.0
    assert 0.0 <= out["rollout_family_accuracy"] <= 1.0


def test_tiny_video_overfit_harness_improves_rollout_proxy(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = tiny_video_overfit_harness(str(manifest), tiny_subset_records=3, epochs=5, rollout_steps=1)
    assert out["after"]["rollout_frame_reconstruction_proxy"] >= out["before"]["rollout_frame_reconstruction_proxy"]


def test_nonempty_video_manifest_rollout_eval_does_not_fallback_to_synthetic(tmp_path: Path) -> None:
    manifest = _video_manifest(tmp_path)
    out = evaluate_rollout_on_video_manifest(dataset_manifest=str(manifest), mode="teacher_forced_rollout")
    assert out["dataset_source"].startswith("manifest_video_rollout_eval_primary")
