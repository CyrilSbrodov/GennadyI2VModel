import json
from pathlib import Path

from learned.parity import graph_change_summary, semantic_parity_checks
from training.stage_scaffolds import (
    LearnedStageDatasetRouter,
    PatchSynthesisStageRunner,
    StageScaffoldConfig,
    TemporalRefinementStageRunner,
)


def test_semantic_parity_is_structured_and_uses_meaningful_graph_change() -> None:
    contract = {
        "graph_before": {"frame_index": 1, "persons": [{"person_id": "p1", "bbox": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.8}}], "objects": [], "relations": []},
        "graph_after": {"frame_index": 1, "persons": [{"person_id": "p1", "bbox": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.8}}], "objects": [], "relations": []},
        "delta_contract": {"affected_regions": ["torso"], "pose_deltas": {"torso_pitch": 0.2}},
        "region_transition_mode": {},
        "state_before": {},
        "state_after": {},
    }
    parity = semantic_parity_checks(stage="dynamics", contract=contract, request={}, output={})
    assert set(parity.keys()) == {"errors", "warnings", "traces"}
    assert "non_empty_delta_without_meaningful_state_change" in parity["errors"]

    summary = graph_change_summary(contract["graph_before"], contract["graph_after"])
    assert summary["meaningful_change"] is False


def test_dataset_router_collects_ingestion_warnings_and_targets(tmp_path: Path) -> None:
    manifest = tmp_path / "patch_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "stage": "patch_synthesis",
                        "graph": {
                            "persons": [
                                {
                                    "person_id": "p1",
                                    "bbox": {"x": -1.0, "y": 0.1, "w": 1.7, "h": "bad"},
                                    "confidence": 2.0,
                                }
                            ],
                            "global_context": {"frame_size": [0, "bad"], "fps": 0},
                        },
                        "region": {"bbox": {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}, "reason": "MOTION SHIFT"},
                        "expected_selected_strategy": "KNOWN_HIDDEN_REVEAL",
                        "expected_synthesis_mode": "retrieval",
                        "hidden_lifecycle_target": "known_hidden_reveal",
                        "retrieval_richness_target": "rich",
                        "expected_region_metadata": {"region_id": "p1:torso"},
                    }
                ]
            }
        )
    )
    sample = LearnedStageDatasetRouter.build("patch_synthesis", size=1, dataset_path=str(manifest))[0]
    assert sample["target_selected_strategy"] == "KNOWN_HIDDEN_REVEAL"
    assert sample["target_synthesis_mode"] == "retrieval"
    assert sample["_ingestion_warnings"]


def test_patch_and_temporal_checkpoints_include_parity_and_warnings(tmp_path: Path) -> None:
    manifest = tmp_path / "stages.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {"stage": "patch_synthesis", "graph": {}, "region": {}, "frame": "bad"},
                    {"stage": "temporal_refinement", "graph": {}, "regions": "bad", "frame_prev": "bad", "frame_cur": "bad"},
                ]
            }
        )
    )

    patch_ckpt = tmp_path / "patch.ckpt"
    patch_runner = PatchSynthesisStageRunner()
    patch_result = patch_runner.run(StageScaffoldConfig(stage_name="patch_synthesis", batch_size=1, dataset_path=str(manifest), checkpoint_path=str(patch_ckpt)))
    patch_payload = json.loads(Path(patch_result.checkpoint_path).read_text())
    assert patch_payload["parity_summary"] != {}
    assert patch_payload["warnings_or_fallbacks"]

    temporal_ckpt = tmp_path / "temporal.ckpt"
    temporal_runner = TemporalRefinementStageRunner()
    temporal_result = temporal_runner.run(StageScaffoldConfig(stage_name="temporal_refinement", batch_size=1, dataset_path=str(manifest), checkpoint_path=str(temporal_ckpt)))
    temporal_payload = json.loads(Path(temporal_result.checkpoint_path).read_text())
    assert temporal_payload["parity_summary"] != {}
    assert temporal_payload["warnings_or_fallbacks"]
