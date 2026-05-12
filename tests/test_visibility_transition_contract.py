from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.schema import BBox, PersonNode, SceneGraph
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from planning.transition_engine import PlannedState
from training.datasets import DynamicsDataset


class _FakeRuntimeStatus:
    usable_for_inference = False
    checkpoint_status = "checkpoint_file_missing"
    fallback_reason = "test_runtime_unavailable"


class _FakeRuntimeBundle:
    model = None

    def load_checkpoint(self) -> None:
        return None

    def runtime_status(self) -> _FakeRuntimeStatus:
        return _FakeRuntimeStatus()


def _predictor(strict_mode: bool) -> GraphDeltaPredictor:
    return GraphDeltaPredictor(strict_mode=strict_mode, runtime_bundle=_FakeRuntimeBundle())


def _write_manifest(tmp_path: Path, record: dict[str, object]) -> Path:
    path = tmp_path / "dynamics_transition_manifest.json"
    path.write_text(json.dumps({"manifest_type": "dynamics_transition_manifest", "records": [record]}), encoding="utf-8")
    return path


def _visibility_record(*, primary_family: str | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "sample_id": "visibility_aux_001",
        "video_id": "vid_visibility",
        "frame_before_index": 1,
        "frame_after_index": 2,
        "family": "visibility_transition",
        "transition_mode": "reveal",
        "phase": "contact_or_reveal",
        "action_tokens": ["visibility_transition"],
        "state_before": {"garment_attachment": "1.0", "garment_coverage": "1.0"},
        "state_after": {"garment_attachment": "0.8", "garment_coverage": "0.7"},
        "graph_delta_target": {
            "garment_deltas": {"attachment_delta": -0.2, "coverage_delta": -0.3},
            "visibility_deltas": {"torso": "partially_visible"},
            "predicted_visibility_changes": {"torso": "partially_visible"},
            "affected_entities": ["p1"],
            "affected_regions": ["garments", "torso"],
            "region_transition_mode": {"garments": "reveal", "torso": "reveal"},
            "semantic_reasons": ["garment_opening"],
        },
    }
    if primary_family is not None:
        record["primary_family"] = primary_family
    return record


def _scene() -> SceneGraph:
    return SceneGraph(
        frame_index=1,
        persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.2, 0.1, 0.5, 0.8), mask_ref=None)],
    )


def test_transition_manifest_rejects_visibility_without_primary_family_in_strict_mode(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _visibility_record())

    with pytest.raises(ValueError, match="visibility_transition_requires_primary_family"):
        DynamicsDataset.from_transition_manifest(str(manifest), strict=True)


def test_transition_manifest_skips_visibility_without_primary_family_in_non_strict_mode(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _visibility_record())

    dataset = DynamicsDataset.from_transition_manifest(str(manifest), strict=False)

    assert len(dataset) == 0
    assert dataset.diagnostics["invalid_records"] == 1
    assert dataset.diagnostics["skipped_records"] == 1
    assert "visibility_transition_requires_primary_family" in dataset.diagnostics["invalid_examples"][0]["reason"]


def test_transition_manifest_accepts_visibility_with_explicit_primary_family(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _visibility_record(primary_family="garment_transition"))

    dataset = DynamicsDataset.from_transition_manifest(str(manifest), strict=True)

    assert len(dataset) == 1
    sample = dataset[0]
    delta = sample["deltas"][0]
    assert sample["temporal_transition_target"]["family"] == "garment_transition"
    assert sample["graph_transition_contract"]["metadata"]["transition_family"] == "garment_transition"
    assert sample["graph_transition_contract"]["metadata"]["manifest_family"] == "visibility_transition"
    assert sample["graph_transition_contract"]["metadata"]["primary_family"] == "garment_transition"
    assert sample["graph_transition_contract"]["metadata"]["auxiliary_families"] == ["visibility_transition"]
    assert "visibility_transition" in delta.semantic_reasons
    assert delta.visibility_deltas == {"torso": "partially_visible"}
    assert delta.predicted_visibility_changes == {"torso": "partially_visible"}
    assert delta.transition_diagnostics["manifest_family"] == "visibility_transition"
    assert delta.transition_diagnostics["primary_family"] == "garment_transition"
    assert delta.transition_diagnostics["auxiliary_families"] == ["visibility_transition"]


def test_graph_delta_predictor_strict_mode_rejects_visibility_only_primary_request() -> None:
    predictor = _predictor(strict_mode=True)

    with pytest.raises(Exception, match="visibility_transition.*auxiliary semantic family"):
        predictor.predict(_scene(), PlannedState(step_index=1, labels=["visibility_transition"]))


def test_graph_delta_predictor_non_strict_visibility_fallback_is_diagnosed() -> None:
    predictor = _predictor(strict_mode=False)

    delta, _ = predictor.predict(_scene(), PlannedState(step_index=1, labels=["visibility_transition"]))

    assert delta.transition_diagnostics["requested_family"] == "visibility_transition"
    assert delta.transition_diagnostics["selected_family"] == "pose_transition"
    assert delta.transition_diagnostics["unsupported_primary_family"] == "visibility_transition"
    assert delta.transition_diagnostics["auxiliary_visibility_without_primary_family"] is True
    assert delta.transition_diagnostics["selected_family_fallback_reason"] == "visibility_transition_is_auxiliary"
    assert "visibility_transition" in delta.semantic_reasons
