from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.schema import BBox, GraphDelta, RegionRef, SceneGraph, VideoMemory
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from training.datasets import RendererDataset
from training.renderer_manifest_exporter import RendererManifestRecordExporter
from training.renderer_trainer import RendererBatchAdapter


def _request(*, transition_context: dict[str, object] | None = None) -> PatchSynthesisRequest:
    return PatchSynthesisRequest(
        region=RegionRef(region_id="person_0:left_arm", bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="sleeve arm transition"),
        scene_state=SceneGraph(frame_index=0),
        memory_summary={"entities": ["person_0"]},
        transition_context=transition_context or {},
        retrieval_summary={"backend": "learned_primary"},
        current_frame=np.zeros((2, 2, 3), dtype=np.float32).tolist(),
    )


def _output(*, trace: dict[str, object] | None = None) -> PatchSynthesisOutput:
    return PatchSynthesisOutput(
        region=RegionRef(region_id="person_0:left_arm", bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="sleeve arm transition"),
        rgb_patch=np.full((2, 2, 3), 0.75, dtype=np.float32).tolist(),
        alpha_mask=np.full((2, 2), 0.8, dtype=np.float32).tolist(),
        height=2,
        width=2,
        channels=3,
        confidence=0.7,
        execution_trace=trace or {"selected_render_strategy": "KNOWN_HIDDEN_REVEAL", "renderer_path": "learned_primary", "synthesis_mode": "learned_primary"},
    )


def _strong_bundle() -> dict[str, object]:
    return {
        "memory_bundle_present": True,
        "memory_support_level": "strong",
        "retrieval_reasons": ["identity_reference"],
        "has_identity_reference": True,
        "has_appearance_reference": True,
    }


def test_exporter_build_record_contains_memory_bundle_and_strategy() -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(transition_context={"region_memory_bundle_serialized": _strong_bundle()}),
        output=_output(trace={"selected_render_strategy": "KNOWN_HIDDEN_REVEAL", "selected_strategy": "legacy"}),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert record["renderer_batch_contract"]["region_memory_bundle_serialized"]["memory_support_level"] == "strong"
    assert record["selected_render_strategy"] == "KNOWN_HIDDEN_REVEAL"
    assert "selected_strategy" not in record


def test_exporter_output_patch_target_is_marked_self_generated() -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert record["roi_after"] == _output().rgb_patch
    assert record["target_source"] == "runtime_output_patch"
    assert record["training_target_quality"] == "self_generated_runtime_target"


def test_exporter_provided_roi_after_marked_external() -> None:
    provided = np.full((2, 2, 3), 0.25, dtype=np.float32)
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
        roi_after=provided,
    )

    assert record["roi_after"] == provided.tolist()
    assert record["target_source"] == "provided_ground_truth_roi"
    assert record["training_target_quality"] == "external_or_observed_target"


def test_exporter_record_json_serializable() -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(
            transition_context={
                "graph_delta": GraphDelta(affected_regions=["person_0:left_arm"], transition_phase="transition"),
                "video_memory": VideoMemory(),
                "target_profile": {"custom_non_json": VideoMemory()},
            }
        ),
        output=_output(),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    json.dumps(record)


def test_exporter_written_manifest_loads_with_renderer_dataset(tmp_path: Path) -> None:
    exporter = RendererManifestRecordExporter()
    record = exporter.build_record(
        request=_request(transition_context={"region_memory_bundle_serialized": _strong_bundle()}),
        output=_output(),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )
    path = tmp_path / "renderer_manifest.json"
    exporter.write_manifest([record], str(path))

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    assert len(ds.samples) == 1
    batch = RendererBatchAdapter().adapt(ds.samples[0])
    assert batch.conditioning_summary["memory_bundle_present"] is True


def test_exporter_does_not_emit_legacy_strategy_fields() -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(
            trace={
                "selected_render_strategy": "LEARNED_PRIMARY",
                "selected_strategy": "legacy_alias",
                "selected_execution_strategy": "legacy_execution",
                "renderer_path": "learned_primary",
            }
        ),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert record["selected_render_strategy"] == "LEARNED_PRIMARY"
    assert "selected_strategy" not in record
    assert "selected_execution_strategy" not in record


def test_exporter_does_not_use_planner_strategy_as_selected_render_strategy() -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(trace={"planner_selected_strategy": "PLANNER_ROUTE_ONLY"}),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert record["selected_render_strategy"] == "unknown"
    assert record["planner_selected_strategy"] == "PLANNER_ROUTE_ONLY"


def test_exporter_does_not_emit_legacy_strategy_fields_in_contracts() -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(
            trace={
                "selected_render_strategy": "LEARNED_PRIMARY",
                "selected_strategy": "legacy_alias",
                "selected_execution_strategy": "legacy_execution",
                "renderer_path": "learned_primary",
            }
        ),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert "selected_strategy" not in record
    assert "selected_execution_strategy" not in record
    assert "selected_strategy" not in record["patch_synthesis_contract"]
    assert "selected_execution_strategy" not in record["patch_synthesis_contract"]
    assert "selected_strategy" not in record["patch_output_contract"]
    assert "selected_execution_strategy" not in record["patch_output_contract"]


def test_exporter_renderer_patch_manifest_type_loads_strict(tmp_path: Path) -> None:
    exporter = RendererManifestRecordExporter()
    record = exporter.build_record(
        request=_request(transition_context={"region_memory_bundle_serialized": _strong_bundle()}),
        output=_output(),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )
    path = tmp_path / "renderer_patch_manifest.json"
    exporter.write_manifest([record], str(path), manifest_type="renderer_patch_manifest")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["manifest_type"] == "renderer_patch_manifest"
    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    assert len(ds.samples) == 1
