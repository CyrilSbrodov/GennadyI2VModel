from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.schema import BBox, RegionRef, SceneGraph
from evaluation.contracts import build_patch_eval_payload
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from training.datasets import RendererDataset
from training.renderer_manifest_exporter import RendererManifestRecordExporter
from training.renderer_trainer import RendererBatchAdapter


def _roi(value: float = 0.25) -> list[list[list[float]]]:
    return [[[value, value, value] for _ in range(2)] for _ in range(2)]


def _request(*, transition_context: dict[str, object] | None = None) -> PatchSynthesisRequest:
    return PatchSynthesisRequest(
        region=RegionRef(region_id="person_0:left_arm", bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="sleeve arm transition"),
        scene_state=SceneGraph(frame_index=0),
        memory_summary={"entities": ["person_0"]},
        transition_context=transition_context or {},
        retrieval_summary={"backend": "learned_primary"},
        current_frame=_roi(0.0),
    )


def _output(*, trace: dict[str, object] | None = None, value: float = 0.75) -> PatchSynthesisOutput:
    return PatchSynthesisOutput(
        region=RegionRef(region_id="person_0:left_arm", bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="sleeve arm transition"),
        rgb_patch=np.full((2, 2, 3), value, dtype=np.float32).tolist(),
        alpha_mask=np.full((2, 2), 0.8, dtype=np.float32).tolist(),
        height=2,
        width=2,
        channels=3,
        confidence=0.7,
        execution_trace=trace or {"selected_render_strategy": "LEARNED_PRIMARY", "renderer_path": "learned_primary", "synthesis_mode": "learned_primary"},
    )


def _record(**updates: object) -> dict[str, object]:
    base: dict[str, object] = {
        "roi_before": _roi(0.25),
        "roi_after": _roi(0.35),
        "semantic_family": "sleeve_arm_transition",
        "region_id": "person_0:left_arm",
    }
    base.update(updates)
    return base


def test_exporter_manifest_metadata_counts_target_quality(tmp_path: Path) -> None:
    exporter = RendererManifestRecordExporter()
    self_generated = exporter.build_record(request=_request(), output=_output(), roi_before=np.zeros((2, 2, 3), dtype=np.float32))
    external = exporter.build_record(
        request=_request(),
        output=_output(),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
        roi_after=np.ones((2, 2, 3), dtype=np.float32) * 0.5,
    )
    path = tmp_path / "renderer_manifest.json"

    exporter.write_manifest([self_generated, external], str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["manifest_type"] == "renderer_patch_manifest"
    assert payload["contract_version"] == "renderer_patch_manifest.v1"
    assert payload["record_count"] == 2
    assert payload["target_quality_counts"] == {"self_generated_runtime_target": 1, "external_or_observed_target": 1}
    assert payload["contains_self_generated_targets"] is True
    assert payload["contains_external_targets"] is True


def test_exporter_execution_trace_summary_is_safe_and_canonical() -> None:
    trace = {
        "selected_render_strategy": "CANONICAL_RENDERER",
        "planner_selected_strategy": "PLANNER_ROUTE",
        "selected_strategy": "LEGACY_ALIAS",
        "selected_execution_strategy": "LEGACY_EXECUTION",
        "renderer_path": "learned_primary",
        "synthesis_mode": "learned_primary",
        "confidence_semantics_by_mode": {"learned_primary": "calibrated_renderer_confidence"},
        "uncertainty_semantics_by_mode": {"learned_primary": "per_pixel_uncertainty"},
        "learnable_mode_surface": {"large_tensor_like": [[1, 2], [3, 4]], "mode": "x"},
    }
    memory_bundle = {
        "memory_bundle_present": True,
        "memory_support_level": "strong",
        "has_current_reuse": True,
        "has_identity_reference": True,
        "has_appearance_reference": True,
        "has_garment_reference": False,
        "has_hidden_slot": True,
        "hidden_type": "known_hidden",
        "hidden_support_active": True,
        "reveal_lifecycle": "stable",
        "retrieval_reasons": ["identity_reference"],
    }

    record = RendererManifestRecordExporter().build_record(
        request=_request(transition_context={"region_memory_bundle_serialized": memory_bundle}),
        output=_output(trace=trace),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    summary = record["execution_trace_summary"]
    assert summary["selected_render_strategy"] == "CANONICAL_RENDERER"
    assert summary["planner_selected_strategy"] == "PLANNER_ROUTE"
    assert summary["memory_bundle_present"] is True
    assert summary["memory_bundle_hidden_support_active"] is True
    assert "selected_strategy" not in summary
    assert "selected_execution_strategy" not in summary
    assert summary["learnable_mode_surface"] == {"keys": ["large_tensor_like", "mode"]}


def test_renderer_dataset_accepts_renderer_patch_manifest_type_strict(tmp_path: Path) -> None:
    path = tmp_path / "renderer_patch_manifest.json"
    path.write_text(json.dumps({"manifest_type": "renderer_patch_manifest", "records": [_record()]}), encoding="utf-8")

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)

    assert len(ds.samples) == 1
    assert ds.diagnostics["manifest_type"] == "renderer_patch_manifest"


def test_renderer_dataset_unknown_manifest_type_non_strict_warns(tmp_path: Path) -> None:
    path = tmp_path / "weird_manifest.json"
    path.write_text(json.dumps({"manifest_type": "weird_renderer_manifest", "records": [_record()]}), encoding="utf-8")

    ds = RendererDataset.from_renderer_manifest(str(path), strict=False)

    assert len(ds.samples) == 1
    assert ds.diagnostics["unknown_manifest_type"] == 1
    assert ds.diagnostics["unsupported_manifest_type"] == 1
    assert ds.diagnostics["warnings"][0]["type"] == "unknown_manifest_type"


def test_eval_strategy_compatibility_semantics() -> None:
    canonical = build_patch_eval_payload({"selected_render_strategy": "CANONICAL", "selected_strategy": "LEGACY"})
    canonical_only = build_patch_eval_payload({"selected_render_strategy": "CANONICAL"})
    canonical_unknown = build_patch_eval_payload({"selected_render_strategy": "unknown", "selected_strategy": "LEGACY"})
    unknown_only = build_patch_eval_payload({"selected_render_strategy": "unknown"})
    legacy_only = build_patch_eval_payload({"selected_strategy": "LEGACY"})
    legacy_equivalent = build_patch_eval_payload({"selected_render_strategy": "LEGACY"})

    assert canonical == canonical_only
    assert canonical_unknown == unknown_only
    assert legacy_only == legacy_equivalent


def test_adapter_absent_memory_bundle_has_full_safe_summary_defaults(tmp_path: Path) -> None:
    path = tmp_path / "old_renderer_manifest.json"
    path.write_text(json.dumps({"records": [_record()]}), encoding="utf-8")

    ds = RendererDataset.from_renderer_manifest(str(path), strict=False)
    batch = RendererBatchAdapter().adapt(ds.samples[0])

    expected = {
        "memory_bundle_present": False,
        "memory_support_level": "none",
        "memory_bundle_reveal_lifecycle": "unknown",
        "memory_bundle_has_current_reuse": False,
        "memory_bundle_has_identity_reference": False,
        "memory_bundle_has_appearance_reference": False,
        "memory_bundle_has_garment_reference": False,
        "memory_bundle_has_hidden_slot": False,
        "memory_bundle_hidden_type": "none",
        "memory_bundle_hidden_support_active": False,
        "memory_bundle_retrieval_reasons": [],
        "memory_bundle_is_revealed_history": False,
        "memory_bundle_low_evidence_newly_revealed": False,
    }
    for key, value in expected.items():
        assert key in batch.conditioning_summary
        assert batch.conditioning_summary[key] == value


def test_orchestrator_exporter_single_instance_pattern_if_easy() -> None:
    source = Path("src/runtime/orchestrator.py").read_text(encoding="utf-8")

    assert "renderer_manifest_exporter = RendererManifestRecordExporter()" in source
    assert "RendererManifestRecordExporter().build_record" not in source
    assert '"renderer_manifest_export"' in source
