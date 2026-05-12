from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.schema import BBox, RegionRef, SceneGraph
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from rendering.trainable_patch_renderer import build_patch_batch
from training.datasets import RendererDataset
from training.renderer_manifest_exporter import RendererManifestRecordExporter
from training.renderer_trainer import RendererBatchAdapter


def _metadata(*, fallback: bool = False) -> dict[str, object]:
    if fallback:
        return {
            "region_id": "person_0:torso",
            "entity_id": "person_0",
            "canonical_region": "torso",
            "bbox_xywh": [0.1, 0.1, 0.5, 0.6],
            "roi_source": "person_bbox_fallback",
            "source_node_type": "fallback",
            "mask_kind": "",
            "metadata_completeness_score": 0.25,
            "evidence_strength_score": 0.1,
            "missing_fields": ["mask_ref", "mask_kind"],
        }
    return {
        "region_id": "person_0:torso",
        "entity_id": "person_0",
        "canonical_region": "torso",
        "bbox_xywh": [0.1, 0.1, 0.5, 0.6],
        "roi_reason": "parser mask region mask_ref=mask_1",
        "roi_source": "parser_mask_bbox",
        "source_node_type": "body_part",
        "source": "parser:fashn",
        "mask_ref": "mask_1",
        "mask_kind": "body_part_mask",
        "parser_class_name": "upper_clothes",
        "metadata_completeness_score": 0.92,
        "evidence_strength_score": 0.87,
        "missing_fields": [],
    }


def _request(*, fallback: bool = False) -> PatchSynthesisRequest:
    metadata = _metadata(fallback=fallback)
    return PatchSynthesisRequest(
        region=RegionRef(region_id="person_0:torso", bbox=BBox(0.1, 0.1, 0.5, 0.6), reason=str(metadata.get("roi_reason", "fallback person bbox"))),
        scene_state=SceneGraph(frame_index=3),
        memory_summary={"entities": ["person_0"]},
        transition_context={
            "frame_index": 3,
            "step_index": 2,
            "graph_delta": {
                "affected_entities": ["person_0"],
                "affected_regions": ["torso"],
                "region_transition_mode": {"torso": "garment_reveal"},
                "transition_phase": "contact_or_reveal",
                "newly_revealed_regions": ["person_0:torso"],
                "semantic_reasons": ["open jacket reveals torso"],
            },
            "region_memory_bundle_serialized": {"memory_bundle_present": True, "memory_support_level": "medium", "retrieval_reasons": ["appearance_match"]},
            "target_profile": {"primary_regions": ["torso"], "secondary_regions": [], "context_regions": ["left_arm"]},
            "region_route_decision": {"decision": "render", "reasons": ["semantic_mask"]},
        },
        retrieval_summary={"reasons": ["appearance_match"]},
        current_frame=np.zeros((4, 4, 3), dtype=np.float32).tolist(),
        region_metadata=metadata,
    )


def _output() -> PatchSynthesisOutput:
    return PatchSynthesisOutput(
        region=RegionRef(region_id="person_0:torso", bbox=BBox(0.1, 0.1, 0.5, 0.6), reason="parser mask region"),
        rgb_patch=np.full((4, 4, 3), 0.6, dtype=np.float32).tolist(),
        alpha_mask=np.ones((4, 4), dtype=np.float32).tolist(),
        height=4,
        width=4,
        channels=3,
        confidence=0.8,
        execution_trace={"selected_render_strategy": "LEARNED_GARMENT_REVEAL", "renderer_path": "learned_primary", "synthesis_mode": "learned_primary"},
    )


def _write_manifest(tmp_path: Path, record: dict[str, object]) -> Path:
    path = tmp_path / "renderer_v2.json"
    RendererManifestRecordExporter().write_manifest([record], str(path))
    return path


def test_runtime_exporter_writes_manifest_v2_record_with_region_metadata() -> None:
    record = RendererManifestRecordExporter().build_record(request=_request(), output=_output(), roi_before=np.zeros((4, 4, 3), dtype=np.float32), frame_index=3, step_index=2)

    assert record["contract_version"] == "renderer_patch_manifest_v2"
    assert record["region_metadata"]["mask_kind"] == "body_part_mask"
    assert record["metadata_completeness_score"] == pytest.approx(0.92)
    assert record["evidence_strength_score"] == pytest.approx(0.87)
    assert record["roi_source"] == "parser_mask_bbox"
    assert record["source_node_type"] == "body_part"
    assert record["mask_kind"] == "body_part_mask"
    json.dumps(record)


def test_explicit_observed_roi_after_is_supervised_external_target(tmp_path: Path) -> None:
    roi_after = np.full((4, 4, 3), 0.35, dtype=np.float32)
    roi_after[1:3, 1:3, :] = [0.8, 0.4, 0.25]
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(),
        roi_before=np.zeros((4, 4, 3), dtype=np.float32),
        roi_after=roi_after,
        frame_index=3,
        step_index=2,
    )

    assert record["target_source"] == "provided_ground_truth_roi"
    assert record["training_target_quality"] == "external_or_observed_target"

    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)
    assert ds.samples[0]["target_training_role"] == "supervised_external"
    assert ds.diagnostics["target_training_role_counts"]["supervised_external"] == 1

def test_dataset_loader_reads_v2_and_reconstructs_request_with_region_metadata(tmp_path: Path) -> None:
    record = RendererManifestRecordExporter().build_record(request=_request(), output=_output(), roi_before=np.zeros((4, 4, 3), dtype=np.float32), frame_index=3, step_index=2)
    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)

    request = RendererDataset.sample_to_patch_request(ds.samples[0])
    assert request.region_metadata["mask_kind"] == "body_part_mask"
    assert request.region_metadata["metadata_completeness_score"] == pytest.approx(0.92)


def test_build_patch_batch_from_loaded_sample_uses_region_metadata(tmp_path: Path) -> None:
    record = RendererManifestRecordExporter().build_record(request=_request(), output=_output(), roi_before=np.zeros((4, 4, 3), dtype=np.float32), frame_index=3, step_index=2)
    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)
    request = RendererDataset.sample_to_patch_request(ds.samples[0])
    batch = build_patch_batch(request, np.asarray(ds.samples[0]["roi_pairs"][0][0], dtype=np.float32))
    adapted = RendererBatchAdapter().adapt(ds.samples[0])

    assert batch.conditioning_summary["region_metadata_used"] is True
    assert "has_parser_mask" in batch.conditioning_summary["metadata_feature_keys"]
    assert adapted.memory_cond.shape == batch.memory_cond.shape
    assert adapted.appearance_cond.shape == batch.appearance_cond.shape


def test_fallback_roi_record_is_valid_and_loadable(tmp_path: Path) -> None:
    record = RendererManifestRecordExporter().build_record(request=_request(fallback=True), output=_output(), roi_before=np.zeros((4, 4, 3), dtype=np.float32), frame_index=3, step_index=2)
    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)

    assert record["roi_source"] == "person_bbox_fallback"
    assert record["evidence_strength_score"] < 0.2
    assert record["mask_ref_present"] is False
    assert record["mask_kind"] == ""
    assert len(ds.samples) == 1


def test_legacy_minimal_manifest_loads_with_low_completeness(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"records": [{"roi_before": np.zeros((2, 2, 3)).tolist(), "roi_after": np.ones((2, 2, 3)).tolist(), "region_id": "person_0:torso"}]}), encoding="utf-8")

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    request = RendererDataset.sample_to_patch_request(ds.samples[0])
    assert request.region_metadata["metadata_completeness_score"] == 0.0
    assert "region_metadata" in request.region_metadata["missing_fields"]


def test_sample_to_patch_request_accepts_dict_bbox_xywh(tmp_path: Path) -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(),
        roi_before=np.zeros((4, 4, 3), dtype=np.float32),
        frame_index=7,
        step_index=4,
    )
    record["region_metadata"]["bbox_xywh"] = {"x": 0.23, "y": 0.34, "w": 0.45, "h": 0.56}
    record["patch_synthesis_contract"]["region_metadata"] = record["region_metadata"]
    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)

    request = RendererDataset.sample_to_patch_request(ds.samples[0])

    assert request.region.bbox.x == pytest.approx(0.23)
    assert request.region.bbox.y == pytest.approx(0.34)
    assert request.region.bbox.w == pytest.approx(0.45)
    assert request.region.bbox.h == pytest.approx(0.56)


def test_dataset_loader_preserves_frame_and_step_indexes(tmp_path: Path) -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(),
        roi_before=np.zeros((4, 4, 3), dtype=np.float32),
        frame_index=11,
        step_index=6,
    )
    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)

    assert ds.samples[0]["frame_index"] == 11
    assert ds.samples[0]["step_index"] == 6
    request = RendererDataset.sample_to_patch_request(ds.samples[0])
    assert request.scene_state.frame_index == 11


def test_dataset_conditioning_summary_does_not_fake_active_metadata_feature_keys(tmp_path: Path) -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(),
        roi_before=np.zeros((4, 4, 3), dtype=np.float32),
        frame_index=3,
        step_index=2,
    )
    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)

    summary = ds.samples[0]["renderer_batch_contract"]["conditioning_summary"]

    assert "metadata_feature_keys" not in summary
    assert "metadata_raw_keys" in summary
    assert "mask_kind" in summary["metadata_raw_keys"]


def test_renderer_batch_adapter_applies_region_metadata_to_conditioning_vectors(tmp_path: Path) -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output(),
        roi_before=np.zeros((4, 4, 3), dtype=np.float32),
        frame_index=3,
        step_index=2,
    )
    record["region_metadata"]["source_confidence"] = 0.73
    record["region_metadata"]["evidence_strength_score"] = 0.91
    record["evidence_strength_score"] = 0.91
    record["patch_synthesis_contract"]["region_metadata"] = record["region_metadata"]
    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)
    adapted = RendererBatchAdapter().adapt(ds.samples[0])

    legacy_path = tmp_path / "legacy_for_adapter.json"
    legacy_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "roi_before": np.zeros((4, 4, 3), dtype=np.float32).tolist(),
                        "roi_after": np.full((4, 4, 3), 0.6, dtype=np.float32).tolist(),
                        "region_id": "person_0:torso",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    legacy = RendererBatchAdapter().adapt(RendererDataset.from_renderer_manifest(str(legacy_path), strict=True).samples[0])

    assert adapted.conditioning_summary["region_metadata_used"] is True
    assert "metadata_feature_keys" not in adapted.conditioning_summary
    assert "metadata_raw_keys" in adapted.conditioning_summary
    assert "metadata_active_feature_keys" in adapted.conditioning_summary
    assert "has_parser_mask" in adapted.conditioning_summary["metadata_active_feature_keys"]
    assert adapted.memory_cond[7] == pytest.approx(1.0)
    assert not np.allclose(adapted.memory_cond, legacy.memory_cond)
    assert not np.allclose(adapted.appearance_cond, legacy.appearance_cond)


def test_renderer_batch_adapter_fallback_metadata_does_not_apply_parser_mask_boost(tmp_path: Path) -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(fallback=True),
        output=_output(),
        roi_before=np.zeros((4, 4, 3), dtype=np.float32),
        frame_index=3,
        step_index=2,
    )
    ds = RendererDataset.from_renderer_manifest(str(_write_manifest(tmp_path, record)), strict=True)
    adapted = RendererBatchAdapter().adapt(ds.samples[0])

    assert adapted.conditioning_summary["roi_source"] == "person_bbox_fallback"
    assert adapted.conditioning_summary["mask_ref_present"] is False
    assert adapted.memory_cond[7] < 1.0
