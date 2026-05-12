from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.schema import BBox, RegionRef, SceneGraph
from training.datasets import RendererDataset
from training.renderer_video_manifest_builder import RendererVideoManifestBuilder, build_renderer_video_manifest


def _region(reason: str = "semantic:face") -> RegionRef:
    return RegionRef(region_id="person_1:face", bbox=BBox(0.25, 0.25, 0.5, 0.5), reason=reason)


def _frames() -> tuple[list, np.ndarray]:
    source = np.zeros((8, 8, 3), dtype=np.float32)
    source[:, :] = [0.1, 0.2, 0.3]
    target = source.copy()
    target[2:6, 2:6, :] = [0.8, 0.1, 0.2]
    return source.tolist(), target


def test_paired_frames_build_manifest_v2_and_dataset_strict_loads(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "observed_renderer_manifest.json"

    result = build_renderer_video_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        scene_graph=SceneGraph(frame_index=7),
        regions=_region(),
        region_metadata={
            "person_1:face": {
                "region_id": "person_1:face",
                "entity_id": "person_1",
                "canonical_region": "face",
                "bbox_xywh": [0.25, 0.25, 0.5, 0.5],
                "roi_source": "parser_mask_bbox",
                "source_node_type": "body_part",
                "mask_kind": "semantic",
                "mask_ref": "mask://face/1",
                "metadata_completeness_score": 0.9,
                "evidence_strength_score": 0.85,
            }
        },
        transition_context={"summary": "observed smile change", "graph_delta": {"affected_regions": ["face"]}},
        frame_index=7,
        step_index=1,
        strict=True,
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    record = manifest["records"][0]
    assert manifest["contract_version"] == "renderer_patch_manifest_v2"
    assert result.diagnostics["total_regions"] == 1
    assert result.diagnostics["exported_records"] == 1
    assert result.diagnostics["supervised_external_count"] == 1
    assert record["target_source"] == "provided_ground_truth_roi"
    assert record["training_target_quality"] == "external_or_observed_target"

    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    assert len(ds.samples) == 1
    assert ds.samples[0]["target_training_role"] == "supervised_external"
    before, after = ds.samples[0]["roi_pairs"][0]
    assert np.mean(np.abs(np.asarray(after) - np.asarray(before))) > 0.05


def test_missing_metadata_still_creates_low_completeness_valid_v2_record(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "low_metadata_manifest.json"

    result = RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=_region(),
        strict=True,
    )

    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    assert result.diagnostics["average_metadata_completeness_score"] <= 0.1
    assert record["metadata_completeness_score"] <= 0.1
    assert record["region_metadata"]["missing_fields"] == ["region_metadata", "mask_ref", "mask_kind"]
    ds = RendererDataset.from_renderer_manifest(str(path), strict=True)
    assert ds.samples[0]["target_training_role"] == "supervised_external"


def test_unrelated_flat_metadata_uses_low_completeness_fallback(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "unrelated_metadata_manifest.json"

    RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=_region(),
        region_metadata={"foo": "bar"},
        strict=True,
    )

    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    assert record["metadata_completeness_score"] <= 0.1
    assert record["evidence_strength_score"] <= 0.05
    assert "foo" not in record["region_metadata"]
    assert "region_metadata" in record["region_metadata"]["missing_fields"]
    assert "mask_ref" in record["region_metadata"]["missing_fields"]
    assert "mask_kind" in record["region_metadata"]["missing_fields"]


def test_fallback_person_bbox_record_is_valid_but_low_evidence(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "fallback_manifest.json"

    result = RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=_region("fallback:person_bbox_template"),
        region_metadata={"person_1:face": {"roi_source": "person_bbox_fallback", "metadata_completeness_score": 0.8, "evidence_strength_score": 0.8}},
        strict=True,
    )

    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    assert result.diagnostics["fallback_person_bbox_record_count"] == 1
    assert record["roi_source"] == "person_bbox_fallback"
    assert record["metadata_completeness_score"] <= 0.2
    assert record["evidence_strength_score"] <= 0.1
    assert RendererDataset.from_renderer_manifest(str(path), strict=True).samples[0]["target_training_role"] == "supervised_external"


def test_target_bbox_resizes_roi_pairs_to_common_shape(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "target_bbox_manifest.json"

    RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=_region(),
        target_bboxes={"person_1:face": BBox(0.25, 0.25, 0.25, 0.25)},
        strict=True,
    )

    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    assert np.asarray(record["roi_before"]).shape == np.asarray(record["roi_after"]).shape
    assert np.asarray(record["roi_before"]).shape[:2] == (2, 2)


def test_absolute_pixel_bbox_is_supported(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "absolute_bbox_manifest.json"

    RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=RegionRef(region_id="person_1:face", bbox=BBox(2, 2, 4, 4), reason="parser_mask_bbox:absolute_pixels"),
        strict=True,
    )

    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    assert np.asarray(record["roi_before"]).shape == (4, 4, 3)
    assert np.asarray(record["roi_after"]).shape == (4, 4, 3)


def test_bbox_units_absolute_pixels_override_disables_auto_normalized_interpretation(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "bbox_units_absolute_manifest.json"

    RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=RegionRef(region_id="person_1:face", bbox=BBox(1, 1, 1, 1), reason="absolute_pixel_override"),
        bbox_units="absolute_pixels",
        target_bbox_units="absolute_pixels",
        strict=True,
    )

    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    assert np.asarray(record["roi_before"]).shape == (1, 1, 3)
    assert np.asarray(record["roi_after"]).shape == (1, 1, 3)


def test_target_bbox_units_inherit_source_units_when_target_bbox_not_explicit(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "bbox_units_inherit_manifest.json"

    RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=RegionRef(region_id="person_1:face", bbox=BBox(1, 1, 1, 1), reason="absolute_pixel_inherit"),
        bbox_units="absolute_pixels",
        strict=True,
    )

    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    assert np.asarray(record["roi_before"]).shape == (1, 1, 3)
    assert np.asarray(record["roi_after"]).shape == (1, 1, 3)


def test_explicit_target_bbox_units_take_precedence(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "target_bbox_units_precedence_manifest.json"

    RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=RegionRef(region_id="person_1:face", bbox=BBox(2, 2, 4, 4), reason="absolute_pixel_source"),
        target_bboxes={"person_1:face": BBox(0.25, 0.25, 0.5, 0.5)},
        bbox_units="absolute_pixels",
        target_bbox_units="normalized",
        strict=True,
    )

    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    assert np.asarray(record["roi_before"]).shape == (4, 4, 3)
    assert np.asarray(record["roi_after"]).shape == (4, 4, 3)


def test_invalid_target_regions_mapping_raises(tmp_path: Path) -> None:
    source, target = _frames()

    with pytest.raises(ValueError, match="target_regions dict values must be RegionRef.*person_1:face"):
        RendererVideoManifestBuilder().build_manifest(
            output_path=str(tmp_path / "invalid_target_regions.json"),
            source_frame=source,
            target_frame=target,
            regions=_region(),
            target_regions={"person_1:face": object()},
            strict=True,
        )


def test_invalid_target_bboxes_mapping_raises(tmp_path: Path) -> None:
    source, target = _frames()

    with pytest.raises(ValueError, match="target_bboxes dict values must be BBox.*person_1:face"):
        RendererVideoManifestBuilder().build_manifest(
            output_path=str(tmp_path / "invalid_target_bboxes.json"),
            source_frame=source,
            target_frame=target,
            regions=_region(),
            target_bboxes={"person_1:face": object()},
            strict=True,
        )


def test_strict_raises_on_invalid_region_even_when_another_region_is_valid(tmp_path: Path) -> None:
    source, target = _frames()
    invalid = RegionRef(region_id="person_1:bad", bbox=BBox(1.2, 1.2, 0.1, 0.1), reason="outside")

    with pytest.raises(ValueError, match="index=1.*person_1:bad.*crop is empty"):
        RendererVideoManifestBuilder().build_manifest(
            output_path=str(tmp_path / "invalid_manifest.json"),
            source_frame=source,
            target_frame=target,
            regions=[_region(), invalid],
            strict=True,
        )


def test_strict_false_exports_valid_regions_and_reports_invalid_examples(tmp_path: Path) -> None:
    source, target = _frames()
    path = tmp_path / "partial_manifest.json"
    invalid = RegionRef(region_id="person_1:bad", bbox=BBox(1.2, 1.2, 0.1, 0.1), reason="outside")

    result = RendererVideoManifestBuilder().build_manifest(
        output_path=str(path),
        source_frame=source,
        target_frame=target,
        regions=[_region(), invalid],
        strict=False,
    )

    assert result.diagnostics["exported_records"] == 1
    assert result.diagnostics["skipped_regions"] == 1
    assert result.diagnostics["invalid_examples"][0]["index"] == 1
    assert result.diagnostics["invalid_examples"][0]["region_id"] == "person_1:bad"
    assert len(json.loads(path.read_text(encoding="utf-8"))["records"]) == 1


def test_strict_empty_regions_raises_no_supervised_records(tmp_path: Path) -> None:
    source, target = _frames()

    with pytest.raises(ValueError, match="no valid supervised_external renderer records"):
        RendererVideoManifestBuilder().build_manifest(
            output_path=str(tmp_path / "empty_manifest.json"),
            source_frame=source,
            target_frame=target,
            regions=[],
            strict=True,
        )
