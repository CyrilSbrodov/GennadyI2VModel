from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from core.schema import BBox, RegionRef, SceneGraph
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from training.renderer_manifest_exporter import RendererManifestRecordExporter


BBoxUnits = Literal["auto", "normalized", "absolute_pixels"]
_REGION_METADATA_HINT_FIELDS = {
    "roi_source",
    "source_node_type",
    "metadata_completeness_score",
    "evidence_strength_score",
    "mask_ref",
    "canonical_region",
    "entity_id",
    "region_id",
    "bbox_xywh",
}


@dataclass(slots=True)
class RendererVideoManifestBuildResult:
    """Records plus builder diagnostics for observed-frame supervised manifests."""

    records: list[dict[str, object]]
    diagnostics: dict[str, object]


@dataclass(slots=True)
class RendererVideoManifestBuilder:
    """Build renderer_patch_manifest_v2 records from observed source/target frame pairs.

    This builder only packages observed paired-frame ROIs as supervised external targets.
    It delegates all v2 record serialization to RendererManifestRecordExporter.
    """

    exporter: RendererManifestRecordExporter = field(default_factory=RendererManifestRecordExporter)

    def build_records(
        self,
        *,
        source_frame: list | np.ndarray,
        target_frame: list | np.ndarray,
        regions: RegionRef | list[RegionRef],
        scene_graph: SceneGraph | dict[str, object] | None = None,
        region_metadata: dict[str, object] | None = None,
        transition_context: dict[str, object] | None = None,
        target_regions: RegionRef | list[RegionRef] | dict[str, RegionRef] | None = None,
        target_bboxes: BBox | dict[str, BBox] | None = None,
        bbox_units: BBoxUnits = "auto",
        target_bbox_units: BBoxUnits = "auto",
        frame_index: int | None = None,
        step_index: int | None = None,
        strict: bool = True,
    ) -> RendererVideoManifestBuildResult:
        """Build supervised v2 records without fabricating synthetic targets.

        roi_before is cropped from source_frame. roi_after is cropped from target_frame
        using the same region bbox unless a target region/bbox is explicitly supplied.
        """

        src = _as_frame_array(source_frame, "source_frame")
        tgt = _as_frame_array(target_frame, "target_frame")
        region_list = _normalize_regions(regions)
        target_region_map = _normalize_target_regions(target_regions)
        target_bbox_map = _normalize_target_bboxes(target_bboxes)
        effective_target_bbox_units = target_bbox_units
        if target_bbox_units == "auto" and target_regions is None and target_bboxes is None and bbox_units != "auto":
            effective_target_bbox_units = bbox_units
        ctx_base = dict(transition_context) if isinstance(transition_context, dict) else {}
        scene = _coerce_scene_graph(scene_graph, frame_index=frame_index if frame_index is not None else _safe_int(ctx_base.get("frame_index"), 0))
        resolved_frame_index = int(frame_index if frame_index is not None else ctx_base.get("frame_index", scene.frame_index))
        resolved_step_index = int(step_index if step_index is not None else ctx_base.get("step_index", -1))

        records: list[dict[str, object]] = []
        invalid_examples: list[dict[str, object]] = []
        completeness_sum = 0.0
        fallback_count = 0
        supervised_external_count = 0
        skipped = 0

        for index, region in enumerate(region_list):
            try:
                roi_before = _crop_region(src, region.bbox, "source_frame", bbox_units=bbox_units)
                target_bbox = _target_bbox_for(region, target_region_map, target_bbox_map)
                roi_after = _crop_region(tgt, target_bbox, "target_frame", bbox_units=effective_target_bbox_units)
                roi_before, roi_after = _match_roi_shapes(roi_before, roi_after)

                metadata = _metadata_for_region(region_metadata, region, total_regions=len(region_list))
                fallback = _is_fallback_person_bbox(region, metadata)
                if fallback:
                    metadata["roi_source"] = "person_bbox_fallback"
                    metadata.setdefault("source_node_type", "fallback")
                    metadata["metadata_completeness_score"] = min(_safe_float(metadata.get("metadata_completeness_score"), 0.0), 0.2)
                    metadata["evidence_strength_score"] = min(_safe_float(metadata.get("evidence_strength_score"), 0.0), 0.1)
                    missing = metadata.get("missing_fields") if isinstance(metadata.get("missing_fields"), list) else []
                    metadata["missing_fields"] = sorted({*map(str, missing), "mask_ref", "semantic_region_metadata"})

                ctx = dict(ctx_base)
                ctx.update(
                    {
                        "frame_index": resolved_frame_index,
                        "step_index": resolved_step_index,
                        "target_source": "provided_ground_truth_roi",
                        "training_target_quality": "external_or_observed_target",
                        "observed_pair_builder": True,
                    }
                )
                request = PatchSynthesisRequest(
                    region=region,
                    scene_state=scene,
                    memory_summary={},
                    transition_context=ctx,
                    retrieval_summary={"source": "observed_frame_pair", "region_id": region.region_id},
                    current_frame=src.tolist(),
                    region_metadata=metadata,
                )
                output = PatchSynthesisOutput(
                    region=region,
                    rgb_patch=roi_after.tolist(),
                    alpha_mask=_ones_mask(roi_after.shape[0], roi_after.shape[1]),
                    height=int(roi_after.shape[0]),
                    width=int(roi_after.shape[1]),
                    channels=3,
                    confidence=1.0,
                    execution_trace={
                        "renderer_path": "supervised_observed_frame_pair",
                        "selected_render_strategy": "SUPERVISED_EXTERNAL_OBSERVED_ROI",
                        "synthesis_mode": "observed_frame_pair_roi",
                    },
                    metadata={"renderer_path": "supervised_observed_frame_pair"},
                )
                record = self.exporter.build_record(
                    request=request,
                    output=output,
                    roi_before=roi_before,
                    roi_after=roi_after,
                    frame_index=resolved_frame_index,
                    step_index=resolved_step_index,
                )
                record["source"] = "observed_frame_pair"
                records.append(record)
                completeness_sum += _safe_float(record.get("metadata_completeness_score"), 0.0)
                if fallback:
                    fallback_count += 1
                if record.get("target_source") == "provided_ground_truth_roi" and record.get("training_target_quality") == "external_or_observed_target":
                    supervised_external_count += 1
            except Exception as exc:  # keep non-strict mode useful for batch manifest building
                skipped += 1
                region_id = str(getattr(region, "region_id", "unknown"))
                if len(invalid_examples) < 8:
                    invalid_examples.append({"index": index, "region_id": region_id, "error": str(exc)})
                if strict:
                    raise ValueError(f"renderer observed-pair region index={index} region_id={region_id!r} is invalid: {exc}") from exc

        diagnostics: dict[str, object] = {
            "total_regions": len(region_list),
            "exported_records": len(records),
            "skipped_regions": skipped,
            "supervised_external_count": supervised_external_count,
            "average_metadata_completeness_score": round(completeness_sum / max(1, len(records)), 6),
            "fallback_person_bbox_record_count": fallback_count,
            "invalid_examples": invalid_examples,
        }
        if strict and supervised_external_count == 0:
            raise ValueError(f"no valid supervised_external renderer records were exported: {diagnostics}")
        return RendererVideoManifestBuildResult(records=records, diagnostics=diagnostics)

    def write_manifest(self, records: list[dict[str, object]], path: str, *, diagnostics: dict[str, object] | None = None) -> None:
        self.exporter.write_manifest(records, path)
        if diagnostics is not None:
            import json

            manifest_path = Path(path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["source"] = "observed_frame_pair"
            payload["builder_diagnostics"] = diagnostics
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_manifest(
        self,
        *,
        output_path: str,
        bbox_units: BBoxUnits = "auto",
        target_bbox_units: BBoxUnits = "auto",
        **kwargs: Any,
    ) -> RendererVideoManifestBuildResult:
        result = self.build_records(bbox_units=bbox_units, target_bbox_units=target_bbox_units, **kwargs)
        self.write_manifest(result.records, output_path, diagnostics=result.diagnostics)
        return result


def build_renderer_video_manifest(
    *,
    output_path: str,
    bbox_units: BBoxUnits = "auto",
    target_bbox_units: BBoxUnits = "auto",
    **kwargs: Any,
) -> RendererVideoManifestBuildResult:
    """Convenience wrapper for building and writing an observed-pair supervised manifest."""

    return RendererVideoManifestBuilder().build_manifest(
        output_path=output_path,
        bbox_units=bbox_units,
        target_bbox_units=target_bbox_units,
        **kwargs,
    )


def _as_frame_array(frame: list | np.ndarray, field_name: str) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{field_name} must be HxWx3-compatible, got shape={list(arr.shape)}")
    if arr.size and float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} contains NaN or inf")
    return np.clip(arr, 0.0, 1.0)


def _normalize_regions(regions: RegionRef | list[RegionRef]) -> list[RegionRef]:
    if isinstance(regions, RegionRef):
        return [regions]
    if isinstance(regions, list) and all(isinstance(item, RegionRef) for item in regions):
        return list(regions)
    raise ValueError("regions must be a RegionRef or list[RegionRef]")


def _normalize_target_regions(target_regions: RegionRef | list[RegionRef] | dict[str, RegionRef] | None) -> dict[str, RegionRef]:
    if target_regions is None:
        return {}
    if isinstance(target_regions, RegionRef):
        return {target_regions.region_id: target_regions}
    if isinstance(target_regions, list):
        if not all(isinstance(item, RegionRef) for item in target_regions):
            raise ValueError("target_regions list must contain RegionRef items")
        return {item.region_id: item for item in target_regions}
    if isinstance(target_regions, dict):
        invalid_keys = [str(k) for k, v in target_regions.items() if not isinstance(v, RegionRef)]
        if invalid_keys:
            raise ValueError(f"target_regions dict values must be RegionRef; invalid keys={invalid_keys}")
        return {str(k): v for k, v in target_regions.items()}
    raise ValueError("target_regions must be RegionRef, list[RegionRef], dict[str, RegionRef], or None")


def _normalize_target_bboxes(target_bboxes: BBox | dict[str, BBox] | None) -> dict[str, BBox]:
    if target_bboxes is None:
        return {}
    if isinstance(target_bboxes, BBox):
        return {"*": target_bboxes}
    if isinstance(target_bboxes, dict):
        invalid_keys = [str(k) for k, v in target_bboxes.items() if not isinstance(v, BBox)]
        if invalid_keys:
            raise ValueError(f"target_bboxes dict values must be BBox; invalid keys={invalid_keys}")
        return {str(k): v for k, v in target_bboxes.items()}
    raise ValueError("target_bboxes must be BBox, dict[str, BBox], or None")


def _target_bbox_for(region: RegionRef, target_region_map: dict[str, RegionRef], target_bbox_map: dict[str, BBox]) -> BBox:
    if region.region_id in target_region_map:
        return target_region_map[region.region_id].bbox
    if region.region_id in target_bbox_map:
        return target_bbox_map[region.region_id]
    if "*" in target_bbox_map:
        return target_bbox_map["*"]
    return region.bbox


def _crop_region(frame: np.ndarray, bbox: BBox, frame_name: str, *, bbox_units: BBoxUnits = "auto") -> np.ndarray:
    if bbox_units not in {"auto", "normalized", "absolute_pixels"}:
        raise ValueError(f"bbox_units must be 'auto', 'normalized', or 'absolute_pixels', got {bbox_units!r}")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    values = np.asarray([bbox.x, bbox.y, bbox.w, bbox.h], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{frame_name} bbox contains NaN or inf: {values.tolist()}")
    x, y, box_w, box_h = (float(v) for v in values.tolist())
    if box_w <= 0.0 or box_h <= 0.0:
        raise ValueError(f"{frame_name} bbox width/height must be > 0, got w={box_w}, h={box_h}")

    normalized = bbox_units == "normalized" or (bbox_units == "auto" and float(np.max(np.abs(values))) <= 1.5)
    if normalized:
        x0 = int(np.floor(x * width))
        y0 = int(np.floor(y * height))
        x1 = int(np.ceil((x + box_w) * width))
        y1 = int(np.ceil((y + box_h) * height))
    else:
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = int(np.ceil(x + box_w))
        y1 = int(np.ceil(y + box_h))

    clipped_x0 = max(0, min(width, x0))
    clipped_y0 = max(0, min(height, y0))
    clipped_x1 = max(0, min(width, x1))
    clipped_y1 = max(0, min(height, y1))
    if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
        mode = "normalized" if normalized else "absolute_pixel"
        raise ValueError(
            f"{frame_name} crop is empty for {mode} bbox={[bbox.x, bbox.y, bbox.w, bbox.h]} "
            f"on frame_shape={[height, width, int(frame.shape[2])]} after clipping="
            f"{[clipped_x0, clipped_y0, clipped_x1, clipped_y1]}"
        )
    roi = frame[clipped_y0:clipped_y1, clipped_x0:clipped_x1, :]
    if roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError(f"{frame_name} crop must be HxWx3, got shape={list(roi.shape)}")
    return roi


def _match_roi_shapes(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if before.shape == after.shape:
        return before, after
    target_h = max(1, min(int(before.shape[0]), int(after.shape[0])))
    target_w = max(1, min(int(before.shape[1]), int(after.shape[1])))
    return _resize_nearest(before, target_h, target_w), _resize_nearest(after, target_h, target_w)


def _resize_nearest(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    if int(arr.shape[0]) == height and int(arr.shape[1]) == width:
        return arr
    y_idx = np.linspace(0, arr.shape[0] - 1, height).round().astype(int)
    x_idx = np.linspace(0, arr.shape[1] - 1, width).round().astype(int)
    return arr[y_idx][:, x_idx, :]


def _metadata_for_region(region_metadata: dict[str, object] | None, region: RegionRef, *, total_regions: int) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if isinstance(region_metadata, dict) and region_metadata:
        by_region = region_metadata.get(region.region_id)
        if isinstance(by_region, dict) and _looks_like_region_metadata(by_region):
            metadata.update(by_region)
        elif total_regions == 1 and _looks_like_region_metadata(region_metadata):
            metadata.update(region_metadata)
    if not metadata:
        metadata = {
            "metadata_source": "observed_pair_builder_low_completeness_fallback",
            "metadata_completeness_score": 0.1,
            "evidence_strength_score": 0.05,
            "missing_fields": ["region_metadata", "mask_ref", "mask_kind"],
            "metadata_source_trace": ["builder:missing_region_metadata"],
        }
    metadata.setdefault("region_id", region.region_id)
    metadata.setdefault("entity_id", _entity_id(region.region_id))
    metadata.setdefault("canonical_region", _region_type(region.region_id))
    metadata.setdefault("bbox_xywh", [float(region.bbox.x), float(region.bbox.y), float(region.bbox.w), float(region.bbox.h)])
    metadata.setdefault("roi_reason", region.reason)
    metadata.setdefault("roi_source", "observed_region_ref")
    metadata.setdefault("source_node_type", "observed_context")
    metadata.setdefault("mask_kind", "")
    metadata.setdefault("metadata_completeness_score", 0.1)
    metadata.setdefault("evidence_strength_score", 0.05)
    metadata.setdefault("missing_fields", [])
    return metadata


def _looks_like_region_metadata(metadata: dict[str, object]) -> bool:
    return bool(_REGION_METADATA_HINT_FIELDS.intersection(metadata.keys()))


def _is_fallback_person_bbox(region: RegionRef, metadata: dict[str, object]) -> bool:
    reason = str(region.reason).lower()
    return "person_bbox" in reason or str(metadata.get("roi_source", "")).lower() == "person_bbox_fallback"


def _coerce_scene_graph(scene_graph: SceneGraph | dict[str, object] | None, *, frame_index: int) -> SceneGraph:
    if isinstance(scene_graph, SceneGraph):
        return scene_graph
    if isinstance(scene_graph, dict):
        return SceneGraph(frame_index=_safe_int(scene_graph.get("frame_index"), frame_index))
    return SceneGraph(frame_index=frame_index)


def _ones_mask(height: int, width: int) -> list[list[float]]:
    return [[1.0 for _ in range(width)] for _ in range(height)]


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _entity_id(region_id: str) -> str:
    return str(region_id).split(":", 1)[0] if ":" in str(region_id) else "unknown"


def _region_type(region_id: str) -> str:
    return str(region_id).split(":", 1)[1] if ":" in str(region_id) else str(region_id)
