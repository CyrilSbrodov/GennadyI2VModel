from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


_SUPERVISED_TARGET_SOURCE = "provided_ground_truth_roi"
_SUPERVISED_TARGET_QUALITY = "external_or_observed_target"
_SUPERVISED_TARGET_ROLE = "supervised_external"

SUPERVISED_RENDERER_SEMANTIC_FAMILIES = {
    "face_expression",
    "torso_reveal",
    "sleeve_arm_transition",
}


def _compact_quality_report(report: dict[str, object]) -> dict[str, object]:
    return {
        "valid": bool(report.get("valid", False)),
        "errors": [str(x) for x in report.get("errors", [])] if isinstance(report.get("errors", []), list) else [],
        "warnings": [str(x) for x in report.get("warnings", [])] if isinstance(report.get("warnings", []), list) else [],
        "roi_height": int(report.get("roi_height", 0) or 0),
        "roi_width": int(report.get("roi_width", 0) or 0),
        "channels": int(report.get("channels", 0) or 0),
        "changed_ratio": float(report.get("changed_ratio", 0.0) or 0.0),
        "mean_abs_delta": float(report.get("mean_abs_delta", 0.0) or 0.0),
        "has_changed_mask": bool(report.get("has_changed_mask", False)),
        "has_alpha_target": bool(report.get("has_alpha_target", False)),
        "has_blend_hint": bool(report.get("has_blend_hint", False)),
        "semantic_family": str(report.get("semantic_family", "")),
        "region_id": str(report.get("region_id", "")),
    }


def _optional_mask_is_valid(value: object, height: int, width: int) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    if arr.ndim == 2:
        return bool(arr.shape == (height, width))
    return bool(arr.ndim == 3 and arr.shape[0] == height and arr.shape[1] == width and arr.shape[2] == 1)


def validate_supervised_renderer_record(record: dict[str, object], *, strict: bool = True) -> dict[str, object]:
    """Validate an external/observed supervised renderer ROI-pair record.

    The validator intentionally works only on JSON/list/numpy-compatible arrays already
    present in the manifest; it does not decode image files or mutate generation
    architecture contracts.
    """
    errors: list[str] = []
    warnings: list[str] = []
    semantic_family = str(record.get("semantic_family", "")).strip().lower()
    region_id = str(record.get("region_id", "")).strip()
    report: dict[str, object] = {
        "valid": False,
        "errors": errors,
        "warnings": warnings,
        "roi_height": 0,
        "roi_width": 0,
        "channels": 0,
        "changed_ratio": 0.0,
        "mean_abs_delta": 0.0,
        "has_changed_mask": "changed_mask" in record,
        "has_alpha_target": "alpha_target" in record,
        "has_blend_hint": "blend_hint" in record,
        "semantic_family": semantic_family,
        "region_id": region_id,
    }

    if "roi_before" not in record:
        errors.append("roi_before missing")
    if "roi_after" not in record:
        errors.append("roi_after missing")

    before: np.ndarray | None = None
    after: np.ndarray | None = None
    if "roi_before" in record:
        try:
            before = np.asarray(record["roi_before"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            errors.append(f"roi_before cannot be converted to numeric np array: {exc}")
    if "roi_after" in record:
        try:
            after = np.asarray(record["roi_after"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            errors.append(f"roi_after cannot be converted to numeric np array: {exc}")

    if before is not None:
        if before.ndim != 3 or before.shape[2] != 3:
            errors.append("roi_before not HxWx3")
        else:
            report["roi_height"] = int(before.shape[0])
            report["roi_width"] = int(before.shape[1])
            report["channels"] = int(before.shape[2])
    if after is not None and (after.ndim != 3 or after.shape[2] != 3):
        errors.append("roi_after not HxWx3")
    if before is not None and after is not None and before.shape != after.shape:
        errors.append("roi_before and roi_after shapes mismatch")
    if before is not None and before.ndim >= 2:
        height = int(before.shape[0])
        width = int(before.shape[1])
        if height <= 0 or width <= 0:
            errors.append("roi height/width must be positive")
        if height < 8 or width < 8:
            warnings.append("tiny ROI: H < 8 or W < 8")

    if before is not None and after is not None:
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            errors.append("roi_before/roi_after contains NaN or inf")
        elif before.ndim == 3 and after.ndim == 3 and before.shape == after.shape and before.shape[2] == 3 and before.shape[0] > 0 and before.shape[1] > 0:
            if float(np.min(before)) < 0.0 or float(np.max(before)) > 1.0 or float(np.min(after)) < 0.0 or float(np.max(after)) > 1.0:
                warnings.append("roi_before/roi_after values outside [0, 1] range")
            delta = np.mean(np.abs(after - before), axis=2)
            mean_abs_delta = float(np.mean(np.abs(after - before)))
            report["mean_abs_delta"] = mean_abs_delta
            changed_mask = record.get("changed_mask")
            if changed_mask is not None and _optional_mask_is_valid(changed_mask, int(before.shape[0]), int(before.shape[1])):
                mask = np.asarray(changed_mask, dtype=np.float32)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                changed_ratio = float(np.mean(mask > 0.05))
            else:
                changed_ratio = float(np.mean(delta > 0.03))
            report["changed_ratio"] = changed_ratio
            if mean_abs_delta < 1e-4:
                warnings.append("low motion: mean_abs_delta < 1e-4; target almost unchanged")
            if changed_ratio < 0.001:
                warnings.append("low changed ratio: changed_ratio < 0.001")

    if semantic_family not in SUPERVISED_RENDERER_SEMANTIC_FAMILIES:
        errors.append("semantic_family not in SUPERVISED_RENDERER_SEMANTIC_FAMILIES")
    if record.get("target_source") != _SUPERVISED_TARGET_SOURCE:
        errors.append(f"not a supervised renderer record: target_source must be {_SUPERVISED_TARGET_SOURCE!r}")
    if record.get("training_target_quality") != _SUPERVISED_TARGET_QUALITY:
        errors.append(f"not a supervised renderer record: training_target_quality must be {_SUPERVISED_TARGET_QUALITY!r}")
    if record.get("target_training_role") != _SUPERVISED_TARGET_ROLE:
        errors.append(f"not a supervised renderer record: target_training_role must be {_SUPERVISED_TARGET_ROLE!r}")

    bbox = record.get("bbox")
    if bbox is None:
        warnings.append("bbox missing")
    else:
        try:
            bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
            if bbox_arr.size != 4:
                warnings.append("bbox length != 4")
            elif not np.all(np.isfinite(bbox_arr)) or float(np.min(bbox_arr)) < 0.0 or float(np.max(bbox_arr)) > 1.0:
                warnings.append("bbox values outside [0, 1]")
        except (TypeError, ValueError):
            warnings.append("bbox length != 4")

    height = int(report["roi_height"] or 0)
    width = int(report["roi_width"] or 0)
    for field in ("changed_mask", "alpha_target", "blend_hint"):
        if field in record and (height <= 0 or width <= 0 or not _optional_mask_is_valid(record.get(field), height, width)):
            warnings.append(f"{field} exists but shape not HxW or HxWx1")

    report["valid"] = len(errors) == 0
    if strict and errors:
        raise ValueError("; ".join(errors))
    return report


def build_supervised_renderer_record(
    *,
    roi_before: list,
    roi_after: list,
    region_id: str,
    semantic_family: str,
    bbox: list[float] | None = None,
    changed_mask: list | None = None,
    alpha_target: list | None = None,
    blend_hint: list | None = None,
    selected_render_strategy: str = "supervised_external_target",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a renderer_patch_manifest.v1 record for an external/observed ROI pair.

    The target provenance labels intentionally identify these samples as provided
    external/observed targets. This helper does not emit legacy strategy aliases
    such as ``selected_strategy`` or ``selected_execution_strategy``.
    """
    family = str(semantic_family).strip().lower()
    if family not in SUPERVISED_RENDERER_SEMANTIC_FAMILIES:
        allowed = ", ".join(sorted(SUPERVISED_RENDERER_SEMANTIC_FAMILIES))
        raise ValueError(
            f"unsupported supervised renderer semantic_family={semantic_family!r}; allowed={allowed}"
        )

    record: dict[str, object] = {
        "roi_before": roi_before,
        "roi_after": roi_after,
        "region_id": str(region_id),
        "semantic_family": family,
        "selected_render_strategy": str(selected_render_strategy),
        "target_source": _SUPERVISED_TARGET_SOURCE,
        "training_target_quality": _SUPERVISED_TARGET_QUALITY,
        "target_training_role": _SUPERVISED_TARGET_ROLE,
        "training_target_origin": "external_observed_pair",
        "source": "supervised_renderer_pair",
        "metadata": dict(metadata or {}),
    }
    if bbox is not None:
        record["bbox"] = [float(x) for x in bbox]
    if changed_mask is not None:
        record["changed_mask"] = changed_mask
    if alpha_target is not None:
        record["alpha_target"] = alpha_target
    if blend_hint is not None:
        record["blend_hint"] = blend_hint
    return record


def write_supervised_renderer_manifest(
    records: list[dict[str, object]],
    output_path: str,
    *,
    source: str = "supervised_renderer_pairs",
) -> str:
    """Write a renderer_patch_manifest.v1 manifest for supervised ROI pair records."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_records = [dict(record) for record in records]
    quality_reports: list[dict[str, object]] = []
    for index, record in enumerate(normalized_records):
        try:
            report = validate_supervised_renderer_record(record, strict=True)
        except ValueError as exc:
            raise ValueError(f"record {index} is not a supervised renderer record: {exc}") from exc
        compact = _compact_quality_report(report)
        record["supervised_quality"] = compact
        quality_reports.append(compact)

    warning_counts: dict[str, int] = {}
    semantic_family_counts: dict[str, int] = {}
    for report in quality_reports:
        family = str(report.get("semantic_family", ""))
        if family:
            semantic_family_counts[family] = semantic_family_counts.get(family, 0) + 1
        for warning in report.get("warnings", []) if isinstance(report.get("warnings", []), list) else []:
            warning_text = str(warning)
            warning_counts[warning_text] = warning_counts.get(warning_text, 0) + 1
    heights = [int(r.get("roi_height", 0) or 0) for r in quality_reports]
    widths = [int(r.get("roi_width", 0) or 0) for r in quality_reports]
    mean_deltas = [float(r.get("mean_abs_delta", 0.0) or 0.0) for r in quality_reports]
    changed_ratios = [float(r.get("changed_ratio", 0.0) or 0.0) for r in quality_reports]

    manifest: dict[str, Any] = {
        "manifest_type": "renderer_patch_manifest",
        "contract_version": "renderer_patch_manifest.v1",
        "source": source,
        "target_quality_counts": {
            "external_or_observed_target": len(normalized_records),
            "self_generated_runtime_target": 0,
            "unknown": 0,
        },
        "record_count": len(normalized_records),
        "valid_record_count": sum(1 for report in quality_reports if bool(report.get("valid", False))),
        "invalid_record_count": sum(1 for report in quality_reports if not bool(report.get("valid", False))),
        "warning_record_count": sum(1 for report in quality_reports if bool(report.get("warnings", []))),
        "min_roi_height": min(heights) if heights else 0,
        "min_roi_width": min(widths) if widths else 0,
        "avg_mean_abs_delta": float(sum(mean_deltas) / max(1, len(mean_deltas))),
        "avg_changed_ratio": float(sum(changed_ratios) / max(1, len(changed_ratios))),
        "semantic_family_counts": semantic_family_counts,
        "quality_warnings_count_by_type": warning_counts,
        "contains_low_motion_records": any(
            float(report.get("mean_abs_delta", 0.0) or 0.0) < 1e-4 or float(report.get("changed_ratio", 0.0) or 0.0) < 0.001
            for report in quality_reports
        ),
        "contains_tiny_roi_records": any(int(report.get("roi_height", 0) or 0) < 8 or int(report.get("roi_width", 0) or 0) < 8 for report in quality_reports),
        "contains_external_targets": bool(normalized_records),
        "contains_self_generated_targets": False,
        "records": normalized_records,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)
