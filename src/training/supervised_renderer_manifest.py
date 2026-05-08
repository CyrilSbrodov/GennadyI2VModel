from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SUPERVISED_TARGET_SOURCE = "provided_ground_truth_roi"
_SUPERVISED_TARGET_QUALITY = "external_or_observed_target"
_SUPERVISED_TARGET_ROLE = "supervised_external"

SUPERVISED_RENDERER_SEMANTIC_FAMILIES = {
    "face_expression",
    "torso_reveal",
    "sleeve_arm_transition",
}


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
    for index, record in enumerate(normalized_records):
        if record.get("target_source") != _SUPERVISED_TARGET_SOURCE:
            raise ValueError(
                f"record {index} is not a supervised renderer record: "
                f"target_source must be {_SUPERVISED_TARGET_SOURCE!r}"
            )
        if record.get("training_target_quality") != _SUPERVISED_TARGET_QUALITY:
            raise ValueError(
                f"record {index} is not a supervised renderer record: "
                f"training_target_quality must be {_SUPERVISED_TARGET_QUALITY!r}"
            )
        if record.get("target_training_role") != _SUPERVISED_TARGET_ROLE:
            raise ValueError(
                f"record {index} is not a supervised renderer record: "
                f"target_training_role must be {_SUPERVISED_TARGET_ROLE!r}"
            )

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
        "contains_external_targets": bool(normalized_records),
        "contains_self_generated_targets": False,
        "records": normalized_records,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)
