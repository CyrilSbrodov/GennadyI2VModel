from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from core.schema import BBox, RegionRef
from training.renderer_video_manifest_builder import _crop_region, _match_roi_shapes, _metadata_for_region


@dataclass(slots=True)
class ObservedPairsBuildOutput:
    manifest_path: str
    diagnostics: dict[str, object]


def build_renderer_manifest_from_observed_pairs(*, observed_pairs_path: str, output_path: str, strict: bool = True) -> ObservedPairsBuildOutput:
    """Build a lightweight supervised renderer manifest from observed image pairs.

    Observed-pair records intentionally externalize ROI tensors into .npy assets so
    manifest JSON stays small and does not pass full frames/ROIs through the generic
    renderer manifest exporter.
    """

    started = time.perf_counter()
    payload = json.loads(Path(observed_pairs_path).read_text(encoding="utf-8"))
    if str(payload.get("contract_version", "")).strip() != "renderer_observed_pair_manifest_input_v1":
        raise ValueError("observed pairs input must use contract_version='renderer_observed_pair_manifest_input_v1'")
    pairs = payload.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("observed pairs input field 'pairs' must be list")

    output = Path(output_path)
    asset_subdir = Path("renderer_roi_assets")
    asset_dir = output.parent / asset_subdir
    asset_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, object]] = []
    skipped_pairs = 0
    invalid_examples: list[dict[str, object]] = []
    roi_asset_count = 0
    roi_asset_total_bytes = 0

    for idx, pair in enumerate(pairs):
        try:
            if not isinstance(pair, dict):
                raise ValueError("pair must be object")
            record_id = str(pair.get("record_id", "")).strip() or f"observed_pair_{idx:06d}"
            source_path = str(pair.get("source_frame", "")).strip()
            target_path = str(pair.get("target_frame", "")).strip()
            if not source_path:
                raise ValueError("source_frame is required")
            if not target_path:
                raise ValueError("target_frame is required")
            source_frame = _read_image(source_path, "source_frame")
            target_frame = _read_image(target_path, "target_frame")
            if source_frame.shape != target_frame.shape:
                raise ValueError(f"source/target frame shapes mismatch: {list(source_frame.shape)} vs {list(target_frame.shape)}")

            regions = _pair_regions(pair)
            transition_context = dict(pair.get("transition_context") or {})
            transition_context.setdefault("summary", str(pair.get("prompt", "")).strip())
            transition_context["record_id"] = record_id
            if isinstance(pair.get("tags"), list):
                transition_context["tags"] = [str(x) for x in pair.get("tags", [])]
            pair_meta = dict(pair.get("region_metadata") or {})

            for rec_idx, region in enumerate(regions):
                roi_before = _crop_region(source_frame, region.bbox, "source_frame", bbox_units="normalized")
                roi_after = _crop_region(target_frame, region.bbox, "target_frame", bbox_units="normalized")
                roi_before, roi_after = _match_roi_shapes(roi_before, roi_after)
                metadata = _metadata_for_region(pair_meta, region, total_regions=len(regions))
                region_id = region.region_id
                full_record_id = f"{record_id}:{region_id}"
                safe_stem = _safe_asset_stem(f"{record_id}_{rec_idx}_{region_id}")
                before_asset_name = f"{safe_stem}_before.npy"
                after_asset_name = f"{safe_stem}_after.npy"
                before_path = asset_dir / before_asset_name
                after_path = asset_dir / after_asset_name
                before_rel_path = asset_subdir / before_asset_name
                after_rel_path = asset_subdir / after_asset_name
                np.save(before_path, np.asarray(roi_before, dtype=np.float32), allow_pickle=False)
                np.save(after_path, np.asarray(roi_after, dtype=np.float32), allow_pickle=False)
                before_bytes = before_path.stat().st_size
                after_bytes = after_path.stat().st_size
                roi_asset_count += 2
                roi_asset_total_bytes += before_bytes + after_bytes

                changed_ratio = float(np.mean(np.any(np.abs(roi_after - roi_before) > 1.0 / 255.0, axis=2))) if roi_before.size else 0.0
                mean_abs_delta = float(np.mean(np.abs(roi_after - roi_before))) if roi_before.size else 0.0
                rec: dict[str, object] = {
                    "contract_version": "renderer_patch_manifest_v2",
                    "record_id": full_record_id,
                    "source_pair_id": record_id,
                    "frame_index": int(transition_context.get("frame_index", 0) or 0),
                    "step_index": int(transition_context.get("step_index", -1) if transition_context.get("step_index") is not None else -1),
                    "region_id": region_id,
                    "semantic_family": _semantic_family_for_region(region_id),
                    "canonical_region": str(metadata.get("canonical_region", _region_type(region_id))),
                    "entity_id": str(metadata.get("entity_id", _entity_id(region_id))),
                    "bbox": [float(region.bbox.x), float(region.bbox.y), float(region.bbox.w), float(region.bbox.h)],
                    "bbox_xywh": [float(region.bbox.x), float(region.bbox.y), float(region.bbox.w), float(region.bbox.h)],
                    "bbox_units": "normalized",
                    "roi_before_path": str(before_rel_path),
                    "roi_after_path": str(after_rel_path),
                    "roi_shape": [int(x) for x in roi_before.shape],
                    "source_frame": source_path,
                    "target_frame": target_path,
                    "region_metadata": metadata,
                    "transition_context_summary": transition_context,
                    "selected_render_strategy": "SUPERVISED_EXTERNAL_OBSERVED_ROI",
                    "synthesis_mode": "observed_frame_pair_roi_external_assets",
                    "execution_trace_summary": {
                        "renderer_path": "supervised_observed_frame_pair_external_assets",
                        "external_roi_asset_mode": True,
                    },
                    "metadata_completeness_score": float(metadata.get("metadata_completeness_score", 0.0) or 0.0),
                    "evidence_strength_score": float(metadata.get("evidence_strength_score", 0.0) or 0.0),
                    "roi_source": str(metadata.get("roi_source", "observed_region_ref")),
                    "source_node_type": str(metadata.get("source_node_type", "observed_context")),
                    "mask_kind": str(metadata.get("mask_kind", "")),
                    "mask_ref_present": bool(metadata.get("mask_ref")),
                    "target_source": "provided_ground_truth_roi",
                    "training_target_quality": "external_or_observed_target",
                    "target_training_role": "supervised_external",
                    "source": "observed_frame_pair",
                    "supervised_quality": {
                        "changed_ratio": changed_ratio,
                        "mean_abs_delta": mean_abs_delta,
                        "semantic_family": _semantic_family_for_region(region_id),
                        "warnings": [],
                    },
                    "diagnostics": {
                        "external_roi_asset_mode": True,
                        "roi_before_bytes": before_bytes,
                        "roi_after_bytes": after_bytes,
                        "source_frame_path": source_path,
                        "target_frame_path": target_path,
                        "region_source": str(metadata.get("roi_source", "observed_region_ref")),
                        "metadata_completeness_score": float(metadata.get("metadata_completeness_score", 0.0) or 0.0),
                        "evidence_strength_score": float(metadata.get("evidence_strength_score", 0.0) or 0.0),
                    },
                }
                _validate_supervised_observed_record_invariants(rec)
                all_records.append(rec)
        except Exception as exc:
            skipped_pairs += 1
            if len(invalid_examples) < 16:
                invalid_examples.append({"index": idx, "error": str(exc)})
            if strict:
                raise

    if not all_records:
        raise ValueError("no supervised records exported from observed pairs")

    diagnostics: dict[str, object] = {
        "input_contract_version": "renderer_observed_pair_manifest_input_v1",
        "external_roi_asset_mode": True,
        "total_pairs": len(pairs),
        "exported_records": len(all_records),
        "skipped_pairs": skipped_pairs,
        "invalid_examples": invalid_examples,
        "strict": bool(strict),
        "roi_asset_count": roi_asset_count,
        "roi_asset_total_bytes": roi_asset_total_bytes,
        "manifest_json_bytes": 0,
        "build_timing_sec": 0.0,
    }
    manifest: dict[str, object] = {
        "manifest_type": "renderer_patch_manifest",
        "contract_version": "renderer_patch_manifest_v2",
        "source": "observed_frame_pair",
        "external_roi_asset_mode": True,
        "record_count": len(all_records),
        "records": all_records,
        "builder_diagnostics": diagnostics,
    }
    text = _stable_manifest_json(manifest, diagnostics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    diagnostics["build_timing_sec"] = round(time.perf_counter() - started, 6)
    text = _stable_manifest_json(manifest, diagnostics)
    output.write_text(text, encoding="utf-8")
    return ObservedPairsBuildOutput(manifest_path=output_path, diagnostics=diagnostics)


def _stable_manifest_json(manifest: dict[str, object], diagnostics: dict[str, object]) -> str:
    text = ""
    for _ in range(6):
        text = json.dumps(manifest, ensure_ascii=False, indent=2)
        byte_count = len(text.encode("utf-8"))
        if diagnostics.get("manifest_json_bytes") == byte_count:
            return text
        diagnostics["manifest_json_bytes"] = byte_count
    return json.dumps(manifest, ensure_ascii=False, indent=2)


def _read_image(path: str, field: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{field} path does not exist: {path}")
    try:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"{field} unreadable image: {path}: {exc}") from exc
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{field} must be HxWx3 image: {path}")
    return arr


def _pair_regions(pair: dict[str, object]) -> list[RegionRef]:
    raw = pair.get("regions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("no valid regions exist; observed-pair input requires non-empty regions")
    regions: list[RegionRef] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"regions[{idx}] must be object")
        region_id = str(item.get("region_id", "")).strip()
        if not region_id:
            raise ValueError(f"regions[{idx}].region_id is required")
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"regions[{idx}].bbox must be [x,y,w,h]")
        vals = [float(x) for x in bbox]
        if vals[2] <= 0.0 or vals[3] <= 0.0:
            raise ValueError(f"regions[{idx}].bbox width/height must be > 0")
        if min(vals) < 0.0 or max(vals) > 1.0:
            raise ValueError(f"regions[{idx}].bbox must be normalized [0..1]")
        regions.append(RegionRef(region_id=region_id, bbox=BBox(*vals), reason=str(item.get("reason", "manual_or_parser_region"))))
    return regions


def _validate_supervised_observed_record_invariants(record: dict[str, object]) -> None:
    if record.get("target_source") != "provided_ground_truth_roi":
        raise ValueError("observed-pair record invariant failed: target_source must be 'provided_ground_truth_roi'")
    if record.get("training_target_quality") != "external_or_observed_target":
        raise ValueError("observed-pair record invariant failed: training_target_quality must be 'external_or_observed_target'")
    if record.get("target_training_role") != "supervised_external":
        raise ValueError("observed-pair record invariant failed: target_training_role must be 'supervised_external'")
    if record.get("target_source") == "self_generated_runtime_target":
        raise ValueError("observed-pair record invariant failed: self_generated_runtime_target is forbidden")


def _safe_asset_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem[:160] or "observed_roi"


def _entity_id(region_id: str) -> str:
    return str(region_id).split(":", 1)[0] if ":" in str(region_id) else "unknown"


def _region_type(region_id: str) -> str:
    return str(region_id).split(":", 1)[1] if ":" in str(region_id) else str(region_id)


def _semantic_family_for_region(region_id: str) -> str:
    lower = str(region_id).lower()
    if "face" in lower or "head" in lower:
        return "face_expression"
    if "torso" in lower or "inner" in lower:
        return "torso_reveal"
    return "sleeve_arm_transition"
