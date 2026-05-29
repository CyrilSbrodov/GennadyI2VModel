from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from core.schema import BBox, RegionRef
from training.renderer_video_manifest_builder import RendererVideoManifestBuilder


@dataclass(slots=True)
class ObservedPairsBuildOutput:
    manifest_path: str
    diagnostics: dict[str, object]


def build_renderer_manifest_from_observed_pairs(*, observed_pairs_path: str, output_path: str, strict: bool = True) -> ObservedPairsBuildOutput:
    payload = json.loads(Path(observed_pairs_path).read_text(encoding="utf-8"))
    if str(payload.get("contract_version", "")).strip() != "renderer_observed_pair_manifest_input_v1":
        raise ValueError("observed pairs input must use contract_version='renderer_observed_pair_manifest_input_v1'")
    pairs = payload.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("observed pairs input field 'pairs' must be list")

    builder = RendererVideoManifestBuilder()
    all_records: list[dict[str, object]] = []
    skipped_pairs = 0
    invalid_examples: list[dict[str, object]] = []

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
            out = builder.build_records(
                source_frame=source_frame,
                target_frame=target_frame,
                regions=regions,
                region_metadata=pair_meta,
                transition_context=transition_context,
                strict=strict,
            )
            for rec_idx, rec in enumerate(out.records):
                region_id = str(rec.get("region_id", "")).strip() or f"region_{rec_idx}"
                rec["source_pair_id"] = record_id
                rec["record_id"] = f"{record_id}:{region_id}"
                rec["target_source"] = "provided_ground_truth_roi"
                rec["training_target_quality"] = "external_or_observed_target"
                rec["target_training_role"] = "supervised_external"
                rec.setdefault("diagnostics", {})
                rec["diagnostics"].update({
                    "source_frame_path": source_path,
                    "target_frame_path": target_path,
                    "region_source": rec.get("roi_source", "observed_region_ref"),
                    "metadata_completeness_score": rec.get("metadata_completeness_score", 0.0),
                    "evidence_strength_score": rec.get("evidence_strength_score", 0.0),
                })
                _validate_supervised_observed_record_invariants(rec)
            all_records.extend(out.records)
        except Exception as exc:
            skipped_pairs += 1
            if len(invalid_examples) < 16:
                invalid_examples.append({"index": idx, "error": str(exc)})
            if strict:
                raise

    if not all_records:
        raise ValueError("no supervised records exported from observed pairs")

    builder.write_manifest(
        all_records,
        output_path,
        diagnostics={
            "input_contract_version": "renderer_observed_pair_manifest_input_v1",
            "total_pairs": len(pairs),
            "exported_records": len(all_records),
            "skipped_pairs": skipped_pairs,
            "invalid_examples": invalid_examples,
            "strict": bool(strict),
        },
    )
    return ObservedPairsBuildOutput(manifest_path=output_path, diagnostics={"total_pairs": len(pairs), "exported_records": len(all_records), "skipped_pairs": skipped_pairs, "invalid_examples": invalid_examples})


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
