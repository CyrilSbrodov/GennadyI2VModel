from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from core.schema import BBox, RegionRef


@dataclass(slots=True)
class TemporalObservedBuildOutput:
    manifest_path: str
    diagnostics: dict[str, object]


def _read_rgb(path: str, field: str) -> np.ndarray:
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


def _parse_regions(raw: object, *, strict: bool) -> list[RegionRef]:
    if not isinstance(raw, list) or not raw:
        if strict:
            raise ValueError("changed_regions are required in strict mode")
        return []
    out: list[RegionRef] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"changed_regions[{idx}] must be object")
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"changed_regions[{idx}].bbox must be [x,y,w,h]")
        vals = [float(v) for v in bbox]
        if vals[2] <= 0.0 or vals[3] <= 0.0:
            raise ValueError(f"changed_regions[{idx}].bbox width/height must be > 0")
        if min(vals) < 0.0 or max(vals) > 1.0:
            raise ValueError(f"changed_regions[{idx}].bbox must be normalized [0..1]")
        out.append(RegionRef(region_id=str(item.get("region_id", f"scene:region_{idx}")), bbox=BBox(*vals), reason=str(item.get("reason", "manual_or_parser_region"))))
    if strict and not out:
        raise ValueError("changed_regions are required in strict mode")
    return out


def build_temporal_manifest_from_observed_sequences(*, observed_sequences_path: str, output_path: str, strict: bool = True) -> TemporalObservedBuildOutput:
    payload = json.loads(Path(observed_sequences_path).read_text(encoding="utf-8"))
    if payload.get("contract_version") != "temporal_observed_sequence_manifest_input_v1":
        raise ValueError("observed sequence input must use contract_version='temporal_observed_sequence_manifest_input_v1'")
    sequences = payload.get("sequences", [])
    if not isinstance(sequences, list):
        raise ValueError("sequences must be list")

    records: list[dict[str, object]] = []
    skipped = 0
    invalid_examples: list[dict[str, object]] = []
    frame_shapes: dict[str, int] = {}
    for idx, seq in enumerate(sequences):
        try:
            if not isinstance(seq, dict):
                raise ValueError("sequence must be object")
            sequence_id = str(seq.get("sequence_id", f"sequence_{idx:06d}"))
            frame_paths = seq.get("frames", [])
            if not isinstance(frame_paths, list) or len(frame_paths) < 2:
                raise ValueError("minimum 2 frames per sequence is required")
            frames = [_read_rgb(str(p), f"frames[{i}]") for i, p in enumerate(frame_paths)]
            if any(f.shape != frames[0].shape for f in frames[1:]):
                raise ValueError("all frames in sequence must share identical shape")
            regions = _parse_regions(seq.get("changed_regions", []), strict=bool(strict))
            fallback_region_used = False
            if not regions:
                fallback_region_used = True
                regions = [RegionRef(region_id="scene:region_0", bbox=BBox(0.2, 0.2, 0.3, 0.3), reason="temporal_drift")]
            for j in range(1, len(frames)):
                rec = {
                    "record_id": f"{sequence_id}:t{j:03d}",
                    "sequence_id": sequence_id,
                    "previous_frame": frames[j - 1].tolist(),
                    "current_composed_frame": frames[j].tolist(),
                    "target_refined_frame": frames[j].tolist(),
                    "changed_regions": [{"region_id": r.region_id, "reason": r.reason, "bbox": {"x": r.bbox.x, "y": r.bbox.y, "w": r.bbox.w, "h": r.bbox.h}} for r in regions],
                    "memory_channels": {},
                    "transition_context": dict(seq.get("transition_context") or {}),
                    "target_source": "provided_ground_truth_temporal_frame",
                    "training_target_quality": "external_or_observed_temporal_target",
                    "target_training_role": "supervised_temporal_external",
                    "diagnostics": {"prompt": str(seq.get("prompt", "")), "frame_paths": [str(frame_paths[j - 1]), str(frame_paths[j])], "fallback_region_used": fallback_region_used, "region_source": "non_strict_default_temporal_region" if fallback_region_used else "provided", "region_quality": "low" if fallback_region_used else "provided"},
                }
                records.append(rec)
            key = f"{frames[0].shape[0]}x{frames[0].shape[1]}x{frames[0].shape[2]}"
            frame_shapes[key] = frame_shapes.get(key, 0) + 1
        except Exception as exc:
            skipped += 1
            if len(invalid_examples) < 16:
                invalid_examples.append({"index": idx, "error": str(exc)})
            if strict:
                raise

    for rec in records:
        if rec["target_source"] != "provided_ground_truth_temporal_frame" or rec["training_target_quality"] != "external_or_observed_temporal_target" or rec["target_training_role"] != "supervised_temporal_external":
            raise ValueError("temporal observed record invariant failed")
    if not records:
        raise ValueError("no temporal supervised records exported")
    out_payload = {
        "contract_version": "temporal_refinement_manifest_v1",
        "manifest_type": "temporal_refinement_manifest",
        "record_count": len(records),
        "records": records,
        "diagnostics": {"input_contract_version": "temporal_observed_sequence_manifest_input_v1", "sequence_count": len(sequences), "exported_records": len(records), "skipped_sequences": skipped, "invalid_examples": invalid_examples, "strict": bool(strict), "frame_shape_summary": frame_shapes},
    }
    Path(output_path).write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return TemporalObservedBuildOutput(manifest_path=output_path, diagnostics=out_payload["diagnostics"])
