from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.input_layer import InputAssetLayer
from perception.image_ops import frame_to_numpy_rgb
from perception.mask_projection import project_mask_to_frame
from perception.mask_store import DEFAULT_MASK_STORE
from perception.parser import ParserBackendConfig, ParserStackConfig
from perception.parser_debug import export_parser_debug_artifacts
from perception.pipeline import ParserOnlyPipeline, PerceptionBackendsConfig, PerceptionPipeline
from perception.profiling import StageTimer

RUNTIME_FAIL_TYPES = {
    "image_decode_failed",
    "frame_tensor_missing",
    "native_backend_missing",
    "fallback_used",
    "validation_invalid_due_to_fallback",
}


def build_parser_config(args: argparse.Namespace) -> ParserStackConfig:
    return ParserStackConfig(
        primary_human_parser=ParserBackendConfig(backend=args.fashn_backend, variant=args.fashn_model, device=args.device),
        structural_body_parser=ParserBackendConfig(backend=args.schp_pascal_backend, variant=args.schp_pascal_model, device=args.device),
        garment_refinement_parser=ParserBackendConfig(backend=args.schp_atr_backend, variant=args.schp_atr_model, device=args.device),
        face_parser=ParserBackendConfig(backend=args.facer_backend, variant=args.facer_model, device=args.device),
    )


def _aliases_for_expected_body_part(part: str) -> set[str]:
    aliases = {"arms": {"arms", "upper_arm", "lower_arm", "hands"}, "legs": {"legs", "upper_leg", "lower_leg", "feet"}, "torso": {"torso"}, "head": {"head"}}
    return aliases.get(part, {part})


def _collect_geometry_stats(frame_size: tuple[int, int], refs: list[str]) -> dict[str, int]:
    stats = {"full": 0, "projected": 0, "local": 0}
    for ref in refs:
        stored = DEFAULT_MASK_STORE.get(ref)
        if stored is None:
            continue
        _, geom = project_mask_to_frame(stored, frame_size=frame_size)
        stats[geom] = stats.get(geom, 0) + 1
    return stats


def _semantic_failures(record: dict[str, Any], out: Any) -> list[str]:
    expected = record.get("expected", {})
    detected_garments = {g["type"] for p in out.persons for g in p.garments}
    detected_body_parts = {bp["part_type"] for p in out.persons for bp in p.body_parts}
    detected_accessories = {a for p in out.persons for a in p.accessory_masks.keys()}
    face_region_mask_count = sum(len(p.face_region_masks) for p in out.persons)
    person_mask_present = any(bool(p.mask_ref) for p in out.persons)

    reasons: list[str] = []
    if expected.get("person") is True and not out.persons:
        reasons.append("person_missing")
    if expected.get("person") is True and not person_mask_present:
        reasons.append("person_mask_missing")
    garments_any_of = expected.get("garments_any_of") or []
    if garments_any_of and not any(g in detected_garments for g in garments_any_of):
        reasons.append("garments_any_of_missing")
    body_expected = expected.get("body_parts_expected") or []
    if body_expected:
        found = any(detected_body_parts.intersection(_aliases_for_expected_body_part(str(exp))) for exp in body_expected)
        if not found:
            reasons.append("body_parts_expected_missing")
    if expected.get("face_regions_expected") is True and face_region_mask_count <= 0:
        reasons.append("face_regions_missing")
    accessories_any_of = expected.get("accessories_any_of") or []
    if accessories_any_of and not any(a in detected_accessories for a in accessories_any_of):
        reasons.append("accessories_any_of_missing")
    return reasons


def _base_report(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "image": record.get("image"),
        "expected": record.get("expected", {}),
        "person_detected": False,
        "person_mask_present": False,
        "garment_mask_count": 0,
        "body_part_mask_count": 0,
        "face_region_mask_count": 0,
        "accessory_mask_count": 0,
        "module_fallbacks": {},
        "runtime_formats": {},
        "geometry_stats": {"full": 0, "projected": 0, "local": 0},
        "warnings": [],
        "detected_garments": [],
        "detected_body_parts": [],
        "detected_face_regions": [],
        "detected_accessories": [],
        "fail_reasons": [],
        "runtime_failure": False,
        "native_backend_missing": False,
        "fallback_used": False,
        "semantic_failure": False,
        "native_parser_available": False,
        "native_parser_used": False,
        "validation_invalid_due_to_fallback": False,
        "image_decode_failed": False,
        "frame_tensor_missing": False,
        "passed": False,
        "timings_ms": {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run parser validation manifest through perception runtime")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strict-native-parser", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--full-path", action="store_true", help="Use full PerceptionPipeline instead of parser-only pipeline")
    ap.add_argument("--detector-backend", default="builtin", choices=["builtin", "ultralytics"])
    ap.add_argument("--detector-model", default="yolov8n.pt")
    ap.add_argument("--fashn-backend", default="fashn")
    ap.add_argument("--fashn-model", default="fashn-ai/fashn-human-parser")
    ap.add_argument("--schp-pascal-backend", default="builtin")
    ap.add_argument("--schp-pascal-model", default="")
    ap.add_argument("--schp-atr-backend", default="builtin")
    ap.add_argument("--schp-atr-model", default="")
    ap.add_argument("--facer-backend", default="builtin")
    ap.add_argument("--facer-model", default="farl/lapa/448")
    args = ap.parse_args()

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    parser_cfg = build_parser_config(args)
    backends = PerceptionBackendsConfig(parser=parser_cfg)
    backends.detector.backend = args.detector_backend
    backends.detector.checkpoint = args.detector_model
    pipe = PerceptionPipeline(backends=backends) if args.full_path else ParserOnlyPipeline(backends=backends)
    input_layer = InputAssetLayer()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures_by_type: Counter[str] = Counter()
    runtime_failures_by_type: Counter[str] = Counter()
    semantic_failures_by_type: Counter[str] = Counter()
    fallback_counts: dict[str, Counter[str]] = defaultdict(Counter)
    runtime_format_counts: dict[str, Counter[str]] = defaultdict(Counter)
    aggregate_geometry_stats: Counter[str] = Counter()
    native_parser_success_count = 0
    fallback_validation_count = 0
    per_image_reports: list[dict[str, Any]] = []
    run_profiler = StageTimer(enabled=bool(args.profile))

    def _accumulate_report(report: dict[str, Any]) -> None:
        nonlocal native_parser_success_count, fallback_validation_count
        if report.get("native_parser_used"):
            native_parser_success_count += 1
        if report.get("validation_invalid_due_to_fallback"):
            fallback_validation_count += 1
        for module, mode in report.get("module_fallbacks", {}).items():
            fallback_counts[module][str(mode)] += 1
        for person_runtime in report.get("runtime_formats", {}).values():
            if isinstance(person_runtime, dict):
                for backend_name, rt in person_runtime.items():
                    runtime_format_counts[backend_name][str(rt)] += 1
        aggregate_geometry_stats.update(report.get("geometry_stats", {}))
        failures_by_type.update(report.get("fail_reasons", []))
        for reason in report.get("fail_reasons", []):
            (runtime_failures_by_type if reason in RUNTIME_FAIL_TYPES else semantic_failures_by_type)[reason] += 1

    for rec in records:
        image_path = rec.get("image")
        image_id = rec.get("id")
        if not image_path or not image_id:
            continue
        report = _base_report(rec)
        frame: Any = None
        rgb = None
        timer = StageTimer(enabled=bool(args.profile))

        try:
            with timer.track("image_decode"):
                req = input_layer.build_request(images=[str(image_path)], text=f"parser-validation:{image_id}")
                frame = req.unified_asset.frames[0] if req.unified_asset and req.unified_asset.frames else None
                if frame is not None:
                    rgb = frame_to_numpy_rgb(frame).rgb
        except Exception as exc:
            report["warnings"].append(f"image_decode_failed:{exc}")
            report["image_decode_failed"] = True
            report["runtime_failure"] = True
            report["fail_reasons"].append("image_decode_failed")

        if frame is None:
            report["frame_tensor_missing"] = True
            report["runtime_failure"] = True
            report["fail_reasons"].append("frame_tensor_missing")
            report["fail_reasons"] = sorted(set(report["fail_reasons"]))
            report["passed"] = False
            report["debug_summary_path"] = None
            per_image_reports.append(report)
            _accumulate_report(report)
            continue

        with timer.track("pipeline_analyze"):
            out = pipe.analyze(frame, profiler=timer) if isinstance(pipe, ParserOnlyPipeline) else pipe.analyze(frame)
        report["warnings"].extend(out.warnings)
        report["module_fallbacks"] = out.module_fallbacks

        parser_mode = out.module_fallbacks.get("parser", "unknown")
        report["native_parser_used"] = parser_mode == "native"
        report["native_parser_available"] = parser_mode in {"native", "builtin", "fallback"}
        report["fallback_used"] = parser_mode != "native"
        if report["fallback_used"]:
            report["runtime_failure"] = True
            report["native_backend_missing"] = True
            report["fail_reasons"].extend(["fallback_used", "native_backend_missing"])
        if args.strict_native_parser and not report["native_parser_used"]:
            report["validation_invalid_due_to_fallback"] = True
            report["runtime_failure"] = True
            report["fail_reasons"].append("validation_invalid_due_to_fallback")

        runtime_formats = getattr(getattr(pipe, "parser", None), "last_runtime_formats", {})
        report["runtime_formats"] = runtime_formats
        image_dir = out_dir / str(image_id)
        with timer.track("debug_export"):
            report["debug_summary"] = export_parser_debug_artifacts(rgb if rgb is not None else [[[0, 0, 0]]], out, image_dir)
        report["debug_summary_path"] = str(image_dir / "fused_summary.json")

        report["person_detected"] = len(out.persons) > 0
        report["person_mask_present"] = any(bool(p.mask_ref) for p in out.persons)
        report["garment_mask_count"] = sum(len(p.garment_masks) for p in out.persons)
        report["body_part_mask_count"] = sum(len(p.body_part_masks) for p in out.persons)
        report["face_region_mask_count"] = sum(len(p.face_region_masks) for p in out.persons)
        report["accessory_mask_count"] = sum(len(p.accessory_masks) for p in out.persons)
        report["detected_garments"] = sorted({g["type"] for p in out.persons for g in p.garments})
        report["detected_body_parts"] = sorted({bp["part_type"] for p in out.persons for bp in p.body_parts})
        report["detected_face_regions"] = sorted({fr["region_type"] for p in out.persons for fr in p.face_regions})
        report["detected_accessories"] = sorted({a for p in out.persons for a in p.accessory_masks.keys()})
        report["geometry_stats"] = _collect_geometry_stats(out.frame_size, [ref for p in out.persons for ref in [*p.garment_masks.values(), *p.body_part_masks.values(), *p.face_region_masks.values(), *p.accessory_masks.values()]])

        semantic_fails = _semantic_failures(rec, out)
        if semantic_fails:
            report["semantic_failure"] = True
            report["fail_reasons"].extend(semantic_fails)

        report["timings_ms"] = {k: round(v["total_ms"], 3) for k, v in timer.summary().items()}
        for stage, stats in timer.summary().items():
            run_profiler.add(stage, stats["total_ms"] / 1000.0)
        report["fail_reasons"] = sorted(set(report["fail_reasons"]))
        report["passed"] = not report["fail_reasons"]
        per_image_reports.append(report)
        _accumulate_report(report)

    validation_report = {
        "pipeline_mode": "full" if args.full_path else "parser_only",
        "strict_native_mode": bool(args.strict_native_parser),
        "detector_backend": args.detector_backend,
        "parser_backend": args.fashn_backend,
        "total_images": len(records),
        "total_failures": sum(1 for r in per_image_reports if not r.get("passed", False)),
        "native_parser_success_count": native_parser_success_count,
        "fallback_validation_count": fallback_validation_count,
        "failures_by_type": dict(failures_by_type),
        "runtime_failures_by_type": dict(runtime_failures_by_type),
        "semantic_failures_by_type": dict(semantic_failures_by_type),
        "fallback_counts_by_module": {k: dict(v) for k, v in fallback_counts.items()},
        "runtime_format_counts_by_backend": {k: dict(v) for k, v in runtime_format_counts.items()},
        "aggregate_geometry_stats": dict(aggregate_geometry_stats),
        "timing_summary": run_profiler.summary() if args.profile else {},
        "per_image_reports": per_image_reports,
    }
    (out_dir / "validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
