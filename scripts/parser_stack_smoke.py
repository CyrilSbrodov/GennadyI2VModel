from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.input_layer import InputAssetLayer
from perception.mask_projection import project_mask_to_frame
from perception.mask_store import DEFAULT_MASK_STORE
from perception.parser import ParserBackendConfig, ParserStackConfig
from perception.parser_debug import export_parser_debug_artifacts
from perception.pipeline import PerceptionBackendsConfig, PerceptionPipeline


def build_parser_config(args: argparse.Namespace) -> ParserStackConfig:
    return ParserStackConfig(
        primary_human_parser=ParserBackendConfig(backend=args.fashn_backend, variant=args.fashn_model, device=args.device),
        structural_body_parser=ParserBackendConfig(backend=args.schp_pascal_backend, variant=args.schp_pascal_model, device=args.device),
        garment_refinement_parser=ParserBackendConfig(backend=args.schp_atr_backend, variant=args.schp_atr_model, device=args.device),
        face_parser=ParserBackendConfig(backend=args.facer_backend, variant=args.facer_model, device=args.device),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke script for real parser stack + debug artifact export")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", default="artifacts/parser_smoke")
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--fashn-backend", default="fashn")
    ap.add_argument("--fashn-model", default="fashn-ai/fashn-human-parser")

    ap.add_argument("--schp-pascal-backend", default="builtin")
    ap.add_argument("--schp-pascal-model", default="")

    ap.add_argument("--schp-atr-backend", default="builtin")
    ap.add_argument("--schp-atr-model", default="")

    ap.add_argument("--facer-backend", default="builtin")
    ap.add_argument("--facer-model", default="farl/lapa/448")

    args = ap.parse_args()

    req = InputAssetLayer().build_request(images=[args.image], text="parser-smoke")
    frame = req.unified_asset.frames[0] if req.unified_asset and req.unified_asset.frames else None
    if frame is None:
        raise RuntimeError("InputAssetLayer returned no frames")

    parser_cfg = build_parser_config(args)
    pipe = PerceptionPipeline(backends=PerceptionBackendsConfig(parser=parser_cfg))
    out = pipe.analyze(frame)

    rgb = [[[int(max(0.0, min(1.0, ch)) * 255.0) for ch in px[:3]] for px in row] for row in frame.tensor]
    summary = export_parser_debug_artifacts(rgb, out, Path(args.out_dir))

    runtime_formats = getattr(pipe.parser, "last_runtime_formats", {})
    mask_geometry: dict[str, dict[str, int]] = {}
    per_person_counts: dict[str, dict[str, int]] = {}
    fw = len(rgb[0]) if rgb else 0
    fh = len(rgb)
    for idx, p in enumerate(out.persons):
        pid = f"person_{idx}"
        per_person_counts[pid] = {
            "garment_masks": len(p.garment_masks),
            "body_part_masks": len(p.body_part_masks),
            "face_region_masks": len(p.face_region_masks),
        }
        geom_counts = {"full": 0, "projected": 0, "local": 0}
        for ref in list(p.garment_masks.values()) + list(p.body_part_masks.values()) + list(p.face_region_masks.values()):
            stored = DEFAULT_MASK_STORE.get(ref)
            if stored is None:
                continue
            _, geom = project_mask_to_frame(stored, frame_size=(fw, fh))
            geom_counts[geom] = geom_counts.get(geom, 0) + 1
        mask_geometry[pid] = geom_counts

    print(json.dumps(
        {
            "persons": len(out.persons),
            "warnings": out.warnings,
            "module_fallbacks": out.module_fallbacks,
            "runtime_formats": runtime_formats,
            "native_backends": {k: v for k, v in out.module_fallbacks.items() if v == "native"},
            "mask_geometry": mask_geometry,
            "per_person_mask_counts": per_person_counts,
            "summary_path": str(Path(args.out_dir) / "fused_summary.json"),
            "summary": summary,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
