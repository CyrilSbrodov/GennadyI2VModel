#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.supervised_renderer_manifest import (
    build_supervised_renderer_record,
    write_supervised_renderer_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert JSON/JSONL external ROI pairs into renderer_patch_manifest.v1."
    )
    parser.add_argument("--input", required=True, help="Input .json array or .jsonl file containing ROI pair records")
    parser.add_argument("--output", required=True, help="Output renderer supervised manifest path")
    return parser


def _load_records(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = payload["records"]
    if not isinstance(payload, list):
        raise ValueError("input must be a JSON array, JSONL file, or object with records[]")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("all input records must be JSON objects")
    return list(payload)


def main() -> None:
    args = build_parser().parse_args()
    records = []
    for rec in _load_records(Path(args.input)):
        records.append(
            build_supervised_renderer_record(
                roi_before=rec["roi_before"],
                roi_after=rec["roi_after"],
                region_id=str(rec["region_id"]),
                semantic_family=str(rec["semantic_family"]),
                bbox=rec.get("bbox"),
                changed_mask=rec.get("changed_mask"),
                alpha_target=rec.get("alpha_target"),
                blend_hint=rec.get("blend_hint"),
                selected_render_strategy=str(rec.get("selected_render_strategy", "supervised_external_target")),
                metadata=rec.get("metadata") if isinstance(rec.get("metadata"), dict) else None,
            )
        )
    output = write_supervised_renderer_manifest(records, args.output)
    print(
        json.dumps(
            {"output": output, "records": len(records), "manifest_type": "renderer_patch_manifest"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
