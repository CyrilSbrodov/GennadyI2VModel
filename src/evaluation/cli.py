from __future__ import annotations

import argparse
import json

from evaluation.benchmark import run_benchmark_suite, run_stage_eval, write_report


def _parse_modes(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_filter(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation and benchmark CLI for single-image pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    stage = sub.add_parser("stage", help="Run stage-level health evaluation on assembled pipeline")
    stage.add_argument("--image", required=True, help="Path to a seed image")
    stage.add_argument("--text", required=True, help="Prompt for runtime plan")
    stage.add_argument("--backend-mode", default="learned_primary", help="learned_primary|legacy|hybrid")
    stage.add_argument("--output", default="artifacts/eval/stage_eval.json", help="JSON output path")

    bench = sub.add_parser("benchmark", help="Run full scenario suite benchmark")
    bench.add_argument("--backend-modes", default="learned_primary,legacy", help="Comma-separated backend modes")
    bench.add_argument("--image", default="", help="Optional path to seed image; by default curated manifest dataset is used")
    bench.add_argument("--dataset-manifest", default="", help="Optional path to curated benchmark manifest")
    bench.add_argument("--scenario-filter", default="", help="Optional comma-separated scenario_id filter")
    bench.add_argument("--asset-filter", default="", help="Optional comma-separated asset_id filter")
    bench.add_argument("--output", default="artifacts/eval/benchmark_report.json", help="JSON output path")

    args = parser.parse_args()

    if args.command == "stage":
        result = run_stage_eval(image_path=args.image, text=args.text, backend_mode=args.backend_mode)
        out = write_report(result, args.output)
        print(json.dumps({"output": out, "summary": result.get("summary", {})}, ensure_ascii=False, indent=2))
        return

    modes = _parse_modes(args.backend_modes)
    image = args.image.strip() if isinstance(args.image, str) else ""
    report = run_benchmark_suite(
        backend_modes=modes,
        image_path=image or None,
        dataset_manifest=(args.dataset_manifest.strip() if isinstance(args.dataset_manifest, str) and args.dataset_manifest.strip() else None),
        scenario_filter=_parse_filter(args.scenario_filter),
        asset_filter=_parse_filter(args.asset_filter),
    )
    out = write_report(report, args.output)
    print(
        json.dumps(
            {
                "output": out,
                "dataset": report.get("dataset", {}),
                "runs": list(report.get("runs", {}).keys()) if isinstance(report.get("runs", {}), dict) else [],
                "comparison": report.get("comparison", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
