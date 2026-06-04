from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from build_renderer_pairs_manifest_auto import build_renderer_pairs_manifest_auto  # noqa: E402
from training.renderer_observed_pairs_builder import build_renderer_manifest_from_observed_pairs  # noqa: E402
from training.orchestrator import train_stage  # noqa: E402
from training.types import TrainingConfig  # noqa: E402


IMAGE_EXTS = [".png", ".jpg", ".jpeg"]

class TimingReport:
    def __init__(self) -> None:
        self.sections: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = perf_counter()

    def stop(self, name: str) -> None:
        started = self._starts.pop(name, None)
        if started is None:
            return
        self.sections[name] = round(perf_counter() - started, 6)

    def as_dict(self) -> dict[str, float]:
        total = sum(v for k, v in self.sections.items() if k != "total_pipeline")
        out = dict(self.sections)
        out["total_measured_sec"] = round(total, 6)
        return out

def _find_image(base: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _discover_pair_ids(pairs_dir: Path) -> list[str]:
    ids: set[str] = set()

    for p in pairs_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if not p.stem.startswith("source_"):
            continue

        idx = p.stem.removeprefix("source_")
        ids.add(idx)

    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def normalize_pair_sizes(
    *,
    pairs_dir: Path,
    backup_originals: bool = True,
    resize_targets_to_source: bool = True,
) -> dict[str, Any]:
    pair_ids = _discover_pair_ids(pairs_dir)

    diagnostics: dict[str, Any] = {
        "total_source_pairs": len(pair_ids),
        "checked_pairs": 0,
        "missing_targets": [],
        "already_same_size": [],
        "resized_targets": [],
        "size_mismatches_left_unfixed": [],
    }

    for idx in pair_ids:
        source_path = _find_image(pairs_dir, f"source_{idx}")
        target_path = _find_image(pairs_dir, f"target_{idx}")

        if source_path is None:
            continue

        if target_path is None:
            diagnostics["missing_targets"].append(idx)
            continue

        diagnostics["checked_pairs"] += 1

        source = Image.open(source_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if source.size == target.size:
            diagnostics["already_same_size"].append(
                {
                    "pair_id": idx,
                    "size": list(source.size),
                }
            )
            continue

        if not resize_targets_to_source:
            diagnostics["size_mismatches_left_unfixed"].append(
                {
                    "pair_id": idx,
                    "source_size": list(source.size),
                    "target_size": list(target.size),
                }
            )
            continue

        if backup_originals:
            backup_path = target_path.with_name(
                f"{target_path.stem}.original{target_path.suffix}"
            )
            if not backup_path.exists():
                target.save(backup_path)

        target_resized = target.resize(source.size, Image.Resampling.LANCZOS)
        target_resized.save(target_path)

        diagnostics["resized_targets"].append(
            {
                "pair_id": idx,
                "source_size": list(source.size),
                "old_target_size": list(target.size),
                "new_target_size": list(source.size),
                "target_path": str(target_path),
            }
        )

    return diagnostics


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    timings = TimingReport()

    timings.start("total_pipeline")

    pairs_dir = Path(args.pairs_dir)
    work_dir = Path(args.work_dir)
    manifests_dir = Path(args.manifests_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    timings.start("mkdirs")
    manifests_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    timings.stop("mkdirs")

    input_manifest_path = manifests_dir / "renderer_observed_pairs.input.json"
    training_manifest_path = manifests_dir / "renderer_observed_pairs.json"
    summary_path = work_dir / "renderer_training_pipeline.summary.json"
    timing_path = work_dir / "renderer_training_pipeline.timing.json"

    timings.start("normalize_pair_sizes")
    size_diagnostics = normalize_pair_sizes(
        pairs_dir=pairs_dir,
        backup_originals=not args.no_backup,
        resize_targets_to_source=not args.no_resize,
    )
    timings.stop("normalize_pair_sizes")

    if size_diagnostics["missing_targets"] and args.strict:
        raise RuntimeError(
            "Missing target files for pair ids: "
            + ", ".join(size_diagnostics["missing_targets"])
        )

    if size_diagnostics["size_mismatches_left_unfixed"] and args.strict:
        raise RuntimeError(
            "Some source/target sizes mismatch and --no-resize was used. "
            "Either remove --no-resize or fix sizes manually."
        )

    timings.start("auto_build_input_manifest_with_perception")
    auto_manifest_result = build_renderer_pairs_manifest_auto(
        pairs_dir=pairs_dir,
        output_path=input_manifest_path,
        prompt=args.prompt,
        transition_family=args.transition_family,
        action_family=args.action_family,
        summary=args.summary,
        strict=args.strict,
        allow_person_fallback_non_strict=args.allow_person_fallback_non_strict,
    )
    timings.stop("auto_build_input_manifest_with_perception")

    timings.start("build_renderer_patch_manifest")
    renderer_manifest_result = build_renderer_manifest_from_observed_pairs(
        observed_pairs_path=str(input_manifest_path),
        output_path=str(training_manifest_path),
        strict=args.strict,
    )
    timings.stop("build_renderer_patch_manifest")

    timings.start("renderer_train_stage")
    config = TrainingConfig(
        epochs=args.epochs,
        checkpoint_dir=str(checkpoint_dir),
        learned_dataset_path=str(training_manifest_path),
        renderer_backend=args.renderer_backend,
    )
    config.renderer_target_role_policy = "supervised_only"

    train_result = train_stage("renderer", config)
    timings.stop("renderer_train_stage")

    timings.stop("total_pipeline")

    timing_payload = timings.as_dict()
    timing_path.write_text(
        json.dumps(timing_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "status": "ok",
        "pairs_dir": str(pairs_dir),
        "timings": timing_payload,
        "timing_path": str(timing_path),
        "size_diagnostics": size_diagnostics,
        "input_manifest_path": str(input_manifest_path),
        "auto_manifest_result": auto_manifest_result,
        "renderer_training_manifest_path": str(training_manifest_path),
        "renderer_manifest_diagnostics": renderer_manifest_result.diagnostics,
        "training": {
            "stage": train_result.stage_name,
            "checkpoint_path": train_result.checkpoint_path,
            "val_metrics": train_result.val_metrics,
            "train_metrics": train_result.train_metrics,
        },
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-command renderer observed-pair manifest build + renderer training pipeline."
    )

    parser.add_argument(
        "--pairs-dir",
        default="data/renderer_pairs",
        help="Directory with source_0001.png / target_0001.png pairs.",
    )
    parser.add_argument(
        "--manifests-dir",
        default="artifacts/manifests",
        help="Directory for generated manifests.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts/renderer_auto_pipeline",
        help="Directory for pipeline summary/debug files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Renderer training epochs.",
    )
    parser.add_argument(
        "--prompt",
        default="person smiles naturally",
    )
    parser.add_argument(
        "--transition-family",
        default="expression_transition",
    )
    parser.add_argument(
        "--action-family",
        default="face_expression_change",
    )
    parser.add_argument(
        "--summary",
        default="auto-generated face expression pair",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid pair/build/training issue.",
    )
    parser.add_argument(
        "--allow-person-fallback-non-strict",
        action="store_true",
        help="Only in non-strict mode: allow person bbox fallback if face detector fails.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Do not resize target images to source size.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not save target_XXXX.original.png backup before resizing.",
    )
    parser.add_argument(
        "--renderer-backend",
        default="numpy_local",
        choices=["numpy_local", "torch_local"],
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()