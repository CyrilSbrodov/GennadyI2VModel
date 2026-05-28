from __future__ import annotations

import argparse
import json

from training.dynamics_eval import evaluate_dynamics
from training.orchestrator import train_pipeline, train_stage
from training.types import TrainingConfig
from training.renderer_observed_pairs_builder import build_renderer_manifest_from_observed_pairs
from training.temporal_observed_sequences_builder import build_temporal_manifest_from_observed_sequences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train modular Gennady stages")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "perception",
            "representation",
            "dynamics",
            "renderer",
            "text_encoder",
            "dynamics_transition",
            "temporal_transition",
            "patch_synthesis",
            "temporal_refinement",
            "renderer_manifest_from_observed_pairs",
            "temporal_manifest_from_observed_sequences",
        ],
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--eval-dynamics", action="store_true")
    parser.add_argument("--weights-path", default="artifacts/checkpoints/dynamics/dynamics_weights.json")
    parser.add_argument("--learned-dataset-path", default="", help="Manifest path for learned dataset (primary for dynamics when provided).")
    parser.add_argument("--observed-pairs-path", default="", help="Observed pairs JSON input path (renderer_observed_pair_manifest_input_v1).")
    parser.add_argument("--output-path", default="", help="Output manifest path for export-only stages.")
    parser.add_argument("--observed-sequences-path", default="", help="Observed temporal sequences JSON input path (temporal_observed_sequence_manifest_input_v1).")
    parser.add_argument("--strict", action="store_true", help="Enable strict validation for export-only manifest builders.")
    parser.add_argument("--strict-dataset", action="store_true", help="Enable strict supervised dataset policy for renderer training.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        train_size=args.train_size,
        val_size=args.val_size,
        checkpoint_dir=args.checkpoint_dir,
        learned_dataset_path=args.learned_dataset_path,
    )
    if args.strict_dataset:
        config.renderer_target_role_policy = "supervised_only"

    if args.stage == "renderer_manifest_from_observed_pairs":
        if not args.observed_pairs_path:
            raise ValueError("--observed-pairs-path is required for renderer_manifest_from_observed_pairs")
        if not args.output_path:
            raise ValueError("--output-path is required for renderer_manifest_from_observed_pairs")
        result = build_renderer_manifest_from_observed_pairs(observed_pairs_path=args.observed_pairs_path, output_path=args.output_path, strict=bool(args.strict))
        payload = [{"stage": "renderer_manifest_from_observed_pairs", "output_path": result.manifest_path, "diagnostics": result.diagnostics}]
    elif args.stage == "temporal_manifest_from_observed_sequences":
        if not args.observed_sequences_path:
            raise ValueError("--observed-sequences-path is required for temporal_manifest_from_observed_sequences")
        if not args.output_path:
            raise ValueError("--output-path is required for temporal_manifest_from_observed_sequences")
        result = build_temporal_manifest_from_observed_sequences(observed_sequences_path=args.observed_sequences_path, output_path=args.output_path, strict=bool(args.strict))
        payload = [{"stage": "temporal_manifest_from_observed_sequences", "output_path": result.manifest_path, "diagnostics": result.diagnostics}]
    elif args.eval_dynamics:
        payload = [{"stage": "dynamics_eval", "metrics": evaluate_dynamics(args.weights_path, dataset_size=args.val_size, dataset_manifest=args.learned_dataset_path)}]
    elif args.stage == "all":
        results = train_pipeline(config)
        payload = [{"stage": r.stage_name, "val": r.val_metrics, "checkpoint": r.checkpoint_path} for r in results]
    else:
        if args.stage in {"renderer", "temporal_refinement"} and args.strict_dataset and not args.learned_dataset_path:
            raise ValueError("--learned-dataset-path is required for supervised training")
        result = train_stage(args.stage, config)
        row = {"stage": result.stage_name, "val": result.val_metrics, "checkpoint": result.checkpoint_path}
        if args.stage in {"renderer", "temporal_refinement"}:
            row["dataset_diagnostics"] = result.train_metrics.get("dataset_diagnostics", {})
        payload = [row]

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
