from __future__ import annotations

import argparse
import json

from training.orchestrator import train_pipeline, train_stage
from training.types import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train modular Gennady stages")
    parser.add_argument("--stage", default="all", choices=["all", "perception", "representation", "dynamics", "renderer"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        train_size=args.train_size,
        val_size=args.val_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.stage == "all":
        results = train_pipeline(config)
        payload = [{"stage": r.stage_name, "val": r.val_metrics, "checkpoint": r.checkpoint_path} for r in results]
    else:
        result = train_stage(args.stage, config)
        payload = [{"stage": result.stage_name, "val": result.val_metrics, "checkpoint": result.checkpoint_path}]

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
