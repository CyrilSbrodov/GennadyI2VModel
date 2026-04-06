from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from training.types import StageResult, TrainingConfig


class BaseTrainer:
    stage_name = "base"

    def build_datasets(self, config: TrainingConfig) -> tuple[object, object]:
        raise NotImplementedError

    def compute_metrics(
        self,
        train_dataset: object,
        val_dataset: object,
        epoch: int,
        total_epochs: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        train_size = len(train_dataset)  # type: ignore[arg-type]
        val_size = len(val_dataset)  # type: ignore[arg-type]
        progress = (epoch + 1) / max(1, total_epochs)
        train_loss = max(0.01, (train_size + 1) / (100.0 * (epoch + 1)))
        val_loss = max(0.01, (val_size + 1) / (95.0 * (epoch + 1)))
        return (
            {"loss": round(train_loss, 4), "progress": round(progress, 4)},
            {"loss": round(val_loss, 4), "score": round(1.0 - val_loss, 4)},
        )

    def train(self, config: TrainingConfig) -> StageResult:
        train_dataset, val_dataset = self.build_datasets(config)
        stage_dir = Path(config.checkpoint_dir) / self.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, dict[str, float] | int]] = []
        last_train_metrics: dict[str, float] = {}
        last_val_metrics: dict[str, float] = {}

        weights = [0.0 for _ in range(8)]
        optimizer_state = {"lr": config.learning_rate, "momentum": 0.9}
        scheduler_state = {"gamma": 0.95, "step": 0}

        for epoch in range(config.epochs):
            train_metrics, val_metrics = self.compute_metrics(train_dataset, val_dataset, epoch, config.epochs)
            grad = [train_metrics["loss"] * 0.1 for _ in weights]
            weights = [w - optimizer_state["lr"] * g for w, g in zip(weights, grad)]
            scheduler_state["step"] += 1
            optimizer_state["lr"] *= scheduler_state["gamma"]

            train_metrics["lr"] = round(optimizer_state["lr"], 6)
            history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            last_train_metrics = train_metrics
            last_val_metrics = val_metrics

        checkpoint_path = stage_dir / "latest.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "stage": self.stage_name,
                    "config": asdict(config),
                    "history": history,
                    "final_train": last_train_metrics,
                    "final_val": last_val_metrics,
                    "optimizer": optimizer_state,
                    "scheduler": scheduler_state,
                    "weights": weights,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return StageResult(
            stage_name=self.stage_name,
            train_metrics=last_train_metrics,
            val_metrics=last_val_metrics,
            checkpoint_path=str(checkpoint_path),
        )
