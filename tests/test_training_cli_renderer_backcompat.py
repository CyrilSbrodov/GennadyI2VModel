from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import pytest

from training import cli as training_cli


@dataclass
class _FakeStageResult:
    stage_name: str = "renderer"
    val_metrics: dict[str, object] = None  # type: ignore[assignment]
    checkpoint_path: str = "/tmp/fake_renderer_ckpt.json"
    train_metrics: dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.val_metrics is None:
            self.val_metrics = {"reconstruction_mae": 0.1}
        if self.train_metrics is None:
            self.train_metrics = {"dataset_diagnostics": {"source": "synthetic_bootstrap"}}


def test_renderer_cli_default_without_learned_dataset_path_uses_backcompat(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["training.cli", "--stage", "renderer", "--epochs", "1"])
    monkeypatch.setattr(training_cli, "train_stage", lambda stage, config: _FakeStageResult())
    training_cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload[0]["stage"] == "renderer"
    assert payload[0]["dataset_diagnostics"]["source"] == "synthetic_bootstrap"


def test_renderer_cli_strict_dataset_without_learned_dataset_path_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["training.cli", "--stage", "renderer", "--epochs", "1", "--strict-dataset"],
    )
    with pytest.raises(ValueError, match="--learned-dataset-path is required for supervised renderer training"):
        training_cli.main()
