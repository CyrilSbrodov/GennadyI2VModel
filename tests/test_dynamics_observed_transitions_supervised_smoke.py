from __future__ import annotations
import json
from pathlib import Path
import pytest

from dynamics.runtime_bundle import DynamicsRuntimeBundle
from training.dynamics_observed_transitions_builder import build_dynamics_manifest_from_observed_transitions
from training.dynamics_trainer import DynamicsTrainer
from training.orchestrator import train_stage
from training.types import TrainingConfig


def _train_supervised(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    build_dynamics_manifest_from_observed_transitions("examples/dynamics_transitions.example.json", str(manifest), strict=True)
    cfg = TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), learned_dataset_path=str(manifest), dynamics_target_role_policy="supervised_only")
    return DynamicsTrainer().train(cfg), cfg


def test_builder_contract_and_expression_delta(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    res = build_dynamics_manifest_from_observed_transitions("examples/dynamics_transitions.example.json", str(out), strict=True)
    payload = json.loads(Path(res.manifest_path).read_text())
    assert payload["contract_version"] == "dynamics_graph_delta_manifest_v1"
    assert payload["manifest_type"] == "dynamics_graph_delta_manifest"
    rec = payload["records"][0]
    assert rec["target_delta"]["expression_deltas"]["smile_intensity"] > 0


def test_builder_strict_rejects_identical_graphs(tmp_path: Path) -> None:
    base = json.loads(Path("examples/dynamics_transitions.example.json").read_text())
    rec = dict(base["transitions"][0]); rec["graph_after"] = rec["graph_before"]; rec.pop("target_delta", None)
    inp = tmp_path / "in.json"; inp.write_text(json.dumps({"contract_version": base["contract_version"], "transitions": [rec]}), encoding="utf-8")
    with pytest.raises(ValueError):
        build_dynamics_manifest_from_observed_transitions(str(inp), str(tmp_path / "o.json"), strict=True)


def test_supervised_training_uses_supervised_loader_and_metrics(tmp_path: Path) -> None:
    result, _ = _train_supervised(tmp_path)
    ckpt_dir = Path(result.checkpoint_path).parent
    latest = json.loads((ckpt_dir / "latest.json").read_text())
    ds_diag = latest["dataset_profile"]["diagnostics"]
    assert ds_diag["source"] == "manifest_dynamics_graph_delta_supervised"
    assert ds_diag["supervised_dynamics_records"] > 0
    assert ds_diag["rejected_runtime_or_heuristic_targets"] == 0
    assert latest["training_source"] == "observed_graph_transition_supervised"
    assert latest["supervised_dynamics_record_count"] > 0
    assert "graph_delta_loss" in latest["val_metrics"]
    assert "supervised_dynamics_record_count" in latest["val_metrics"]


def test_strict_single_record_no_synthetic_val_fallback(tmp_path: Path) -> None:
    result, _ = _train_supervised(tmp_path)
    latest = json.loads((Path(result.checkpoint_path).parent / "latest.json").read_text())
    assert latest["dataset_profile"]["source"] == "manifest_single_sample_train_val_reuse"
    assert latest["dataset_profile"]["diagnostics"].get("split_strategy") == "single_sample_train_val_reuse"


def test_checkpoint_path_reloadable(tmp_path: Path) -> None:
    result, _ = _train_supervised(tmp_path)
    status = DynamicsRuntimeBundle(checkpoint_path=result.checkpoint_path).load_checkpoint()
    assert status.usable_for_inference is True


def test_strict_supervised_without_manifest_raises() -> None:
    cfg = TrainingConfig(epochs=1, dynamics_target_role_policy="supervised_only")
    with pytest.raises(ValueError):
        DynamicsTrainer().train(cfg)


def test_debug_training_metadata_is_synthetic(tmp_path: Path) -> None:
    cfg = TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"))
    result = DynamicsTrainer().train(cfg)
    latest = json.loads((Path(result.checkpoint_path).parent / "latest.json").read_text())
    assert latest["training_source"] == "synthetic_dynamics_bootstrap"
    assert latest["supervised_dynamics_record_count"] == 0


def test_orchestrator_dynamics_transition_routes_to_dynamics_trainer(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    build_dynamics_manifest_from_observed_transitions("examples/dynamics_transitions.example.json", str(manifest), strict=True)
    cfg = TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), learned_dataset_path=str(manifest), dynamics_target_role_policy="supervised_only")
    result = train_stage("dynamics_transition", cfg)
    assert result.stage_name == "dynamics_transition"
    assert result.checkpoint_path.endswith("dynamics_weights.json")
