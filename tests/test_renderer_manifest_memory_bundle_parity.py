from __future__ import annotations

import json
from pathlib import Path


from rendering.patch_conditioning_contract import (
    APPEARANCE_DIM,
    BBOX_DIM,
    DELTA_DIM,
    GRAPH_DIM,
    MEMORY_DIM,
    PLANNER_DIM,
    SEMANTIC_DIM,
)
from training.datasets import RendererDataset
from training.renderer_trainer import RendererBatchAdapter, RendererTrainer


def _roi(value: float = 0.25) -> list[list[list[float]]]:
    return [[[value, value, value] for _ in range(4)] for _ in range(4)]


def _manifest(tmp_path: Path, sample: dict[str, object]) -> Path:
    path = tmp_path / "renderer_manifest.json"
    base = {
        "roi_before": _roi(0.25),
        "roi_after": _roi(0.35),
        "semantic_family": "face_expression",
        "region_id": "p1:face",
    }
    base.update(sample)
    path.write_text(json.dumps({"records": [base]}), encoding="utf-8")
    return path


def _adapt(path: Path):
    ds = RendererDataset.from_renderer_manifest(str(path), strict=False)
    batch = RendererBatchAdapter().adapt(ds.samples[0])
    return ds, batch


def test_renderer_manifest_memory_bundle_summary_reaches_patch_batch(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        {
            "renderer_memory_bundle": {
                "memory_bundle_present": True,
                "memory_support_level": "strong",
                "reveal_lifecycle": "stable",
                "has_identity_reference": True,
                "retrieval_reasons": ["identity_reference"],
            }
        },
    )

    _, batch = _adapt(path)

    assert batch.conditioning_summary["memory_bundle_present"] is True
    assert batch.conditioning_summary["memory_support_level"] == "strong"
    assert batch.conditioning_summary["memory_bundle_has_identity_reference"] is True


def test_renderer_manifest_revealed_history_not_active_hidden(tmp_path: Path) -> None:
    baseline_path = _manifest(tmp_path, {"memory_cond": [0.0] * 10})
    _, baseline = _adapt(baseline_path)
    path = _manifest(
        tmp_path,
        {
            "memory_cond": [0.0] * 10,
            "renderer_memory_bundle": {
                "memory_bundle_present": True,
                "memory_support_level": "medium",
                "reveal_lifecycle": "stable",
                "has_hidden_slot": True,
                "hidden_type": "revealed_history",
                "hidden_support_active": False,
            },
        },
    )

    _, batch = _adapt(path)

    assert batch.conditioning_summary["memory_bundle_is_revealed_history"] is True
    assert batch.conditioning_summary["memory_bundle_hidden_support_active"] is False
    assert batch.memory_cond[9] == baseline.memory_cond[9]


def test_renderer_manifest_low_evidence_newly_revealed_sets_risk(tmp_path: Path) -> None:
    baseline_path = _manifest(tmp_path, {"appearance_cond": [0.25, 0.25, 0.25, 0.02, 0.02, 0.02, 0.25, 0.02]})
    _, baseline = _adapt(baseline_path)
    path = _manifest(
        tmp_path,
        {
            "appearance_cond": [0.25, 0.25, 0.25, 0.02, 0.02, 0.02, 0.25, 0.02],
            "renderer_memory_bundle": {
                "memory_bundle_present": True,
                "memory_support_level": "weak",
                "reveal_lifecycle": "newly_revealed",
                "retrieval_reasons": ["low_evidence"],
            },
        },
    )

    _, batch = _adapt(path)

    assert batch.conditioning_summary["memory_bundle_low_evidence_newly_revealed"] is True
    assert batch.appearance_cond[7] > baseline.appearance_cond[7]


def test_old_renderer_manifest_without_memory_bundle_still_loads(tmp_path: Path) -> None:
    path = _manifest(tmp_path, {})

    _, batch = _adapt(path)

    assert batch.conditioning_summary["memory_bundle_present"] is False
    assert batch.conditioning_summary["memory_support_level"] == "none"


def test_invalid_memory_support_level_records_diagnostic(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        {
            "renderer_memory_bundle": {
                "memory_bundle_present": True,
                "memory_support_level": "super_strong",
            }
        },
    )

    ds, batch = _adapt(path)

    assert ds.diagnostics["invalid_memory_bundle_records"] == 1
    assert ds.diagnostics["memory_bundle_warnings"]
    assert batch.conditioning_summary["memory_support_level"] == "none"
    assert batch.memory_cond[6] < 1.0


def test_renderer_training_eval_metrics_uses_manifest_memory_bundle(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        {
            "renderer_memory_bundle": {
                "memory_bundle_present": True,
                "memory_support_level": "strong",
                "has_current_reuse": True,
            }
        },
    )
    ds = RendererDataset.from_renderer_manifest(str(path), strict=False)
    trainer = RendererTrainer()
    batch = RendererBatchAdapter().adapt(ds.samples[0])

    metrics = trainer.evaluate_model(trainer.model, [batch], renderer_backend="numpy_local")

    assert metrics["memory_bundle_conditioning_present"] > 0.0
    assert metrics["memory_support_level_strong_ratio"] > 0.0


def test_renderer_adapter_outputs_patch_batch_contract_dims() -> None:
    sample = RendererDataset.synthetic(1).samples[0]

    batch = RendererBatchAdapter().adapt(sample)

    assert len(batch.semantic_embed) == SEMANTIC_DIM
    assert len(batch.delta_cond) == DELTA_DIM
    assert len(batch.planner_cond) == PLANNER_DIM
    assert len(batch.graph_cond) == GRAPH_DIM
    assert len(batch.memory_cond) == MEMORY_DIM
    assert len(batch.appearance_cond) == APPEARANCE_DIM
    assert len(batch.bbox_cond) == BBOX_DIM


def test_renderer_adapter_memory_bundle_with_missing_region_id_does_not_crash(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        {
            "region_id": "",
            "renderer_memory_bundle": {
                "memory_bundle_present": True,
                "memory_support_level": "strong",
                "has_identity_reference": True,
            },
        },
    )
    ds = RendererDataset.from_renderer_manifest(str(path), strict=False)

    batch = RendererBatchAdapter().adapt(ds.samples[0])

    assert batch.conditioning_summary["memory_bundle_present"] is True
    assert batch.conditioning_summary["memory_support_level"] == "strong"
    assert batch.conditioning_summary["memory_bundle_has_identity_reference"] is True
