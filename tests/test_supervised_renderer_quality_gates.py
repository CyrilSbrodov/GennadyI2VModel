from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.datasets import RendererDataset
from training.renderer_trainer import RendererTrainer
from training.supervised_renderer_manifest import (
    build_supervised_renderer_record,
    validate_supervised_renderer_record,
    write_supervised_renderer_manifest,
)
from training.types import TrainingConfig


def _roi(value: float, shape: tuple[int, int, int] = (8, 8, 3)) -> list:
    return np.full(shape, value, dtype=np.float32).tolist()


def _changed_roi(
    *,
    before_value: float = 0.1,
    after_value: float = 0.1,
    family: str = "face_expression",
    region_id: str = "p1:face",
    shape: tuple[int, int, int] = (8, 8, 3),
) -> dict[str, object]:
    before = np.full(shape, before_value, dtype=np.float32)
    after = np.full(shape, after_value, dtype=np.float32)
    h, w, _ = shape
    after[h // 4 : h // 2, w // 4 : w // 2, :] = min(1.0, after_value + 0.4)
    changed_mask = (np.mean(np.abs(after - before), axis=2) > 0.03).astype(np.float32).tolist()
    return build_supervised_renderer_record(
        roi_before=before.tolist(),
        roi_after=after.tolist(),
        region_id=region_id,
        semantic_family=family,
        bbox=[0.1, 0.2, 0.6, 0.7],
        changed_mask=changed_mask,
    )


def _bootstrap_record() -> dict[str, object]:
    return {
        "roi_before": _roi(0.1),
        "roi_after": _roi(0.2),
        "region_id": "p1:arm",
        "semantic_family": "sleeve_arm_transition",
        "target_source": "runtime_output_patch",
        "training_target_quality": "self_generated_runtime_target",
        "target_training_role": "bootstrap_self_generated",
    }


def test_validate_supervised_record_valid_report() -> None:
    record = _changed_roi(family="face_expression")

    report = validate_supervised_renderer_record(record, strict=True)

    assert report["valid"] is True
    assert report["roi_height"] == 8
    assert report["roi_width"] == 8
    assert report["channels"] == 3
    assert report["mean_abs_delta"] > 0
    assert report["changed_ratio"] > 0
    assert report["semantic_family"] == "face_expression"


def test_validate_supervised_record_rejects_shape_mismatch() -> None:
    record = build_supervised_renderer_record(
        roi_before=_roi(0.1, (4, 4, 3)),
        roi_after=_roi(0.2, (5, 4, 3)),
        region_id="p1:face",
        semantic_family="face_expression",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )

    with pytest.raises(ValueError, match="shapes mismatch"):
        validate_supervised_renderer_record(record, strict=True)


def test_validate_supervised_record_rejects_nan() -> None:
    after = np.full((8, 8, 3), 0.2, dtype=np.float32)
    after[0, 0, 0] = np.nan
    record = build_supervised_renderer_record(
        roi_before=_roi(0.1),
        roi_after=after.tolist(),
        region_id="p1:face",
        semantic_family="face_expression",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )

    with pytest.raises(ValueError, match="NaN or inf"):
        validate_supervised_renderer_record(record, strict=True)


def test_validate_supervised_record_warns_tiny_roi_and_low_motion() -> None:
    record = build_supervised_renderer_record(
        roi_before=_roi(0.1, (4, 4, 3)),
        roi_after=_roi(0.1, (4, 4, 3)),
        region_id="p1:face",
        semantic_family="face_expression",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )

    report = validate_supervised_renderer_record(record, strict=True)
    warnings = "\n".join(report["warnings"])

    assert report["valid"] is True
    assert "tiny ROI" in warnings
    assert "mean_abs_delta" in warnings
    assert "changed_ratio" in warnings


def test_write_manifest_includes_quality_metadata(tmp_path: Path) -> None:
    records = [
        _changed_roi(family="face_expression", region_id="p1:face"),
        _changed_roi(family="torso_reveal", region_id="p1:torso"),
    ]
    path = Path(write_supervised_renderer_manifest(records, str(tmp_path / "supervised.json")))

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["valid_record_count"] == 2
    assert payload["invalid_record_count"] == 0
    assert payload["avg_changed_ratio"] > 0
    assert payload["semantic_family_counts"]["face_expression"] == 1
    assert payload["semantic_family_counts"]["torso_reveal"] == 1
    assert payload["records"][0]["supervised_quality"]["valid"] is True


def test_dataset_preserves_supervised_quality_diagnostics(tmp_path: Path) -> None:
    records = [
        _changed_roi(family="face_expression", region_id="p1:face"),
        _changed_roi(family="torso_reveal", region_id="p1:torso"),
    ]
    path = write_supervised_renderer_manifest(records, str(tmp_path / "supervised.json"))

    ds = RendererDataset.from_renderer_manifest(path, strict=True)
    sample = ds.samples[0]
    contract = sample["renderer_batch_contract"]

    assert "supervised_quality" in sample
    assert "supervised_quality" in contract
    assert ds.diagnostics["supervised_quality_present_count"] == len(records)
    assert "supervised_quality_avg_changed_ratio" in ds.diagnostics
    assert "supervised_quality_avg_mean_abs_delta" in ds.diagnostics


def test_trainer_checkpoint_metadata_includes_supervised_quality(tmp_path: Path) -> None:
    records = [
        _changed_roi(family="face_expression", region_id="p1:face"),
        _changed_roi(family="torso_reveal", region_id="p1:torso"),
    ]
    manifest_path = write_supervised_renderer_manifest(records, str(tmp_path / "supervised.json"))

    result = RendererTrainer().train(
        TrainingConfig(
            learned_dataset_path=manifest_path,
            renderer_target_role_policy="supervised_only",
            renderer_backend="numpy_local",
            epochs=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
    )
    metadata = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))["renderer_model_metadata"]

    assert "supervised_quality_avg_changed_ratio" in metadata
    assert metadata["supervised_quality_avg_changed_ratio"] > 0
    assert metadata["supervised_quality_semantic_family_counts"]


def test_writer_rejects_non_supervised_record_quality_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a supervised renderer record"):
        write_supervised_renderer_manifest([_bootstrap_record()], str(tmp_path / "bad.json"))


def test_invalid_mask_shape_warns_not_fails() -> None:
    record = _changed_roi()
    record["changed_mask"] = [[1.0, 0.0]]

    report = validate_supervised_renderer_record(record, strict=True)

    assert report["valid"] is True
    assert any("changed_mask" in warning and "shape" in warning for warning in report["warnings"])
