from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.datasets import RendererDataset
from training.renderer_observed_pairs_builder import build_renderer_manifest_from_observed_pairs


def _png(path: Path, value: float, *, shape: tuple[int, int] = (4, 4)) -> None:
    arr = np.full((shape[0], shape[1], 3), int(value * 255), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _input(tmp_path: Path, pairs: list[dict[str, object]]) -> Path:
    p = tmp_path / "observed_pairs.json"
    p.write_text(json.dumps({"contract_version": "renderer_observed_pair_manifest_input_v1", "pairs": pairs}), encoding="utf-8")
    return p


def _pair(src: str, tgt: str, *, regions: list[dict[str, object]] | None = None) -> dict[str, object]:
    d: dict[str, object] = {"record_id": "sample_1", "source_frame": src, "target_frame": tgt, "prompt": "smile"}
    d["regions"] = regions if regions is not None else [{"region_id": "person_1:face", "bbox": [0.0, 0.0, 1.0, 1.0], "reason": "manual"}]
    return d


def test_observed_pair_export_and_dataset_load(tmp_path: Path) -> None:
    src = tmp_path / "s.png"; tgt = tmp_path / "t.png"
    _png(src, 0.1); _png(tgt, 0.8)
    out = tmp_path / "manifest.json"
    build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, [_pair(str(src), str(tgt))])), output_path=str(out), strict=True)
    payload = json.loads(out.read_text(encoding="utf-8"))
    rec = payload["records"][0]
    assert payload["contract_version"] == "renderer_patch_manifest_v2"
    assert rec["target_source"] == "provided_ground_truth_roi"
    assert rec["training_target_quality"] == "external_or_observed_target"
    assert rec["target_training_role"] == "supervised_external"
    ds = RendererDataset.from_renderer_manifest(str(out), strict=True)
    before, after = ds.samples[0]["roi_pairs"][0]
    assert np.mean(np.asarray(before)) < np.mean(np.asarray(after))


def test_strict_missing_source_raises(tmp_path: Path) -> None:
    tgt = tmp_path / "t.png"; _png(tgt, 0.7)
    with pytest.raises(Exception):
        build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, [_pair(str(tmp_path/'missing.png'), str(tgt))])), output_path=str(tmp_path / "o.json"), strict=True)


def test_strict_missing_target_raises(tmp_path: Path) -> None:
    src = tmp_path / "s.png"; _png(src, 0.1)
    with pytest.raises(Exception):
        build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, [_pair(str(src), str(tmp_path/'missing.png'))])), output_path=str(tmp_path / "o.json"), strict=True)


def test_strict_no_regions_raises(tmp_path: Path) -> None:
    src = tmp_path / "s.png"; tgt = tmp_path / "t.png"
    _png(src, 0.1); _png(tgt, 0.2)
    with pytest.raises(ValueError, match="no valid regions"):
        build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, [_pair(str(src), str(tgt), regions=[])])), output_path=str(tmp_path / "o.json"), strict=True)


def test_strict_invalid_bbox_raises(tmp_path: Path) -> None:
    src = tmp_path / "s.png"; tgt = tmp_path / "t.png"
    _png(src, 0.1); _png(tgt, 0.2)
    bad = [{"region_id": "person_1:face", "bbox": [0.1, 0.1, -0.2, 0.3], "reason": "manual"}]
    with pytest.raises(ValueError, match="width/height"):
        build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, [_pair(str(src), str(tgt), regions=bad)])), output_path=str(tmp_path / "o.json"), strict=True)


def test_strict_shape_mismatch_raises(tmp_path: Path) -> None:
    src = tmp_path / "s.png"; tgt = tmp_path / "t.png"
    _png(src, 0.1, shape=(4, 4)); _png(tgt, 0.2, shape=(5, 4))
    with pytest.raises(ValueError, match="shapes mismatch"):
        build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, [_pair(str(src), str(tgt))])), output_path=str(tmp_path / "o.json"), strict=True)


def test_non_strict_skips_invalid_pair(tmp_path: Path) -> None:
    src = tmp_path / "s.png"; tgt = tmp_path / "t.png"
    _png(src, 0.1); _png(tgt, 0.9)
    pairs = [_pair(str(tmp_path / "missing.png"), str(tgt)), _pair(str(src), str(tgt))]
    out = tmp_path / "manifest.json"
    result = build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, pairs)), output_path=str(out), strict=False)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert result.diagnostics["skipped_pairs"] == 1
    assert payload["record_count"] == 1


def test_non_strict_all_invalid_raises_no_supervised_records(tmp_path: Path) -> None:
    tgt = tmp_path / "t.png"; _png(tgt, 0.9)
    with pytest.raises(ValueError, match="no supervised records exported"):
        build_renderer_manifest_from_observed_pairs(
            observed_pairs_path=str(_input(tmp_path, [_pair(str(tmp_path / "missing.png"), str(tgt))])),
            output_path=str(tmp_path / "manifest.json"),
            strict=False,
        )


def test_no_record_uses_self_generated_runtime_target(tmp_path: Path) -> None:
    src = tmp_path / "s.png"; tgt = tmp_path / "t.png"
    _png(src, 0.2); _png(tgt, 0.7)
    out = tmp_path / "manifest.json"
    build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, [_pair(str(src), str(tgt))])), output_path=str(out), strict=True)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert all(rec.get("target_source") != "self_generated_runtime_target" for rec in payload["records"])


def test_two_regions_have_unique_record_ids_and_shared_source_pair_id(tmp_path: Path) -> None:
    src = tmp_path / "s.png"; tgt = tmp_path / "t.png"
    _png(src, 0.2); _png(tgt, 0.7)
    regions = [
        {"region_id": "person_1:face", "bbox": [0.0, 0.0, 0.5, 0.5], "reason": "manual"},
        {"region_id": "person_1:torso", "bbox": [0.5, 0.5, 0.5, 0.5], "reason": "manual"},
    ]
    out = tmp_path / "manifest.json"
    build_renderer_manifest_from_observed_pairs(observed_pairs_path=str(_input(tmp_path, [_pair(str(src), str(tgt), regions=regions)])), output_path=str(out), strict=True)
    records = json.loads(out.read_text(encoding="utf-8"))["records"]
    assert len(records) == 2
    ids = [rec["record_id"] for rec in records]
    assert len(set(ids)) == 2
    assert all(rec["source_pair_id"] == "sample_1" for rec in records)
