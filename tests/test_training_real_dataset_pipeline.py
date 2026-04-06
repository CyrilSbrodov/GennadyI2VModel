import json
from pathlib import Path

from training.datasets import MemoryDataset, PerceptionDataset, RepresentationDataset, TextActionDataset


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_real_manifest_to_representation_and_memory(tmp_path: Path) -> None:
    img = tmp_path / "img.ppm"
    _write_ppm(img, 32, 24, (20, 50, 200))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [{"image": str(img), "text": "улыбается"}]}))

    pds = PerceptionDataset.from_image_manifest(str(manifest), quality_profile="debug")
    rds = RepresentationDataset.from_perception_dataset(pds)
    mds = MemoryDataset.from_representation_dataset(rds)

    assert len(pds) == 1
    assert len(rds[0]["graphs"]) == 1
    assert mds[0]["memory_records"][0]["person_count"] >= 0


def test_text_action_dataset_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    path.write_text(json.dumps({"text": "садится на стул", "actions": [{"type": "sit_down", "target_object": "chair"}]}) + "\n")

    ds = TextActionDataset.from_jsonl(str(path))
    assert len(ds) == 1
    assert ds[0]["actions"][0].type == "sit_down"
