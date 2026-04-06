from pathlib import Path

import pytest

from core.input_layer import InputAssetLayer


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_single_image_input(tmp_path: Path) -> None:
    img = tmp_path / "a.ppm"
    _write_ppm(img, 32, 16, (255, 0, 0))

    layer = InputAssetLayer()
    req = layer.build_request(images=[str(img)], text="smile", quality_profile="debug")

    assert req.input_type == "single_image"
    assert req.orig_size == (32, 16)
    assert req.unified_asset is not None
    assert len(req.unified_asset.frames) == 1


def test_multiple_image_input(tmp_path: Path) -> None:
    img1 = tmp_path / "a.ppm"
    img2 = tmp_path / "b.ppm"
    _write_ppm(img1, 16, 16, (0, 255, 0))
    _write_ppm(img2, 16, 16, (0, 0, 255))

    layer = InputAssetLayer()
    req = layer.build_request(images=[str(img1), str(img2)], text="smile")

    assert req.input_type == "multi_image"
    assert req.frame_count == 2
    assert len(req.unified_asset.frames) == 2


def test_missing_file_handling(tmp_path: Path) -> None:
    layer = InputAssetLayer()
    with pytest.raises(FileNotFoundError):
        layer.build_request(images=[str(tmp_path / "missing.ppm")], text="smile")


def test_video_metadata_extraction_fallback() -> None:
    layer = InputAssetLayer()
    req = layer.build_request(images=[], video="missing.mp4", text="smile", fps=10, duration=2.0)
    assert req.input_type == "video"
    assert req.frame_count >= 1
    assert req.orig_size is not None


def test_profile_dependent_resizing_behavior(tmp_path: Path) -> None:
    img = tmp_path / "big.ppm"
    _write_ppm(img, 300, 100, (128, 128, 128))
    layer = InputAssetLayer()
    debug_req = layer.build_request(images=[str(img)], text="", quality_profile="debug")
    quality_req = layer.build_request(images=[str(img)], text="", quality_profile="quality")

    assert debug_req.normalized_size != quality_req.normalized_size
