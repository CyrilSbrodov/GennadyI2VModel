from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from PIL import Image, ImageDraw

from core.schema import BBox
from perception.detector import BackendConfig, PersonDetection
from perception.mask_store import DEFAULT_MASK_STORE
from perception.pipeline import PerceptionPipeline, real_human_parsing_config
from perception.pose import MediaPipePoseAdapter
from representation.graph_builder import SceneGraphBuilder


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_image(path: Path, downscale: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if downscale and downscale > 0 and max(img.size) > downscale:
        scale = downscale / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return img


def _mask_to_image(mask_payload: object, size: tuple[int, int]) -> Image.Image:
    arr = np.asarray(mask_payload, dtype=np.uint8)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.size == 0:
        arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr = (arr > 0).astype(np.uint8) * 255
    img = Image.fromarray(arr, mode="L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return img


def _bbox_from_mask_payload(mask_payload: object) -> tuple[float, float, float, float] | None:
    arr = np.asarray(mask_payload)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.size == 0 or arr.ndim != 2:
        return None
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = arr.shape
    return (float(xs.min()) / max(1, w), float(ys.min()) / max(1, h), float(xs.max() + 1) / max(1, w), float(ys.max() + 1) / max(1, h))


def _clamp_bbox_xyxy(bbox: object) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    vals = [max(0.0, min(1.0, float(v))) for v in bbox]
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _overlay(
    base: Image.Image,
    mask_refs: list[str],
    *,
    kinds: set[str] | None = None,
    exclude_kinds: set[str] | None = None,
    background_alpha: int = 24,
) -> Image.Image:
    out = base.convert("RGBA")
    colors = [(255, 0, 0, 92), (0, 255, 0, 92), (0, 128, 255, 92), (255, 200, 0, 92), (255, 0, 255, 92), (0, 255, 255, 92)]
    draw = ImageDraw.Draw(out)
    ordered_refs = sorted(mask_refs, key=lambda r: 1 if (DEFAULT_MASK_STORE.get(r) and DEFAULT_MASK_STORE.get(r).mask_kind == "background_mask") else 0)
    for idx, ref in enumerate(ordered_refs):
        item = DEFAULT_MASK_STORE.get(ref)
        if item is None:
            continue
        if kinds is not None and item.mask_kind not in kinds:
            continue
        if exclude_kinds is not None and item.mask_kind in exclude_kinds:
            continue
        color = colors[idx % len(colors)]
        if item.mask_kind == "background_mask":
            color = color[:3] + (background_alpha,)
        mask = _mask_to_image(item.payload, base.size)
        layer = Image.new("RGBA", base.size, color)
        out.alpha_composite(Image.composite(layer, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask))
        bbox = _clamp_bbox_xyxy(item.extra.get("bbox_xyxy") if isinstance(item.extra, dict) else None) or _bbox_from_mask_payload(item.payload)
        bbox = _clamp_bbox_xyxy(bbox)
        if bbox:
            x1, y1, x2, y2 = bbox
            box = (int(x1 * base.width), int(y1 * base.height), int(x2 * base.width), int(y2 * base.height))
            draw.rectangle(box, outline=color[:3] + (255,), width=2)
            draw.text((box[0], max(0, box[1] - 12)), f"{item.mask_kind}:{ref.split('/')[-1]}", fill=color[:3] + (255,))
    return out.convert("RGB")


def _draw_pose(base: Image.Image, persons: list[Any]) -> Image.Image:
    out = base.copy()
    draw = ImageDraw.Draw(out)
    for person in persons:
        for kp in person.pose.keypoints:
            if kp.confidence < 0.2:
                continue
            x, y = int(kp.x * base.width), int(kp.y * base.height)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 0))
            draw.text((x + 4, y), kp.name, fill=(0, 255, 0))
    return out


def _draw_pose_predictions(base: Image.Image, predictions: dict[str, Any]) -> Image.Image:
    out = base.copy()
    draw = ImageDraw.Draw(out)
    for pred in predictions.values():
        for kp in pred.pose.keypoints:
            if kp.confidence < 0.2:
                continue
            x, y = int(kp.x * base.width), int(kp.y * base.height)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 180, 0))
            draw.text((x + 4, y), kp.name, fill=(255, 180, 0))
    return out


def _grid(base: Image.Image, mask_refs: list[str]) -> Image.Image:
    cells = []
    for ref in mask_refs:
        item = DEFAULT_MASK_STORE.get(ref)
        if item is None:
            continue
        mask = _mask_to_image(item.payload, base.size).convert("RGB")
        thumb = mask.resize((180, 180), Image.Resampling.NEAREST)
        d = ImageDraw.Draw(thumb)
        d.text((4, 4), f"{item.mask_kind}\n{item.source}\n{ref.split('/')[-1]}", fill=(255, 0, 0))
        cells.append(thumb)
    if not cells:
        return base.copy()
    cols = min(4, len(cells))
    rows = (len(cells) + cols - 1) // cols
    out = Image.new("RGB", (cols * 180, rows * 180), (30, 30, 30))
    for i, cell in enumerate(cells):
        out.paste(cell, ((i % cols) * 180, (i // cols) * 180))
    return out


def _summary_text(output, refs: list[str]) -> str:
    lines = [
        f"persons count: {len(output.persons)}",
        f"mask refs count: {len(refs)}",
        f"module_fallbacks: {json.dumps(output.module_fallbacks, ensure_ascii=False)}",
        f"warnings: {json.dumps(output.warnings, ensure_ascii=False)}",
        f"parser_summary: {json.dumps(output.parser_summary, ensure_ascii=False)}",
        "masks:",
    ]
    for ref in refs:
        item = DEFAULT_MASK_STORE.get(ref)
        if item is None:
            continue
        extra = item.extra if isinstance(item.extra, dict) else {}
        lines.append(
            f"- {ref}: mask_kind={item.mask_kind} source={item.source} backend={item.backend} "
            f"confidence={item.confidence:.3f} bbox_xyxy={extra.get('bbox_xyxy')} "
            f"pixel_count={extra.get('pixel_count')} tags={item.tags}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Project-native debug export for real human perception layers.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--parser-model", default="fashn-ai/fashn-human-parser")
    ap.add_argument("--disable-yolo", action="store_true")
    ap.add_argument("--disable-yolo-pose", action="store_true")
    ap.add_argument("--disable-parser", action="store_true")
    ap.add_argument("--disable-mediapipe", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--min-parser-pixels", type=int, default=100)
    ap.add_argument("--downscale", type=int, default=None)
    args = ap.parse_args()

    DEFAULT_MASK_STORE.clear()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = _load_image(Path(args.image), args.downscale)
    img.save(out_dir / "00_original.jpg")

    cfg = real_human_parsing_config(device=args.device, parser_model=args.parser_model, strict_perception=args.strict)
    if args.disable_yolo:
        cfg.detector.backend = "builtin"
    if args.disable_yolo_pose:
        cfg.pose.backend = "builtin"
    if args.disable_parser:
        from perception.parser import ParserStackConfig

        cfg.parser = ParserStackConfig()

    pipe = PerceptionPipeline(backends=cfg)
    frame = np.asarray(img).tolist()
    output = pipe.analyze(frame)
    graph = SceneGraphBuilder().build(output)

    mediapipe_summary: dict[str, Any] = {"enabled": not args.disable_mediapipe, "warnings": []}
    if not args.disable_mediapipe:
        try:
            persons = [
                PersonDetection(p.track_id or f"p{i}", p.bbox, p.bbox_confidence, p.bbox_source, p.mask_ref, p.mask_confidence, p.mask_source)
                for i, p in enumerate(output.persons, start=1)
            ]
            mp_preds = MediaPipePoseAdapter(BackendConfig(backend="mediapipe", device=cfg.perception_device)).estimate(frame, persons)
            _draw_pose_predictions(img, mp_preds).save(out_dir / "21_mediapipe_pose_overlay.jpg")
            mediapipe_summary["poses"] = _jsonify(mp_preds)
        except Exception as exc:
            mediapipe_summary["warnings"].append(str(exc))
            if args.strict:
                raise RuntimeError(f"strict mediapipe perception backend failed: {exc}") from exc

    refs = sorted(output.mask_store.keys())
    _overlay(img, refs, kinds={"person_mask"}).save(out_dir / "10_yolo_person_seg_overlay.jpg")
    _draw_pose(img, output.persons).save(out_dir / "20_yolo_pose_overlay.jpg")
    _overlay(img, refs, exclude_kinds={"person_mask"}, background_alpha=24).save(out_dir / "30_parser_all_masks_overlay.jpg")
    _grid(img, refs).save(out_dir / "31_parser_masks_grid.jpg")
    _overlay(img, refs, kinds={"body_part_mask"}).save(out_dir / "32_parser_body_parts_overlay.jpg")
    _overlay(img, refs, kinds={"garment_mask", "accessory_mask"}).save(out_dir / "33_parser_garments_overlay.jpg")
    _overlay(img, [r for r in refs if any(tag in r.lower() for tag in ("face", "hair"))], kinds={"body_part_mask", "face_region_mask"}).save(out_dir / "34_parser_face_hair_overlay.jpg")
    _overlay(img, refs, kinds={"background_mask"}, background_alpha=90).save(out_dir / "35_parser_background_overlay.jpg")
    _overlay(_draw_pose(img, output.persons), refs, background_alpha=18).save(out_dir / "40_fusion_preview.jpg")
    _overlay(img, refs, exclude_kinds={"background_mask"}).save(out_dir / "50_scene_graph_overlay.jpg")

    summary = {"perception": _jsonify(output), "mask_store": output.mask_store, "parser_summary": output.parser_summary, "diagnostics": output.diagnostics, "mediapipe": mediapipe_summary, "scene_graph": _jsonify(graph)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "summary.txt").write_text(_summary_text(output, refs))
    for ref in refs:
        item = DEFAULT_MASK_STORE.get(ref)
        if item is None:
            continue
        safe = ref.replace("mask://", "").replace("/", "_")
        _mask_to_image(item.payload, img.size).save(out_dir / f"mask_{safe}.png")


if __name__ == "__main__":
    main()
