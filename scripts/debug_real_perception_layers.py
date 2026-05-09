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




def _project_local_xyxy_to_frame(
    bbox: tuple[float, float, float, float],
    roi_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if roi_bbox is None:
        return bbox
    rx, ry, rw, rh = [float(v) for v in roi_bbox]
    x1, y1, x2, y2 = bbox
    return (rx + x1 * rw, ry + y1 * rh, rx + x2 * rw, ry + y2 * rh)


def _bbox_for_stored_mask(item: Any) -> tuple[float, float, float, float] | None:
    bbox = _clamp_bbox_xyxy(item.extra.get("bbox_xyxy") if isinstance(item.extra, dict) else None)
    if bbox is not None:
        return bbox
    payload_bbox = _clamp_bbox_xyxy(_bbox_from_mask_payload(item.payload))
    if payload_bbox is None:
        return None
    return _clamp_bbox_xyxy(_project_local_xyxy_to_frame(payload_bbox, item.roi_bbox))


def _filter_refs_by_min_pixels(refs: list[str], min_pixels: int) -> list[str]:
    if min_pixels <= 0:
        return refs
    filtered: list[str] = []
    parser_kinds = {"body_part_mask", "face_region_mask", "garment_mask", "accessory_mask", "background_mask"}
    for ref in refs:
        item = DEFAULT_MASK_STORE.get(ref)
        if item is None:
            continue
        if item.mask_kind not in parser_kinds:
            filtered.append(ref)
            continue
        pixel_count = 0
        if isinstance(item.extra, dict):
            try:
                pixel_count = int(item.extra.get("pixel_count", 0) or 0)
            except Exception:
                pixel_count = 0
        if pixel_count >= min_pixels:
            filtered.append(ref)
    return filtered

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
        bbox = _bbox_for_stored_mask(item)
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




def _refs_from_parser_summary(output: Any, region_keys: set[str], *, fallback_kinds: set[str] | None = None) -> list[str]:
    """Resolve semantic overlay refs from parser_summary, not fragile ref names."""
    refs: list[str] = []
    summary = getattr(output, "parser_summary", {})
    refs_by_region = summary.get("region_mask_refs", {}) if isinstance(summary, dict) else {}
    if isinstance(refs_by_region, dict):
        for key in region_keys:
            values = refs_by_region.get(key, [])
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                refs.extend(str(v) for v in values)
    if not refs and fallback_kinds:
        store = getattr(output, "mask_store", {})
        if isinstance(store, dict):
            for ref, meta in store.items():
                if isinstance(meta, dict) and meta.get("mask_kind") in fallback_kinds:
                    refs.append(str(ref))
    return sorted(dict.fromkeys(refs))


def _mask_counts_by_kind(refs: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ref in refs:
        item = DEFAULT_MASK_STORE.get(ref)
        if item is None:
            continue
        counts[item.mask_kind] = counts.get(item.mask_kind, 0) + 1
    return counts

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
    module_fallbacks = getattr(output, "module_fallbacks", {})
    parser_summary = getattr(output, "parser_summary", {})
    lines = [
        f"persons count: {len(output.persons)}",
        f"mask refs count: {len(refs)}",
        f"detector mode/fallback: {module_fallbacks.get('detector', 'unknown')}",
        f"pose mode/fallback: {module_fallbacks.get('pose', 'unknown')}",
        f"parser mode/fallback: {module_fallbacks.get('parser', 'unknown')}",
        f"parser canonical regions: {json.dumps(parser_summary.get('canonical_regions', []), ensure_ascii=False) if isinstance(parser_summary, dict) else '[]'}",
        f"mask counts by kind: {json.dumps(_mask_counts_by_kind(refs), ensure_ascii=False)}",
        f"warnings: {json.dumps(output.warnings, ensure_ascii=False)}",
        f"diagnostics: {json.dumps(output.diagnostics, ensure_ascii=False)}",
        f"module_fallbacks: {json.dumps(module_fallbacks, ensure_ascii=False)}",
        f"parser_summary: {json.dumps(parser_summary, ensure_ascii=False)}",
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

    refs = _filter_refs_by_min_pixels(sorted(output.mask_store.keys()), args.min_parser_pixels)
    _overlay(img, refs, kinds={"person_mask"}).save(out_dir / "10_yolo_person_seg_overlay.jpg")
    _draw_pose(img, output.persons).save(out_dir / "20_yolo_pose_overlay.jpg")
    _overlay(img, refs, exclude_kinds={"person_mask", "background_mask"}).save(out_dir / "30_parser_all_masks_overlay.jpg")
    _grid(img, refs).save(out_dir / "31_parser_masks_grid.jpg")
    _overlay(img, refs, kinds={"body_part_mask"}).save(out_dir / "32_parser_body_parts_overlay.jpg")
    _overlay(img, refs, kinds={"garment_mask", "accessory_mask"}).save(out_dir / "33_parser_garments_overlay.jpg")
    face_hair_refs = _refs_from_parser_summary(
        output,
        {"face:face", "face:hair", "body:face", "body:hair", "canonical:face", "canonical:hair"},
        fallback_kinds={"face_region_mask"},
    )
    _overlay(img, face_hair_refs, kinds={"body_part_mask", "face_region_mask"}).save(out_dir / "34_parser_face_hair_overlay.jpg")
    _overlay(img, refs, kinds={"background_mask"}, background_alpha=90).save(out_dir / "35_parser_background_overlay.jpg")
    _overlay(_draw_pose(img, output.persons), refs, exclude_kinds={"background_mask"}).save(out_dir / "40_fusion_preview.jpg")
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
