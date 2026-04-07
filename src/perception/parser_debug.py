from __future__ import annotations

import json
from pathlib import Path

from perception.mask_store import DEFAULT_MASK_STORE
from perception.mask_projection import project_mask_to_frame
from perception.pipeline import PerceptionOutput


def _mask_to_rgba(mask: object, color: tuple[int, int, int], alpha: int = 120):
    arr = mask.tolist() if hasattr(mask, "tolist") else mask
    if not isinstance(arr, list) or not arr or not isinstance(arr[0], list):
        arr = [[0]]
    h = len(arr)
    w = len(arr[0]) if h else 1
    out = [[[color[0], color[1], color[2], alpha if int(arr[y][x]) > 0 else 0] for x in range(w)] for y in range(h)]
    return out


def _save_rgb_image(rgb: object, path: Path) -> None:
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(rgb).save(path)
        return
    except Exception:
        pass

    arr = rgb.tolist() if hasattr(rgb, "tolist") else rgb
    if not isinstance(arr, list) or not arr:
        arr = [[[0, 0, 0]]]
    h = len(arr)
    w = len(arr[0]) if h else 1
    with path.open("w", encoding="utf-8") as f:
        f.write(f"P3\n{w} {h}\n255\n")
        for row in arr:
            for px in row:
                r, g, b = (px + [0, 0, 0])[:3]
                f.write(f"{int(r)} {int(g)} {int(b)} ")
            f.write("\n")


def export_parser_debug_artifacts(frame_rgb: object, output: PerceptionOutput, out_dir: str | Path) -> dict[str, object]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    base = frame_rgb.tolist() if hasattr(frame_rgb, "tolist") else frame_rgb
    if not isinstance(base, list) or not base or not isinstance(base[0], list):
        base = [[[0, 0, 0]]]
    _save_rgb_image(base, target / "frame.png")

    summary: dict[str, object] = {"persons": []}
    palette = {
        "fashn": (255, 120, 0),
        "schp_pascal": (0, 220, 120),
        "schp_atr": (60, 180, 255),
        "facer": (255, 70, 180),
        "fusion": (255, 255, 0),
    }

    for pidx, person in enumerate(output.persons):
        person_dir = target / f"person_{pidx}"
        person_dir.mkdir(exist_ok=True)
        person_meta = {
            "mask_ref": person.mask_ref,
            "garment_masks": person.garment_masks,
            "body_part_masks": person.body_part_masks,
            "face_region_masks": person.face_region_masks,
            "accessory_masks": person.accessory_masks,
            "provenance_by_region": person.provenance_by_region,
        }
        (person_dir / "summary.json").write_text(json.dumps(person_meta, ensure_ascii=False, indent=2))

        layers: list[tuple[object, tuple[int, int, int]]] = []

        def _dump_mask_group(group_name: str, refs: dict[str, str]):
            group_dir = person_dir / group_name
            group_dir.mkdir(exist_ok=True)
            for label, ref in refs.items():
                stored = DEFAULT_MASK_STORE.get(ref)
                if stored is None:
                    continue
                payload, geometry = project_mask_to_frame(stored, frame_size=(len(base[0]), len(base)))
                source_key = stored.source.split(":")[-1]
                color = palette.get(source_key, (200, 200, 200))
                rgba = _mask_to_rgba(payload, color)
                _save_rgb_image([[px[:3] for px in row] for row in rgba], group_dir / f"{label}.png")
                layers.append((payload, color))
                person_meta.setdefault("mask_geometry", {})[label] = geometry

        _dump_mask_group("primary_fashn_masks", {k: v for k, v in person.garment_masks.items() if person.provenance_by_region.get(f"garment:{k}") == "parser:fashn"})
        _dump_mask_group("schp_pascal_masks", person.body_part_masks)
        _dump_mask_group("schp_atr_masks", {k: v for k, v in person.garment_masks.items() if person.provenance_by_region.get(f"garment:{k}") == "parser:schp_atr"})
        _dump_mask_group("facer_masks", person.face_region_masks)

        overlay = [[px[:] for px in row] for row in base]
        for payload, color in layers:
            arr = payload.tolist() if hasattr(payload, "tolist") else payload
            if not isinstance(arr, list) or not arr or len(arr) != len(overlay) or len(arr[0]) != len(overlay[0]):
                continue
            for y, row in enumerate(arr):
                for x, v in enumerate(row):
                    if int(v) > 0:
                        overlay[y][x] = [int(0.55 * overlay[y][x][i] + 0.45 * color[i]) for i in range(3)]
        _save_rgb_image(overlay, person_dir / "fused_masks_overlay.png")
        summary["persons"].append(person_meta)

    (target / "fused_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
