from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
PAIR_RE = re.compile(r"^(source|target)_(\d+)$")


class PairBuildError(RuntimeError):
    pass


# =========================
# Generic helpers
# =========================

def _elapsed(start: float) -> float:
    return round(perf_counter() - start, 6)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _first_attr(obj: Any, names: list[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _bbox_to_xywh_tuple(bbox_obj: Any) -> tuple[float, float, float, float] | None:
    """
    Accepts:
    - dataclass/object with x,y,w,h
    - dict with x,y,w,h
    - list/tuple [x,y,w,h]
    Returns pixel or normalized xywh depending on source object.
    """
    if bbox_obj is None:
        return None

    if isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) == 4:
        try:
            x, y, w, h = [float(v) for v in bbox_obj]
            return x, y, w, h
        except Exception:
            return None

    if isinstance(bbox_obj, dict):
        keys = ["x", "y", "w", "h"]
        if all(k in bbox_obj for k in keys):
            try:
                return (
                    float(bbox_obj["x"]),
                    float(bbox_obj["y"]),
                    float(bbox_obj["w"]),
                    float(bbox_obj["h"]),
                )
            except Exception:
                return None

    x = _first_attr(bbox_obj, ["x"])
    y = _first_attr(bbox_obj, ["y"])
    w = _first_attr(bbox_obj, ["w"])
    h = _first_attr(bbox_obj, ["h"])
    if None not in (x, y, w, h):
        try:
            return float(x), float(y), float(w), float(h)
        except Exception:
            return None

    return None


def _validate_normalized_bbox(bbox: list[float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 elements [x, y, w, h]")

    x, y, w, h = bbox

    if not (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 <= w <= 1.0
        and 0.0 <= h <= 1.0
    ):
        raise ValueError(f"bbox values must be normalized within [0,1], got {bbox}")

    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"bbox width/height must be > 0, got {bbox}")

    if x + w > 1.0 + 1e-6 or y + h > 1.0 + 1e-6:
        raise ValueError(f"bbox extends outside image bounds, got {bbox}")


def _union_boxes_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left = min(a[0], b[0])
    top = min(a[1], b[1])
    right = max(a[2], b[2])
    bottom = max(a[3], b[3])
    return left, top, right, bottom


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return x, y, x + w, y + h


def _xyxy_to_normalized_xywh(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_w: int,
    image_h: int,
) -> list[float]:
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w and image_h must be > 0")

    x1 = max(0.0, min(float(image_w), float(x1)))
    y1 = max(0.0, min(float(image_h), float(y1)))
    x2 = max(0.0, min(float(image_w), float(x2)))
    y2 = max(0.0, min(float(image_h), float(y2)))

    x = x1 / image_w
    y = y1 / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h

    out = [
        round(float(x), 6),
        round(float(y), 6),
        round(float(w), 6),
        round(float(h), 6),
    ]
    _validate_normalized_bbox(out)
    return out


def _maybe_normalized_to_pixels(
    bbox_xywh: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """
    Some repo bboxes may already be normalized. Image detector bboxes are pixel-based.
    Heuristic:
    - if all values are <= 1.0, treat as normalized and convert to pixels.
    - otherwise treat as pixel bbox.
    """
    x, y, w, h = bbox_xywh
    image_w, image_h = image_size

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
        return x * image_w, y * image_h, w * image_w, h * image_h

    return bbox_xywh


# =========================
# Pair discovery
# =========================

def discover_pairs(pairs_dir: Path) -> list[dict[str, Any]]:
    """
    Finds files like:
      source_0001.png
      target_0001.png

    Supports:
      .png .jpg .jpeg
    """
    groups: dict[str, dict[str, Path]] = {}

    for path in pairs_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        match = PAIR_RE.match(path.stem)
        if not match:
            continue

        kind, idx = match.groups()
        slot = groups.setdefault(idx, {})
        slot[kind] = path

    out: list[dict[str, Any]] = []
    for idx in sorted(groups.keys(), key=lambda s: int(s) if s.isdigit() else s):
        item = groups[idx]
        out.append(
            {
                "index": idx,
                "source": item.get("source"),
                "target": item.get("target"),
            }
        )

    return out


# =========================
# Perception adapter
# =========================

class PerceptionAdapter:
    """
    Lightweight adapter around existing repo perception code.

    It intentionally tries several entry point styles, because the local repo
    perception API may evolve.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._backend_name = "unknown"
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        try:
            from perception import pipeline as perception_pipeline  # type: ignore
        except Exception as err:
            raise PairBuildError(
                "Failed to import perception.pipeline. "
                "Run this script from repo root with PYTHONPATH=src or activate project environment."
            ) from err

        factory_names = [
            "build_default_pipeline",
            "get_default_pipeline",
            "create_default_pipeline",
            "build_pipeline",
        ]
        for name in factory_names:
            fn = getattr(perception_pipeline, name, None)
            if callable(fn):
                self._pipeline = fn()
                self._backend_name = name
                return

        class_names = [
            "PerceptionPipeline",
            "Pipeline",
            "DefaultPerceptionPipeline",
        ]
        for name in class_names:
            cls = getattr(perception_pipeline, name, None)
            if cls is not None:
                self._pipeline = cls()
                self._backend_name = name
                return

        module_level_callables = [
            "run_perception",
            "analyze_image",
            "infer",
            "process_image",
        ]
        for name in module_level_callables:
            fn = getattr(perception_pipeline, name, None)
            if callable(fn):
                self._pipeline = fn
                self._backend_name = name
                return

        raise PairBuildError(
            "Could not initialize perception pipeline from src/perception/pipeline.py. "
            "Expected one of factory/class/module-level entry points."
        )

    def run(self, image_path: Path) -> dict[str, Any]:
        """
        Returns:
          {
            "scene_graph": ...,
            "raw_result": ...,
            "backend_name": ...
          }
        """
        result = None

        for method_name in ["run", "infer", "process", "analyze"]:
            method = getattr(self._pipeline, method_name, None)
            if callable(method):
                try:
                    result = method(str(image_path))
                    break
                except TypeError:
                    img = Image.open(image_path).convert("RGB")
                    result = method(img)
                    break
                except Exception:
                    continue

        if result is None and callable(self._pipeline):
            try:
                result = self._pipeline(str(image_path))
            except TypeError:
                img = Image.open(image_path).convert("RGB")
                result = self._pipeline(img)

        if result is None:
            raise PairBuildError(f"Perception pipeline could not process image: {image_path}")

        scene_graph = _first_attr(result, ["scene_graph", "graph"], default=None)
        if scene_graph is None:
            scene_graph = result

        return {
            "scene_graph": scene_graph,
            "raw_result": result,
            "backend_name": self._backend_name,
        }


# =========================
# Face bbox extraction from scene graph
# =========================

def _iter_persons(scene_graph: Any) -> list[Any]:
    persons = _first_attr(scene_graph, ["persons", "people"], default=None)
    if persons is None:
        return []
    return list(persons)


def _iter_body_parts(person: Any) -> list[Any]:
    body_parts = _first_attr(person, ["body_parts", "parts"], default=None)
    if body_parts is None:
        return []
    return list(body_parts)


def _person_bbox_xywh(person: Any) -> tuple[float, float, float, float] | None:
    return _bbox_to_xywh_tuple(_first_attr(person, ["bbox"]))


def _face_like_score(body_part: Any) -> int:
    part_type = str(_first_attr(body_part, ["part_type", "type"], "") or "").lower()
    part_id = str(_first_attr(body_part, ["part_id", "id"], "") or "").lower()

    score = 0
    if "face" in part_type:
        score += 20
    if "face" in part_id:
        score += 15
    if "head" in part_type:
        score += 5
    if "head" in part_id:
        score += 3
    return score


def extract_face_bbox_from_scene_graph(
    scene_graph: Any,
    *,
    strict: bool,
    allow_person_fallback_non_strict: bool = False,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    """
    Returns bbox xywh. It may be pixel or normalized depending on graph contract.
    Later it is converted to pixel coordinates by _maybe_normalized_to_pixels().
    """
    persons = _iter_persons(scene_graph)
    if not persons:
        raise PairBuildError("No persons found in scene graph")

    best_person = None
    best_part = None
    best_score = -1

    for person in persons:
        for part in _iter_body_parts(person):
            score = _face_like_score(part)
            if score > best_score:
                bbox = _bbox_to_xywh_tuple(_first_attr(part, ["bbox"]))
                if bbox is not None:
                    best_score = score
                    best_person = person
                    best_part = part

    if best_part is not None and best_score > 0:
        bbox = _bbox_to_xywh_tuple(_first_attr(best_part, ["bbox"]))
        if bbox is None:
            raise PairBuildError("Face body part exists but bbox is missing/invalid")

        confidence = _first_attr(best_part, ["confidence"], default=None)
        person_id = str(_first_attr(best_person, ["person_id", "id"], default="person_1"))
        part_id = str(_first_attr(best_part, ["part_id", "id"], default="face"))
        part_type = str(_first_attr(best_part, ["part_type", "type"], default="face"))

        meta = {
            "person_id": person_id or "person_1",
            "part_id": part_id or "face",
            "part_type": part_type or "face",
            "confidence": float(confidence) if confidence is not None else None,
            "source_node_type": "body_part",
            "roi_source": "scene_graph_face_body_part",
            "fallback_used": False,
        }
        return bbox, meta

    if strict or not allow_person_fallback_non_strict:
        raise PairBuildError("Face body part bbox not found in scene graph")

    person = persons[0]
    person_bbox = _person_bbox_xywh(person)
    if person_bbox is None:
        raise PairBuildError("Non-strict fallback failed: first person bbox missing")

    meta = {
        "person_id": str(_first_attr(person, ["person_id", "id"], default="person_1")),
        "part_id": "face",
        "part_type": "fallback_person_bbox",
        "confidence": float(_first_attr(person, ["confidence"], default=0.0) or 0.0),
        "source_node_type": "person",
        "roi_source": "fallback_person_bbox_for_face_missing",
        "fallback_used": True,
    }
    return person_bbox, meta


# =========================
# Face bbox extraction from image detector
# =========================

def extract_face_bbox_from_image_detector(
    image_path: Path,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    """
    Image-level face fallback detector.
    Returns pixel xywh bbox.

    Priority:
    1. MediaPipe face detection if installed
    2. OpenCV Haar cascade if installed
    """
    img = Image.open(image_path).convert("RGB")
    image_w, image_h = img.size

    # 1) MediaPipe
    try:
        import numpy as np
        import mediapipe as mp  # type: ignore

        arr = np.asarray(img)
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.35,
        ) as face_detection:
            result = face_detection.process(arr)

        detections = result.detections or []
        if detections:
            best = max(
                detections,
                key=lambda d: float(d.score[0]) if getattr(d, "score", None) else 0.0,
            )
            box = best.location_data.relative_bounding_box

            x = max(0.0, float(box.xmin) * image_w)
            y = max(0.0, float(box.ymin) * image_h)
            w = max(1.0, float(box.width) * image_w)
            h = max(1.0, float(box.height) * image_h)

            pad_x = 0.12 * w
            pad_y = 0.18 * h

            x1 = max(0.0, x - pad_x)
            y1 = max(0.0, y - pad_y)
            x2 = min(float(image_w), x + w + pad_x)
            y2 = min(float(image_h), y + h + pad_y)

            confidence = float(best.score[0]) if getattr(best, "score", None) else 0.75

            return (
                (x1, y1, x2 - x1, y2 - y1),
                {
                    "person_id": "person_1",
                    "part_id": "face",
                    "part_type": "face",
                    "confidence": confidence,
                    "source_node_type": "image_face_detector",
                    "roi_source": "mediapipe_face_detection",
                    "fallback_used": False,
                },
            )
    except Exception:
        pass

    # 2) OpenCV Haar
    try:
        import cv2  # type: ignore
        import numpy as np

        arr = np.asarray(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(32, 32),
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))

            pad_x = 0.12 * float(w)
            pad_y = 0.18 * float(h)

            x1 = max(0.0, float(x) - pad_x)
            y1 = max(0.0, float(y) - pad_y)
            x2 = min(float(image_w), float(x + w) + pad_x)
            y2 = min(float(image_h), float(y + h) + pad_y)

            return (
                (x1, y1, x2 - x1, y2 - y1),
                {
                    "person_id": "person_1",
                    "part_id": "face",
                    "part_type": "face",
                    "confidence": 0.7,
                    "source_node_type": "image_face_detector",
                    "roi_source": "opencv_haar_face_detection",
                    "fallback_used": False,
                },
            )
    except Exception:
        pass

    raise PairBuildError(f"No face detected by image-level face detectors: {image_path}")


# =========================
# Record building
# =========================

def build_pair_record(
    *,
    index: str,
    source_path: Path,
    target_path: Path,
    source_img_size: tuple[int, int],
    target_img_size: tuple[int, int],
    source_face_bbox_xywh: tuple[float, float, float, float],
    target_face_bbox_xywh: tuple[float, float, float, float],
    source_face_meta: dict[str, Any],
    target_face_meta: dict[str, Any],
    prompt: str,
    transition_family: str,
    action_family: str,
    summary: str | None,
) -> dict[str, Any]:
    image_w, image_h = source_img_size

    if target_img_size != source_img_size:
        raise PairBuildError(
            f"Image size mismatch for pair {index}: "
            f"source={source_img_size}, target={target_img_size}"
        )

    source_face_bbox_xywh = _maybe_normalized_to_pixels(source_face_bbox_xywh, source_img_size)
    target_face_bbox_xywh = _maybe_normalized_to_pixels(target_face_bbox_xywh, target_img_size)

    sx, sy, sw, sh = source_face_bbox_xywh
    tx, ty, tw, th = target_face_bbox_xywh

    s_xyxy = _xywh_to_xyxy(sx, sy, sw, sh)
    t_xyxy = _xywh_to_xyxy(tx, ty, tw, th)
    ux1, uy1, ux2, uy2 = _union_boxes_xyxy(s_xyxy, t_xyxy)

    bbox = _xyxy_to_normalized_xywh(ux1, uy1, ux2, uy2, image_w, image_h)

    source_conf = source_face_meta.get("confidence")
    target_conf = target_face_meta.get("confidence")
    conf_values = [float(v) for v in [source_conf, target_conf] if v is not None]
    mean_conf = sum(conf_values) / len(conf_values) if conf_values else 0.8

    fallback_used = bool(source_face_meta.get("fallback_used")) or bool(target_face_meta.get("fallback_used"))

    metadata_completeness_score = 0.55 if fallback_used else 0.95
    evidence_strength_score = 0.45 if fallback_used else max(0.5, min(0.99, mean_conf))

    region_id = "person_1:face"
    record_id = f"auto_pair_{int(index):04d}" if str(index).isdigit() else f"auto_pair_{index}"

    return {
        "record_id": record_id,
        "source_frame": str(source_path).replace("\\", "/"),
        "target_frame": str(target_path).replace("\\", "/"),
        "prompt": prompt,
        "transition_context": {
            "summary": summary or "auto-generated face expression pair",
            "family": transition_family,
            "action_family": action_family,
        },
        "tags": ["face", "observed_pair", "auto_generated_pair"],
        "regions": [
            {
                "region_id": region_id,
                "bbox": bbox,
                "reason": (
                    "fallback_person_bbox_for_face_missing"
                    if fallback_used
                    else "auto_face_region_from_perception_union"
                ),
            }
        ],
        "region_metadata": {
            region_id: {
                "roi_source": (
                    "fallback_person_bbox_for_face_missing"
                    if fallback_used
                    else "auto_face_region_from_perception_union"
                ),
                "source_node_type": "body_part" if not fallback_used else "person",
                "metadata_completeness_score": round(metadata_completeness_score, 4),
                "evidence_strength_score": round(evidence_strength_score, 4),
                "mask_kind": "face_bbox",
                "source_frame_face_detected": True,
                "target_frame_face_detected": True,
                "source_detection_backend": source_face_meta.get("roi_source", "unknown"),
                "target_detection_backend": target_face_meta.get("roi_source", "unknown"),
                "source_node_type_raw": source_face_meta.get("source_node_type", "unknown"),
                "target_node_type_raw": target_face_meta.get("source_node_type", "unknown"),
                "notes": "auto-generated from perception/image-detector face bbox union",
            }
        },
    }


# =========================
# Main builder
# =========================

def build_renderer_pairs_manifest_auto(
    *,
    pairs_dir: Path,
    output_path: Path,
    prompt: str,
    transition_family: str,
    action_family: str,
    summary: str | None,
    strict: bool,
    allow_person_fallback_non_strict: bool,
) -> dict[str, Any]:
    if not pairs_dir.exists():
        raise PairBuildError(f"pairs_dir does not exist: {pairs_dir}")

    discovered = discover_pairs(pairs_dir)
    if not discovered:
        raise PairBuildError(f"No source/target pairs found in: {pairs_dir}")

    diagnostics: dict[str, Any] = {
        "strict": strict,
        "pairs_dir": str(pairs_dir),
        "total_pairs_found": len(discovered),
        "exported_pairs": 0,
        "skipped_pairs": 0,
        "skipped_by_reason": Counter(),
        "source_face_missing_count": 0,
        "target_face_missing_count": 0,
        "size_mismatch_count": 0,
        "missing_target_count": 0,
        "missing_source_count": 0,
        "perception_failed_count": 0,
        "timing": {
            "total_sec": 0.0,
            "pipeline_init_sec": 0.0,
            "pairs": [],
            "sum_image_open_sec": 0.0,
            "sum_perception_source_sec": 0.0,
            "sum_perception_target_sec": 0.0,
            "sum_face_source_sec": 0.0,
            "sum_face_target_sec": 0.0,
            "sum_record_build_sec": 0.0,
        },
    }

    total_start = perf_counter()

    pipeline = None
    pipeline_start = perf_counter()
    try:
        pipeline = PerceptionAdapter()
    except Exception as err:
        # Do not fail immediately: image-level detector may still work.
        diagnostics["perception_adapter_init_error"] = str(err)
        pipeline = None
    diagnostics["timing"]["pipeline_init_sec"] = _elapsed(pipeline_start)

    pairs: list[dict[str, Any]] = []

    def _fail_or_skip(reason: str, message: str) -> None:
        diagnostics["skipped_pairs"] += 1
        diagnostics["skipped_by_reason"][reason] += 1
        if strict:
            raise PairBuildError(message)

    for item in discovered:
        idx = item["index"]
        source_path = item.get("source")
        target_path = item.get("target")

        pair_timing = {
            "pair_id": str(idx),
            "image_open_sec": 0.0,
            "perception_source_sec": 0.0,
            "perception_target_sec": 0.0,
            "face_source_sec": 0.0,
            "face_target_sec": 0.0,
            "record_build_sec": 0.0,
            "total_pair_sec": 0.0,
        }
        pair_start = perf_counter()

        if source_path is None:
            diagnostics["missing_source_count"] += 1
            _fail_or_skip("missing_source", f"Pair {idx}: missing source image")
            continue

        if target_path is None:
            diagnostics["missing_target_count"] += 1
            _fail_or_skip("missing_target", f"Pair {idx}: missing target image")
            continue

        try:
            image_open_start = perf_counter()
            src_img = Image.open(source_path).convert("RGB")
            tgt_img = Image.open(target_path).convert("RGB")
            pair_timing["image_open_sec"] = _elapsed(image_open_start)
            diagnostics["timing"]["sum_image_open_sec"] += pair_timing["image_open_sec"]
        except Exception as err:
            _fail_or_skip("image_open_failed", f"Pair {idx}: failed to open images: {err}")
            continue

        if src_img.size != tgt_img.size:
            diagnostics["size_mismatch_count"] += 1
            _fail_or_skip(
                "size_mismatch",
                f"Pair {idx}: image size mismatch source={src_img.size}, target={tgt_img.size}",
            )
            continue

        src_result: dict[str, Any] | None = None
        tgt_result: dict[str, Any] | None = None

        if pipeline is not None:
            try:
                src_perception_start = perf_counter()
                src_result = pipeline.run(source_path)
                pair_timing["perception_source_sec"] = _elapsed(src_perception_start)
                diagnostics["timing"]["sum_perception_source_sec"] += pair_timing["perception_source_sec"]
            except Exception as err:
                diagnostics["perception_failed_count"] += 1
                src_result = {
                    "scene_graph": None,
                    "raw_result": None,
                    "backend_name": "perception_failed",
                    "error": str(err),
                }

            try:
                tgt_perception_start = perf_counter()
                tgt_result = pipeline.run(target_path)
                pair_timing["perception_target_sec"] = _elapsed(tgt_perception_start)
                diagnostics["timing"]["sum_perception_target_sec"] += pair_timing["perception_target_sec"]
            except Exception as err:
                diagnostics["perception_failed_count"] += 1
                tgt_result = {
                    "scene_graph": None,
                    "raw_result": None,
                    "backend_name": "perception_failed",
                    "error": str(err),
                }
        else:
            src_result = {"scene_graph": None, "raw_result": None, "backend_name": "no_perception_adapter"}
            tgt_result = {"scene_graph": None, "raw_result": None, "backend_name": "no_perception_adapter"}

        face_source_start = perf_counter()
        try:
            src_face_bbox, src_face_meta = extract_face_bbox_from_scene_graph(
                src_result["scene_graph"],
                strict=strict,
                allow_person_fallback_non_strict=allow_person_fallback_non_strict,
            )
        except Exception as graph_err:
            try:
                src_face_bbox, src_face_meta = extract_face_bbox_from_image_detector(source_path)
                src_face_meta["graph_error"] = str(graph_err)
            except Exception as detector_err:
                diagnostics["source_face_missing_count"] += 1
                pair_timing["face_source_sec"] = _elapsed(face_source_start)
                diagnostics["timing"]["sum_face_source_sec"] += pair_timing["face_source_sec"]
                _fail_or_skip(
                    "source_face_missing",
                    f"Pair {idx}: source face missing. graph_error={graph_err}; detector_error={detector_err}",
                )
                continue
        finally:
            if pair_timing["face_source_sec"] == 0.0:
                pair_timing["face_source_sec"] = _elapsed(face_source_start)
                diagnostics["timing"]["sum_face_source_sec"] += pair_timing["face_source_sec"]

        face_target_start = perf_counter()
        try:
            tgt_face_bbox, tgt_face_meta = extract_face_bbox_from_scene_graph(
                tgt_result["scene_graph"],
                strict=strict,
                allow_person_fallback_non_strict=allow_person_fallback_non_strict,
            )
        except Exception as graph_err:
            try:
                tgt_face_bbox, tgt_face_meta = extract_face_bbox_from_image_detector(target_path)
                tgt_face_meta["graph_error"] = str(graph_err)
            except Exception as detector_err:
                diagnostics["target_face_missing_count"] += 1
                pair_timing["face_target_sec"] = _elapsed(face_target_start)
                diagnostics["timing"]["sum_face_target_sec"] += pair_timing["face_target_sec"]
                _fail_or_skip(
                    "target_face_missing",
                    f"Pair {idx}: target face missing. graph_error={graph_err}; detector_error={detector_err}",
                )
                continue
        finally:
            if pair_timing["face_target_sec"] == 0.0:
                pair_timing["face_target_sec"] = _elapsed(face_target_start)
                diagnostics["timing"]["sum_face_target_sec"] += pair_timing["face_target_sec"]

        try:
            record_build_start = perf_counter()
            record = build_pair_record(
                index=idx,
                source_path=source_path,
                target_path=target_path,
                source_img_size=src_img.size,
                target_img_size=tgt_img.size,
                source_face_bbox_xywh=src_face_bbox,
                target_face_bbox_xywh=tgt_face_bbox,
                source_face_meta=src_face_meta,
                target_face_meta=tgt_face_meta,
                prompt=prompt,
                transition_family=transition_family,
                action_family=action_family,
                summary=summary,
            )
            pair_timing["record_build_sec"] = _elapsed(record_build_start)
            diagnostics["timing"]["sum_record_build_sec"] += pair_timing["record_build_sec"]
        except Exception as err:
            _fail_or_skip("record_build_failed", f"Pair {idx}: record build failed: {err}")
            continue

        pair_timing["total_pair_sec"] = _elapsed(pair_start)
        diagnostics["timing"]["pairs"].append(pair_timing)

        pairs.append(record)
        diagnostics["exported_pairs"] += 1

    diagnostics["timing"]["total_sec"] = _elapsed(total_start)

    if not pairs:
        raise PairBuildError("No valid pairs exported")

    payload = {
        "contract_version": "renderer_observed_pair_manifest_input_v1",
        "pairs": pairs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sidecar_path = output_path.with_suffix(output_path.suffix + ".diagnostics.json")
    sidecar_payload = dict(diagnostics)
    sidecar_payload["skipped_by_reason"] = dict(sidecar_payload["skipped_by_reason"])
    sidecar_path.write_text(
        json.dumps(sidecar_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_path": str(output_path),
        "diagnostics_path": str(sidecar_path),
        "diagnostics": sidecar_payload,
    }


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-build renderer observed-pair input manifest from image pairs "
            "using repo perception stack plus image-level face detector fallback."
        )
    )
    parser.add_argument(
        "--pairs-dir",
        required=True,
        help="Directory with source_XXXX / target_XXXX image pairs",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output JSON path for renderer_observed_pair_manifest_input_v1",
    )
    parser.add_argument(
        "--prompt",
        default="person smiles naturally",
        help="Prompt text for all exported pairs",
    )
    parser.add_argument(
        "--transition-family",
        default="expression_transition",
        help="transition_context.family",
    )
    parser.add_argument(
        "--action-family",
        default="face_expression_change",
        help="transition_context.action_family",
    )
    parser.add_argument(
        "--summary",
        default="auto-generated face expression pair",
        help="transition_context.summary",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first invalid pair",
    )
    parser.add_argument(
        "--allow-person-fallback-non-strict",
        action="store_true",
        help=(
            "Only for non-strict mode: if no face body part exists, "
            "allow person bbox fallback with explicit diagnostics"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_renderer_pairs_manifest_auto(
        pairs_dir=Path(args.pairs_dir),
        output_path=Path(args.output_path),
        prompt=args.prompt,
        transition_family=args.transition_family,
        action_family=args.action_family,
        summary=args.summary,
        strict=bool(args.strict),
        allow_person_fallback_non_strict=bool(args.allow_person_fallback_non_strict),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()