from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise ValueError(f"Не удалось сохранить изображение: {path}")


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb.astype(np.uint8))


def stable_color(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (30, 144, 255),
        (238, 130, 238),
        (255, 215, 0),
        (72, 209, 204),
        (255, 140, 0),
        (154, 205, 50),
        (0, 191, 255),
        (199, 21, 133),
        (70, 130, 180),
        (205, 92, 92),
        (0, 206, 209),
        (244, 164, 96),
        (186, 85, 211),
        (46, 139, 87),
    ]
    return palette[index % len(palette)]


def resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def alpha_blend_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    out = image_bgr.copy()
    mask_bool = mask.astype(bool)
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]

    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[:, :] = color
    out[mask_bool] = cv2.addWeighted(out[mask_bool], 1.0 - alpha, overlay[mask_bool], alpha, 0)
    return out


def contour_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_mask_contours(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    contours = contour_from_mask(mask)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color, thickness)


def draw_text_box(
    image_bgr: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = max(0, min(image_bgr.shape[1] - 1, x))
    y = max(h + baseline + 6, min(image_bgr.shape[0] - 1, y))

    cv2.rectangle(image_bgr, (x, y - h - baseline - 4), (x + w + 6, y + 2), color, -1)
    cv2.putText(image_bgr, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str | None = None,
    thickness: int = 2,
) -> None:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))

    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
    if label:
        draw_text_box(image_bgr, label, (x1, max(18, y1)), color)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_center(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(np.median(xs)), int(np.median(ys))


# ============================================================
# Results dataclasses
# ============================================================

@dataclass
class DetectionInstance:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray | None = None


@dataclass
class PoseResult:
    source: str
    keypoints_xy: list[tuple[int, int]] = field(default_factory=list)
    keypoints_conf: list[float] = field(default_factory=list)
    person_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class ParserClassMask:
    class_id: int
    class_name: str
    mask: np.ndarray
    pixel_count: int
    bbox_xyxy: tuple[int, int, int, int] | None = None


@dataclass
class ParserResult:
    model_name: str
    masks: list[ParserClassMask] = field(default_factory=list)
    raw_label_map_shape: tuple[int, int] | None = None


@dataclass
class ImageProcessSummary:
    image_name: str
    width: int
    height: int
    detector_ok: bool = False
    pose_yolo_ok: bool = False
    pose_mediapipe_ok: bool = False
    parser_ok: bool = False
    errors: list[str] = field(default_factory=list)
    detector_instances: list[dict[str, Any]] = field(default_factory=list)
    pose_yolo: dict[str, Any] = field(default_factory=dict)
    pose_mediapipe: dict[str, Any] = field(default_factory=dict)
    parser: dict[str, Any] = field(default_factory=dict)


# ============================================================
# YOLO segmentation + YOLO pose
# ============================================================

class YoloHumanStack:
    def __init__(
        self,
        seg_model_name: str = "yolo11n-seg.pt",
        pose_model_name: str = "yolo11n-pose.pt",
        device: str = "cuda",
        conf: float = 0.25,
    ) -> None:
        from ultralytics import YOLO

        self.seg_model_name = seg_model_name
        self.pose_model_name = pose_model_name
        self.device = device
        self.conf = conf

        self.seg_model = YOLO(seg_model_name)
        self.pose_model = YOLO(pose_model_name)

    def detect_and_segment_persons(self, image_bgr: np.ndarray) -> list[DetectionInstance]:
        results = self.seg_model.predict(source=image_bgr, conf=self.conf, device=self.device, verbose=False)
        if not results:
            return []

        res = results[0]
        if res.boxes is None:
            return []

        boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(int)
        confs = res.boxes.conf.detach().cpu().numpy().tolist()
        classes = res.boxes.cls.detach().cpu().numpy().astype(int).tolist()
        names = res.names if hasattr(res, "names") else {}

        masks = None
        if res.masks is not None and res.masks.data is not None:
            masks = res.masks.data.detach().cpu().numpy()

        out: list[DetectionInstance] = []

        for idx, (bbox, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, classes)):
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            if class_name != "person":
                continue

            mask = None
            if masks is not None and idx < len(masks):
                mask = resize_mask_nearest(masks[idx] > 0.5, image_bgr.shape[1], image_bgr.shape[0])

            out.append(
                DetectionInstance(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox_xyxy=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    mask=mask,
                )
            )

        return out

    def detect_pose(self, image_bgr: np.ndarray) -> PoseResult:
        results = self.pose_model.predict(source=image_bgr, conf=self.conf, device=self.device, verbose=False)
        if not results:
            return PoseResult(source="yolo")

        res = results[0]
        out = PoseResult(source="yolo")

        if res.boxes is not None:
            boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(int)
            classes = res.boxes.cls.detach().cpu().numpy().astype(int).tolist()
            names = res.names if hasattr(res, "names") else {}

            for bbox, cls_id in zip(boxes_xyxy, classes):
                class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                if class_name == "person":
                    out.person_boxes.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

        if res.keypoints is not None and res.keypoints.xy is not None:
            xy = res.keypoints.xy.detach().cpu().numpy()
            conf = None
            if res.keypoints.conf is not None:
                conf = res.keypoints.conf.detach().cpu().numpy()

            if len(xy) > 0:
                kp_xy = xy[0]
                kp_conf = conf[0] if conf is not None and len(conf) > 0 else np.ones((kp_xy.shape[0],), dtype=np.float32)
                out.keypoints_xy = [(int(x), int(y)) for x, y in kp_xy]
                out.keypoints_conf = [float(v) for v in kp_conf.tolist()]

        return out


# ============================================================
# MediaPipe Pose Landmarker
# ============================================================

class MediaPipePoseModule:
    def __init__(self, model_path: Path = Path("models/pose_landmarker_heavy.task")) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл {model_path}\n"
                "Скачай MediaPipe Pose Landmarker model и положи его туда, "
                "или запускай с --disable-mediapipe"
            )

        self.mp = mp
        self.model_path = model_path

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_segmentation_masks=True,
            num_poses=4,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> PoseResult:
        rgb = bgr_to_rgb(image_bgr)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        out = PoseResult(source="mediapipe")

        if not result.pose_landmarks:
            return out

        height, width = image_bgr.shape[:2]
        first = result.pose_landmarks[0]

        for lm in first:
            x = int(lm.x * width)
            y = int(lm.y * height)
            out.keypoints_xy.append((x, y))
            out.keypoints_conf.append(float(getattr(lm, "visibility", 1.0)))

        return out


# ============================================================
# HF SegFormer Human Parser / Clothes Parser
# ============================================================

class HFSegFormerParser:
    def __init__(
        self,
        model_name: str = "fashn-ai/fashn-human-parser",
        device: str = "cuda",
        min_pixels: int = 100,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.model_name = model_name
        self.min_pixels = min_pixels
        self.torch = torch
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.id2label = {int(k): str(v) for k, v in cfg.id2label.items()} if hasattr(cfg, "id2label") else {}

    def parse(self, image_bgr: np.ndarray) -> ParserResult:
        rgb = bgr_to_rgb(image_bgr)
        pil = to_pil(rgb)
        height, width = image_bgr.shape[:2]

        with self.torch.no_grad():
            inputs = self.processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            label_maps = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])

        label_map = label_maps[0].detach().cpu().numpy().astype(np.int32)

        masks: list[ParserClassMask] = []
        for class_id in np.unique(label_map).tolist():
            mask = label_map == int(class_id)
            pixel_count = int(mask.sum())
            if pixel_count < self.min_pixels:
                continue

            class_name = self.id2label.get(int(class_id), f"class_{class_id}")
            masks.append(
                ParserClassMask(
                    class_id=int(class_id),
                    class_name=str(class_name),
                    mask=mask,
                    pixel_count=pixel_count,
                    bbox_xyxy=mask_bbox(mask),
                )
            )

        masks.sort(key=lambda item: item.pixel_count, reverse=True)
        return ParserResult(model_name=self.model_name, masks=masks, raw_label_map_shape=label_map.shape)


# ============================================================
# Overlay builders
# ============================================================

def build_detector_overlay(image_bgr: np.ndarray, detections: list[DetectionInstance]) -> np.ndarray:
    out = image_bgr.copy()

    for idx, det in enumerate(detections):
        color = stable_color(idx)
        if det.mask is not None:
            out = alpha_blend_mask(out, det.mask, color=color, alpha=0.35)
            draw_mask_contours(out, det.mask, color=color, thickness=2)
        draw_bbox(out, det.bbox_xyxy, color=color, label=f"{det.class_name} {det.confidence:.2f}")

    return out


def build_person_mask_only_overlay(image_bgr: np.ndarray, detections: list[DetectionInstance]) -> np.ndarray:
    out = image_bgr.copy()

    for idx, det in enumerate(detections):
        if det.mask is None:
            continue
        color = stable_color(idx)
        out = alpha_blend_mask(out, det.mask, color=color, alpha=0.55)
        draw_mask_contours(out, det.mask, color=color, thickness=3)
        center = mask_center(det.mask)
        if center:
            draw_text_box(out, f"person_mask {det.confidence:.2f}", center, color)

    return out


def build_pose_overlay(image_bgr: np.ndarray, pose: PoseResult) -> np.ndarray:
    out = image_bgr.copy()

    for bbox in pose.person_boxes:
        draw_bbox(out, bbox, (0, 255, 255), label=f"{pose.source}_person")

    for idx, (xy, conf) in enumerate(zip(pose.keypoints_xy, pose.keypoints_conf)):
        x, y = xy
        color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
        cv2.circle(out, (x, y), 4, color, -1)
        cv2.putText(out, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return out


def build_parser_overlay(
    image_bgr: np.ndarray,
    parser_result: ParserResult,
    title_prefix: str = "",
    only_names: set[str] | None = None,
) -> np.ndarray:
    out = image_bgr.copy()

    items = parser_result.masks
    if only_names is not None:
        lowered = {x.lower() for x in only_names}
        items = [m for m in items if m.class_name.lower() in lowered]

    for idx, item in enumerate(items):
        color = stable_color(idx)
        out = alpha_blend_mask(out, item.mask, color=color, alpha=0.35)
        draw_mask_contours(out, item.mask, color=color, thickness=2)

        center = mask_center(item.mask)
        if center:
            label = f"{title_prefix}{item.class_name} ({item.pixel_count})"
            draw_text_box(out, label, center, color)

    return out


def build_parser_binary_masks_grid(parser_result: ParserResult, image_shape: tuple[int, int, int], max_items: int = 18) -> np.ndarray:
    h, w, _ = image_shape
    items = parser_result.masks[:max_items]

    if not items:
        return np.full((h, w, 3), 20, dtype=np.uint8)

    cols = 3
    rows = int(math.ceil(len(items) / cols))
    tile_h = max(220, h // max(1, rows))
    tile_w = max(220, w // cols)

    canvas = np.full((rows * tile_h, cols * tile_w, 3), 18, dtype=np.uint8)

    for idx, item in enumerate(items):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_h
        x0 = col * tile_w

        tile = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
        mask_u8 = (item.mask.astype(np.uint8) * 255)
        mask_resized = cv2.resize(mask_u8, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

        color = np.array(stable_color(idx), dtype=np.uint8)
        tile[mask_resized > 0] = color

        draw_text_box(tile, f"{item.class_name} ({item.pixel_count})", (8, 24), tuple(int(x) for x in color))
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    return canvas


def build_fusion_overlay(
    image_bgr: np.ndarray,
    detections: list[DetectionInstance],
    parser_result: ParserResult | None,
    pose_yolo: PoseResult | None,
    pose_mediapipe: PoseResult | None,
) -> np.ndarray:
    out = image_bgr.copy()

    if detections:
        out = build_detector_overlay(out, detections)

    if parser_result is not None:
        out = build_parser_overlay(out, parser_result)

    if pose_yolo is not None:
        out = build_pose_overlay(out, pose_yolo)

    if pose_mediapipe is not None:
        out = build_pose_overlay(out, pose_mediapipe)

    return out


# ============================================================
# Heuristic group names for separate overlays
# ============================================================

GARMENT_HINTS = {
    "top", "upper-clothes", "upper clothes", "shirt", "t-shirt", "coat", "jacket",
    "dress", "skirt", "pants", "trousers", "shorts", "belt", "scarf", "sleeve",
    "left-shoe", "right-shoe", "shoe", "bag", "hat", "glove", "socks",
}

BODY_PART_HINTS = {
    "left-arm", "right-arm", "arm", "arms", "left-leg", "right-leg", "leg", "legs",
    "left-hand", "right-hand", "hand", "hands", "skin", "torso", "neck", "foot", "feet",
}

FACE_HAIR_HINTS = {
    "face", "hair", "hat", "sunglasses", "glasses", "ear", "neck",
}


def names_matching(masks: list[ParserClassMask], hints: set[str]) -> set[str]:
    out: set[str] = set()
    for item in masks:
        name = item.class_name.lower()
        normalized = name.replace("_", "-")
        if name in hints or normalized in hints:
            out.add(item.class_name)
            continue
        for hint in hints:
            if hint in name or hint in normalized:
                out.add(item.class_name)
                break
    return out


# ============================================================
# Service
# ============================================================

class RealHumanParsingLayersService:
    def __init__(
        self,
        *,
        input_dir: Path | None,
        image: Path | None,
        output_dir: Path,
        device: str,
        yolo_seg_model: str,
        yolo_pose_model: str,
        parser_model: str,
        conf: float,
        min_parser_pixels: int,
        disable_yolo: bool,
        disable_yolo_pose: bool,
        disable_mediapipe: bool,
        disable_parser: bool,
        mediapipe_model_path: Path,
    ) -> None:
        self.input_dir = input_dir
        self.image = image
        self.output_dir = output_dir
        self.device = device
        self.yolo_seg_model = yolo_seg_model
        self.yolo_pose_model = yolo_pose_model
        self.parser_model = parser_model
        self.conf = conf
        self.min_parser_pixels = min_parser_pixels
        self.disable_yolo = disable_yolo
        self.disable_yolo_pose = disable_yolo_pose
        self.disable_mediapipe = disable_mediapipe
        self.disable_parser = disable_parser
        self.mediapipe_model_path = mediapipe_model_path

        self.yolo: YoloHumanStack | None = None
        self.parser: HFSegFormerParser | None = None
        self.mediapipe_pose: MediaPipePoseModule | None = None

        self.init_errors: list[str] = []
        self._init_modules()

    def _init_modules(self) -> None:
        if not self.disable_yolo:
            try:
                self.yolo = YoloHumanStack(
                    seg_model_name=self.yolo_seg_model,
                    pose_model_name=self.yolo_pose_model,
                    device=self.device,
                    conf=self.conf,
                )
            except Exception as exc:
                self.init_errors.append(f"yolo_init_error: {type(exc).__name__}: {exc}")
                self.yolo = None

        if not self.disable_parser:
            try:
                self.parser = HFSegFormerParser(
                    model_name=self.parser_model,
                    device=self.device,
                    min_pixels=self.min_parser_pixels,
                )
            except Exception as exc:
                self.init_errors.append(f"parser_init_error: {type(exc).__name__}: {exc}")
                self.parser = None

        if not self.disable_mediapipe:
            try:
                self.mediapipe_pose = MediaPipePoseModule(model_path=self.mediapipe_model_path)
            except Exception as exc:
                self.init_errors.append(f"mediapipe_init_error: {type(exc).__name__}: {exc}")
                self.mediapipe_pose = None

    def resolve_images(self) -> list[Path]:
        if self.image is not None:
            if not self.image.exists():
                raise FileNotFoundError(f"Image does not exist: {self.image}")
            return [self.image]

        if self.input_dir is None:
            raise ValueError("Either --image or --input-dir is required")

        images = list_images(self.input_dir)
        if not images:
            raise ValueError(f"В папке нет изображений: {self.input_dir}")
        return images

    def process_all(self) -> dict[str, Any]:
        ensure_dir(self.output_dir)

        images = self.resolve_images()
        global_summary: dict[str, Any] = {
            "input_dir": str(self.input_dir) if self.input_dir else "",
            "image": str(self.image) if self.image else "",
            "output_dir": str(self.output_dir),
            "device": self.device,
            "yolo_seg_model": self.yolo_seg_model,
            "yolo_pose_model": self.yolo_pose_model,
            "parser_model": self.parser_model,
            "disable_yolo": self.disable_yolo,
            "disable_yolo_pose": self.disable_yolo_pose,
            "disable_mediapipe": self.disable_mediapipe,
            "disable_parser": self.disable_parser,
            "init_errors": self.init_errors,
            "images_total": len(images),
            "images": [],
        }

        for image_path in images:
            summary = self.process_one(image_path)
            global_summary["images"].append(asdict(summary))

        (self.output_dir / "summary.json").write_text(
            json.dumps(global_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return global_summary

    def process_one(self, image_path: Path) -> ImageProcessSummary:
        image_bgr = load_bgr_image(image_path)
        height, width = image_bgr.shape[:2]

        image_out_dir = self.output_dir / image_path.stem
        ensure_dir(image_out_dir)

        summary = ImageProcessSummary(image_name=image_path.name, width=width, height=height)
        summary.errors.extend(self.init_errors)

        save_image(image_out_dir / "00_original.jpg", image_bgr)

        detections: list[DetectionInstance] = []
        pose_yolo: PoseResult | None = None
        pose_mp: PoseResult | None = None
        parser_result: ParserResult | None = None

        # ---------- YOLO detector / segmentation ----------
        if self.yolo is not None:
            try:
                detections = self.yolo.detect_and_segment_persons(image_bgr)
                save_image(image_out_dir / "10_yolo_person_seg_overlay.jpg", build_detector_overlay(image_bgr, detections))
                save_image(image_out_dir / "11_yolo_person_mask_only.jpg", build_person_mask_only_overlay(image_bgr, detections))

                summary.detector_ok = True
                summary.detector_instances = [
                    {
                        "class_name": item.class_name,
                        "confidence": item.confidence,
                        "bbox_xyxy": list(item.bbox_xyxy),
                        "has_mask": item.mask is not None,
                        "mask_pixels": int(item.mask.sum()) if item.mask is not None else 0,
                    }
                    for item in detections
                ]
            except Exception as exc:
                summary.errors.append(f"detector_error: {type(exc).__name__}: {exc}")

        # ---------- YOLO pose ----------
        if self.yolo is not None and not self.disable_yolo_pose:
            try:
                pose_yolo = self.yolo.detect_pose(image_bgr)
                save_image(image_out_dir / "20_yolo_pose_overlay.jpg", build_pose_overlay(image_bgr, pose_yolo))
                summary.pose_yolo_ok = True
                summary.pose_yolo = {
                    "source": pose_yolo.source,
                    "keypoints_count": len(pose_yolo.keypoints_xy),
                    "person_boxes_count": len(pose_yolo.person_boxes),
                    "keypoints": [
                        {"x": x, "y": y, "confidence": conf}
                        for (x, y), conf in zip(pose_yolo.keypoints_xy, pose_yolo.keypoints_conf)
                    ],
                }
            except Exception as exc:
                summary.errors.append(f"pose_yolo_error: {type(exc).__name__}: {exc}")

        # ---------- MediaPipe pose ----------
        if self.mediapipe_pose is not None:
            try:
                pose_mp = self.mediapipe_pose.detect(image_bgr)
                save_image(image_out_dir / "21_mediapipe_pose_overlay.jpg", build_pose_overlay(image_bgr, pose_mp))
                summary.pose_mediapipe_ok = True
                summary.pose_mediapipe = {
                    "source": pose_mp.source,
                    "keypoints_count": len(pose_mp.keypoints_xy),
                    "person_boxes_count": len(pose_mp.person_boxes),
                    "keypoints": [
                        {"x": x, "y": y, "confidence": conf}
                        for (x, y), conf in zip(pose_mp.keypoints_xy, pose_mp.keypoints_conf)
                    ],
                }
            except Exception as exc:
                summary.errors.append(f"pose_mediapipe_error: {type(exc).__name__}: {exc}")

        # ---------- HF parser ----------
        if self.parser is not None:
            try:
                parser_result = self.parser.parse(image_bgr)

                save_image(image_out_dir / "30_parser_all_masks_overlay.jpg", build_parser_overlay(image_bgr, parser_result))
                save_image(image_out_dir / "31_parser_masks_grid.jpg", build_parser_binary_masks_grid(parser_result, image_bgr.shape))

                garment_names = names_matching(parser_result.masks, GARMENT_HINTS)
                body_names = names_matching(parser_result.masks, BODY_PART_HINTS)
                face_hair_names = names_matching(parser_result.masks, FACE_HAIR_HINTS)

                save_image(
                    image_out_dir / "32_parser_garments_overlay.jpg",
                    build_parser_overlay(image_bgr, parser_result, title_prefix="garment:", only_names=garment_names),
                )
                save_image(
                    image_out_dir / "33_parser_body_parts_overlay.jpg",
                    build_parser_overlay(image_bgr, parser_result, title_prefix="body:", only_names=body_names),
                )
                save_image(
                    image_out_dir / "34_parser_face_hair_overlay.jpg",
                    build_parser_overlay(image_bgr, parser_result, title_prefix="face:", only_names=face_hair_names),
                )

                summary.parser_ok = True
                summary.parser = {
                    "model_name": parser_result.model_name,
                    "raw_label_map_shape": list(parser_result.raw_label_map_shape) if parser_result.raw_label_map_shape else None,
                    "mask_count": len(parser_result.masks),
                    "garment_class_names": sorted(garment_names),
                    "body_part_class_names": sorted(body_names),
                    "face_hair_class_names": sorted(face_hair_names),
                    "classes": [
                        {
                            "class_id": item.class_id,
                            "class_name": item.class_name,
                            "pixel_count": item.pixel_count,
                            "bbox_xyxy": list(item.bbox_xyxy) if item.bbox_xyxy else None,
                        }
                        for item in parser_result.masks
                    ],
                }
            except Exception as exc:
                summary.errors.append(f"parser_error: {type(exc).__name__}: {exc}")

        # ---------- fusion ----------
        try:
            fusion = build_fusion_overlay(image_bgr, detections, parser_result, pose_yolo, pose_mp)
            save_image(image_out_dir / "40_fusion_preview.jpg", fusion)
        except Exception as exc:
            summary.errors.append(f"fusion_error: {type(exc).__name__}: {exc}")

        # ---------- one-image summary ----------
        (image_out_dir / "summary.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._write_summary_txt(image_out_dir, summary)

        return summary

    @staticmethod
    def _write_summary_txt(image_out_dir: Path, summary: ImageProcessSummary) -> None:
        lines: list[str] = []
        lines.append(f"image: {summary.image_name}")
        lines.append(f"size: {summary.width}x{summary.height}")
        lines.append("")
        lines.append(f"detector_ok: {summary.detector_ok}")
        lines.append(f"pose_yolo_ok: {summary.pose_yolo_ok}")
        lines.append(f"pose_mediapipe_ok: {summary.pose_mediapipe_ok}")
        lines.append(f"parser_ok: {summary.parser_ok}")
        lines.append("")

        if summary.detector_instances:
            lines.append("detector_instances:")
            for item in summary.detector_instances:
                lines.append(
                    f"  - {item['class_name']} conf={item['confidence']:.3f} "
                    f"bbox={item['bbox_xyxy']} has_mask={item['has_mask']} mask_pixels={item.get('mask_pixels', 0)}"
                )
            lines.append("")

        if summary.pose_yolo:
            lines.append(f"yolo_pose_keypoints: {summary.pose_yolo.get('keypoints_count', 0)}")
        if summary.pose_mediapipe:
            lines.append(f"mediapipe_pose_keypoints: {summary.pose_mediapipe.get('keypoints_count', 0)}")
        lines.append("")

        if summary.parser:
            lines.append(f"parser_model: {summary.parser.get('model_name')}")
            lines.append(f"parser_mask_count: {summary.parser.get('mask_count')}")
            lines.append(f"garment_class_names: {summary.parser.get('garment_class_names')}")
            lines.append(f"body_part_class_names: {summary.parser.get('body_part_class_names')}")
            lines.append(f"face_hair_class_names: {summary.parser.get('face_hair_class_names')}")
            lines.append("")
            lines.append("parser_classes:")
            for item in summary.parser.get("classes", []):
                lines.append(
                    f"  - id={item['class_id']} name={item['class_name']} "
                    f"pixels={item['pixel_count']} bbox={item['bbox_xyxy']}"
                )
            lines.append("")

        if summary.errors:
            lines.append("errors:")
            for error in summary.errors:
                lines.append(f"  - {error}")

        (image_out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real human parsing/layers debug service")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Путь к одному изображению")
    source.add_argument("--input-dir", type=Path, help="Папка с изображениями")

    parser.add_argument("--output-dir", type=Path, default=Path("output/real_human_parsing_layers"))

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf", type=float, default=0.25)

    parser.add_argument("--yolo-seg-model", type=str, default="yolo11n-seg.pt")
    parser.add_argument("--yolo-pose-model", type=str, default="yolo11n-pose.pt")

    parser.add_argument("--parser-model", type=str, default="fashn-ai/fashn-human-parser")
    parser.add_argument("--min-parser-pixels", type=int, default=100)

    parser.add_argument("--mediapipe-model-path", type=Path, default=Path("models/pose_landmarker_heavy.task"))

    parser.add_argument("--disable-yolo", action="store_true")
    parser.add_argument("--disable-yolo-pose", action="store_true")
    parser.add_argument("--disable-mediapipe", action="store_true")
    parser.add_argument("--disable-parser", action="store_true")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dir(args.output_dir)

    service = RealHumanParsingLayersService(
        input_dir=args.input_dir,
        image=args.image,
        output_dir=args.output_dir,
        device=args.device,
        yolo_seg_model=args.yolo_seg_model,
        yolo_pose_model=args.yolo_pose_model,
        parser_model=args.parser_model,
        conf=args.conf,
        min_parser_pixels=args.min_parser_pixels,
        disable_yolo=args.disable_yolo,
        disable_yolo_pose=args.disable_yolo_pose,
        disable_mediapipe=args.disable_mediapipe,
        disable_parser=args.disable_parser,
        mediapipe_model_path=args.mediapipe_model_path,
    )

    summary = service.process_all()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()