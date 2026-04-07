from __future__ import annotations

from dataclasses import dataclass

from core.input_layer import AssetFrame
from core.schema import BBox
from perception.frame_context import FrameLike, PerceptionFrameContext, unwrap_frame
from utils_tensor import shape


@dataclass(slots=True)
class FrameImage:
    rgb: "object"
    width: int
    height: int


def _empty_image() -> FrameImage:
    try:
        import numpy as np  # type: ignore

        return FrameImage(rgb=np.zeros((1, 1, 3), dtype=np.uint8), width=1, height=1)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for real perception backends") from exc


def frame_to_numpy_rgb(frame: FrameLike) -> FrameImage:
    if isinstance(frame, PerceptionFrameContext):
        cached = frame.get("frame_rgb")
        if cached is not None:
            return cached
        image = frame_to_numpy_rgb(frame.frame)
        frame.put("frame_rgb", image)
        return image

    frame = unwrap_frame(frame)
    if isinstance(frame, str):
        return _empty_image()

    tensor = frame.tensor if isinstance(frame, AssetFrame) else frame
    h, w, c = shape(tensor)
    if h <= 0 or w <= 0 or c < 3:
        return _empty_image()

    import numpy as np  # type: ignore

    arr = np.asarray(tensor)
    if arr.dtype == np.uint8:
        # For uint8 runtime tensors we keep a zero-copy view when possible.
        # Create a compact copy only when the slice is not contiguous.
        rgb = arr[..., :3]
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
    else:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        rgb = (arr[..., :3] * 255.0).astype(np.uint8)
    return FrameImage(rgb=rgb, width=w, height=h)


def rgb_to_bgr(image: "object") -> "object":
    import numpy as np  # type: ignore

    rgb = np.asarray(image)
    return rgb[:, :, ::-1].copy()


def clamp_bbox(bbox: BBox) -> BBox:
    x = min(max(0.0, float(bbox.x)), 1.0)
    y = min(max(0.0, float(bbox.y)), 1.0)
    w = min(max(0.0, float(bbox.w)), 1.0 - x)
    h = min(max(0.0, float(bbox.h)), 1.0 - y)
    return BBox(x=x, y=y, w=w, h=h)


def xyxy_to_norm_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> BBox:
    if width <= 0 or height <= 0:
        return BBox(0.0, 0.0, 0.0, 0.0)
    nx1 = min(max(0.0, x1 / width), 1.0)
    ny1 = min(max(0.0, y1 / height), 1.0)
    nx2 = min(max(0.0, x2 / width), 1.0)
    ny2 = min(max(0.0, y2 / height), 1.0)
    return clamp_bbox(BBox(nx1, ny1, max(0.0, nx2 - nx1), max(0.0, ny2 - ny1)))


def crop_rgb(image: "object", bbox: BBox) -> "object":
    import numpy as np  # type: ignore

    arr = np.asarray(image)
    if arr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = arr.shape[:2]
    b = clamp_bbox(bbox)
    x1 = int(round(b.x * w))
    y1 = int(round(b.y * h))
    x2 = max(x1 + 1, int(round((b.x + b.w) * w)))
    y2 = max(y1 + 1, int(round((b.y + b.h) * h)))
    return arr[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)].copy()
