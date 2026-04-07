from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from utils_tensor import zeros


@dataclass(slots=True)
class InputOptions:
    target_duration_sec: float = 4.0
    fps: int = 16
    quality_profile: str = "balanced"
    aspect_ratio_policy: str = "pad"
    color_normalization: str = "none"
    duration_cap_sec: float | None = 12.0
    keyframe_stride: int = 8


@dataclass(slots=True)
class AssetFrame:
    frame_id: str
    tensor: object
    width: int
    height: int
    timestamp: float = 0.0
    source: str = "image"


@dataclass(slots=True)
class UnifiedAsset:
    input_type: str
    frames: list[AssetFrame] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class InputRequest:
    input_type: str
    images: list[str] = field(default_factory=list)
    video: str | None = None
    text: str = ""
    options: InputOptions = field(default_factory=InputOptions)
    orig_size: tuple[int, int] | None = None
    normalized_size: tuple[int, int] | None = None
    timestamps: list[float] = field(default_factory=list)
    frame_count: int = 0
    reference_set: list[str] = field(default_factory=list)
    unified_asset: UnifiedAsset | None = None


class InputAssetLayer:
    """Loads real input assets (image/video) and converts them to unified runtime DTO."""

    _PROFILE_TARGET = {
        "debug": 256,
        "mobile": 512,
        "balanced": 768,
        "quality": 1024,
    }

    def build_request(
        self,
        images: list[str] | None,
        text: str,
        video: str | None = None,
        fps: int = 16,
        duration: float = 4.0,
        quality_profile: str = "balanced",
        aspect_ratio_policy: str = "pad",
        color_normalization: str = "none",
        duration_cap_sec: float | None = 12.0,
    ) -> InputRequest:
        image_list = [str(Path(p)) for p in (images or [])]
        input_type = "video" if video else ("single_image" if len(image_list) <= 1 else "multi_image")
        req = InputRequest(
            input_type=input_type,
            images=image_list,
            video=str(Path(video)) if video else None,
            text=text,
            options=InputOptions(
                target_duration_sec=duration,
                fps=fps,
                quality_profile=quality_profile,
                aspect_ratio_policy=aspect_ratio_policy,
                color_normalization=color_normalization,
                duration_cap_sec=duration_cap_sec,
            ),
        )
        if req.video:
            self._enrich_video(req)
        else:
            self._enrich_image(req)
        return req

    def _load_image_tensor(self, image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        width, height = image.size

        arr = np.asarray(image, dtype=np.uint8)
        arr = arr.astype(np.float32) / 255.0
        return arr, (width, height)

    def _normalize_tensor(self, tensor: np.ndarray, profile: str) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = tensor.shape[:2]
        target = self._PROFILE_TARGET.get(profile, self._PROFILE_TARGET["balanced"])
        nw, nh = self._normalize_size(w, h, target=target)

        if (nw, nh) == (w, h):
            return self._color_normalize(tensor), (nw, nh)

        # Через PIL это будет на порядки быстрее, чем твои Python-циклы
        img = Image.fromarray((np.clip(tensor, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
        img = img.resize((nw, nh), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
        return self._color_normalize(arr), (nw, nh)

    def _color_normalize(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def _enrich_image(self, request: InputRequest) -> None:
        request.frame_count = len(request.images)
        request.timestamps = [0.0 for _ in request.images]
        request.reference_set = request.images[:]

        frames: list[AssetFrame] = []
        first_size: tuple[int, int] | None = None
        normalized_size: tuple[int, int] | None = None

        for idx, image_path in enumerate(request.images):
            tensor, size = self._load_image_tensor(image_path)
            normalized_tensor, ns = self._normalize_tensor(tensor, request.options.quality_profile)

            if first_size is None:
                first_size = size
            normalized_size = ns

            frames.append(
                AssetFrame(
                    frame_id=f"img_{idx}",
                    tensor=normalized_tensor,
                    width=ns[0],
                    height=ns[1],
                    timestamp=0.0,
                    source=image_path,
                )
            )

        if not frames:
            debug_tensor = np.full((256, 256, 3), 0.5, dtype=np.float32)
            frames = [AssetFrame(frame_id="debug_0", tensor=debug_tensor, width=256, height=256, source="debug://blank")]
            first_size = (256, 256)
            normalized_size = (256, 256)

        request.orig_size = first_size
        request.normalized_size = normalized_size
        request.unified_asset = UnifiedAsset(
            input_type=request.input_type,
            frames=frames,
            references=request.reference_set,
            metadata={"profile": request.options.quality_profile, "count": len(frames)},
        )

    def _decode_video_frames(self, video_path: str, fps: int, duration: float, stride: int) -> tuple[list[AssetFrame], dict[str, object]]:
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("unable to open video")

            native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            max_seconds = max(0.1, duration)
            max_frames = int(max_seconds * fps)

            frames: list[AssetFrame] = []
            frame_idx = 0

            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % max(1, stride) == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    nt, ns = self._normalize_tensor(frame, "balanced")

                    frames.append(
                        AssetFrame(
                            frame_id=f"video_{len(frames)}",
                            tensor=nt,
                            width=ns[0],
                            height=ns[1],
                            timestamp=frame_idx / max(1.0, native_fps),
                            source=video_path,
                        )
                    )

                frame_idx += 1

            cap.release()
            return frames, {
                "native_fps": native_fps,
                "frame_total": frame_total,
                "orig_size": (width, height),
            }
        except Exception:
            frame_count = max(1, int(round(duration * fps)))
            frames = [
                AssetFrame(
                    frame_id=f"video_{i}",
                    tensor=np.zeros((256, 256, 3), dtype=np.float32),
                    width=256,
                    height=256,
                    timestamp=i / fps,
                    source=video_path,
                )
                for i in range(0, frame_count, max(1, stride))
            ]
            return frames, {"native_fps": fps, "frame_total": frame_count, "orig_size": (1280, 720), "metadata_only": True}

    def _enrich_video(self, request: InputRequest) -> None:
        fps = max(1, request.options.fps)
        duration = request.options.target_duration_sec
        if request.options.duration_cap_sec is not None:
            duration = min(duration, request.options.duration_cap_sec)
        stride = max(1, request.options.keyframe_stride)

        frames, metadata = self._decode_video_frames(str(request.video), fps=fps, duration=duration, stride=stride)
        request.frame_count = len(frames)
        request.timestamps = [f.timestamp for f in frames]
        request.reference_set = [f"{request.video}#t={ts:.3f}" for ts in request.timestamps]

        orig_size = metadata.get("orig_size", (1280, 720))
        request.orig_size = (int(orig_size[0]), int(orig_size[1]))
        request.normalized_size = (frames[0].width, frames[0].height) if frames else self._normalize_size(*request.orig_size)
        request.unified_asset = UnifiedAsset(
            input_type="video",
            frames=frames,
            references=request.reference_set,
            metadata=metadata,
        )

    def _normalize_size(self, w: int, h: int, target: int = 1024) -> tuple[int, int]:
        if w <= 0 or h <= 0:
            return target, target
        scale = target / max(w, h)
        nw = max(64, int(round(w * scale / 32) * 32))
        nh = max(64, int(round(h * scale / 32) * 32))
        return nw, nh