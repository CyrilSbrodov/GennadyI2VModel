from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


class InputAssetLayer:
    """Normalizes request assets and constructs canonical request DTO."""

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

    def _enrich_image(self, request: InputRequest) -> None:
        request.frame_count = len(request.images)
        request.timestamps = [0.0 for _ in request.images]
        request.reference_set = request.images[:]
        request.orig_size = (1024, 1024)
        request.normalized_size = self._normalize_size(*request.orig_size)

    def _enrich_video(self, request: InputRequest) -> None:
        # cv2-free fallback metadata based on requested runtime settings.
        fps = max(1, request.options.fps)
        duration = request.options.target_duration_sec
        if request.options.duration_cap_sec is not None:
            duration = min(duration, request.options.duration_cap_sec)
        frame_count = max(1, int(round(duration * fps)))
        request.frame_count = frame_count
        request.timestamps = [i / fps for i in range(frame_count)]
        request.orig_size = (1280, 720)
        request.normalized_size = self._normalize_size(*request.orig_size)

        stride = max(1, request.options.keyframe_stride)
        request.reference_set = [
            f"{request.video}#t={request.timestamps[i]:.3f}" for i in range(0, frame_count, stride)
        ]

    def _normalize_size(self, w: int, h: int, target: int = 1024) -> tuple[int, int]:
        if w <= 0 or h <= 0:
            return target, target
        scale = target / max(w, h)
        nw = max(64, int(round(w * scale / 32) * 32))
        nh = max(64, int(round(h * scale / 32) * 32))
        return nw, nh
