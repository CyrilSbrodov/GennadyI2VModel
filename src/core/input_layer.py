from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class InputOptions:
    target_duration_sec: float = 4.0
    fps: int = 16
    quality_profile: str = "balanced"


@dataclass(slots=True)
class InputRequest:
    input_type: str
    images: list[str] = field(default_factory=list)
    video: str | None = None
    text: str = ""
    options: InputOptions = field(default_factory=InputOptions)


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
    ) -> InputRequest:
        image_list = images or []
        input_type = "video" if video else ("single_image" if len(image_list) <= 1 else "multi_image")
        existing_images = [str(Path(p)) for p in image_list]
        return InputRequest(
            input_type=input_type,
            images=existing_images,
            video=str(Path(video)) if video else None,
            text=text,
            options=InputOptions(target_duration_sec=duration, fps=fps, quality_profile=quality_profile),
        )
