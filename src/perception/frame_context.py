from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.input_layer import AssetFrame


@dataclass(slots=True)
class PerceptionFrameContext:
    """Общий контекст кадра: кеширует тяжелые преобразования внутри одного прохода."""

    frame: AssetFrame | list[list[list[float]]] | str
    cache: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def put(self, key: str, value: Any) -> Any:
        self.cache[key] = value
        return value


FrameLike = AssetFrame | list[list[list[float]]] | str | PerceptionFrameContext


def ensure_frame_context(frame: FrameLike) -> PerceptionFrameContext:
    if isinstance(frame, PerceptionFrameContext):
        return frame
    return PerceptionFrameContext(frame=frame)


def unwrap_frame(frame: FrameLike) -> AssetFrame | list[list[list[float]]] | str:
    if isinstance(frame, PerceptionFrameContext):
        return frame.frame
    return frame
