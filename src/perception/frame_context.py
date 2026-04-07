from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from core.input_layer import AssetFrame

if TYPE_CHECKING:
    import numpy as np

    NumpyFrameArray = np.ndarray[Any, Any]
else:
    NumpyFrameArray = Any

LegacyFrameTensor = list[list[list[float]]]


@dataclass(slots=True)
class PerceptionFrameContext:
    """Общий контекст кадра: кеширует тяжелые преобразования внутри одного прохода."""

    frame: AssetFrame | NumpyFrameArray | LegacyFrameTensor | str
    cache: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def put(self, key: str, value: Any) -> Any:
        self.cache[key] = value
        return value


FrameLike = AssetFrame | NumpyFrameArray | LegacyFrameTensor | str | PerceptionFrameContext


def ensure_frame_context(frame: FrameLike) -> PerceptionFrameContext:
    if isinstance(frame, PerceptionFrameContext):
        return frame
    return PerceptionFrameContext(frame=frame)


def unwrap_frame(frame: FrameLike) -> AssetFrame | NumpyFrameArray | LegacyFrameTensor | str:
    if isinstance(frame, PerceptionFrameContext):
        return frame.frame
    return frame
