from __future__ import annotations

from dataclasses import dataclass, field
import itertools


@dataclass(slots=True)
class StoredMask:
    ref: str
    payload: "object"
    confidence: float
    source: str
    mask_kind: str = "generic"
    backend: str = "unknown"
    roi_bbox: tuple[float, float, float, float] | None = None
    frame_size: tuple[int, int] | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, "object"] = field(default_factory=dict)


class InMemoryMaskStore:
    """Простой runtime-store масок, чтобы mask_ref ссылался на реальный payload."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._items: dict[str, StoredMask] = {}

    def put(
        self,
        payload: "object",
        confidence: float,
        source: str,
        prefix: str,
        *,
        mask_kind: str = "generic",
        backend: str = "unknown",
        roi_bbox: tuple[float, float, float, float] | None = None,
        frame_size: tuple[int, int] | None = None,
        tags: list[str] | None = None,
        extra: dict[str, "object"] | None = None,
    ) -> str:
        mid = next(self._counter)
        ref = f"mask://{prefix}/{mid}"
        self._items[ref] = StoredMask(
            ref=ref,
            payload=payload,
            confidence=confidence,
            source=source,
            mask_kind=mask_kind,
            backend=backend,
            roi_bbox=roi_bbox,
            frame_size=frame_size,
            tags=tags or [],
            extra=extra or {},
        )
        return ref

    def get(self, ref: str) -> StoredMask | None:
        return self._items.get(ref)


DEFAULT_MASK_STORE = InMemoryMaskStore()
