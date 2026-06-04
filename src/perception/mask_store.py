from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Any


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
    # For perception masks, extra["bbox_xyxy"] is normalized frame-space xyxy
    # (x1, y1, x2, y2 in [0, 1]) and extra["pixel_count"] is payload-local.
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
        ref: str | None = None,
    ) -> str:
        mid = next(self._counter)
        ref = ref or f"mask://{prefix}/{mid}"
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

    def clear(self) -> None:
        """Remove all runtime masks and reset deterministic per-run numbering."""
        self._counter = itertools.count(1)
        self._items.clear()

    def snapshot_metadata(self) -> dict[str, dict[str, object]]:
        return {
            ref: {
                "confidence": item.confidence,
                "source": item.source,
                "mask_kind": item.mask_kind,
                "backend": item.backend,
                "roi_bbox": item.roi_bbox,
                "frame_size": item.frame_size,
                "tags": list(item.tags),
                "extra": dict(item.extra),
            }
            for ref, item in self._items.items()
        }




def mask_store_from_frame(frame: Any, *, allow_legacy_default: bool = False) -> InMemoryMaskStore:
    """Return the explicit per-run mask store carried by a frame context.

    Production perception paths must receive an InMemoryMaskStore via
    PerceptionFrameContext["mask_store"]. Legacy helper/tests may opt into
    DEFAULT_MASK_STORE by passing allow_legacy_default=True.
    """

    store = frame.get("mask_store") if hasattr(frame, "get") else None
    if isinstance(store, InMemoryMaskStore):
        return store
    if allow_legacy_default:
        return DEFAULT_MASK_STORE
    raise RuntimeError("Perception frame context is missing explicit InMemoryMaskStore; production mask writes cannot use DEFAULT_MASK_STORE")


def legacy_mask_store_from_frame(frame: Any) -> InMemoryMaskStore:
    return mask_store_from_frame(frame, allow_legacy_default=True)


DEFAULT_MASK_STORE = InMemoryMaskStore()
