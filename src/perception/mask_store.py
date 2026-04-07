from __future__ import annotations

from dataclasses import dataclass
import itertools


@dataclass(slots=True)
class StoredMask:
    ref: str
    payload: "object"
    confidence: float
    source: str


class InMemoryMaskStore:
    """Простой runtime-store масок, чтобы mask_ref ссылался на реальный payload."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._items: dict[str, StoredMask] = {}

    def put(self, payload: "object", confidence: float, source: str, prefix: str) -> str:
        mid = next(self._counter)
        ref = f"mask://{prefix}/{mid}"
        self._items[ref] = StoredMask(ref=ref, payload=payload, confidence=confidence, source=source)
        return ref

    def get(self, ref: str) -> StoredMask | None:
        return self._items.get(ref)


DEFAULT_MASK_STORE = InMemoryMaskStore()
