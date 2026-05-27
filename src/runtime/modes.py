from __future__ import annotations

from typing import Literal

RuntimeMode = Literal["debug_stub", "trainable_stub", "strict_learned", "production_eval"]


def normalize_runtime_mode(value: str) -> RuntimeMode:
    mode = str(value or "").strip().lower()
    if mode in {"debug_stub", "trainable_stub", "strict_learned", "production_eval"}:
        return mode  # type: ignore[return-value]
    raise ValueError(f"Unsupported runtime_mode: {value}")


def runtime_forbids_fallbacks(mode: RuntimeMode) -> bool:
    return mode in {"strict_learned", "production_eval"}

