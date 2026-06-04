from __future__ import annotations

import numpy as np


def map_to_shape(value: np.ndarray | None, shape_hw: tuple[int, int], fill: float = 0.0) -> np.ndarray:
    h, w = shape_hw
    if value is None:
        return np.full((h, w, 1), fill, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[:2] != (h, w):
        raise RuntimeError(f"Unexpected map shape {arr.shape}, expected {(h, w)}")
    if arr.shape[2] != 1:
        arr = np.mean(arr, axis=2, keepdims=True)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)
