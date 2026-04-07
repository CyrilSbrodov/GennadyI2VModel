from __future__ import annotations

import math


def zeros(h: int, w: int, c: int = 3, value: float = 0.0) -> list[list[list[float]]]:
    return [[[value for _ in range(c)] for _ in range(w)] for _ in range(h)]


def shape(x):
    if x is None:
        return 0, 0, 0

    if hasattr(x, "shape"):
        s = x.shape
        if len(s) == 0:
            return 0, 0, 0
        if len(s) == 1:
            return int(s[0]), 0, 0
        if len(s) == 2:
            return int(s[0]), int(s[1]), 0
        return int(s[0]), int(s[1]), int(s[2])

    h = len(x)
    w = len(x[0]) if h > 0 else 0
    c = len(x[0][0]) if h > 0 and w > 0 else 0
    return h, w, c


def clip01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def mean_color(patch: list[list[list[float]]]) -> list[float]:
    h, w, c = shape(patch)
    sums = [0.0] * c
    n = max(1, h * w)
    for row in patch:
        for px in row:
            for i, v in enumerate(px):
                sums[i] += v
    return [s / n for s in sums]


def crop(frame: list[list[list[float]]], x0: int, y0: int, x1: int, y1: int) -> list[list[list[float]]]:
    return [row[x0:x1] for row in frame[y0:y1]]


def blend(dst: list[list[list[float]]], src: list[list[list[float]]], alpha: list[list[float]]) -> list[list[list[float]]]:
    h = min(len(dst), len(src), len(alpha))
    w = min(len(dst[0]), len(src[0]), len(alpha[0])) if h else 0
    out = [row[:] for row in dst]
    for y in range(h):
        for x in range(w):
            a = clip01(alpha[y][x])
            out[y][x] = [clip01(src[y][x][k] * a + dst[y][x][k] * (1 - a)) for k in range(len(dst[y][x]))]
    return out


def roll_x(frame: list[list[list[float]]], shift: int = 1) -> list[list[list[float]]]:
    out = []
    for row in frame:
        if not row:
            out.append(row)
            continue
        s = shift % len(row)
        out.append(row[-s:] + row[:-s])
    return out


def alpha_radial(h: int, w: int) -> list[list[float]]:
    cy = (h - 1) / 2 if h else 0
    cx = (w - 1) / 2 if w else 0
    out = []
    for y in range(h):
        line = []
        for x in range(w):
            dy = (y - cy) / max(1.0, cy or 1.0)
            dx = (x - cx) / max(1.0, cx or 1.0)
            d = math.sqrt(dx * dx + dy * dy)
            line.append(clip01(1.0 - d))
        out.append(line)
    return out


def to_uint8_bytes(frame: list[list[list[float]]]) -> bytes:
    b = bytearray()
    for row in frame:
        for px in row:
            for ch in px:
                b.append(int(clip01(ch) * 255))
    return bytes(b)
