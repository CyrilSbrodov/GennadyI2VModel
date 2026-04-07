from __future__ import annotations

from perception.mask_store import StoredMask


def _safe2d(payload: object) -> list[list[int]]:
    arr = payload.tolist() if hasattr(payload, "tolist") else payload
    if not isinstance(arr, list) or not arr:
        return [[0]]
    if isinstance(arr[0], list):
        return [[1 if int(v) > 0 else 0 for v in row] for row in arr]
    return [([1 if int(v) > 0 else 0 for v in arr])]


def project_mask_to_frame(stored: StoredMask, frame_size: tuple[int, int] | None = None) -> tuple[list[list[int]], str]:
    base = _safe2d(stored.payload)
    target = frame_size or stored.frame_size
    if not target:
        return base, "local"

    fw, fh = target
    if fw <= 0 or fh <= 0:
        return base, "local"
    if len(base) == fh and len(base[0]) == fw:
        return base, "full"

    if not stored.roi_bbox:
        return base, "local"

    x, y, w, h = stored.roi_bbox
    x1 = max(0, min(fw - 1, int(round(x * fw))))
    y1 = max(0, min(fh - 1, int(round(y * fh))))
    x2 = max(x1 + 1, min(fw, int(round((x + w) * fw))))
    y2 = max(y1 + 1, min(fh, int(round((y + h) * fh))))

    rh = max(1, y2 - y1)
    rw = max(1, x2 - x1)
    out = [[0 for _ in range(fw)] for _ in range(fh)]

    src_h = len(base)
    src_w = len(base[0]) if src_h else 1
    for yy in range(rh):
        sy = min(src_h - 1, int(round((yy / max(1, rh - 1)) * max(0, src_h - 1))))
        for xx in range(rw):
            sx = min(src_w - 1, int(round((xx / max(1, rw - 1)) * max(0, src_w - 1))))
            out[y1 + yy][x1 + xx] = 1 if int(base[sy][sx]) > 0 else 0
    return out, "projected"
