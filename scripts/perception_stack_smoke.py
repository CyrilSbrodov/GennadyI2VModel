from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perception.detector import BackendConfig
from perception.pipeline import PerceptionBackendsConfig, PerceptionPipeline


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke utility for perception backend stack.")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--detector-backend", default="builtin", help="detector backend (builtin/ultralytics)")
    parser.add_argument("--detector-checkpoint", default="yolov8n-seg.pt", help="detector checkpoint")
    parser.add_argument("--pose-backend", default="builtin", help="pose backend (builtin/ultralytics/yolo_pose/mediapipe)")
    parser.add_argument("--pose-checkpoint", default="yolov8n-pose.pt", help="pose checkpoint")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    frame = _load_rgb(args.image)
    cfg = PerceptionBackendsConfig(
        detector=BackendConfig(backend=args.detector_backend, checkpoint=args.detector_checkpoint),
        pose=BackendConfig(backend=args.pose_backend, checkpoint=args.pose_checkpoint),
    )
    out = PerceptionPipeline(backends=cfg).analyze(frame)

    payload = {
        "frame_size": out.frame_size,
        "warnings": out.warnings,
        "module_fallbacks": out.module_fallbacks,
        "module_confidence": out.module_confidence,
        "persons": [
            {
                "bbox": {"x": p.bbox.x, "y": p.bbox.y, "w": p.bbox.w, "h": p.bbox.h},
                "bbox_source": p.bbox_source,
                "mask_ref": p.mask_ref,
                "mask_source": p.mask_source,
                "pose_source": p.pose_source,
                "num_keypoints": len(p.pose.keypoints),
                "num_garments": len(p.garments),
                "num_body_parts": len(p.body_parts),
                "num_face_regions": len(p.face_regions),
            }
            for p in out.persons
        ],
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
