from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.video_transition_manifest import VideoTransitionBuilderConfig, VideoTransitionManifestBuilder, save_video_transition_manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract video transition training manifest from video.")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output manifest path")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--quality-profile", default="debug")
    p.add_argument("--text", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = VideoTransitionBuilderConfig(fps=args.fps, duration=args.duration, quality_profile=args.quality_profile)
    builder = VideoTransitionManifestBuilder(config=cfg)
    manifest = builder.build_from_video(args.video, text=args.text)
    save_video_transition_manifest(manifest, args.output)
    print(json.dumps({"output": args.output, "records": len(manifest.get("records", [])), "manifest_type": manifest.get("manifest_type")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
