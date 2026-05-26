from __future__ import annotations
import argparse, json
import dataclasses
import sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from runtime.orchestrator import GennadyEngine


def _save_frame(frame: list, path: Path) -> None:
    arr = np.asarray(frame, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def _safe_json(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {k: _safe_json(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    duration = max(1.0 / max(1, args.fps), float(args.frames) / max(1, args.fps))
    engine = GennadyEngine()
    artifacts = engine.run([args.image], args.prompt, fps=args.fps, duration=duration, quality_profile='debug')

    frames_to_save = artifacts.frames[: max(0, min(args.frames, len(artifacts.frames)))]
    for i, frame in enumerate(frames_to_save):
        _save_frame(frame, out / f'frame_{i:04d}.png')

    try:
        imgs = [Image.open(out / f'frame_{i:04d}.png') for i in range(len(frames_to_save))]
        if imgs:
            imgs[0].save(out / 'preview.gif', save_all=True, append_images=imgs[1:], duration=max(1, int(1000 / max(1, args.fps))), loop=0)
    except Exception:
        pass

    report = {
        'input_image': args.image,
        'prompt': args.prompt,
        'frames_requested': args.frames,
        'frames_generated': len(artifacts.frames),
        'frames_saved': len(frames_to_save),
        'fps': args.fps,
        'state_plan_steps': len(artifacts.state_plan.steps),
        'i2v_frame_plan': artifacts.debug.get('i2v_frame_plan', []),
        'step_execution': artifacts.debug.get('step_execution', []),
        'reference_coverage_summary': artifacts.debug.get('reference_coverage_summary', {}),
        'video_export': artifacts.debug.get('video_export'),
    }
    (out / 'debug_report.json').write_text(json.dumps(_safe_json(report), indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
