from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from learned.factory import BackendConfig
from runtime.orchestrator import GennadyEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Gennady image-to-video scene engine demo.")
    parser.add_argument("--patch-checkpoint-path", default="", help="Renderer patch checkpoint JSON to load for runtime inference.")
    parser.add_argument("--patch-strict-mode", action="store_true", help="Raise renderer inference errors instead of using legacy runtime fallback.")
    parser.add_argument(
        "--allow-patch-checkpoint-fallback",
        action="store_true",
        help="Allow runtime to continue with normal patch backend creation if the requested patch checkpoint cannot be loaded.",
    )
    parser.add_argument("--prompt", default="Снимает пальто и садится на стул. Улыбается.", help="Text prompt for the demo run.")
    parser.add_argument("--export-renderer-manifest-path", default=None, help="Write renderer_patch_manifest_v2 training records from the runtime patch loop.")
    parser.add_argument("inputs", nargs="*", default=["ref_0001.png"], help="Input frame paths for the demo run.")
    return parser


def backend_config_from_args(args: argparse.Namespace) -> BackendConfig:
    return BackendConfig(
        patch_checkpoint_path=str(args.patch_checkpoint_path or ""),
        patch_strict_mode=bool(args.patch_strict_mode),
        patch_strict_checkpoint=not bool(args.allow_patch_checkpoint_fallback),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = GennadyEngine(backend_config=backend_config_from_args(args))
    result = engine.run(list(args.inputs), args.prompt, export_renderer_manifest_path=args.export_renderer_manifest_path)
    print(json.dumps({"frames": len(result.frames), "state_steps": len(result.state_plan.steps)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
