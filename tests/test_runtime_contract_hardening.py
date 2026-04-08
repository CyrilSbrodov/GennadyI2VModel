from __future__ import annotations

from pathlib import Path

import pytest

from learned.factory import BackendConfig, LearnedBackendFactory
from learned.interfaces import PatchSynthesisOutput
from runtime.orchestrator import GennadyEngine


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_engine_step_debug_contains_contract_validation(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    _write_ppm(img, 24, 24, (35, 90, 170))

    engine = GennadyEngine()
    artifacts = engine.run([str(img)], "Поворачивается и улыбается.", quality_profile="debug")

    first_step = artifacts.debug["step_execution"][0]
    assert "contract_validation" in first_step["temporal"]
    assert isinstance(first_step["patch"], list)
    if first_step["patch"]:
        assert "contract_validation" in first_step["patch"][0]


def test_engine_fails_fast_on_patch_contract_violation(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    _write_ppm(img, 24, 24, (120, 40, 90))

    bundle = LearnedBackendFactory(BackendConfig()).build()

    class BrokenPatchBackend:
        def synthesize_patch(self, request):
            return PatchSynthesisOutput(
                region=request.region,
                rgb_patch=[[[0.2, 0.2, 0.2] for _ in range(3)] for _ in range(3)],
                alpha_mask=[[1.0, 1.0]],
                height=3,
                width=3,
                channels=3,
                confidence=0.9,
                execution_trace={"renderer_path": "learned_primary", "selected_render_strategy": "BROKEN"},
            )

    bundle.patch_backend = BrokenPatchBackend()
    engine = GennadyEngine(backend_bundle=bundle)

    with pytest.raises(ValueError, match="Patch contract violation"):
        engine.run([str(img)], "Снимает куртку.", quality_profile="debug")


def test_backend_baseline_aliases_are_explicitly_legacy() -> None:
    bundle = LearnedBackendFactory(
        BackendConfig(dynamics_backend="baseline", patch_backend="baseline", temporal_backend="baseline")
    ).build()

    assert bundle.backend_names["dynamics_backend"] == "legacy_heuristic"
    assert bundle.backend_names["patch_backend"] == "legacy_deterministic"
    assert bundle.backend_names["temporal_backend"] == "legacy_baseline"
