from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    internal_resolution: tuple[int, int]
    max_transition_steps: int
    temporal_refinement: bool
    roi_only: bool
    backend: str
    model_variant: str
    precision: str
    max_roi_count: int
    temporal_window: int
    frame_budget_ms: int
    debug_artifacts: bool = False


LIGHTWEIGHT = RuntimeProfile(
    "lightweight",
    (512, 512),
    8,
    False,
    True,
    backend="numpy",
    model_variant="tiny",
    precision="fp16",
    max_roi_count=3,
    temporal_window=2,
    frame_budget_ms=25,
)
BALANCED = RuntimeProfile(
    "balanced",
    (768, 768),
    16,
    True,
    True,
    backend="torch",
    model_variant="base",
    precision="fp16",
    max_roi_count=6,
    temporal_window=4,
    frame_budget_ms=40,
)
QUALITY = RuntimeProfile(
    "quality",
    (1024, 1024),
    24,
    True,
    True,
    backend="torch",
    model_variant="large",
    precision="fp32",
    max_roi_count=10,
    temporal_window=8,
    frame_budget_ms=80,
)
DEBUG = RuntimeProfile(
    "debug",
    (768, 768),
    16,
    True,
    True,
    backend="numpy",
    model_variant="debug",
    precision="fp32",
    max_roi_count=8,
    temporal_window=6,
    frame_budget_ms=200,
    debug_artifacts=True,
)

PROFILES = {p.name: p for p in (LIGHTWEIGHT, BALANCED, QUALITY, DEBUG)}
