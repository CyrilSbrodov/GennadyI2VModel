from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    internal_resolution: tuple[int, int]
    max_transition_steps: int
    temporal_refinement: bool
    roi_only: bool


LIGHTWEIGHT = RuntimeProfile("lightweight", (512, 512), 8, False, True)
BALANCED = RuntimeProfile("balanced", (768, 768), 16, True, True)
QUALITY = RuntimeProfile("quality", (1024, 1024), 24, True, True)
DEBUG = RuntimeProfile("debug", (768, 768), 16, True, True)

PROFILES = {p.name: p for p in (LIGHTWEIGHT, BALANCED, QUALITY, DEBUG)}
