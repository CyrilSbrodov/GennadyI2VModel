from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class StageTimer:
    enabled: bool = False
    entries: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @contextmanager
    def track(self, stage: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.entries[stage].append(time.perf_counter() - t0)

    def add(self, stage: str, duration_sec: float) -> None:
        if self.enabled:
            self.entries[stage].append(float(duration_sec))

    def merge(self, other: "StageTimer") -> None:
        self.merge_entries(other.entries)

    def merge_entries(self, entries: dict[str, list[float]]) -> None:
        if not self.enabled:
            return
        for stage, durations in entries.items():
            if durations:
                self.entries[stage].extend(float(v) for v in durations)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, values in self.entries.items():
            if not values:
                continue
            total = sum(values)
            out[stage] = {
                "count": float(len(values)),
                "total_ms": round(total * 1000.0, 3),
                "avg_ms": round(total * 1000.0 / len(values), 3),
                "max_ms": round(max(values) * 1000.0, 3),
            }
        return out
