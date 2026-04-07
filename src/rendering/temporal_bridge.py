from __future__ import annotations

from learned.interfaces import TemporalConsistencyModel, TemporalRefinementOutput, TemporalRefinementRequest
from rendering.compositor import TemporalStabilizer


class BaselineTemporalConsistencyModel(TemporalConsistencyModel):
    def __init__(self, stabilizer: TemporalStabilizer | None = None) -> None:
        self.stabilizer = stabilizer or TemporalStabilizer()

    def refine_temporal(self, request: TemporalRefinementRequest) -> TemporalRefinementOutput:
        changed = request.changed_regions
        region_conf = 0.7 if not changed else min(0.95, 0.5 + 0.05 * len(changed))
        hidden_signal = request.memory_channels.get("hidden_regions", {})
        identity_signal = request.memory_channels.get("identity", {})
        body_signal = request.memory_channels.get("body_regions", {})
        if isinstance(hidden_signal, dict) and hidden_signal:
            region_conf = min(0.98, region_conf + 0.04)
        if isinstance(identity_signal, dict) and identity_signal:
            region_conf = min(0.98, region_conf + 0.02)
        drift = float(hidden_signal.get("drift", 0.0)) if isinstance(hidden_signal, dict) else 0.0
        if drift > 0.08:
            region_conf = max(0.55, region_conf - 0.08)
        elif drift > 0.03:
            region_conf = max(0.62, region_conf - 0.03)
        refined = self.stabilizer.refine(
            previous_frame=request.previous_frame,
            new_frame=request.current_composed_frame,
            memory=request.memory_state,
            enabled=True,
            updated_regions=changed,
            region_confidence=region_conf,
        )
        body_weight = float(body_signal.get("roi_count", 0)) if isinstance(body_signal, dict) else 0.0
        scores = {}
        for idx, r in enumerate(changed):
            region_weight = 0.03 if ("face" in r.region_id or "head" in r.region_id) else 0.0
            region_weight += 0.02 if ("torso" in r.region_id and body_weight > 1) else 0.0
            drift_penalty = 0.04 * min(1.0, drift * 8.0) * idx
            scores[r.region_id] = max(0.0, min(1.0, region_conf - 0.02 * idx + region_weight - drift_penalty))
        return TemporalRefinementOutput(
            refined_frame=refined,
            region_consistency_scores=scores,
            metadata={
                "backend": "deterministic_temporal_stabilizer",
                "learned_ready_usage": {
                    "memory_channels_used": [k for k, v in request.memory_channels.items() if v],
                    "hidden_regions_signal_used": bool(hidden_signal),
                    "drift_mode": "aggressive_anti_drift" if drift > 0.08 else ("moderate_anti_drift" if drift > 0.03 else "stable"),
                },
            },
        )
