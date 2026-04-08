from __future__ import annotations

from learned.interfaces import TemporalConsistencyModel, TemporalRefinementOutput, TemporalRefinementRequest
from rendering.compositor import TemporalStabilizer
from rendering.trainable_temporal_consistency import (
    TemporalInferenceError,
    TemporalInputError,
    TrainableTemporalConsistencyModel,
    build_temporal_batch,
    extract_history_frame,
    output_from_temporal_prediction,
)


class LegacyBaselineTemporalConsistencyModel(TemporalConsistencyModel):
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
                "temporal_path": "legacy_fallback",
                "learned_ready_usage": {
                    "memory_channels_used": [k for k, v in request.memory_channels.items() if v],
                    "hidden_regions_signal_used": bool(hidden_signal),
                    "drift_mode": "aggressive_anti_drift" if drift > 0.08 else ("moderate_anti_drift" if drift > 0.03 else "stable"),
                },
            },
        )


class TrainableTemporalConsistencyBackend(TemporalConsistencyModel):
    def __init__(
        self,
        model: TrainableTemporalConsistencyModel | None = None,
        fallback: LegacyBaselineTemporalConsistencyModel | None = None,
        strict_mode: bool = False,
    ) -> None:
        self.model = model or TrainableTemporalConsistencyModel()
        self.fallback = fallback or LegacyBaselineTemporalConsistencyModel()
        self.strict_mode = strict_mode

    def refine_temporal(self, request: TemporalRefinementRequest) -> TemporalRefinementOutput:
        try:
            history_frame = extract_history_frame(request.memory_state)
            batch = build_temporal_batch(request, history_frame=history_frame)
            pred = self.model.infer(batch)
            return output_from_temporal_prediction(
                request,
                pred,
                temporal_path="learned_primary",
                metadata={
                    "fallback_used": False,
                    "strict_mode": self.strict_mode,
                    "history_used": history_frame is not None,
                    "conditioning": {
                        "transition_cond_mean": float(batch.transition_cond.mean()),
                        "memory_cond_mean": float(batch.memory_cond.mean()),
                        "history_cond_mean": float(batch.history_cond.mean()),
                        "changed_ratio": float(batch.changed_mask.mean()),
                        "alpha_hint_mean": float(batch.alpha_hint.mean()),
                        "confidence_hint_mean": float(batch.confidence_hint.mean()),
                    },
                },
            )
        except (TemporalInputError, TemporalInferenceError, ValueError) as err:
            if self.strict_mode:
                raise
            fb = self.fallback.refine_temporal(request)
            fb.metadata = dict(fb.metadata)
            fb.metadata.update(
                {
                    "temporal_path": "legacy_fallback",
                    "fallback_reason": type(err).__name__,
                    "fallback_message": str(err),
                    "strict_mode": self.strict_mode,
                }
            )
            usage = dict(fb.metadata.get("learned_ready_usage", {}))
            usage.update({"fallback_reason": type(err).__name__, "fallback_message": str(err)})
            fb.metadata["learned_ready_usage"] = usage
            return fb


BaselineTemporalConsistencyModel = LegacyBaselineTemporalConsistencyModel
