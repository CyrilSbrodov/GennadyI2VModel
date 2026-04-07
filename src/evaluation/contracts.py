from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalPayload:
    stage: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)


def text_action_alignment_eval(payload: dict[str, object]) -> EvalPayload:
    score = float(payload.get("alignment_score", 0.0))
    return EvalPayload(stage="text_action_alignment", metrics={"alignment_score": score}, artifacts=payload)


def graph_transition_eval(payload: dict[str, object]) -> EvalPayload:
    correctness = float(payload.get("delta_match", 0.0))
    return EvalPayload(stage="graph_transition", metrics={"transition_correctness": correctness}, artifacts=payload)


def hidden_region_reconstruction_eval(payload: dict[str, object]) -> EvalPayload:
    quality = float(payload.get("reconstruction_quality", 0.0))
    return EvalPayload(stage="hidden_region_reconstruction", metrics={"reconstruction_quality": quality}, artifacts=payload)


def patch_synthesis_eval(payload: dict[str, object]) -> EvalPayload:
    quality = float(payload.get("patch_quality", 0.0))
    identity = float(payload.get("identity_consistency", 0.0))
    return EvalPayload(stage="patch_synthesis", metrics={"patch_quality": quality, "identity_consistency": identity}, artifacts=payload)


def temporal_consistency_eval(payload: dict[str, object]) -> EvalPayload:
    drift = float(payload.get("temporal_drift", 1.0))
    consistency = max(0.0, 1.0 - drift)
    return EvalPayload(stage="temporal_consistency", metrics={"temporal_consistency": consistency, "temporal_drift": drift}, artifacts=payload)
