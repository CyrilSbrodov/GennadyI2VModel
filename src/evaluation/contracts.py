from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalPayload:
    stage: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)


def build_text_eval_payload(contract: dict[str, object]) -> dict[str, object]:
    tokens = contract.get("parsed_actions", []) if isinstance(contract, dict) else []
    embedding = contract.get("action_embedding", []) if isinstance(contract, dict) else []
    return {
        "alignment_score": min(1.0, len(tokens) / max(1.0, float(len(embedding) or 1))),
        "token_count": float(len(tokens)),
    }


def build_graph_eval_payload(contract: dict[str, object]) -> dict[str, object]:
    delta = contract.get("delta_contract", {}) if isinstance(contract, dict) else {}
    magnitude = float(delta.get("magnitude", 0.0)) if isinstance(delta, dict) else 0.0
    return {
        "delta_match": max(0.0, 1.0 - min(1.0, magnitude)),
        "delta_magnitude": magnitude,
    }


def build_patch_eval_payload(contract: dict[str, object]) -> dict[str, object]:
    strategy = str(contract.get("selected_strategy", "unknown")) if isinstance(contract, dict) else "unknown"
    quality = 0.7 if strategy and strategy != "unknown" else 0.4
    return {
        "patch_quality": quality,
        "identity_consistency": 0.75 if contract.get("hidden_lifecycle_state") else 0.55,
    }


def build_temporal_eval_payload(contract: dict[str, object]) -> dict[str, object]:
    changed = contract.get("changed_regions", []) if isinstance(contract, dict) else []
    drift = 0.2 + 0.05 * max(0, len(changed) - 1)
    return {
        "temporal_drift": min(1.0, drift),
        "changed_count": float(len(changed)),
    }


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
