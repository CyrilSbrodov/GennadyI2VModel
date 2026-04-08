from __future__ import annotations

from dataclasses import dataclass

from core.schema import GraphDelta, SceneGraph, VideoMemory
from dynamics.graph_delta_predictor import DynamicsMetrics, GraphDeltaPredictor
from learned.interfaces import DynamicsTransitionModel, DynamicsTransitionOutput, DynamicsTransitionRequest
from text.encoder_contracts import TextEncodingDiagnostics


@dataclass(slots=True)
class TransitionPrediction:
    delta: GraphDelta
    metrics: DynamicsMetrics
    backend: str


class LearnedDynamicsTransitionModel(DynamicsTransitionModel):
    """Primary runtime dynamics backend: learned graph-delta predictor."""

    def __init__(self, *, strict_mode: bool = False) -> None:
        self.predictor = GraphDeltaPredictor(strict_mode=strict_mode)

    def predict_transition(self, request: DynamicsTransitionRequest) -> DynamicsTransitionOutput:
        labels = request.text_action_summary.structured_action_tokens or ["micro_adjust"]
        step_index = int(request.step_context.get("step_index", 1))
        planned_state = type("_PS", (), {"labels": labels, "step_index": step_index})()
        strict = bool(request.step_context.get("strict_dynamics", False))
        if strict and not self.predictor.strict_mode:
            self.predictor = GraphDeltaPredictor(strict_mode=True)
        delta, metrics = self.predictor.predict(
            scene_graph=request.graph_state,
            target_state=planned_state,
            planner_context={"step_index": float(step_index), "total_steps": float(max(step_index + 1, 2))},
            memory=request.step_context.get("memory") if isinstance(request.step_context.get("memory"), VideoMemory) else None,
        )
        used_channels = [name for name, payload in request.memory_channels.items() if payload]
        return DynamicsTransitionOutput(
            delta=delta,
            confidence=max(0.0, min(1.0, 1.0 - 0.05 * metrics.constraint_violations)),
            diagnostics={
                "delta_magnitude": metrics.delta_magnitude,
                "temporal_smoothness_proxy": metrics.temporal_smoothness_proxy,
                "constraint_violations": float(metrics.constraint_violations),
            },
            metadata={
                "backend": "learned_graph_delta_predictor",
                "runtime_path": delta.transition_diagnostics.get("runtime_path", "learned_primary"),
                "learned_ready_usage": {
                    "memory_channels_used": used_channels,
                    "graph_encoding_used": bool(request.graph_encoding and request.graph_encoding.graph_embedding),
                    "identity_embeddings_used": bool(request.identity_embeddings),
                },
            },
        )


class LegacyHeuristicDynamicsTransitionModel(DynamicsTransitionModel):
    """Explicit legacy fallback backend for debug only."""

    def __init__(self, *, strict_mode: bool = False) -> None:
        self.predictor = GraphDeltaPredictor(strict_mode=strict_mode)

    def predict_transition(self, request: DynamicsTransitionRequest) -> DynamicsTransitionOutput:
        labels = request.text_action_summary.structured_action_tokens or ["micro_adjust"]
        step_index = int(request.step_context.get("step_index", 1))
        planned_state = type("_PS", (), {"labels": labels, "step_index": step_index})()
        delta, metrics = self.predictor._predict_legacy(
            scene_graph=request.graph_state,
            target_state=planned_state,
            planner_context={"step_index": float(step_index)},
            memory=request.step_context.get("memory") if isinstance(request.step_context.get("memory"), VideoMemory) else None,
        )
        return DynamicsTransitionOutput(delta=delta, confidence=0.5, diagnostics={"delta_magnitude": metrics.delta_magnitude}, metadata={"backend": "legacy_heuristic_fallback"})


class LearnedReadyTransitionEngine:
    def __init__(self, model: DynamicsTransitionModel | None = None) -> None:
        self.model = model or LearnedDynamicsTransitionModel()

    def predict(
        self,
        graph_state: SceneGraph,
        memory_summary: dict[str, object],
        text_tokens: list[str],
        step_context: dict[str, object],
    ) -> TransitionPrediction:
        text_summary = request_text_summary(text_tokens)
        request = DynamicsTransitionRequest(
            graph_state=graph_state,
            memory_summary=memory_summary,
            text_action_summary=text_summary,
            step_context=step_context,
        )
        out = self.model.predict_transition(request)
        return TransitionPrediction(delta=out.delta, metrics=_metrics_from_delta(out.delta), backend=str(out.metadata.get("backend", "unknown")))


def request_text_summary(tokens: list[str]):
    from learned.interfaces import TextEncodingOutput

    return TextEncodingOutput(
        global_text_embedding=[1.0],
        action_embedding=[float(len(tokens))],
        target_embedding=[0.0],
        modifier_embedding=[0.0],
        temporal_embedding=[0.0],
        constraint_embedding=[0.0],
        grounding_embedding=[0.0],
        structured_action_tokens=tokens,
        grounded_targets=[],
        parser_confidence=0.5,
        encoder_confidence=0.5,
        diagnostics=TextEncodingDiagnostics(action_count=len(tokens), parser_confidence=0.5, encoder_confidence=0.5),
        confidence=0.5,
    )


def _metrics_from_delta(delta: GraphDelta) -> DynamicsMetrics:
    magnitude = sum(abs(v) for v in delta.pose_deltas.values()) + sum(abs(v) for v in delta.interaction_deltas.values())
    return DynamicsMetrics(
        delta_magnitude=magnitude,
        constraint_violations=0,
        temporal_smoothness_proxy=1.0 / (1.0 + magnitude),
    )
