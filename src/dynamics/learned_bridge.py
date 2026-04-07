from __future__ import annotations

from dataclasses import dataclass

from core.schema import ActionStep, GraphDelta, SceneGraph, VideoMemory
from dynamics.graph_delta_predictor import DynamicsMetrics, GraphDeltaPredictor
from learned.interfaces import DynamicsTransitionModel, DynamicsTransitionOutput, DynamicsTransitionRequest


@dataclass(slots=True)
class TransitionPrediction:
    delta: GraphDelta
    metrics: DynamicsMetrics
    backend: str


class BaselineDynamicsTransitionModel(DynamicsTransitionModel):
    """Fallback deterministic backend implementing learned transition contract."""

    def __init__(self) -> None:
        self.predictor = GraphDeltaPredictor()

    def predict_transition(self, request: DynamicsTransitionRequest) -> DynamicsTransitionOutput:
        labels = request.text_action_summary.structured_action_tokens or ["micro_adjust"]
        step_index = int(request.step_context.get("step_index", 1))
        planned_state = type("_PS", (), {"labels": labels, "step_index": step_index})()
        delta, metrics = self.predictor.predict(
            scene_graph=request.graph_state,
            target_state=planned_state,
            planner_context={"step_index": float(step_index)},
            memory=request.step_context.get("memory") if isinstance(request.step_context.get("memory"), VideoMemory) else None,
        )
        return DynamicsTransitionOutput(
            delta=delta,
            confidence=max(0.0, min(1.0, 1.0 - 0.1 * metrics.constraint_violations)),
            metadata={
                "backend": "deterministic_graph_delta_predictor",
                "delta_magnitude": metrics.delta_magnitude,
                "smoothness": metrics.temporal_smoothness_proxy,
            },
        )


class LearnedReadyTransitionEngine:
    def __init__(self, model: DynamicsTransitionModel | None = None) -> None:
        self.model = model or BaselineDynamicsTransitionModel()

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

    return TextEncodingOutput(action_embedding=[float(len(tokens))], structured_action_tokens=tokens, confidence=0.5)


def _metrics_from_delta(delta: GraphDelta) -> DynamicsMetrics:
    magnitude = sum(abs(v) for v in delta.pose_deltas.values()) + sum(abs(v) for v in delta.interaction_deltas.values())
    return DynamicsMetrics(
        delta_magnitude=magnitude,
        constraint_violations=0,
        temporal_smoothness_proxy=1.0 / (1.0 + magnitude),
    )
