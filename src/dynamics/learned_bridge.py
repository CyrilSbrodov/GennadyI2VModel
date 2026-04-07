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
        graph_dim = len(request.graph_encoding.graph_embedding) if request.graph_encoding else 0
        identity_dim = sum(len(v) for v in request.identity_embeddings.values())
        memory_signal = float(sum(1 for v in request.memory_channels.values() if v))
        phase_bias = "settle" if memory_signal > 1 else ("motion" if graph_dim > 0 else delta.transition_phase)
        if request.graph_encoding and request.graph_encoding.graph_embedding:
            relation_signal = request.graph_encoding.graph_embedding[2] if len(request.graph_encoding.graph_embedding) > 2 else 0.0
            if relation_signal > 0.35:
                delta.region_transition_mode = {
                    region: ("deform_relation_aware" if "arm" in region or "torso" in region else mode)
                    for region, mode in delta.region_transition_mode.items()
                }
        if identity_dim:
            smooth = max(0.85, 1.0 - min(0.15, identity_dim * 0.002))
            delta.pose_deltas = {k: float(v) * smooth for k, v in delta.pose_deltas.items()}
        if memory_signal and delta.affected_regions:
            delta.state_after = dict(delta.state_after)
            delta.state_after.setdefault("transition_bias", "memory_smoothed")
            emphasis = delta.affected_regions[0]
            delta.state_after.setdefault("region_emphasis", emphasis)
        delta.transition_phase = phase_bias
        confidence_boost = min(0.15, 0.01 * graph_dim + 0.005 * identity_dim + 0.03 * memory_signal)
        used_channels = [name for name, payload in request.memory_channels.items() if payload]
        return DynamicsTransitionOutput(
            delta=delta,
            confidence=max(0.0, min(1.0, 1.0 - 0.1 * metrics.constraint_violations + confidence_boost)),
            diagnostics={
                "delta_magnitude": metrics.delta_magnitude,
                "temporal_smoothness_proxy": metrics.temporal_smoothness_proxy,
                "constraint_violations": float(metrics.constraint_violations),
                "graph_encoding_signal": float(graph_dim),
                "identity_signal": float(identity_dim),
                "memory_channel_signal": memory_signal,
            },
            metadata={
                "backend": "deterministic_graph_delta_predictor",
                "delta_magnitude": metrics.delta_magnitude,
                "smoothness": metrics.temporal_smoothness_proxy,
                "learned_ready_usage": {
                    "graph_encoding_used": bool(request.graph_encoding and request.graph_encoding.graph_embedding),
                    "identity_embeddings_used": bool(request.identity_embeddings),
                    "memory_channels_used": used_channels,
                    "ignored_fields": [name for name, payload in request.memory_channels.items() if not payload],
                },
                "delta_postprocess": {
                    "phase_bias": phase_bias,
                    "identity_smoothing_applied": bool(identity_dim),
                    "region_emphasis": delta.state_after.get("region_emphasis"),
                },
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
