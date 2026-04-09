from __future__ import annotations

from dataclasses import dataclass

from core.schema import GraphDelta, SceneGraph, VideoMemory
from dynamics.graph_delta_predictor import DynamicsMetrics, GraphDeltaPredictor
from dynamics.temporal_contract_alignment import compute_temporal_contract_alignment
from dynamics.temporal_transition_encoder import TemporalTransitionEncoder
from dynamics.transition_contracts import LearnedTemporalTransitionContract
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
        self.temporal_transition_encoder = TemporalTransitionEncoder()

    @staticmethod
    def _transition_feature_vector(graph_state: SceneGraph, delta: GraphDelta, step_index: int) -> list[float]:
        person = graph_state.persons[0] if graph_state.persons else None
        base = [
            float(person.bbox.x) if person else 0.0,
            float(person.bbox.y) if person else 0.0,
            float(person.bbox.w) if person else 0.0,
            float(person.bbox.h) if person else 0.0,
            float(len(graph_state.persons)) / 4.0,
            float(len(graph_state.objects)) / 8.0,
            float(len(delta.pose_deltas)),
            float(len(delta.garment_deltas)),
            float(len(delta.visibility_deltas)),
            float(len(delta.expression_deltas)),
            float(len(delta.interaction_deltas)),
            float(step_index) / 16.0,
            1.0 if delta.transition_phase == "prepare" else 0.0,
            1.0 if delta.transition_phase == "transition" else 0.0,
            1.0 if delta.transition_phase == "contact_or_reveal" else 0.0,
            1.0 if delta.transition_phase == "stabilize" else 0.0,
            float(delta.interaction_deltas.get("support_contact", 0.0)),
            float(delta.visibility_deltas.get("revealed_regions_score", 0.0)) if isinstance(delta.visibility_deltas, dict) else 0.0,
            float(delta.visibility_deltas.get("occluded_regions_score", 0.0)) if isinstance(delta.visibility_deltas, dict) else 0.0,
        ]
        if len(base) < 128:
            base.extend([0.0] * (128 - len(base)))
        return base[:128]

    @staticmethod
    def _apply_learned_contract_to_delta(delta: GraphDelta, contract: LearnedTemporalTransitionContract) -> None:
        weak_phase = str(delta.transition_phase or "")
        weak_target_profile = dict(delta.transition_diagnostics.get("target_profile", {})) if isinstance(delta.transition_diagnostics, dict) else {}
        delta.transition_phase = contract.predicted_phase
        if not isinstance(delta.transition_diagnostics, dict):
            delta.transition_diagnostics = {}
        delta.transition_diagnostics["weak_manifest_fallback"] = {
            "phase": weak_phase,
            "target_profile": weak_target_profile,
        }
        delta.transition_diagnostics["target_profile"] = contract.to_metadata()["target_profile"]
        delta.transition_diagnostics["learned_temporal_primary"] = True
        delta.transition_diagnostics["teacher_source"] = contract.teacher_source
        if contract.reveal_score >= max(contract.occlusion_score, 0.55):
            for region in contract.target_profile.primary_regions:
                delta.region_transition_mode.setdefault(region, "garment_reveal")
        elif contract.occlusion_score > 0.55:
            for region in contract.target_profile.primary_regions:
                delta.region_transition_mode.setdefault(region, "visibility_occlusion")
        else:
            for region in contract.target_profile.primary_regions:
                delta.region_transition_mode.setdefault(region, "pose_exposure")
        delta.semantic_reasons = [contract.predicted_family] + [x for x in delta.semantic_reasons if x != contract.predicted_family]
        contract_regions = contract.target_profile.primary_regions + contract.target_profile.secondary_regions + contract.target_profile.context_regions
        for region in contract_regions:
            if region not in delta.affected_regions:
                delta.affected_regions.append(region)

    def predict_transition(self, request: DynamicsTransitionRequest) -> DynamicsTransitionOutput:
        labels = request.text_action_summary.structured_action_tokens or ["micro_adjust"]
        step_index = int(request.step_context.get("step_index", 1))
        planned_state = type(
            "_PS",
            (),
            {
                "labels": labels,
                "step_index": step_index,
                "semantic_transition": request.step_context.get("semantic_transition"),
            },
        )()
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
        temporal_features = self._transition_feature_vector(request.graph_state, delta, step_index)
        temporal_prediction = self.temporal_transition_encoder.forward(temporal_features)
        typed_contract = self.temporal_transition_encoder.to_typed_contract(temporal_prediction)
        temporal_contract = self.temporal_transition_encoder.to_contract(temporal_prediction)
        self._apply_learned_contract_to_delta(delta, typed_contract)
        alignment = compute_temporal_contract_alignment(typed_contract, delta)
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
                "temporal_transition_contract": temporal_contract,
                "temporal_contract_alignment": alignment,
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
        planned_state = type(
            "_PS",
            (),
            {
                "labels": labels,
                "step_index": step_index,
                "semantic_transition": request.step_context.get("semantic_transition"),
            },
        )()
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
