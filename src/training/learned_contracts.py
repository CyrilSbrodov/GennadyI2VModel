from __future__ import annotations

from typing import TypedDict

from core.schema import ActionStep, GraphDelta, RegionRef, SceneGraph
from training.datasets import _serialize_delta_contract, _serialize_graph


class TextActionStateContract(TypedDict):
    text: str
    parsed_actions: list[dict[str, object]]
    action_embedding: list[float]
    target_entities: list[str]
    target_objects: list[str]
    temporal_decomposition: list[dict[str, object]]
    constraints: list[str]


class GraphTransitionContract(TypedDict):
    graph_before: dict[str, object]
    graph_after: dict[str, object]
    delta_contract: dict[str, object]
    transition_context: dict[str, object]
    state_before: dict[str, str]
    state_after: dict[str, str]
    region_transition_mode: dict[str, str]
    semantic_reasons: list[str]


class PatchSynthesisContract(TypedDict):
    roi_before: object
    roi_after: object
    region_metadata: dict[str, object]
    retrieval_explanation_summary: str
    selected_render_strategy: str
    selected_strategy: str
    hidden_lifecycle_state: dict[str, object]
    synthesis_mode: str
    transition_context: dict[str, object]


class TemporalConsistencyContract(TypedDict):
    previous_frame: object
    composed_frame: object
    target_frame: object
    changed_regions: list[dict[str, object]]
    region_consistency_metadata: dict[str, object]
    scene_transition_context: dict[str, object]
    memory_transition_context: dict[str, object]


def build_text_action_state_contract(text: str, actions: list[ActionStep], encoding: dict[str, object]) -> TextActionStateContract:
    return {
        "text": text,
        "parsed_actions": [
            {
                "type": a.type,
                "priority": a.priority,
                "target_entity": a.target_entity,
                "target_object": a.target_object,
                "body_part": a.body_part,
            }
            for a in actions
        ],
        "action_embedding": list(encoding.get("action_embedding", [])),
        "target_entities": list(encoding.get("target_hints", {}).get("entities", [])),
        "target_objects": list(encoding.get("target_hints", {}).get("objects", [])),
        "temporal_decomposition": list(encoding.get("decomposition_hints", [])),
        "constraints": list(encoding.get("constraints", [])),
    }


def build_graph_transition_contract(before: SceneGraph, after: SceneGraph, delta: GraphDelta, transition_context: dict[str, object]) -> GraphTransitionContract:
    return {
        "graph_before": _serialize_graph(before),
        "graph_after": _serialize_graph(after),
        "delta_contract": _serialize_delta_contract(delta),
        "transition_context": transition_context,
        "state_before": delta.state_before,
        "state_after": delta.state_after,
        "region_transition_mode": delta.region_transition_mode,
        "semantic_reasons": delta.semantic_reasons,
    }


def build_patch_synthesis_contract(roi_before: object, roi_after: object, region: RegionRef, retrieval_summary: str, selected_strategy: str, hidden_state: dict[str, object], synthesis_mode: str, transition_context: dict[str, object]) -> PatchSynthesisContract:
    return {
        "roi_before": roi_before,
        "roi_after": roi_after,
        "region_metadata": {"region_id": region.region_id, "reason": region.reason},
        "retrieval_explanation_summary": retrieval_summary,
        "selected_render_strategy": selected_strategy,
        # backward-compatible alias for older contract consumers
        "selected_strategy": selected_strategy,
        "hidden_lifecycle_state": hidden_state,
        "synthesis_mode": synthesis_mode,
        "transition_context": transition_context,
    }


def build_temporal_consistency_contract(previous_frame: object, composed_frame: object, target_frame: object, changed_regions: list[RegionRef], region_consistency_metadata: dict[str, object], scene_transition_context: dict[str, object], memory_transition_context: dict[str, object]) -> TemporalConsistencyContract:
    return {
        "previous_frame": previous_frame,
        "composed_frame": composed_frame,
        "target_frame": target_frame,
        "changed_regions": [{"region_id": r.region_id, "reason": r.reason} for r in changed_regions],
        "region_consistency_metadata": region_consistency_metadata,
        "scene_transition_context": scene_transition_context,
        "memory_transition_context": memory_transition_context,
    }
