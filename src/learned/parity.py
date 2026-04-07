from __future__ import annotations

import copy
from dataclasses import asdict
from typing import TypedDict

from dynamics.state_update import apply_delta
from learned.interfaces import (
    DynamicsTransitionOutput,
    DynamicsTransitionRequest,
    PatchSynthesisOutput,
    PatchSynthesisRequest,
    TemporalRefinementOutput,
    TemporalRefinementRequest,
    TextEncodingOutput,
)
from training.learned_contracts import (
    GraphTransitionContract,
    PatchSynthesisContract,
    TemporalConsistencyContract,
    TextActionStateContract,
)

class ParityResult(TypedDict):
    errors: list[str]
    warnings: list[str]
    traces: list[str]


class StructuredParityResult(TypedDict):
    missing_fields: list[str]
    errors: list[str]
    warnings: list[str]
    traces: list[str]


def _empty_parity() -> ParityResult:
    return {"errors": [], "warnings": [], "traces": []}


def empty_structured_parity() -> StructuredParityResult:
    return {"missing_fields": [], "errors": [], "warnings": [], "traces": []}


def _push(parity: ParityResult, severity: str, issue: str) -> None:
    parity[severity].append(issue)


def graph_semantic_signature(graph: object) -> dict[str, object]:
    if not isinstance(graph, dict):
        return {"frame_index": -1, "person_count": 0, "object_count": 0, "relation_count": 0, "entity_bbox": {}, "state_markers": {}}
    persons = graph.get("persons", [])
    objects = graph.get("objects", [])
    relations = graph.get("relations", [])
    state_markers = {"visibility": {}, "garment_state": {}}
    entity_bbox: dict[str, tuple[float, float, float, float]] = {}
    for person in persons if isinstance(persons, list) else []:
        if not isinstance(person, dict):
            continue
        pid = str(person.get("person_id", ""))
        bbox = person.get("bbox", {})
        if isinstance(bbox, dict):
            entity_bbox[pid] = tuple(float(bbox.get(k, 0.0)) for k in ("x", "y", "w", "h"))
        for garment in person.get("garments", []) if isinstance(person.get("garments"), list) else []:
            if not isinstance(garment, dict):
                continue
            gid = str(garment.get("garment_id", ""))
            state_markers["garment_state"][gid] = str(garment.get("garment_state", "unknown"))
            state_markers["visibility"][gid] = str(garment.get("visibility", "unknown"))
        for part in person.get("body_parts", []) if isinstance(person.get("body_parts"), list) else []:
            if not isinstance(part, dict):
                continue
            part_id = str(part.get("part_id", ""))
            state_markers["visibility"][part_id] = str(part.get("visibility", "unknown"))
    for rel in relations if isinstance(relations, list) else []:
        if not isinstance(rel, dict):
            continue
        rid = f"{rel.get('source','')}->{rel.get('relation','')}->{rel.get('target','')}"
        state_markers["visibility"][rid] = str(rel.get("relation", "unknown"))
    return {
        "frame_index": int(graph.get("frame_index", -1)),
        "person_count": len(persons) if isinstance(persons, list) else 0,
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "relation_count": len(relations) if isinstance(relations, list) else 0,
        "entity_bbox": entity_bbox,
        "state_markers": state_markers,
    }


def graph_change_summary(before_graph: object, after_graph: object) -> dict[str, object]:
    before_sig = graph_semantic_signature(before_graph)
    after_sig = graph_semantic_signature(after_graph)
    changed = {
        "frame_progressed": after_sig["frame_index"] > before_sig["frame_index"],
        "person_count_changed": before_sig["person_count"] != after_sig["person_count"],
        "object_count_changed": before_sig["object_count"] != after_sig["object_count"],
        "relation_count_changed": before_sig["relation_count"] != after_sig["relation_count"],
        "entity_bbox_changed": before_sig["entity_bbox"] != after_sig["entity_bbox"],
        "state_markers_changed": before_sig["state_markers"] != after_sig["state_markers"],
    }
    return {
        "before_signature": before_sig,
        "after_signature": after_sig,
        "changed_flags": changed,
        "meaningful_change": any(changed.values()),
    }


def text_output_to_contract(text: str, output: TextEncodingOutput) -> TextActionStateContract:
    return {
        "text": text,
        "parsed_actions": [{"type": t, "priority": idx + 1} for idx, t in enumerate(output.structured_action_tokens)],
        "action_embedding": output.action_embedding[:],
        "target_entities": list(output.target_hints.get("entities", [])),
        "target_objects": list(output.target_hints.get("objects", [])),
        "temporal_decomposition": output.decomposition_hints[:],
        "constraints": output.constraints[:],
    }


def _build_graph_after(request: DynamicsTransitionRequest, output: DynamicsTransitionOutput):
    predicted = output.metadata.get("predicted_graph_after")
    if predicted is not None:
        return predicted
    before = copy.deepcopy(request.graph_state)
    return apply_delta(before, output.delta)


def dynamics_io_to_contract(request: DynamicsTransitionRequest, output: DynamicsTransitionOutput) -> GraphTransitionContract:
    from training.learned_contracts import build_graph_transition_contract

    predicted_after = _build_graph_after(request, output)
    ground_truth_after = request.step_context.get("ground_truth_graph_after") if isinstance(request.step_context, dict) else None
    after = ground_truth_after if ground_truth_after is not None else predicted_after
    supervision_mode = "supervision" if ground_truth_after is not None else "inference"
    change_summary = graph_change_summary(to_debug_dict(request.graph_state), to_debug_dict(after))
    return build_graph_transition_contract(
        before=request.graph_state,
        after=after,
        delta=output.delta,
        transition_context={
            "step_context": request.step_context,
            "text_tokens": request.text_action_summary.structured_action_tokens,
            "diagnostics": output.diagnostics,
            "backend": output.metadata.get("backend", "unknown"),
            "predicted_graph_after": to_debug_dict(predicted_after),
            "ground_truth_graph_after": to_debug_dict(ground_truth_after) if ground_truth_after is not None else {},
            "graph_after_target_source": "step_context_ground_truth" if ground_truth_after is not None else "predicted_fallback",
            "supervision_mode": supervision_mode,
            "has_ground_truth_targets": ground_truth_after is not None,
            "graph_change_summary": change_summary,
        },
    )


def patch_io_to_contract(request: PatchSynthesisRequest, output: PatchSynthesisOutput) -> PatchSynthesisContract:
    from training.learned_contracts import build_patch_synthesis_contract

    return build_patch_synthesis_contract(
        roi_before=request.current_frame,
        roi_after=output.rgb_patch,
        region=request.region,
        retrieval_summary=str(request.retrieval_summary),
        selected_strategy=str(output.execution_trace.get("selected_render_strategy", "unknown")),
        hidden_state=request.memory_channels.get("hidden_regions", {}),
        synthesis_mode=str(output.execution_trace.get("synthesis_mode", "deterministic")),
        transition_context=request.transition_context,
    )


def temporal_io_to_contract(request: TemporalRefinementRequest, output: TemporalRefinementOutput) -> TemporalConsistencyContract:
    from training.learned_contracts import build_temporal_consistency_contract

    return build_temporal_consistency_contract(
        previous_frame=request.previous_frame,
        composed_frame=request.current_composed_frame,
        target_frame=output.refined_frame,
        changed_regions=request.changed_regions,
        region_consistency_metadata={"scores": output.region_consistency_scores},
        scene_transition_context={"frame_index": request.scene_state.frame_index},
        memory_transition_context={"channels": list(request.memory_channels.keys())},
    )


def validate_parity(payload: dict[str, object], required_fields: list[str]) -> list[str]:
    missing = [name for name in required_fields if name not in payload]
    return missing


def build_parity_result(
    *,
    contract: dict[str, object],
    required_fields: list[str],
    stage: str,
    request: object,
    output: object,
    changed_regions_count: int | None = None,
) -> StructuredParityResult:
    missing = validate_parity(contract, required_fields)
    semantic = semantic_parity_checks(
        stage=stage,
        contract=contract,
        request=request,
        output=output,
        changed_regions_count=changed_regions_count,
    )
    result = empty_structured_parity()
    result["missing_fields"] = missing
    result["errors"].extend([f"missing_field:{m}" for m in missing])
    for severity in ("errors", "warnings", "traces"):
        result[severity].extend(semantic.get(severity, []))
    return result


def semantic_parity_checks(
    *,
    stage: str,
    contract: dict[str, object],
    request: object,
    output: object,
    changed_regions_count: int | None = None,
) -> ParityResult:
    issues = _empty_parity()
    if stage == "text":
        parsed = contract.get("parsed_actions", []) if isinstance(contract, dict) else []
        embedding = contract.get("action_embedding", []) if isinstance(contract, dict) else []
        if isinstance(parsed, list) and not parsed:
            _push(issues, "errors", "structured_action_tokens_empty")
        if parsed and not embedding:
            _push(issues, "errors", "embedding_empty_for_non_empty_actions")
        if isinstance(parsed, list) and len(parsed) > 1 and not contract.get("temporal_decomposition"):
            _push(issues, "warnings", "decomposition_hints_missing_for_multi_action_input")
        target_refs = [a for a in parsed if isinstance(a, dict) and (a.get("target_entity") or a.get("target_object"))]
        target_entities = contract.get("target_entities", []) if isinstance(contract, dict) else []
        target_objects = contract.get("target_objects", []) if isinstance(contract, dict) else []
        if target_refs and not (target_entities or target_objects):
            _push(issues, "warnings", "target_hints_missing_despite_action_targets")
    if stage == "dynamics":
        delta_contract = contract.get("delta_contract", {}) if isinstance(contract, dict) else {}
        if isinstance(delta_contract, dict):
            affected = list(delta_contract.get("affected_regions", []))
            before_idx = ((contract.get("graph_before") or {}).get("frame_index", -1)) if isinstance(contract.get("graph_before"), dict) else -1
            after_idx = ((contract.get("graph_after") or {}).get("frame_index", -1)) if isinstance(contract.get("graph_after"), dict) else -1
            if affected and after_idx <= before_idx:
                _push(issues, "traces", "graph_after_frame_index_not_progressed")
            region_mode = contract.get("region_transition_mode", {})
            if affected and (not isinstance(region_mode, dict) or not region_mode):
                _push(issues, "warnings", "region_transition_mode_missing_for_affected_regions")
            state_after = contract.get("state_after", {}) if isinstance(contract, dict) else {}
            state_before = contract.get("state_before", {}) if isinstance(contract, dict) else {}
            if affected and (not isinstance(state_after, dict) or not state_after):
                _push(issues, "warnings", "state_after_missing_for_non_empty_delta")
            before_graph = contract.get("graph_before", {}) if isinstance(contract, dict) else {}
            after_graph = contract.get("graph_after", {}) if isinstance(contract, dict) else {}
            has_non_empty_delta = bool(affected or delta_contract.get("pose_deltas") or delta_contract.get("interaction_deltas") or delta_contract.get("visibility_deltas"))
            graph_delta = graph_change_summary(before_graph, after_graph)
            graph_unchanged = not graph_delta["meaningful_change"]
            state_changed = isinstance(state_before, dict) and isinstance(state_after, dict) and state_before != state_after
            if has_non_empty_delta and graph_unchanged and (not state_changed) and (not state_after):
                _push(issues, "errors", "non_empty_delta_without_meaningful_state_change")
            visibility = delta_contract.get("predicted_visibility_changes", {})
            if isinstance(visibility, dict) and visibility and not state_after:
                _push(issues, "warnings", "visibility_delta_without_state_after_transition")
            if isinstance(region_mode, dict):
                missing_modes = [r for r in affected if r not in region_mode]
                if missing_modes:
                    _push(issues, "warnings", "region_transition_mode_missing_specific_regions")
    if stage == "patch":
        selected = str(contract.get("selected_strategy", "")) if isinstance(contract, dict) else ""
        patch_obj = getattr(output, "rgb_patch", None)
        if patch_obj and selected.strip() in {"", "unknown"}:
            _push(issues, "warnings", "selected_strategy_missing_for_non_empty_patch")
        if isinstance(request, PatchSynthesisRequest) and request.memory_channels.get("identity") and not request.identity_embedding:
            _push(issues, "warnings", "identity_channel_requested_but_embedding_empty")
    if stage == "temporal":
        changed = contract.get("changed_regions", []) if isinstance(contract, dict) else []
        if (changed_regions_count or 0) > 0 and not changed:
            _push(issues, "warnings", "changed_regions_missing_despite_patch_updates")
    return issues


def to_debug_dict(obj: object) -> dict[str, object]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)  # type: ignore[arg-type]
    if isinstance(obj, dict):
        return obj
    return {"repr": repr(obj)}
