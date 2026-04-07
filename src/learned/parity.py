from __future__ import annotations

import copy
from dataclasses import asdict

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


def semantic_parity_checks(
    *,
    stage: str,
    contract: dict[str, object],
    request: object,
    output: object,
    changed_regions_count: int | None = None,
) -> list[str]:
    def _stable_serialized(value: object) -> str:
        import json

        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)

    issues: list[str] = []
    if stage == "text":
        parsed = contract.get("parsed_actions", []) if isinstance(contract, dict) else []
        embedding = contract.get("action_embedding", []) if isinstance(contract, dict) else []
        if isinstance(parsed, list) and not parsed:
            issues.append("structured_action_tokens_empty")
        if parsed and not embedding:
            issues.append("embedding_empty_for_non_empty_actions")
        if isinstance(parsed, list) and len(parsed) > 1 and not contract.get("temporal_decomposition"):
            issues.append("decomposition_hints_missing_for_multi_action_input")
        target_refs = [a for a in parsed if isinstance(a, dict) and (a.get("target_entity") or a.get("target_object"))]
        target_entities = contract.get("target_entities", []) if isinstance(contract, dict) else []
        target_objects = contract.get("target_objects", []) if isinstance(contract, dict) else []
        if target_refs and not (target_entities or target_objects):
            issues.append("target_hints_missing_despite_action_targets")
    if stage == "dynamics":
        delta_contract = contract.get("delta_contract", {}) if isinstance(contract, dict) else {}
        if isinstance(delta_contract, dict):
            affected = list(delta_contract.get("affected_regions", []))
            before_idx = ((contract.get("graph_before") or {}).get("frame_index", -1)) if isinstance(contract.get("graph_before"), dict) else -1
            after_idx = ((contract.get("graph_after") or {}).get("frame_index", -1)) if isinstance(contract.get("graph_after"), dict) else -1
            if affected and after_idx <= before_idx:
                issues.append("graph_after_frame_index_not_progressed_trace")
            region_mode = contract.get("region_transition_mode", {})
            if affected and (not isinstance(region_mode, dict) or not region_mode):
                issues.append("region_transition_mode_missing_for_affected_regions")
            state_after = contract.get("state_after", {}) if isinstance(contract, dict) else {}
            state_before = contract.get("state_before", {}) if isinstance(contract, dict) else {}
            if affected and (not isinstance(state_after, dict) or not state_after):
                issues.append("state_after_missing_for_non_empty_delta")
            before_graph = contract.get("graph_before", {}) if isinstance(contract, dict) else {}
            after_graph = contract.get("graph_after", {}) if isinstance(contract, dict) else {}
            has_non_empty_delta = bool(affected or delta_contract.get("pose_deltas") or delta_contract.get("interaction_deltas") or delta_contract.get("visibility_deltas"))
            graph_unchanged = _stable_serialized(before_graph) == _stable_serialized(after_graph)
            state_changed = isinstance(state_before, dict) and isinstance(state_after, dict) and state_before != state_after
            if has_non_empty_delta and graph_unchanged and (not state_changed) and (not state_after):
                issues.append("non_empty_delta_without_meaningful_state_change")
            visibility = delta_contract.get("predicted_visibility_changes", {})
            if isinstance(visibility, dict) and visibility and not state_after:
                issues.append("visibility_delta_without_state_after_transition")
            if isinstance(region_mode, dict):
                missing_modes = [r for r in affected if r not in region_mode]
                if missing_modes:
                    issues.append("region_transition_mode_missing_specific_regions")
    if stage == "patch":
        selected = str(contract.get("selected_strategy", "")) if isinstance(contract, dict) else ""
        patch_obj = getattr(output, "rgb_patch", None)
        if patch_obj and selected.strip() in {"", "unknown"}:
            issues.append("selected_strategy_missing_for_non_empty_patch")
        if isinstance(request, PatchSynthesisRequest) and request.memory_channels.get("identity") and not request.identity_embedding:
            issues.append("identity_channel_requested_but_embedding_empty")
    if stage == "temporal":
        changed = contract.get("changed_regions", []) if isinstance(contract, dict) else []
        if (changed_regions_count or 0) > 0 and not changed:
            issues.append("changed_regions_missing_despite_patch_updates")
    return issues


def to_debug_dict(obj: object) -> dict[str, object]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)  # type: ignore[arg-type]
    if isinstance(obj, dict):
        return obj
    return {"repr": repr(obj)}
