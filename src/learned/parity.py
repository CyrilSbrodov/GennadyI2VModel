from __future__ import annotations

from dataclasses import asdict

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


def dynamics_io_to_contract(request: DynamicsTransitionRequest, output: DynamicsTransitionOutput) -> GraphTransitionContract:
    from training.learned_contracts import build_graph_transition_contract

    after = request.graph_state
    return build_graph_transition_contract(
        before=request.graph_state,
        after=after,
        delta=output.delta,
        transition_context={
            "step_context": request.step_context,
            "text_tokens": request.text_action_summary.structured_action_tokens,
            "diagnostics": output.diagnostics,
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


def to_debug_dict(obj: object) -> dict[str, object]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)  # type: ignore[arg-type]
    if isinstance(obj, dict):
        return obj
    return {"repr": repr(obj)}
