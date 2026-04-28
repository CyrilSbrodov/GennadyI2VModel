from __future__ import annotations

from typing import Literal


DecisionKind = Literal[
    "direct_reuse",
    "temporal_stabilize",
    "local_deform_or_update",
    "reveal_from_memory",
    "reveal_partial_memory_assist",
    "reveal_requires_synthesis",
    "garment_transition_update",
    "expression_refine",
    "pose_exposure_update",
    "fallback_unknown",
]

RUNTIME_ROUTING_DECISION_KINDS: tuple[DecisionKind, ...] = (
    "direct_reuse",
    "temporal_stabilize",
    "local_deform_or_update",
    "reveal_from_memory",
    "reveal_partial_memory_assist",
    "reveal_requires_synthesis",
    "garment_transition_update",
    "expression_refine",
    "pose_exposure_update",
)

ExecutionStrategy = Literal[
    "EXISTING_REGION_UPDATE",
    "KNOWN_HIDDEN_REVEAL",
    "PARTIAL_MEMORY_ASSIST_REVEAL",
    "UNKNOWN_HIDDEN_SYNTHESIS",
    "NEW_ENTITY_INSERTION",
]

RoutingInputStatus = Literal["authoritative_runtime_plan", "partial_runtime_plan", "no_runtime_plan"]
