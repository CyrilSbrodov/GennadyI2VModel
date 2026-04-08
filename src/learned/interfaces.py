from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.schema import ActionPlan, GraphDelta, RegionRef, SceneGraph, VideoMemory
from text.encoder_contracts import TextEncodingOutput


@dataclass(slots=True)
class GraphEncodingOutput:
    graph_embedding: list[float]
    serialized_graph: dict[str, object]
    confidence: float = 0.0


@dataclass(slots=True)
class DynamicsTransitionRequest:
    graph_state: SceneGraph
    memory_summary: dict[str, object]
    text_action_summary: TextEncodingOutput
    memory_channels: dict[str, object] = field(default_factory=dict)
    graph_encoding: GraphEncodingOutput | None = None
    identity_embeddings: dict[str, list[float]] = field(default_factory=dict)
    step_context: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class DynamicsTransitionOutput:
    delta: GraphDelta
    confidence: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PatchSynthesisRequest:
    region: RegionRef
    scene_state: SceneGraph
    memory_summary: dict[str, object]
    transition_context: dict[str, object]
    retrieval_summary: dict[str, object]
    current_frame: list
    memory_channels: dict[str, object] = field(default_factory=dict)
    graph_encoding: GraphEncodingOutput | None = None
    identity_embedding: list[float] = field(default_factory=list)


@dataclass(slots=True)
class PatchSynthesisOutput:
    region: RegionRef
    rgb_patch: list[list[list[float]]]
    alpha_mask: list[list[float]]
    height: int
    width: int
    channels: int
    confidence: float
    z_index: int = 1
    debug_trace: list[str] = field(default_factory=list)
    execution_trace: dict[str, object] = field(default_factory=dict)
    uncertainty_map: list[list[float]] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TemporalRefinementRequest:
    previous_frame: list
    current_composed_frame: list
    changed_regions: list[RegionRef]
    scene_state: SceneGraph
    memory_state: VideoMemory
    memory_channels: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TemporalRefinementOutput:
    refined_frame: list
    region_consistency_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


class TextEncoder(Protocol):
    def encode(self, text: str, scene_graph: SceneGraph | None = None, action_plan: ActionPlan | None = None) -> TextEncodingOutput:
        ...


class GraphEncoder(Protocol):
    def encode(self, scene_graph: SceneGraph) -> GraphEncodingOutput:
        ...


class IdentityAppearanceEncoder(Protocol):
    def encode_identity(self, memory_summary: dict[str, object], entity_id: str) -> list[float]:
        ...


class DynamicsTransitionModel(Protocol):
    def predict_transition(self, request: DynamicsTransitionRequest) -> DynamicsTransitionOutput:
        ...


class PatchSynthesisModel(Protocol):
    def synthesize_patch(self, request: PatchSynthesisRequest) -> PatchSynthesisOutput:
        ...


class TemporalConsistencyModel(Protocol):
    def refine_temporal(self, request: TemporalRefinementRequest) -> TemporalRefinementOutput:
        ...
