from __future__ import annotations

from enum import Enum
from typing import Iterable, Mapping

from core.routing_contracts import RUNTIME_ROUTING_DECISION_KINDS


class ContractValidationError(RuntimeError):
    """Raised when an architecture contract is violated."""


class PipelineStage(str, Enum):
    INPUT = "input"
    PERCEPTION = "perception"
    SCENE_GRAPH = "scene_graph"
    MEMORY = "memory"
    INTENT = "intent"
    PLANNING = "planning"
    DYNAMICS = "dynamics"
    REGION_ROUTING = "region_routing"
    RENDERING = "rendering"
    COMPOSITING = "compositing"
    TEMPORAL_REFINEMENT = "temporal_refinement"
    OUTPUT = "output"


CANONICAL_PIPELINE_ORDER: tuple[PipelineStage, ...] = (
    PipelineStage.INPUT,
    PipelineStage.PERCEPTION,
    PipelineStage.SCENE_GRAPH,
    PipelineStage.MEMORY,
    PipelineStage.INTENT,
    PipelineStage.PLANNING,
    PipelineStage.DYNAMICS,
    PipelineStage.REGION_ROUTING,
    PipelineStage.RENDERING,
    PipelineStage.COMPOSITING,
    PipelineStage.TEMPORAL_REFINEMENT,
    PipelineStage.OUTPUT,
)

CANONICAL_STAGE_NAMES: tuple[str, ...] = tuple(stage.value for stage in CANONICAL_PIPELINE_ORDER)
MANDATORY_PIPELINE_STAGE_NAMES: tuple[str, ...] = CANONICAL_STAGE_NAMES
_ALLOWED_TRANSITIONS: set[tuple[str, str]] = set(zip(CANONICAL_STAGE_NAMES, CANONICAL_STAGE_NAMES[1:]))
_MATERIAL_PROVENANCE = {"observed", "memory_assisted", "generated", "inferred", "fallback", "unknown"}


def normalize_stage_name(stage: str | PipelineStage) -> str:
    if isinstance(stage, PipelineStage):
        return stage.value
    if not isinstance(stage, str) or not stage.strip():
        raise ContractValidationError(f"Pipeline stage must be a non-empty canonical string, got {stage!r}")
    return stage.strip()


def validate_stage_name(stage: str | PipelineStage) -> str:
    name = normalize_stage_name(stage)
    if name not in CANONICAL_STAGE_NAMES:
        raise ContractValidationError(f"Unknown pipeline stage '{name}'. Canonical stages: {list(CANONICAL_STAGE_NAMES)}")
    return name


def validate_transition(stage_from: str | PipelineStage, stage_to: str | PipelineStage) -> tuple[str, str]:
    src = validate_stage_name(stage_from)
    dst = validate_stage_name(stage_to)
    if (src, dst) not in _ALLOWED_TRANSITIONS:
        raise ContractValidationError(f"Invalid pipeline transition '{src}' -> '{dst}'")
    return src, dst


def _trace_stage_name(item: object) -> str:
    if isinstance(item, Mapping):
        raw = item.get("stage")
    else:
        raw = item
    return validate_stage_name(raw)  # type: ignore[arg-type]


def validate_runtime_trace(trace: Iterable[object], *, mandatory_stages: Iterable[str | PipelineStage] = MANDATORY_PIPELINE_STAGE_NAMES) -> list[str]:
    names = [_trace_stage_name(item) for item in trace]
    required = [validate_stage_name(stage) for stage in mandatory_stages]
    if not names:
        raise ContractValidationError("Runtime trace must not be empty")
    positions = {name: idx for idx, name in enumerate(CANONICAL_STAGE_NAMES)}
    previous = -1
    seen: set[str] = set()
    for name in names:
        pos = positions[name]
        if pos < previous:
            raise ContractValidationError(f"Runtime trace stage '{name}' appears out of canonical order")
        previous = pos
        seen.add(name)
    missing = [stage for stage in required if stage not in seen]
    if missing:
        raise ContractValidationError(f"Runtime trace skipped mandatory stage(s): {missing}")
    return names


def validate_rendering_context(*, region_id: str, transition_context: Mapping[str, object] | None, region_metadata: Mapping[str, object] | None = None) -> dict[str, object]:
    """Validate that ROI rendering is backed by an explicit region-routing decision."""
    if not isinstance(region_id, str) or ":" not in region_id:
        raise ContractValidationError(f"Rendering requires a canonical entity:region id, got {region_id!r}")
    if not isinstance(transition_context, Mapping):
        raise ContractValidationError(f"Rendering for {region_id} requires transition_context with region routing")
    route = transition_context.get("region_route_decision")
    if not isinstance(route, Mapping):
        raise ContractValidationError(f"Rendering for {region_id} requires region_route_decision")
    route_region_id = str(route.get("region_id", "") or "")
    if route_region_id != region_id:
        raise ContractValidationError(f"Rendering route region mismatch for {region_id}: {route_region_id!r}")
    canonical_region_id = str(route.get("canonical_region_id", "") or "")
    if canonical_region_id != region_id:
        raise ContractValidationError(f"Rendering route canonical_region_id mismatch for {region_id}: {canonical_region_id!r}")
    decision = str(route.get("decision", "") or "")
    if decision not in RUNTIME_ROUTING_DECISION_KINDS:
        raise ContractValidationError(f"Rendering for {region_id} requires an authoritative routing decision, got {decision!r}")
    render_mode = str(route.get("render_mode", route.get("renderer_mode_hint", "")) or "")
    if not render_mode or render_mode in {"unknown", "none"}:
        raise ContractValidationError(f"Rendering for {region_id} requires an explicit render mode")
    source_provenance = str(route.get("source_provenance", "") or "")
    if not source_provenance or source_provenance == "unknown":
        raise ContractValidationError(f"Rendering for {region_id} requires source provenance")
    material_provenance = str(route.get("material_provenance", "") or "")
    if material_provenance not in _MATERIAL_PROVENANCE:
        raise ContractValidationError(f"Rendering for {region_id} has invalid material_provenance={material_provenance!r}")
    if isinstance(region_metadata, Mapping):
        meta_region = str(region_metadata.get("region_id", region_id) or region_id)
        if meta_region != region_id:
            raise ContractValidationError(f"Rendering metadata region mismatch for {region_id}: {meta_region!r}")
    return dict(route)
