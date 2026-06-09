from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from core.body_ontology import BODY_ONTOLOGY, BodyRegionGroup, is_canonical_body_region
from core.region_ids import is_known_canonical_region_type, make_region_id
from core.schema import CanonicalRegionMemoryEntry, SceneGraph, VideoMemory
from dynamics.graph_delta_contract import (
    ALLOWED_GARMENT_REGIONS,
    ALLOWED_OBJECT_REGIONS,
    FUTURE_FACE_SUBREGIONS,
    GraphDeltaContract,
    RegionDelta,
    RegionDeltaType,
    validate_graph_delta_contract,
)
from memory.memory_policy import (
    MemoryAuthority,
    MemoryFamily,
    MemoryMaterialProvenance,
    assess_memory_candidate,
    classify_memory_family,
    memory_reference_kind,
)


class RevealValidationError(ValueError):
    """Raised when Sprint-6 reveal / occlusion continuity invariants are violated."""

    def __init__(self, code: str, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details or {}


class RevealDecisionType(str, Enum):
    PRESERVE_VISIBLE = "preserve_visible"
    NEWLY_OCCLUDED = "newly_occluded"
    OCCLUSION_REASONING_REQUIRED = "occlusion_reasoning_required"
    REVEAL_FROM_OBSERVED_MEMORY = "reveal_from_observed_memory"
    REVEAL_FROM_WEAK_MEMORY = "reveal_from_weak_memory"
    REVEAL_UNKNOWN_DEFER = "reveal_unknown_defer"
    REVEAL_BLOCKED_PRIVATE = "reveal_blocked_private"
    REVEAL_BLOCKED_NO_EVIDENCE = "reveal_blocked_no_evidence"
    REVEAL_BLOCKED_IDENTITY_RISK = "reveal_blocked_identity_risk"
    REVEAL_BLOCKED_UNSUPPORTED_REGION = "reveal_blocked_unsupported_region"


class OcclusionLifecycleState(str, Enum):
    VISIBLE_STABLE = "visible_stable"
    NEWLY_OCCLUDED = "newly_occluded"
    OCCLUDED_KNOWN = "occluded_known"
    OCCLUDED_UNKNOWN = "occluded_unknown"
    NEWLY_REVEALED_KNOWN = "newly_revealed_known"
    NEWLY_REVEALED_WEAK = "newly_revealed_weak"
    NEWLY_REVEALED_UNKNOWN = "newly_revealed_unknown"
    REVEAL_BLOCKED = "reveal_blocked"
    PRIVATE_BLOCKED = "private_blocked"
    IDENTITY_RISK_BLOCKED = "identity_risk_blocked"


IDENTITY_REVEAL_REGIONS: frozenset[str] = frozenset({"face", "head", "hair", "scalp"}) | FUTURE_FACE_SUBREGIONS
_SAFE_OBSERVED_MATERIALS: frozenset[str] = frozenset(
    {
        MemoryMaterialProvenance.OBSERVED_INPUT.value,
        MemoryMaterialProvenance.OBSERVED_PARSER.value,
        MemoryMaterialProvenance.OBSERVED_DETECTOR.value,
        MemoryMaterialProvenance.OBSERVED_FACE.value,
    }
)
_UNSAFE_OBSERVED_CLAIM_MATERIALS: frozenset[str] = frozenset(
    {
        MemoryMaterialProvenance.GENERATED.value,
        MemoryMaterialProvenance.INFERRED.value,
        MemoryMaterialProvenance.FALLBACK.value,
        MemoryMaterialProvenance.SYNTHETIC.value,
        MemoryMaterialProvenance.HIDDEN.value,
        MemoryMaterialProvenance.UNKNOWN.value,
        MemoryMaterialProvenance.UNSUPPORTED.value,
    }
)
_ALLOWED_DECISION_VALUES = {item.value for item in RevealDecisionType}
_ALLOWED_LIFECYCLE_VALUES = {item.value for item in OcclusionLifecycleState}
_ALLOWED_NON_BODY_REGIONS = ALLOWED_GARMENT_REGIONS | ALLOWED_OBJECT_REGIONS
_ALLOWED_ROUTING_DECISION_TYPES = {
    RevealDecisionType.PRESERVE_VISIBLE.value,
    RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value,
    RevealDecisionType.REVEAL_FROM_WEAK_MEMORY.value,
}
_EXPLICIT_NEWLY_OCCLUDED_MARKERS = frozenset({"newly_occluded", "became_occluded", "becoming_occluded", "newly_hidden", "became_hidden"})


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _value(value: object) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _known_region(region: str) -> bool:
    return region in _ALLOWED_NON_BODY_REGIONS or is_canonical_body_region(region) or is_known_canonical_region_type(region)


def _is_private_or_optional(region: str) -> bool:
    meta = BODY_ONTOLOGY.get(str(region or ""))
    if meta is None:
        return False
    return meta.memory_family == "private" or meta.group in {BodyRegionGroup.OPTIONAL_PRIVATE, BodyRegionGroup.OPTIONAL_SEX_SPECIFIC}


def _is_identity_region(region: str) -> bool:
    return region in IDENTITY_REVEAL_REGIONS or classify_memory_family(region) == MemoryFamily.IDENTITY


@dataclass(frozen=True, slots=True)
class RevealMemoryEvidence:
    memory_family: str = MemoryFamily.UNKNOWN.value
    reference_kind: str = "none"
    authority: str = MemoryAuthority.REJECTED.value
    material_provenance: str = MemoryMaterialProvenance.UNKNOWN.value
    memory_support_level: str = "none"
    evidence_score: float = 0.0
    confidence: float = 0.0
    observed_directly: bool = False
    generated: bool = False
    inferred: bool = False
    policy_reasons: tuple[str, ...] = ()
    record_id: str | None = None
    memory_kind: str | None = None
    mask_evidence_type: str = "missing"
    supports_observed_reveal: bool = False
    supports_weak_reveal: bool = False


@dataclass(frozen=True, slots=True)
class RevealCandidate:
    entity_id: str
    canonical_region_id: str
    region_id: str
    source_delta_type: str
    action_type: str
    phase_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class RevealRoutingCandidate:
    region_id: str
    canonical_region_id: str
    reveal_decision_type: str
    reason: str
    source_delta_type: str
    action_type: str
    phase_id: str
    requires_rendering_candidate: bool
    identity_locked: bool
    reveal_allowed: bool
    memory_reference_kind: str
    memory_authority: str
    routing_decision_made: bool = False
    render_strategy_selected: bool = False


@dataclass(frozen=True, slots=True)
class RevealDecision:
    entity_id: str
    canonical_region_id: str
    region_id: str
    decision_type: RevealDecisionType | str
    lifecycle_state: OcclusionLifecycleState | str
    reason: str
    source_delta_type: str
    action_type: str
    phase_id: str
    action_order: int = 0
    phase_order: int = 0
    identity_locked: bool = False
    allowed_to_modify_identity: bool = False
    reveal_allowed: bool = False
    requires_rendering_candidate: bool = False
    private_or_optional_region: bool = False
    memory_evidence: RevealMemoryEvidence = field(default_factory=RevealMemoryEvidence)
    policy_reasons: tuple[str, ...] = ()
    rendered_pixels_generated: bool = False
    observed_evidence_created: bool = False
    memory_write_performed: bool = False
    region_routing_called: bool = False
    scene_graph_mutation_performed: bool = False
    mask_created: bool = False
    routing_decision_made: bool = False
    render_strategy_selected: bool = False

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RevealTrace:
    reveal_source: str = "sprint6_reveal_occlusion_contract"
    provenance: str = "dynamics_to_reveal_policy_handoff"
    support_level: str = "supported"
    reasons: tuple[str, ...] = ()
    unsupported_dynamics_code: str | None = None
    unsupported_dynamics_reasons: tuple[str, ...] = ()
    unsupported_planner_fragments: tuple[str, ...] = ()
    planner_trace_reasons: tuple[str, ...] = ()
    dynamics_trace_reasons: tuple[str, ...] = ()
    forbidden_operations_asserted_absent: tuple[str, ...] = (
        "rendering",
        "renderer_call",
        "region_routing_call",
        "region_routing_decision",
        "scene_graph_mutation",
        "memory_write",
        "mask_creation",
        "observed_perception_evidence_creation",
        "pixel_generation",
        "hidden_anatomy_generation",
        "clothing_removal",
        "physics_simulation",
    )


@dataclass(frozen=True, slots=True)
class RevealContract:
    contract_version: str = "reveal_occlusion_contract_v1"
    supported: bool = True
    decisions: tuple[RevealDecision, ...] = ()
    reveal_candidates: tuple[RevealCandidate, ...] = ()
    routing_candidates: tuple[RevealRoutingCandidate, ...] = ()
    trace: RevealTrace = field(default_factory=RevealTrace)
    rendered_pixels_generated: bool = False
    observed_evidence_created: bool = False
    memory_write_performed: bool = False
    region_routing_called: bool = False
    scene_graph_mutation_performed: bool = False
    mask_creation_performed: bool = False
    identity_memory_created: bool = False
    identity_embedding_created: bool = False

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RevealHandoffResult:
    supported: bool
    reveal_contract: RevealContract
    routing_candidates: tuple[RevealRoutingCandidate, ...]
    trace: RevealTrace

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def _memory_entries_for_region(memory: VideoMemory | Mapping[str, object] | None, entity_id: str, canonical_region: str) -> list[CanonicalRegionMemoryEntry | Mapping[str, object]]:
    if memory is None:
        return []
    store: Mapping[str, object]
    if isinstance(memory, Mapping):
        raw = memory.get("canonical_region_memory", memory)
        store = raw if isinstance(raw, Mapping) else {}
    else:
        store = getattr(memory, "canonical_region_memory", {}) or {}
    keys = (make_region_id(entity_id, canonical_region), f"{entity_id}:{canonical_region}", canonical_region)
    entries = [store[key] for key in keys if key in store]
    for value in store.values():
        entry_entity = _entry_get(value, "entity_id", "")
        entry_region = _entry_get(value, "canonical_region", _entry_get(value, "canonical_region_id", ""))
        if entry_entity == entity_id and entry_region == canonical_region and value not in entries:
            entries.append(value)
    return entries


def _entry_get(entry: object, name: str, default: Any = None) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _evidence_from_entry(entry: CanonicalRegionMemoryEntry | Mapping[str, object] | None, canonical_region: str) -> RevealMemoryEvidence:
    if entry is None:
        family = classify_memory_family(canonical_region)
        return RevealMemoryEvidence(memory_family=family.value, reference_kind=memory_reference_kind(family), policy_reasons=("memory_missing",))
    assessment = assess_memory_candidate(
        canonical_region=canonical_region,
        confidence=float(_entry_get(entry, "confidence", 0.0) or 0.0),
        evidence_score=float(_entry_get(entry, "evidence_score", 0.0) or 0.0),
        observed_directly=bool(_entry_get(entry, "observed_directly", False)),
        generated=bool(_entry_get(entry, "generated", False)),
        inferred=bool(_entry_get(entry, "inferred", False)),
        provenance=str(_entry_get(entry, "provenance", "unknown") or "unknown"),
        visibility_state=str(_entry_get(entry, "visibility_state", "unknown_expected_region") or "unknown_expected_region"),
        mask_ref=_entry_get(entry, "mask_ref", None),
        applicability=str(_entry_get(entry, "applicability", "applicable") or "applicable"),
        observation_status=str(_entry_get(entry, "observation_status", "unknown") or "unknown"),
        mask_evidence_type=str(_entry_get(entry, "mask_evidence_type", "missing") or "missing"),
        parser_support_level=str(_entry_get(entry, "parser_support_level", "unknown") or "unknown"),
        reveal_lifecycle=str(_entry_get(entry, "reveal_lifecycle", "unknown") or "unknown"),
        source_frame_kind=str(_entry_get(entry, "source_frame_kind", "unknown") or "unknown"),
    )
    authority = assessment.authority.value
    material = assessment.material_provenance.value
    support = "none"
    supports_observed = False
    supports_weak = False
    if assessment.can_seed_reveal and authority in {MemoryAuthority.AUTHORITATIVE.value, MemoryAuthority.REUSABLE.value}:
        support = "strong" if authority == MemoryAuthority.AUTHORITATIVE.value else "medium"
        supports_observed = material in _SAFE_OBSERVED_MATERIALS and bool(_entry_get(entry, "observed_directly", False)) and not bool(_entry_get(entry, "generated", False)) and not bool(_entry_get(entry, "inferred", False))
    elif authority == MemoryAuthority.WEAK or assessment.can_seed_appearance:
        support = "weak"
        supports_weak = (
            authority == MemoryAuthority.WEAK.value
            and assessment.reference_kind != "none"
            and material in _SAFE_OBSERVED_MATERIALS
            and bool(_entry_get(entry, "observed_directly", False))
            and not bool(_entry_get(entry, "generated", False))
            and not bool(_entry_get(entry, "inferred", False))
        )
    safe_observed_flag = bool(_entry_get(entry, "observed_directly", False)) and material not in _UNSAFE_OBSERVED_CLAIM_MATERIALS
    return RevealMemoryEvidence(
        memory_family=assessment.memory_family.value,
        reference_kind=assessment.reference_kind,
        authority=authority,
        material_provenance=material,
        memory_support_level=support,
        evidence_score=float(_entry_get(entry, "evidence_score", 0.0) or 0.0),
        confidence=float(_entry_get(entry, "confidence", 0.0) or 0.0),
        observed_directly=safe_observed_flag,
        generated=bool(_entry_get(entry, "generated", False)),
        inferred=bool(_entry_get(entry, "inferred", False)),
        policy_reasons=tuple(assessment.policy_reasons),
        record_id=_entry_get(entry, "record_id", None),
        memory_kind=_entry_get(entry, "memory_kind", None),
        mask_evidence_type=str(_entry_get(entry, "mask_evidence_type", "missing") or "missing"),
        supports_observed_reveal=supports_observed,
        supports_weak_reveal=supports_weak,
    )


def _best_memory_evidence(memory: VideoMemory | Mapping[str, object] | None, entity_id: str, canonical_region: str) -> RevealMemoryEvidence:
    evidences = [_evidence_from_entry(entry, canonical_region) for entry in _memory_entries_for_region(memory, entity_id, canonical_region)]
    if not evidences:
        return _evidence_from_entry(None, canonical_region)
    rank = {"strong": 3, "medium": 2, "weak": 1, "none": 0}
    return max(evidences, key=lambda e: (rank.get(e.memory_support_level, 0), e.evidence_score, e.confidence))


def _delta_has_explicit_newly_occluded_signal(delta: RegionDelta) -> bool:
    values: list[object] = [
        delta.expected_motion_role,
        delta.phase_type,
        *delta.validation_reasons,
        *delta.source_planner_trace,
    ]
    if delta.visibility_intent is not None:
        values.append(delta.visibility_intent.expected_change)
    if delta.occlusion_intent is not None:
        values.append(delta.occlusion_intent.expected_change)
    text = " ".join(str(value or "").strip().lower() for value in values)
    return any(marker in text for marker in _EXPLICIT_NEWLY_OCCLUDED_MARKERS)


def _decision_kind_for_delta(delta: RegionDelta, evidence: RevealMemoryEvidence) -> tuple[str, str, bool, bool, tuple[str, ...]]:
    region = delta.canonical_region_id
    delta_type = _value(delta.delta_type)
    reasons = list(evidence.policy_reasons)
    identity_region = _is_identity_region(region) or bool(delta.identity_locked)
    if _is_private_or_optional(region) or delta.private_or_optional_region:
        return (RevealDecisionType.REVEAL_BLOCKED_PRIVATE.value, OcclusionLifecycleState.PRIVATE_BLOCKED.value, False, False, tuple(reasons + ["private_optional_no_reveal_no_render"]))
    if not _known_region(region):
        return (RevealDecisionType.REVEAL_BLOCKED_UNSUPPORTED_REGION.value, OcclusionLifecycleState.REVEAL_BLOCKED.value, False, False, tuple(reasons + ["unknown_region_no_reveal_candidate"]))
    if _delta_has_explicit_newly_occluded_signal(delta):
        return (RevealDecisionType.NEWLY_OCCLUDED.value, OcclusionLifecycleState.NEWLY_OCCLUDED.value, False, False, tuple(reasons + ["explicit_newly_occluded_signal"]))
    if delta_type == RegionDeltaType.OCCLUSION_DELTA.value or delta.occlusion_reasoning_required:
        return (RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value, OcclusionLifecycleState.OCCLUDED_UNKNOWN.value, False, False, tuple(reasons + ["dynamics_occlusion_reasoning_required_no_visibility_state_claim"]))
    if not delta.reveal_may_be_required and delta_type not in {RegionDeltaType.VISIBILITY_DELTA.value, RegionDeltaType.GARMENT_INTENT_DELTA.value}:
        return (RevealDecisionType.PRESERVE_VISIBLE.value, OcclusionLifecycleState.VISIBLE_STABLE.value, True, bool(delta.requires_rendering_candidate), tuple(reasons + ["no_reveal_visibility_change_claimed"]))
    if evidence.supports_observed_reveal:
        if identity_region and evidence.authority != MemoryAuthority.AUTHORITATIVE.value:
            return (RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK.value, OcclusionLifecycleState.IDENTITY_RISK_BLOCKED.value, False, False, tuple(reasons + ["identity_reveal_requires_authoritative_identity_memory"]))
        return (RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value, OcclusionLifecycleState.NEWLY_REVEALED_KNOWN.value, True, True, tuple(reasons + ["safe_observed_memory_supports_reveal"]))
    if identity_region:
        return (RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK.value, OcclusionLifecycleState.IDENTITY_RISK_BLOCKED.value, False, False, tuple(reasons + ["identity_memory_missing_or_not_authoritative"]))
    if evidence.memory_support_level == "weak" or evidence.supports_weak_reveal:
        return (RevealDecisionType.REVEAL_FROM_WEAK_MEMORY.value, OcclusionLifecycleState.NEWLY_REVEALED_WEAK.value, True, True, tuple(reasons + ["weak_memory_requires_later_routing_validation"]))
    return (RevealDecisionType.REVEAL_UNKNOWN_DEFER.value, OcclusionLifecycleState.NEWLY_REVEALED_UNKNOWN.value, False, False, tuple(reasons + ["no_safe_memory_defer_unknown_region"]))


def _decision_from_delta(delta: RegionDelta, *, action_order: int, phase_order: int, memory: VideoMemory | Mapping[str, object] | None) -> RevealDecision:
    evidence = _best_memory_evidence(memory, delta.entity_id, delta.canonical_region_id)
    kind, lifecycle, reveal_allowed, requires_render, reasons = _decision_kind_for_delta(delta, evidence)
    identity_locked = bool(delta.identity_locked or _is_identity_region(delta.canonical_region_id))
    return RevealDecision(
        entity_id=delta.entity_id,
        canonical_region_id=delta.canonical_region_id,
        region_id=delta.region_id,
        decision_type=kind,
        lifecycle_state=lifecycle,
        reason=reasons[-1] if reasons else "reveal_policy_applied",
        source_delta_type=_value(delta.delta_type),
        action_type=_value(delta.action_type),
        phase_id=delta.phase_id,
        action_order=action_order,
        phase_order=phase_order,
        identity_locked=identity_locked,
        allowed_to_modify_identity=False if identity_locked else bool(delta.allowed_to_modify_identity),
        reveal_allowed=reveal_allowed,
        requires_rendering_candidate=requires_render,
        private_or_optional_region=bool(delta.private_or_optional_region or _is_private_or_optional(delta.canonical_region_id)),
        memory_evidence=evidence,
        policy_reasons=tuple(reasons),
    )


def _routing_candidate_from_decision(decision: RevealDecision) -> RevealRoutingCandidate | None:
    decision_type = _value(decision.decision_type)
    if decision_type not in _ALLOWED_ROUTING_DECISION_TYPES:
        return None
    if not decision.reveal_allowed or not decision.requires_rendering_candidate:
        return None
    return RevealRoutingCandidate(
        region_id=decision.region_id,
        canonical_region_id=decision.canonical_region_id,
        reveal_decision_type=decision_type,
        reason=decision.reason,
        source_delta_type=decision.source_delta_type,
        action_type=decision.action_type,
        phase_id=decision.phase_id,
        requires_rendering_candidate=decision.requires_rendering_candidate,
        identity_locked=decision.identity_locked,
        reveal_allowed=decision.reveal_allowed,
        memory_reference_kind=decision.memory_evidence.reference_kind,
        memory_authority=decision.memory_evidence.authority,
    )


def build_reveal_handoff(
    *,
    scene_graph: SceneGraph,
    memory: VideoMemory | Mapping[str, object] | None,
    graph_delta_contract: GraphDeltaContract,
    current_frame_index: int | None = None,
    previous_reveal_state: Mapping[str, object] | None = None,
    strict: bool = True,
) -> RevealHandoffResult:
    """Build the reveal / occlusion policy handoff without rendering, routing, masks, or memory writes."""
    del scene_graph, current_frame_index, previous_reveal_state
    if not isinstance(graph_delta_contract, GraphDeltaContract):
        raise RevealValidationError("invalid_graph_delta_contract", "reveal requires a Sprint-5 GraphDeltaContract")
    if not graph_delta_contract.supported:
        trace = RevealTrace(
            support_level="unsupported_dynamics_diagnostic_only",
            reasons=("unsupported_dynamics_contract_no_normal_reveal_candidates",),
            unsupported_dynamics_code=graph_delta_contract.trace.unsupported_planner_code,
            unsupported_dynamics_reasons=tuple(graph_delta_contract.trace.unsupported_planner_reasons),
            unsupported_planner_fragments=tuple(graph_delta_contract.trace.unsupported_planner_fragments),
            planner_trace_reasons=tuple(graph_delta_contract.trace.planner_trace_reasons),
            dynamics_trace_reasons=tuple(graph_delta_contract.trace.reasons),
        )
        contract = RevealContract(supported=False, trace=trace)
        return RevealHandoffResult(supported=False, reveal_contract=contract, routing_candidates=(), trace=trace)
    if strict:
        try:
            validate_graph_delta_contract(graph_delta_contract)
        except Exception as exc:  # noqa: BLE001 - normalize Sprint-5 errors behind reveal validation
            raise RevealValidationError("invalid_graph_delta_contract", str(exc)) from exc
    decisions: list[RevealDecision] = []
    candidates: list[RevealCandidate] = []
    for step in graph_delta_contract.steps:
        for delta in step.region_deltas:
            decision = _decision_from_delta(delta, action_order=step.action_order, phase_order=step.phase_order, memory=memory)
            decisions.append(decision)
            if decision.reveal_allowed:
                candidates.append(
                    RevealCandidate(
                        entity_id=decision.entity_id,
                        canonical_region_id=decision.canonical_region_id,
                        region_id=decision.region_id,
                        source_delta_type=decision.source_delta_type,
                        action_type=decision.action_type,
                        phase_id=decision.phase_id,
                        reason=decision.reason,
                    )
                )
    routing_candidates = tuple(candidate for decision in decisions if (candidate := _routing_candidate_from_decision(decision)) is not None)
    trace = RevealTrace(
        reasons=("graph_delta_contract_consumed", "memory_policy_evidence_evaluated", "routing_candidates_are_handoff_only"),
        unsupported_planner_fragments=tuple(graph_delta_contract.trace.unsupported_planner_fragments),
        planner_trace_reasons=tuple(graph_delta_contract.trace.planner_trace_reasons),
        dynamics_trace_reasons=tuple(graph_delta_contract.trace.reasons),
    )
    contract = RevealContract(decisions=tuple(decisions), reveal_candidates=tuple(candidates), routing_candidates=routing_candidates, trace=trace)
    validate_reveal_contract(contract)
    return RevealHandoffResult(supported=True, reveal_contract=contract, routing_candidates=routing_candidates, trace=trace)


def validate_reveal_decision(decision: RevealDecision) -> RevealDecision:
    if not isinstance(decision, RevealDecision):
        raise RevealValidationError("invalid_decision", "expected RevealDecision")
    if not decision.entity_id or not decision.region_id or not decision.canonical_region_id:
        raise RevealValidationError("missing_region_provenance", "RevealDecision requires entity_id, region_id, and canonical_region_id")
    if _value(decision.decision_type) not in _ALLOWED_DECISION_VALUES:
        raise RevealValidationError("unknown_decision_type", f"unknown reveal decision type={decision.decision_type!r}")
    if _value(decision.lifecycle_state) not in _ALLOWED_LIFECYCLE_VALUES:
        raise RevealValidationError("unknown_lifecycle_state", f"unknown occlusion lifecycle={decision.lifecycle_state!r}")
    region = decision.canonical_region_id
    private = _is_private_or_optional(region) or decision.private_or_optional_region
    if not _known_region(region):
        if _value(decision.decision_type) != RevealDecisionType.REVEAL_BLOCKED_UNSUPPORTED_REGION.value:
            raise RevealValidationError("unknown_region", f"unknown reveal region={region!r}")
    if private:
        if _value(decision.decision_type) != RevealDecisionType.REVEAL_BLOCKED_PRIVATE.value:
            raise RevealValidationError("private_reveal_forbidden", "private/optional region must be reveal_blocked_private")
        if decision.reveal_allowed or decision.requires_rendering_candidate:
            raise RevealValidationError("private_render_candidate_forbidden", "private/optional region cannot be reveal/render candidate")
    if decision.identity_locked and decision.allowed_to_modify_identity:
        raise RevealValidationError("identity_modification_forbidden", "identity-locked region cannot allow identity modification during reveal")
    if any((decision.rendered_pixels_generated, decision.observed_evidence_created, decision.memory_write_performed, decision.region_routing_called, decision.scene_graph_mutation_performed, decision.mask_created)):
        raise RevealValidationError("forbidden_reveal_operation", "reveal decision claims rendering/routing/memory/mask/evidence/scene mutation")
    if decision.routing_decision_made or decision.render_strategy_selected:
        raise RevealValidationError("routing_decision_forbidden", "reveal routing candidates are handoff only, not routing/render decisions")
    evidence = decision.memory_evidence
    material = str(evidence.material_provenance)
    if evidence.observed_directly and material in _UNSAFE_OBSERVED_CLAIM_MATERIALS:
        raise RevealValidationError("unsafe_material_marked_observed", "generated/inferred/fallback/synthetic/hidden/unknown material cannot be marked observed reveal evidence")
    if _value(decision.decision_type) == RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value:
        if private:
            raise RevealValidationError("private_reveal_from_memory_forbidden", "private memory cannot support reveal")
        if not evidence.supports_observed_reveal:
            raise RevealValidationError("observed_reveal_without_safe_evidence", "reveal_from_observed_memory requires safe observed memory evidence")
        if decision.identity_locked and evidence.authority != MemoryAuthority.AUTHORITATIVE.value:
            raise RevealValidationError("identity_reveal_not_authoritative", "identity observed reveal requires authoritative identity memory")
    if _value(decision.decision_type) == RevealDecisionType.NEWLY_OCCLUDED.value:
        if decision.reveal_allowed or decision.requires_rendering_candidate:
            raise RevealValidationError("newly_occluded_reveal_candidate_forbidden", "newly occluded regions cannot be reveal/render candidates")
    if _value(decision.decision_type) == RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value:
        if decision.reveal_allowed or decision.requires_rendering_candidate:
            raise RevealValidationError("occlusion_reasoning_reveal_candidate_forbidden", "occlusion reasoning requirements cannot be reveal/render candidates")
    if _value(decision.decision_type) == RevealDecisionType.REVEAL_FROM_WEAK_MEMORY.value:
        if evidence.authority == MemoryAuthority.AUTHORITATIVE.value or evidence.memory_support_level in {"strong", "authoritative"}:
            raise RevealValidationError("weak_reveal_claims_authoritative", "weak reveal cannot claim authoritative evidence")
        if evidence.authority in {MemoryAuthority.REJECTED.value, MemoryAuthority.DIAGNOSTIC_ONLY.value} or evidence.reference_kind == "none":
            raise RevealValidationError("weak_reveal_without_reusable_reference", "weak reveal requires reusable non-private reference evidence")
        if not evidence.supports_weak_reveal and evidence.memory_support_level != "weak":
            raise RevealValidationError("weak_reveal_without_weak_support", "weak reveal requires real weak reusable support")
        if evidence.material_provenance not in _SAFE_OBSERVED_MATERIALS:
            raise RevealValidationError("weak_reveal_unsafe_material", "unsafe material cannot support renderable weak reveal candidates")
    if _value(decision.decision_type) == RevealDecisionType.REVEAL_UNKNOWN_DEFER.value:
        if decision.requires_rendering_candidate or evidence.authority == MemoryAuthority.AUTHORITATIVE.value or evidence.supports_observed_reveal:
            raise RevealValidationError("unknown_defer_claims_known_content", "unknown defer cannot require known-content rendering or authoritative evidence")
    return decision


def validate_reveal_contract(contract: RevealContract) -> RevealContract:
    if not isinstance(contract, RevealContract):
        raise RevealValidationError("invalid_contract", "expected RevealContract")
    if any((contract.rendered_pixels_generated, contract.observed_evidence_created, contract.memory_write_performed, contract.region_routing_called, contract.scene_graph_mutation_performed, contract.mask_creation_performed, contract.identity_memory_created, contract.identity_embedding_created)):
        raise RevealValidationError("forbidden_reveal_operation", "reveal contract claims forbidden operation")
    if not contract.supported:
        if contract.decisions or contract.reveal_candidates or contract.routing_candidates:
            raise RevealValidationError("unsupported_contract_has_candidates", "unsupported dynamics cannot produce normal reveal candidates")
        return contract
    decision_keys = set()
    for decision in contract.decisions:
        validate_reveal_decision(decision)
        decision_keys.add((decision.region_id, _value(decision.decision_type), decision.action_type, decision.phase_id, decision.source_delta_type))
    for candidate in contract.routing_candidates:
        if candidate.routing_decision_made or candidate.render_strategy_selected:
            raise RevealValidationError("routing_decision_forbidden", "RevealRoutingCandidate cannot select routing or render strategy")
        if not candidate.reveal_allowed or not candidate.requires_rendering_candidate:
            raise RevealValidationError("invalid_routing_candidate", "handoff routing candidates must be reveal-allowed and require later candidate validation")
        if _is_private_or_optional(candidate.canonical_region_id):
            raise RevealValidationError("private_routing_candidate_forbidden", "private/optional regions cannot be reveal routing candidates")
        key = (candidate.region_id, candidate.reveal_decision_type, candidate.action_type, candidate.phase_id, candidate.source_delta_type)
        if key not in decision_keys:
            raise RevealValidationError("routing_candidate_without_decision", "RevealRoutingCandidate must be backed by a RevealDecision")
    return contract
