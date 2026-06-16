from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping

from core.body_ontology import BODY_ONTOLOGY, BodyRegionGroup, is_canonical_body_region
from core.region_ids import is_known_canonical_region_type, parse_region_id
from core.schema import SceneGraph, VideoMemory
from dynamics.graph_delta_contract import (
    ALLOWED_GARMENT_REGIONS,
    ALLOWED_OBJECT_REGIONS,
    FUTURE_FACE_SUBREGIONS,
    GraphDeltaContract,
    RegionDeltaType,
    RoutingCandidate,
    validate_graph_delta_contract,
)
from memory.memory_policy import MemoryAuthority, MemoryFamily, classify_memory_family
from reveal.reveal_contract import RevealContract, RevealDecision, RevealDecisionType, RevealRoutingCandidate, validate_reveal_contract


class RegionRoutingValidationError(ValueError):
    """Raised when the Sprint-7 Region Routing v2 contract is violated."""

    def __init__(self, code: str, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details or {}


class RegionRoutingDecisionType(str, Enum):
    ROUTE_POSE_UPDATE = "route_pose_update"
    ROUTE_EXPRESSION_UPDATE = "route_expression_update"
    ROUTE_VISIBILITY_UPDATE = "route_visibility_update"
    ROUTE_INTERACTION_UPDATE = "route_interaction_update"
    ROUTE_GARMENT_INTENT = "route_garment_intent"
    ROUTE_REVEAL_OBSERVED_MEMORY = "route_reveal_observed_memory"
    ROUTE_REVEAL_WEAK_MEMORY = "route_reveal_weak_memory"
    ROUTE_PRESERVE_VISIBLE = "route_preserve_visible"
    ROUTE_MEMORY_ASSISTED_IDENTITY_LOCKED = "route_memory_assisted_identity_locked"
    ROUTE_OCCLUSION_REASONING_ONLY = "route_occlusion_reasoning_only"
    ROUTE_NEWLY_OCCLUDED_TRACKING_ONLY = "route_newly_occluded_tracking_only"
    BLOCK_PRIVATE_REGION = "block_private_region"
    BLOCK_UNKNOWN_DEFER = "block_unknown_defer"
    BLOCK_IDENTITY_RISK = "block_identity_risk"
    BLOCK_NO_SAFE_MEMORY = "block_no_safe_memory"
    BLOCK_UNSUPPORTED_REGION = "block_unsupported_region"
    BLOCK_NO_ROUTE_EVIDENCE = "block_no_route_evidence"
    BLOCK_POLICY_REJECTED = "block_policy_rejected"


class RegionRoutingSource(str, Enum):
    DYNAMICS = "dynamics_graph_delta"
    REVEAL = "reveal_occlusion"
    MERGED = "dynamics_reveal_merge"
    VALIDATION = "routing_validation"


class RegionRoutingBlockReason(str, Enum):
    PRIVATE_REGION = "private_region"
    UNKNOWN_DEFER_NO_SAFE_MEMORY = "unknown_defer_no_safe_memory"
    IDENTITY_RISK = "identity_risk"
    NO_SAFE_MEMORY = "no_safe_memory"
    UNSUPPORTED_REGION = "unsupported_region"
    NO_ROUTE_EVIDENCE = "no_route_evidence"
    POLICY_REJECTED = "policy_rejected"


class RegionRoutingRenderRequirement(str, Enum):
    NONE = "none"
    ROI_REQUIRED = "roi_required"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    TRACKING_ONLY = "tracking_only"


class RegionRoutingIdentityRequirement(str, Enum):
    NONE = "none"
    PRESERVE_IDENTITY = "preserve_identity"
    AUTHORITATIVE_IDENTITY_MEMORY_REQUIRED = "authoritative_identity_memory_required"


class RegionRoutingMemoryRequirement(str, Enum):
    NONE = "none"
    APPEARANCE_MEMORY = "appearance_memory"
    REVEAL_MEMORY = "reveal_memory"
    IDENTITY_MEMORY = "identity_memory"
    WEAK_MEMORY = "weak_memory"


_IDENTITY_REGIONS = frozenset({"face", "head", "hair", "scalp"}) | FUTURE_FACE_SUBREGIONS
_ALLOWED_NON_BODY_REGIONS = ALLOWED_GARMENT_REGIONS | ALLOWED_OBJECT_REGIONS
_LEARNED_RENDERER_STRATEGIES = frozenset({"learned_primary", "diffusion", "video_diffusion", "latent_diffusion", "generative_fill"})
_ROUTEABLE_TYPES = frozenset(
    {
        RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value,
        RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value,
        RegionRoutingDecisionType.ROUTE_VISIBILITY_UPDATE.value,
        RegionRoutingDecisionType.ROUTE_INTERACTION_UPDATE.value,
        RegionRoutingDecisionType.ROUTE_GARMENT_INTENT.value,
        RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value,
        RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value,
        RegionRoutingDecisionType.ROUTE_PRESERVE_VISIBLE.value,
        RegionRoutingDecisionType.ROUTE_MEMORY_ASSISTED_IDENTITY_LOCKED.value,
    }
)
_DIAGNOSTIC_TYPES = frozenset(
    {
        RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value,
        RegionRoutingDecisionType.ROUTE_NEWLY_OCCLUDED_TRACKING_ONLY.value,
    }
)
_BLOCKED_TYPES = frozenset(
    {
        RegionRoutingDecisionType.BLOCK_PRIVATE_REGION.value,
        RegionRoutingDecisionType.BLOCK_UNKNOWN_DEFER.value,
        RegionRoutingDecisionType.BLOCK_IDENTITY_RISK.value,
        RegionRoutingDecisionType.BLOCK_NO_SAFE_MEMORY.value,
        RegionRoutingDecisionType.BLOCK_UNSUPPORTED_REGION.value,
        RegionRoutingDecisionType.BLOCK_NO_ROUTE_EVIDENCE.value,
        RegionRoutingDecisionType.BLOCK_POLICY_REJECTED.value,
    }
)


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
    return region in _IDENTITY_REGIONS or classify_memory_family(region) == MemoryFamily.IDENTITY


@dataclass(frozen=True, slots=True)
class RegionRoutingCandidate:
    region_id: str
    entity_id: str
    canonical_region_id: str
    source: RegionRoutingSource | str
    source_delta_type: str
    source_reveal_decision_type: str | None
    action_type: str
    phase_id: str
    action_order: int = 0
    phase_order: int = 0
    requires_rendering_candidate: bool = False
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionRoutingTrace:
    routing_source: str = "sprint7_region_routing_v2_contract"
    provenance: str = "dynamics_reveal_to_renderer_safe_route_handoff"
    support_level: str = "supported"
    reasons: tuple[str, ...] = ()
    dynamics_candidate_count: int = 0
    reveal_decision_count: int = 0
    reveal_candidate_count: int = 0
    route_allowed_count: int = 0
    render_candidate_allowed_count: int = 0
    blocked_count: int = 0
    diagnostic_count: int = 0
    forbidden_operations_asserted_absent: tuple[str, ...] = (
        "rendering",
        "compositing",
        "scene_graph_mutation",
        "memory_write",
        "mask_creation",
        "observed_perception_evidence_creation",
        "pixel_generation",
        "learned_renderer_selection",
        "private_anatomical_rendering",
    )


@dataclass(frozen=True, slots=True)
class RegionRoutingDecision:
    region_id: str
    entity_id: str
    canonical_region_id: str
    decision_type: RegionRoutingDecisionType | str
    source: RegionRoutingSource | str
    source_delta_type: str | None
    source_reveal_decision_type: str | None
    action_type: str
    phase_id: str
    action_order: int
    phase_order: int
    route_allowed: bool
    render_candidate_allowed: bool
    blocked: bool
    block_reason: RegionRoutingBlockReason | str | None
    diagnostic_only: bool
    identity_locked: bool
    allowed_to_modify_identity: bool
    requires_identity_memory: bool
    requires_appearance_memory: bool
    requires_reveal_memory: bool
    memory_reference_kind: str
    memory_authority: str
    memory_family: str
    weak_memory_candidate: bool
    preserve_visible: bool
    occlusion_reasoning_only: bool
    newly_occluded_tracking_only: bool
    private_or_optional_region: bool
    roi_required: bool
    roi_source_policy: str
    renderer_strategy_allowed_values: tuple[str, ...]
    selected_render_strategy: str | None
    renderer_must_preserve_identity: bool
    renderer_must_not_create_observed_evidence: bool
    renderer_must_not_create_identity_memory: bool
    validation_reasons: tuple[str, ...]
    provenance: tuple[str, ...]
    render_requirement: str = RegionRoutingRenderRequirement.NONE.value
    identity_requirement: str = RegionRoutingIdentityRequirement.NONE.value
    memory_requirement: str = RegionRoutingMemoryRequirement.NONE.value
    output_reference_authority: str = "none"
    memory_write_allowed: bool = False
    identity_memory_write_allowed: bool = False
    observed_evidence_claim_allowed: bool = False
    rendered_pixels_generated: bool = False
    memory_write_performed: bool = False
    observed_evidence_created: bool = False
    scene_graph_mutation_performed: bool = False

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RegionRoutingContract:
    contract_version: str = "region_routing_contract_v2"
    supported: bool = True
    candidates: tuple[RegionRoutingCandidate, ...] = ()
    decisions: tuple[RegionRoutingDecision, ...] = ()
    renderable_region_ids: tuple[str, ...] = ()
    blocked_region_ids: tuple[str, ...] = ()
    diagnostic_region_ids: tuple[str, ...] = ()
    trace: RegionRoutingTrace = field(default_factory=RegionRoutingTrace)
    rendered_pixels_generated: bool = False
    observed_evidence_created: bool = False
    memory_write_performed: bool = False
    scene_graph_mutation_performed: bool = False
    mask_creation_performed: bool = False
    learned_renderer_selected: bool = False

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]

    def decision_for_region_id(self, region_id: str) -> RegionRoutingDecision | None:
        for decision in self.decisions:
            if decision.region_id == region_id:
                return decision
        return None

    def renderable_decision_for_region_id(self, region_id: str) -> RegionRoutingDecision | None:
        for decision in self.decisions:
            if (
                decision.region_id == region_id
                and decision.route_allowed
                and decision.render_candidate_allowed
                and not decision.blocked
                and not decision.diagnostic_only
                and decision.roi_required
            ):
                return decision
        return None


@dataclass(frozen=True, slots=True)
class RegionRoutingHandoffResult:
    supported: bool
    region_routing_contract: RegionRoutingContract
    renderable_region_ids: tuple[str, ...]
    trace: RegionRoutingTrace

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def _entity_from_region_id(region_id: str, fallback: str = "scene") -> str:
    try:
        entity, _ = parse_region_id(region_id)
        return entity
    except ValueError:
        return fallback


def _step_orders(graph_delta_contract: GraphDeltaContract) -> dict[tuple[str, str, str], tuple[int, int]]:
    orders: dict[tuple[str, str, str], tuple[int, int]] = {}
    for step in graph_delta_contract.steps:
        for delta in step.region_deltas:
            orders[(delta.region_id, _value(delta.delta_type), delta.phase_id)] = (step.action_order, step.phase_order)
    return orders


def _candidate_from_dynamics(candidate: RoutingCandidate, orders: Mapping[tuple[str, str, str], tuple[int, int]]) -> RegionRoutingCandidate:
    action_order, phase_order = orders.get((candidate.region_id, candidate.delta_type, candidate.phase_id), (0, 0))
    return RegionRoutingCandidate(
        region_id=candidate.region_id,
        entity_id=_entity_from_region_id(candidate.region_id),
        canonical_region_id=candidate.canonical_region_id,
        source=RegionRoutingSource.DYNAMICS,
        source_delta_type=candidate.delta_type,
        source_reveal_decision_type=None,
        action_type=candidate.action_type,
        phase_id=candidate.phase_id,
        action_order=action_order,
        phase_order=phase_order,
        requires_rendering_candidate=candidate.requires_rendering_candidate,
        provenance=(candidate.reason, "graph_delta.routing_candidates"),
    )


def _candidate_from_reveal(candidate: RevealRoutingCandidate, decision: RevealDecision | None) -> RegionRoutingCandidate:
    return RegionRoutingCandidate(
        region_id=candidate.region_id,
        entity_id=_entity_from_region_id(candidate.region_id, getattr(decision, "entity_id", "scene")),
        canonical_region_id=candidate.canonical_region_id,
        source=RegionRoutingSource.REVEAL,
        source_delta_type=candidate.source_delta_type,
        source_reveal_decision_type=candidate.reveal_decision_type,
        action_type=candidate.action_type,
        phase_id=candidate.phase_id,
        action_order=int(getattr(decision, "action_order", 0) or 0),
        phase_order=int(getattr(decision, "phase_order", 0) or 0),
        requires_rendering_candidate=candidate.requires_rendering_candidate,
        provenance=(candidate.reason, "reveal.routing_candidates"),
    )


def _base_decision(
    *,
    candidate: RegionRoutingCandidate,
    decision_type: str,
    route_allowed: bool,
    render_candidate_allowed: bool,
    blocked: bool = False,
    block_reason: str | None = None,
    diagnostic_only: bool = False,
    memory_reference_kind: str = "none",
    memory_authority: str = MemoryAuthority.REJECTED.value,
    memory_family: str = MemoryFamily.UNKNOWN.value,
    validation_reasons: tuple[str, ...] = (),
    provenance: tuple[str, ...] = (),
) -> RegionRoutingDecision:
    region = candidate.canonical_region_id
    identity_locked = _is_identity_region(region) and not _is_private_or_optional(region)
    private = _is_private_or_optional(region)
    weak = decision_type == RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value
    preserve = decision_type == RegionRoutingDecisionType.ROUTE_PRESERVE_VISIBLE.value
    occ_reasoning = decision_type == RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value
    newly_occ = decision_type == RegionRoutingDecisionType.ROUTE_NEWLY_OCCLUDED_TRACKING_ONLY.value
    requires_identity_memory = identity_locked and route_allowed and decision_type in {
        RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value,
        RegionRoutingDecisionType.ROUTE_MEMORY_ASSISTED_IDENTITY_LOCKED.value,
        RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value,
        RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value,
        RegionRoutingDecisionType.ROUTE_VISIBILITY_UPDATE.value,
    }
    requires_reveal_memory = decision_type in {RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value, RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value}
    output_authority = "weak" if weak else (memory_authority if requires_reveal_memory else "none")
    render_requirement = (
        RegionRoutingRenderRequirement.DIAGNOSTIC_ONLY.value
        if diagnostic_only and occ_reasoning
        else RegionRoutingRenderRequirement.TRACKING_ONLY.value
        if diagnostic_only and newly_occ
        else RegionRoutingRenderRequirement.ROI_REQUIRED.value
        if render_candidate_allowed
        else RegionRoutingRenderRequirement.NONE.value
    )
    identity_requirement = (
        RegionRoutingIdentityRequirement.AUTHORITATIVE_IDENTITY_MEMORY_REQUIRED.value
        if requires_identity_memory
        else RegionRoutingIdentityRequirement.PRESERVE_IDENTITY.value
        if identity_locked
        else RegionRoutingIdentityRequirement.NONE.value
    )
    memory_requirement = (
        RegionRoutingMemoryRequirement.IDENTITY_MEMORY.value
        if requires_identity_memory
        else RegionRoutingMemoryRequirement.WEAK_MEMORY.value
        if weak
        else RegionRoutingMemoryRequirement.REVEAL_MEMORY.value
        if requires_reveal_memory
        else RegionRoutingMemoryRequirement.NONE.value
    )
    return RegionRoutingDecision(
        region_id=candidate.region_id,
        entity_id=candidate.entity_id,
        canonical_region_id=region,
        decision_type=decision_type,
        source=candidate.source,
        source_delta_type=candidate.source_delta_type,
        source_reveal_decision_type=candidate.source_reveal_decision_type,
        action_type=candidate.action_type,
        phase_id=candidate.phase_id,
        action_order=candidate.action_order,
        phase_order=candidate.phase_order,
        route_allowed=route_allowed,
        render_candidate_allowed=render_candidate_allowed,
        blocked=blocked,
        block_reason=block_reason,
        diagnostic_only=diagnostic_only,
        identity_locked=identity_locked,
        allowed_to_modify_identity=False,
        requires_identity_memory=requires_identity_memory,
        requires_appearance_memory=route_allowed and classify_memory_family(region) in {MemoryFamily.SKIN, MemoryFamily.BODY_SHAPE, MemoryFamily.SOFT_TISSUE, MemoryFamily.GARMENT, MemoryFamily.ACCESSORY},
        requires_reveal_memory=requires_reveal_memory,
        memory_reference_kind=memory_reference_kind,
        memory_authority=memory_authority,
        memory_family=memory_family,
        weak_memory_candidate=weak,
        preserve_visible=preserve,
        occlusion_reasoning_only=occ_reasoning,
        newly_occluded_tracking_only=newly_occ,
        private_or_optional_region=private,
        roi_required=render_candidate_allowed,
        roi_source_policy="existing_roi_selector_only" if render_candidate_allowed else "no_roi_request",
        renderer_strategy_allowed_values=("keep", "warp", "deform", "refine", "reveal"),
        selected_render_strategy=None,
        renderer_must_preserve_identity=identity_locked,
        renderer_must_not_create_observed_evidence=True,
        renderer_must_not_create_identity_memory=True,
        validation_reasons=validation_reasons,
        provenance=tuple((*candidate.provenance, *provenance)),
        render_requirement=render_requirement,
        identity_requirement=identity_requirement,
        memory_requirement=memory_requirement,
        output_reference_authority=output_authority,
        memory_write_allowed=False,
        identity_memory_write_allowed=False,
        observed_evidence_claim_allowed=False,
    )


def _block_decision(candidate: RegionRoutingCandidate, decision_type: str, reason: str, *extra: str) -> RegionRoutingDecision:
    return _base_decision(
        candidate=candidate,
        decision_type=decision_type,
        route_allowed=False,
        render_candidate_allowed=False,
        blocked=True,
        block_reason=reason,
        diagnostic_only=True,
        validation_reasons=(reason, *extra),
        provenance=("reveal_or_policy_block_overrides_dynamics",),
    )


def _decision_for_reveal(candidate: RegionRoutingCandidate, reveal_decision: RevealDecision) -> RegionRoutingDecision:
    reveal_type = _value(reveal_decision.decision_type)
    evidence = reveal_decision.memory_evidence
    memory_authority = str(evidence.authority)
    memory_reference_kind = str(evidence.reference_kind)
    memory_family = str(evidence.memory_family)
    if reveal_type == RevealDecisionType.REVEAL_BLOCKED_PRIVATE.value:
        return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_PRIVATE_REGION.value, RegionRoutingBlockReason.PRIVATE_REGION.value, reveal_type)
    if reveal_type == RevealDecisionType.REVEAL_UNKNOWN_DEFER.value:
        return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_UNKNOWN_DEFER.value, RegionRoutingBlockReason.UNKNOWN_DEFER_NO_SAFE_MEMORY.value, reveal_type)
    if reveal_type == RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK.value:
        return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_IDENTITY_RISK.value, RegionRoutingBlockReason.IDENTITY_RISK.value, reveal_type)
    if reveal_type == RevealDecisionType.REVEAL_BLOCKED_UNSUPPORTED_REGION.value:
        return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_UNSUPPORTED_REGION.value, RegionRoutingBlockReason.UNSUPPORTED_REGION.value, reveal_type)
    if reveal_type == RevealDecisionType.REVEAL_BLOCKED_NO_EVIDENCE.value:
        return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_NO_SAFE_MEMORY.value, RegionRoutingBlockReason.NO_SAFE_MEMORY.value, reveal_type)
    if reveal_type == RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value:
        return _base_decision(
            candidate=candidate,
            decision_type=RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value,
            route_allowed=False,
            render_candidate_allowed=False,
            diagnostic_only=True,
            validation_reasons=("occlusion_reasoning_required_no_render",),
            provenance=("reveal_decision_diagnostic_only",),
        )
    if reveal_type == RevealDecisionType.NEWLY_OCCLUDED.value:
        return _base_decision(
            candidate=candidate,
            decision_type=RegionRoutingDecisionType.ROUTE_NEWLY_OCCLUDED_TRACKING_ONLY.value,
            route_allowed=False,
            render_candidate_allowed=False,
            diagnostic_only=True,
            validation_reasons=("newly_occluded_tracking_only_no_render",),
            provenance=("reveal_decision_tracking_only",),
        )
    if reveal_type == RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value:
        if _is_identity_region(candidate.canonical_region_id) and memory_authority != MemoryAuthority.AUTHORITATIVE.value:
            return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_IDENTITY_RISK.value, RegionRoutingBlockReason.IDENTITY_RISK.value, "identity_observed_reveal_requires_authoritative_memory")
        return _base_decision(
            candidate=candidate,
            decision_type=RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value,
            route_allowed=True,
            render_candidate_allowed=True,
            memory_reference_kind=memory_reference_kind,
            memory_authority=memory_authority,
            memory_family=memory_family,
            validation_reasons=("observed_memory_reveal_route",),
            provenance=("reveal_decision_takes_priority",),
        )
    if reveal_type == RevealDecisionType.REVEAL_FROM_WEAK_MEMORY.value:
        if _is_identity_region(candidate.canonical_region_id):
            return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_IDENTITY_RISK.value, RegionRoutingBlockReason.IDENTITY_RISK.value, "weak_identity_reveal_forbidden")
        return _base_decision(
            candidate=candidate,
            decision_type=RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value,
            route_allowed=True,
            render_candidate_allowed=True,
            memory_reference_kind=memory_reference_kind,
            memory_authority=MemoryAuthority.WEAK.value,
            memory_family=memory_family,
            validation_reasons=("weak_memory_non_authoritative_route",),
            provenance=("reveal_decision_takes_priority",),
        )
    if reveal_type == RevealDecisionType.PRESERVE_VISIBLE.value:
        return _base_decision(
            candidate=candidate,
            decision_type=RegionRoutingDecisionType.ROUTE_PRESERVE_VISIBLE.value,
            route_allowed=bool(reveal_decision.reveal_allowed),
            render_candidate_allowed=bool(reveal_decision.requires_rendering_candidate),
            memory_reference_kind=memory_reference_kind,
            memory_authority=memory_authority,
            memory_family=memory_family,
            validation_reasons=("preserve_visible_no_reveal_generation",),
            provenance=("reveal_preserve_visible",),
        )
    return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_POLICY_REJECTED.value, RegionRoutingBlockReason.POLICY_REJECTED.value, f"unsupported_reveal_decision:{reveal_type}")


def _decision_for_dynamics(candidate: RegionRoutingCandidate) -> RegionRoutingDecision:
    if _is_private_or_optional(candidate.canonical_region_id):
        return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_PRIVATE_REGION.value, RegionRoutingBlockReason.PRIVATE_REGION.value, "private_optional_dynamics_candidate")
    if not _known_region(candidate.canonical_region_id):
        return _block_decision(candidate, RegionRoutingDecisionType.BLOCK_UNSUPPORTED_REGION.value, RegionRoutingBlockReason.UNSUPPORTED_REGION.value, "unknown_dynamics_region")
    mapping = {
        RegionDeltaType.POSE_DELTA.value: RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value,
        RegionDeltaType.EXPRESSION_DELTA.value: RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value,
        RegionDeltaType.VISIBILITY_DELTA.value: RegionRoutingDecisionType.ROUTE_VISIBILITY_UPDATE.value,
        RegionDeltaType.INTERACTION_DELTA.value: RegionRoutingDecisionType.ROUTE_INTERACTION_UPDATE.value,
        RegionDeltaType.GARMENT_INTENT_DELTA.value: RegionRoutingDecisionType.ROUTE_GARMENT_INTENT.value,
    }
    if candidate.source_delta_type == RegionDeltaType.OCCLUSION_DELTA.value:
        return _base_decision(
            candidate=candidate,
            decision_type=RegionRoutingDecisionType.ROUTE_OCCLUSION_REASONING_ONLY.value,
            route_allowed=False,
            render_candidate_allowed=False,
            diagnostic_only=True,
            validation_reasons=("occlusion_delta_requires_reveal_reasoning_no_render",),
            provenance=("ordinary_dynamics_occlusion_delta_diagnostic_only",),
        )
    decision_type = mapping.get(candidate.source_delta_type, RegionRoutingDecisionType.BLOCK_NO_ROUTE_EVIDENCE.value)
    if decision_type == RegionRoutingDecisionType.BLOCK_NO_ROUTE_EVIDENCE.value:
        return _block_decision(candidate, decision_type, RegionRoutingBlockReason.NO_ROUTE_EVIDENCE.value, candidate.source_delta_type)
    return _base_decision(
        candidate=candidate,
        decision_type=decision_type,
        route_allowed=True,
        render_candidate_allowed=bool(candidate.requires_rendering_candidate),
        memory_family=classify_memory_family(candidate.canonical_region_id).value,
        validation_reasons=("ordinary_dynamics_route_without_reveal_block",),
        provenance=("no_reveal_override_for_candidate",),
    )


def _candidate_from_reveal_decision(decision: RevealDecision) -> RegionRoutingCandidate:
    return RegionRoutingCandidate(
        region_id=decision.region_id,
        entity_id=decision.entity_id,
        canonical_region_id=decision.canonical_region_id,
        source=RegionRoutingSource.REVEAL,
        source_delta_type=decision.source_delta_type,
        source_reveal_decision_type=_value(decision.decision_type),
        action_type=decision.action_type,
        phase_id=decision.phase_id,
        action_order=decision.action_order,
        phase_order=decision.phase_order,
        requires_rendering_candidate=decision.requires_rendering_candidate,
        provenance=(decision.reason, "reveal.decisions"),
    )



_HARD_REVEAL_OVERRIDE_PRIORITY: dict[str, int] = {
    RevealDecisionType.REVEAL_BLOCKED_PRIVATE.value: 1,
    RevealDecisionType.REVEAL_BLOCKED_IDENTITY_RISK.value: 2,
    RevealDecisionType.REVEAL_UNKNOWN_DEFER.value: 3,
    RevealDecisionType.REVEAL_BLOCKED_UNSUPPORTED_REGION.value: 4,
    RevealDecisionType.REVEAL_BLOCKED_NO_EVIDENCE.value: 5,
}
_DIAGNOSTIC_REVEAL_PRIORITY: dict[str, int] = {
    RevealDecisionType.NEWLY_OCCLUDED.value: 10,
    RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value: 11,
}
_ROUTEABLE_REVEAL_PRIORITY: dict[str, int] = {
    RevealDecisionType.REVEAL_FROM_OBSERVED_MEMORY.value: 20,
    RevealDecisionType.REVEAL_FROM_WEAK_MEMORY.value: 21,
    RevealDecisionType.PRESERVE_VISIBLE.value: 22,
}


def _reveal_priority(decision: RevealDecision) -> tuple[int, int, int]:
    decision_type = _value(decision.decision_type)
    if decision_type in _HARD_REVEAL_OVERRIDE_PRIORITY:
        group = 0
        priority = _HARD_REVEAL_OVERRIDE_PRIORITY[decision_type]
    elif decision_type in _DIAGNOSTIC_REVEAL_PRIORITY:
        group = 1
        priority = _DIAGNOSTIC_REVEAL_PRIORITY[decision_type]
    else:
        group = 2
        priority = _ROUTEABLE_REVEAL_PRIORITY.get(decision_type, 99)
    return (group, priority, int(decision.action_order) * 1000 + int(decision.phase_order))


def _select_reveal_decision(decisions: list[RevealDecision]) -> RevealDecision | None:
    if not decisions:
        return None
    return sorted(decisions, key=_reveal_priority)[0]


def _select_region_level_hard_override(decisions: list[RevealDecision]) -> RevealDecision | None:
    hard = [decision for decision in decisions if _value(decision.decision_type) in _HARD_REVEAL_OVERRIDE_PRIORITY]
    return _select_reveal_decision(hard)

def build_region_routing_handoff(
    *,
    scene_graph: SceneGraph,
    graph_delta_contract: GraphDeltaContract,
    reveal_contract: RevealContract,
    memory: VideoMemory | Mapping[str, object] | None = None,
) -> RegionRoutingHandoffResult:
    del scene_graph, memory  # read-only inputs reserved for future route-context lookups.
    validate_graph_delta_contract(graph_delta_contract, allow_unsupported=True)
    validate_reveal_contract(reveal_contract)

    orders = _step_orders(graph_delta_contract)
    dynamics_candidates = [_candidate_from_dynamics(candidate, orders) for candidate in graph_delta_contract.routing_candidates]
    reveal_decisions_by_key: dict[tuple[str, str, str], list[RevealDecision]] = {}
    reveal_decisions_by_region: dict[str, list[RevealDecision]] = {}
    for decision in reveal_contract.decisions:
        reveal_decisions_by_key.setdefault((decision.region_id, decision.source_delta_type, decision.phase_id), []).append(decision)
        reveal_decisions_by_region.setdefault(decision.region_id, []).append(decision)
    reveal_candidate_by_key = {(candidate.region_id, candidate.source_delta_type, candidate.phase_id): candidate for candidate in reveal_contract.routing_candidates}

    candidates: list[RegionRoutingCandidate] = []
    decisions: list[RegionRoutingDecision] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for reveal_decision in reveal_contract.decisions:
        key = (reveal_decision.region_id, reveal_decision.source_delta_type, reveal_decision.phase_id)
        reveal_candidate = reveal_candidate_by_key.get(key)
        candidate = _candidate_from_reveal(reveal_candidate, reveal_decision) if reveal_candidate is not None else _candidate_from_reveal_decision(reveal_decision)
        candidates.append(candidate)
        decisions.append(_decision_for_reveal(candidate, reveal_decision))
        seen_keys.add(key)

    for candidate in dynamics_candidates:
        key = (candidate.region_id, candidate.source_delta_type, candidate.phase_id)
        if key in seen_keys:
            continue
        reveal_decision = _select_reveal_decision(reveal_decisions_by_key.get(key, []))
        if reveal_decision is None:
            reveal_decision = _select_region_level_hard_override(reveal_decisions_by_region.get(candidate.region_id, []))
        if reveal_decision is not None:
            merged = RegionRoutingCandidate(
                region_id=candidate.region_id,
                entity_id=candidate.entity_id,
                canonical_region_id=candidate.canonical_region_id,
                source=RegionRoutingSource.MERGED,
                source_delta_type=candidate.source_delta_type,
                source_reveal_decision_type=_value(reveal_decision.decision_type),
                action_type=candidate.action_type,
                phase_id=candidate.phase_id,
                action_order=candidate.action_order,
                phase_order=candidate.phase_order,
                requires_rendering_candidate=candidate.requires_rendering_candidate,
                provenance=(*candidate.provenance, "reveal_decision_overrode_dynamics_candidate"),
            )
            candidates.append(merged)
            decisions.append(_decision_for_reveal(merged, reveal_decision))
        else:
            candidates.append(candidate)
            decisions.append(_decision_for_dynamics(candidate))

    hard_blocked_region_ids = tuple(dict.fromkeys(decision.region_id for decision in decisions if decision.blocked))
    hard_blocked_set = set(hard_blocked_region_ids)
    renderable_region_ids = tuple(dict.fromkeys(decision.region_id for decision in decisions if decision.render_candidate_allowed and decision.region_id not in hard_blocked_set))
    contract = RegionRoutingContract(
        candidates=tuple(candidates),
        decisions=tuple(decisions),
        renderable_region_ids=renderable_region_ids,
        blocked_region_ids=hard_blocked_region_ids,
        diagnostic_region_ids=tuple(dict.fromkeys(decision.region_id for decision in decisions if decision.diagnostic_only)),
        trace=RegionRoutingTrace(
            reasons=("reveal_decisions_processed_before_uncovered_dynamics_candidates",),
            dynamics_candidate_count=len(graph_delta_contract.routing_candidates),
            reveal_decision_count=len(reveal_contract.decisions),
            reveal_candidate_count=len(reveal_contract.routing_candidates),
            route_allowed_count=sum(1 for decision in decisions if decision.route_allowed),
            render_candidate_allowed_count=sum(1 for decision in decisions if decision.render_candidate_allowed),
            blocked_count=sum(1 for decision in decisions if decision.blocked),
            diagnostic_count=sum(1 for decision in decisions if decision.diagnostic_only),
        ),
    )
    validate_region_routing_contract(contract)
    return RegionRoutingHandoffResult(
        supported=contract.supported,
        region_routing_contract=contract,
        renderable_region_ids=contract.renderable_region_ids,
        trace=contract.trace,
    )


def validate_region_routing_decision(decision: RegionRoutingDecision) -> RegionRoutingDecision:
    if not isinstance(decision, RegionRoutingDecision):
        raise RegionRoutingValidationError("invalid_decision", "expected RegionRoutingDecision")
    if not decision.region_id or not decision.entity_id or not decision.canonical_region_id:
        raise RegionRoutingValidationError("missing_region_provenance", "routing decision requires entity_id, region_id, and canonical_region_id")
    if not decision.action_type or not decision.phase_id:
        raise RegionRoutingValidationError("missing_action_phase_provenance", "routing decision requires action_type and phase_id")
    region = decision.canonical_region_id
    if not _known_region(region):
        if _value(decision.decision_type) != RegionRoutingDecisionType.BLOCK_UNSUPPORTED_REGION.value:
            raise RegionRoutingValidationError("unsupported_region_not_blocked", f"unsupported routing region must be blocked: {region}")
    if _is_private_or_optional(region):
        if not decision.blocked or decision.route_allowed or decision.render_candidate_allowed or _value(decision.block_reason) != RegionRoutingBlockReason.PRIVATE_REGION.value:
            raise RegionRoutingValidationError("private_region_route_forbidden", f"private/optional region must be explicitly blocked: {region}")
    if decision.blocked and decision.route_allowed:
        raise RegionRoutingValidationError("blocked_route_allowed", "blocked routing decision cannot be route_allowed")
    if decision.diagnostic_only and decision.render_candidate_allowed:
        raise RegionRoutingValidationError("diagnostic_render_candidate", "diagnostic routing decision cannot be a render candidate")
    if _value(decision.decision_type) in _BLOCKED_TYPES and not decision.blocked:
        raise RegionRoutingValidationError("block_type_not_blocked", "blocked decision type must set blocked=True")
    if _value(decision.decision_type) in _DIAGNOSTIC_TYPES and not decision.diagnostic_only:
        raise RegionRoutingValidationError("diagnostic_type_not_diagnostic", "diagnostic decision type must set diagnostic_only=True")
    if _value(decision.decision_type) in _ROUTEABLE_TYPES and decision.blocked:
        raise RegionRoutingValidationError("routeable_type_blocked", "routeable decision type cannot be blocked")
    if decision.source_reveal_decision_type == RevealDecisionType.REVEAL_UNKNOWN_DEFER.value and decision.render_candidate_allowed:
        raise RegionRoutingValidationError("unknown_defer_render_candidate", "reveal_unknown_defer cannot render")
    if decision.source_reveal_decision_type == RevealDecisionType.OCCLUSION_REASONING_REQUIRED.value and decision.render_candidate_allowed:
        raise RegionRoutingValidationError("occlusion_reasoning_render_candidate", "occlusion reasoning cannot render")
    if decision.source_reveal_decision_type == RevealDecisionType.NEWLY_OCCLUDED.value and decision.render_candidate_allowed:
        raise RegionRoutingValidationError("newly_occluded_render_candidate", "newly occluded tracking cannot render")
    if decision.identity_locked and decision.allowed_to_modify_identity:
        raise RegionRoutingValidationError("identity_modification_forbidden", "identity-locked route cannot allow identity modification")
    if _is_identity_region(region) and _value(decision.decision_type) == RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value:
        raise RegionRoutingValidationError("weak_identity_reveal_forbidden", "weak reveal route is not allowed for identity regions")
    if _is_identity_region(region) and _value(decision.decision_type) == RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value and decision.memory_authority != MemoryAuthority.AUTHORITATIVE.value:
        raise RegionRoutingValidationError("identity_observed_reveal_requires_authoritative", "identity reveal observed memory requires authoritative identity memory")
    if decision.weak_memory_candidate and decision.output_reference_authority != "weak":
        raise RegionRoutingValidationError("weak_memory_authority_mismatch", "weak memory route must remain weak")
    if decision.memory_write_allowed or decision.identity_memory_write_allowed or decision.observed_evidence_claim_allowed:
        raise RegionRoutingValidationError("downstream_write_or_observed_claim_allowed", "routing must forbid memory writes and observed evidence claims")
    if not decision.renderer_must_not_create_observed_evidence or not decision.renderer_must_not_create_identity_memory:
        raise RegionRoutingValidationError("renderer_constraints_missing", "renderer constraints must forbid observed evidence and identity memory creation")
    if decision.rendered_pixels_generated:
        raise RegionRoutingValidationError("rendered_pixels_claimed", "routing decision cannot claim rendered pixels")
    if decision.memory_write_performed:
        raise RegionRoutingValidationError("memory_write_claimed", "routing decision cannot claim a memory write")
    if decision.observed_evidence_created:
        raise RegionRoutingValidationError("observed_evidence_claimed", "routing decision cannot claim observed evidence creation")
    if decision.scene_graph_mutation_performed:
        raise RegionRoutingValidationError("scene_graph_mutation_claimed", "routing decision cannot mutate SceneGraph")
    if decision.selected_render_strategy in _LEARNED_RENDERER_STRATEGIES:
        raise RegionRoutingValidationError("learned_renderer_selected", "routing contract cannot select a learned renderer implementation")
    return decision


def validate_region_routing_contract(contract: RegionRoutingContract) -> RegionRoutingContract:
    if not isinstance(contract, RegionRoutingContract):
        raise RegionRoutingValidationError("invalid_contract", "expected RegionRoutingContract")
    if any((contract.rendered_pixels_generated, contract.observed_evidence_created, contract.memory_write_performed, contract.scene_graph_mutation_performed, contract.mask_creation_performed, contract.learned_renderer_selected)):
        raise RegionRoutingValidationError("forbidden_routing_claim", "routing contract claims a forbidden operation")
    for decision in contract.decisions:
        validate_region_routing_decision(decision)
    renderable_set = set(contract.renderable_region_ids)
    decision_renderable_set = {decision.region_id for decision in contract.decisions if decision.render_candidate_allowed}
    hard_blocked_set = {decision.region_id for decision in contract.decisions if decision.blocked}
    missing_decisions = renderable_set - decision_renderable_set
    if missing_decisions:
        raise RegionRoutingValidationError("renderable_without_decision", "every renderable region id must have a render_candidate_allowed decision")
    expected_renderable = tuple(dict.fromkeys(decision.region_id for decision in contract.decisions if decision.render_candidate_allowed and decision.region_id not in hard_blocked_set))
    if tuple(contract.renderable_region_ids) != expected_renderable:
        raise RegionRoutingValidationError("renderable_region_ids_mismatch", "renderable_region_ids must match render_candidate_allowed decisions after hard blocks")
    expected_blocked = tuple(dict.fromkeys(decision.region_id for decision in contract.decisions if decision.blocked))
    if tuple(contract.blocked_region_ids) != expected_blocked:
        raise RegionRoutingValidationError("blocked_region_ids_mismatch", "blocked_region_ids must contain only hard blocked decisions")
    diagnostic_only_ids = {decision.region_id for decision in contract.decisions if decision.diagnostic_only and not decision.blocked}
    if any(region_id in diagnostic_only_ids and region_id not in decision_renderable_set for region_id in contract.renderable_region_ids):
        raise RegionRoutingValidationError("diagnostic_only_renderable", "diagnostic-only regions cannot be renderable without a separate renderable decision")
    if hard_blocked_set & renderable_set:
        raise RegionRoutingValidationError("hard_blocked_renderable", "hard-blocked regions cannot be renderable")
    return contract
