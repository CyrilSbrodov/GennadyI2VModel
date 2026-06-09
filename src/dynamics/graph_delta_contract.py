from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from core.body_ontology import BODY_ONTOLOGY, BodyRegionGroup, is_canonical_body_region
from core.region_ids import is_known_canonical_region_type, make_region_id, parse_region_id
from planning.action_plan import ActionPlan, ActionType, RegionPlanTarget, validate_action_plan


class DynamicsValidationError(ValueError):
    """Raised when Dynamics cannot produce a valid Sprint-5 GraphDelta contract."""

    def __init__(self, code: str, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details or {}


class RegionDeltaType(str, Enum):
    POSE_DELTA = "pose_delta"
    EXPRESSION_DELTA = "expression_delta"
    VISIBILITY_DELTA = "visibility_delta"
    OCCLUSION_DELTA = "occlusion_delta"
    INTERACTION_DELTA = "interaction_delta"
    GARMENT_INTENT_DELTA = "garment_intent_delta"
    SECONDARY_MOTION_HINT = "secondary_motion_hint"


SUPPORTED_REGION_DELTA_TYPES: frozenset[str] = frozenset(item.value for item in RegionDeltaType)
IDENTITY_LOCK_REGION_IDS: frozenset[str] = frozenset({"face", "head", "hair", "scalp"})
FUTURE_FACE_SUBREGIONS: frozenset[str] = frozenset({"forehead", "left_eye", "right_eye", "nose", "mouth", "lips", "jaw", "chin", "left_ear", "right_ear"})
ALLOWED_GARMENT_REGIONS: frozenset[str] = frozenset({"upper_garment", "lower_garment", "outer_garment", "inner_garment", "garments", "sleeves", "accessories"})
ALLOWED_OBJECT_REGIONS: frozenset[str] = frozenset({"reachable_object", "held_object", "support_object"})
FORBIDDEN_PLANNER_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "render_mode",
        "renderer",
        "rendered_frame",
        "rgb_patch",
        "alpha_mask",
        "composited_frame",
        "final_pose_coordinates",
        "observed_region_created",
        "observation_status",
        "mask_evidence_type",
        "generated_observed_region",
    }
)


@dataclass(frozen=True, slots=True)
class PoseDeltaIntent:
    intent_kind: str = "intent_only_pose_change"
    target_state_hint: str = "unresolved_by_dynamics"
    no_final_coordinates: bool = True


@dataclass(frozen=True, slots=True)
class VisibilityDeltaIntent:
    intent_kind: str = "visibility_reasoning_requirement"
    expected_change: str = "may_change_visibility"
    no_visibility_state_claim: bool = True


@dataclass(frozen=True, slots=True)
class OcclusionDeltaIntent:
    intent_kind: str = "occlusion_reasoning_requirement"
    expected_change: str = "occlusion_reasoning_required"
    no_occlusion_mask_claim: bool = True


@dataclass(frozen=True, slots=True)
class InteractionDeltaIntent:
    intent_kind: str = "interaction_intent_only"
    target_role: str = "unresolved"
    no_contact_solution: bool = True


@dataclass(frozen=True, slots=True)
class GarmentIntentDelta:
    intent_kind: str = "garment_interaction_intent_only"
    may_require_future_reveal_reasoning: bool = False
    removes_clothing: bool = False
    reveals_hidden_anatomy: bool = False
    synthesizes_underlying_content: bool = False


@dataclass(frozen=True, slots=True)
class RoutingCandidate:
    region_id: str
    canonical_region_id: str
    reason: str
    delta_type: str
    action_type: str
    phase_id: str
    requires_rendering_candidate: bool


@dataclass(frozen=True, slots=True)
class DynamicsTrace:
    dynamics_source: str = "sprint5_graph_delta_contract"
    provenance: str = "planner_to_dynamics_contract_builder"
    support_level: str = "supported"
    reasons: tuple[str, ...] = ()
    unsupported_planner_code: str | None = None
    unsupported_planner_reasons: tuple[str, ...] = ()
    unsupported_planner_fragments: tuple[str, ...] = ()
    planner_trace_reasons: tuple[str, ...] = ()
    forbidden_operations_asserted_absent: tuple[str, ...] = (
        "rendering",
        "compositing",
        "renderer_call",
        "region_routing_call",
        "memory_write",
        "observed_perception_evidence_creation",
        "pixel_generation",
        "learned_video_generation",
        "physics_simulation",
        "private_anatomical_rendering",
    )


@dataclass(frozen=True, slots=True)
class RegionDelta:
    entity_id: str
    canonical_region_id: str
    region_id: str
    delta_type: RegionDeltaType | str
    action_type: ActionType | str
    phase_id: str
    phase_type: str
    region_role: str
    expected_motion_role: str
    confidence: float
    provenance: str
    source_planner_trace: tuple[str, ...]
    identity_locked: bool
    protected_region: bool
    requires_routing: bool
    requires_rendering_candidate: bool
    reveal_may_be_required: bool
    occlusion_reasoning_required: bool
    secondary_motion_required: bool
    private_or_optional_region: bool
    allowed_to_modify_geometry: bool
    allowed_to_modify_visibility: bool
    allowed_to_modify_identity: bool
    validation_reasons: tuple[str, ...]
    pose_intent: PoseDeltaIntent | None = None
    visibility_intent: VisibilityDeltaIntent | None = None
    occlusion_intent: OcclusionDeltaIntent | None = None
    interaction_intent: InteractionDeltaIntent | None = None
    garment_intent: GarmentIntentDelta | None = None
    rendered_pixels_generated: bool = False
    observed_evidence_created: bool = False
    memory_write_performed: bool = False
    region_routing_called: bool = False


@dataclass(frozen=True, slots=True)
class GraphDeltaStep:
    step_id: str
    action_order: int
    phase_order: int
    action_type: ActionType | str
    phase_id: str
    phase_type: str
    region_deltas: tuple[RegionDelta, ...]
    provenance: str = "sprint5_graph_delta_step"
    source_planner_trace: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphDeltaContract:
    contract_version: str = "graph_delta_contract_v1"
    supported: bool = True
    steps: tuple[GraphDeltaStep, ...] = ()
    identity_lock_regions: tuple[str, ...] = ()
    read_only_memory_requirements: tuple[dict[str, object], ...] = ()
    routing_candidates: tuple[RoutingCandidate, ...] = ()
    trace: DynamicsTrace = field(default_factory=DynamicsTrace)
    rendered_pixels_generated: bool = False
    observed_evidence_created: bool = False
    memory_write_performed: bool = False
    region_routing_called: bool = False
    scene_graph_mutation_performed: bool = False
    learned_motion_claimed: bool = False

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DynamicsHandoffResult:
    supported: bool
    graph_delta_contract: GraphDeltaContract
    routing_candidates: tuple[RoutingCandidate, ...]
    trace: DynamicsTrace

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


# Backwards-compatible spelling for callers that use the requested concept name.
GraphDeltaHandoffResult = DynamicsHandoffResult


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


def _region_name(target: RegionPlanTarget | str) -> str:
    if isinstance(target, RegionPlanTarget):
        return target.canonical_region_id or target.region_id.rsplit(":", 1)[-1]
    return str(target).rsplit(":", 1)[-1]


def _target_entity(target: RegionPlanTarget, fallback: str) -> str:
    if target.entity_id:
        return target.entity_id
    if ":" in target.region_id:
        return parse_region_id(target.region_id)[0]
    return fallback


def _is_private_or_optional(region: str) -> bool:
    meta = BODY_ONTOLOGY.get(str(region or ""))
    if meta is None:
        return False
    return meta.memory_family == "private" or meta.group in {BodyRegionGroup.OPTIONAL_PRIVATE, BodyRegionGroup.OPTIONAL_SEX_SPECIFIC}


def _is_identity_locked(region: str, action_identity_locks: set[str]) -> bool:
    return region in action_identity_locks or region in IDENTITY_LOCK_REGION_IDS or (region in FUTURE_FACE_SUBREGIONS and "face" in action_identity_locks)


def _delta_types_for_target(action_type: str, target: RegionPlanTarget, secondary_motion_required: bool, required_dynamics_capabilities: tuple[str, ...]) -> tuple[str, ...]:
    region = _region_name(target)
    role = _value(target.role)
    deltas: list[str]
    if action_type == "expression_change":
        deltas = [RegionDeltaType.EXPRESSION_DELTA.value]
    elif action_type == "garment_adjust":
        if role == "garment" or region in ALLOWED_GARMENT_REGIONS:
            deltas = [RegionDeltaType.GARMENT_INTENT_DELTA.value]
        else:
            deltas = [RegionDeltaType.INTERACTION_DELTA.value]
    else:
        deltas = [RegionDeltaType.POSE_DELTA.value]

    if "visibility_update" in required_dynamics_capabilities:
        deltas.append(RegionDeltaType.VISIBILITY_DELTA.value)
    if "occlusion_update" in required_dynamics_capabilities:
        deltas.append(RegionDeltaType.OCCLUSION_DELTA.value)
    if secondary_motion_required and str(getattr(target, "memory_family", "")) == "soft_tissue":
        deltas.append(RegionDeltaType.SECONDARY_MOTION_HINT.value)
    return tuple(dict.fromkeys(deltas))


def _intent_payload(delta_type: str, *, phase_type: str, reveal_may_be_required: bool) -> dict[str, object]:
    if delta_type == RegionDeltaType.POSE_DELTA.value:
        return {"pose_intent": PoseDeltaIntent(target_state_hint=phase_type)}
    if delta_type == RegionDeltaType.VISIBILITY_DELTA.value:
        return {"visibility_intent": VisibilityDeltaIntent()}
    if delta_type == RegionDeltaType.OCCLUSION_DELTA.value:
        return {"occlusion_intent": OcclusionDeltaIntent()}
    if delta_type == RegionDeltaType.INTERACTION_DELTA.value:
        return {"interaction_intent": InteractionDeltaIntent(target_role=phase_type)}
    if delta_type == RegionDeltaType.GARMENT_INTENT_DELTA.value:
        return {"garment_intent": GarmentIntentDelta(may_require_future_reveal_reasoning=reveal_may_be_required)}
    return {}


def _action_order_from_trace(reasons: tuple[str, ...] | list[str], fallback: int) -> int:
    for reason in reasons:
        text = str(reason)
        if text.startswith("action_order:"):
            try:
                return int(text.rsplit(":", 1)[-1])
            except ValueError:
                return fallback
    return fallback


def build_dynamics_handoff(action_plan: ActionPlan) -> DynamicsHandoffResult:
    """Build and validate a Sprint-5 Dynamics/GraphDelta handoff from a Sprint-4 ActionPlan."""

    forbidden_keys = set(getattr(action_plan, "forbidden_outputs", {}) or {}) & FORBIDDEN_PLANNER_OUTPUT_KEYS
    if forbidden_keys:
        raise DynamicsValidationError("forbidden_planner_output", f"planner forbidden output cannot enter dynamics: {sorted(forbidden_keys)}")

    if not isinstance(action_plan, ActionPlan):
        raise DynamicsValidationError("invalid_action_plan", "dynamics input must be planning.action_plan.ActionPlan")

    if not action_plan.supported:
        trace = DynamicsTrace(
            support_level="unsupported_planner_input",
            reasons=("unsupported_planner_input_no_normal_graph_delta",),
            unsupported_planner_code=action_plan.unsupported_code,
            unsupported_planner_reasons=tuple(action_plan.unsupported_reasons),
            unsupported_planner_fragments=tuple(action_plan.unsupported_fragments),
            planner_trace_reasons=tuple(action_plan.trace.reasons),
        )
        contract = GraphDeltaContract(supported=False, steps=(), routing_candidates=(), trace=trace)
        validate_graph_delta_contract(contract, allow_unsupported=True)
        return DynamicsHandoffResult(supported=False, graph_delta_contract=contract, routing_candidates=(), trace=trace)

    try:
        validate_action_plan(action_plan)
    except Exception as exc:
        raise DynamicsValidationError("invalid_action_plan", str(exc)) from exc

    steps: list[GraphDeltaStep] = []
    all_identity_locks: list[str] = []
    memory_requirements: list[dict[str, object]] = []
    for action_index, action in enumerate(action_plan.actions):
        action_type = _value(action.action_type)
        action_order = _action_order_from_trace(action.trace.reasons, action_index)
        action_identity_locks = set(action.dynamics_requirement.identity_lock_regions)
        all_identity_locks.extend(sorted(action_identity_locks))
        for req in action.memory_requirements:
            memory_requirements.append({"action_order": action_order, "action_type": action_type, "family": req.family, "regions": list(req.regions), "required": req.required, "reason": req.reason})
        for phase_index, phase in enumerate(action.phases):
            region_deltas: list[RegionDelta] = []
            for target in phase.affected_regions:
                canonical_region = _region_name(target)
                entity_id = _target_entity(target, action.target_entity_id)
                identity_locked = _is_identity_locked(canonical_region, action_identity_locks)
                protected = identity_locked or _is_private_or_optional(canonical_region)
                private = _is_private_or_optional(canonical_region)
                if private:
                    raise DynamicsValidationError("private_auto_target_forbidden", f"private/sex-specific region cannot enter dynamics: {canonical_region}")
                for delta_type in _delta_types_for_target(action_type, target, action.dynamics_requirement.secondary_motion_required, tuple(action.dynamics_requirement.required_dynamics_capabilities)):
                    requires_rendering_candidate = delta_type != RegionDeltaType.SECONDARY_MOTION_HINT.value
                    region_delta = RegionDelta(
                        entity_id=entity_id,
                        canonical_region_id=canonical_region,
                        region_id=target.region_id or make_region_id(entity_id, canonical_region),
                        delta_type=delta_type,
                        action_type=action_type,
                        phase_id=phase.phase_id,
                        phase_type=phase.phase_type,
                        region_role=_value(target.role),
                        expected_motion_role=phase.expected_motion_role,
                        confidence=float(phase.confidence),
                        provenance="planner_to_dynamics_intent_translation",
                        source_planner_trace=tuple((*action.trace.reasons, *action_plan.trace.reasons)),
                        identity_locked=identity_locked,
                        protected_region=protected,
                        requires_routing=True,
                        requires_rendering_candidate=requires_rendering_candidate,
                        reveal_may_be_required=bool(action.dynamics_requirement.reveal_may_be_required and not private),
                        occlusion_reasoning_required=bool(action.dynamics_requirement.occlusion_reasoning_required),
                        secondary_motion_required=delta_type == RegionDeltaType.SECONDARY_MOTION_HINT.value,
                        private_or_optional_region=private,
                        allowed_to_modify_geometry=(not private and delta_type in {RegionDeltaType.POSE_DELTA.value, RegionDeltaType.INTERACTION_DELTA.value, RegionDeltaType.GARMENT_INTENT_DELTA.value}),
                        allowed_to_modify_visibility=(not private and delta_type in {RegionDeltaType.VISIBILITY_DELTA.value, RegionDeltaType.OCCLUSION_DELTA.value}),
                        allowed_to_modify_identity=False,
                        validation_reasons=("intent_only_no_pixels", "identity_not_modified" if identity_locked else "identity_unchanged"),
                        **_intent_payload(delta_type, phase_type=phase.phase_type, reveal_may_be_required=action.dynamics_requirement.reveal_may_be_required),
                    )
                    region_deltas.append(region_delta)
            steps.append(
                GraphDeltaStep(
                    step_id=f"action_{action_order}:phase_{phase_index}:{phase.phase_type}",
                    action_order=action_order,
                    phase_order=phase_index,
                    action_type=action_type,
                    phase_id=phase.phase_id,
                    phase_type=phase.phase_type,
                    region_deltas=tuple(region_deltas),
                    source_planner_trace=tuple((*action.trace.reasons, *action_plan.trace.reasons)),
                )
            )

    trace = DynamicsTrace(
        support_level="supported_with_partial_unsupported" if action_plan.unsupported_fragments else "supported",
        reasons=tuple((f"unsupported_intent_fragment:{f}" for f in action_plan.unsupported_fragments)),
        unsupported_planner_fragments=tuple(action_plan.unsupported_fragments),
        planner_trace_reasons=tuple(action_plan.trace.reasons),
    )
    routing_candidates = _routing_candidates_from_steps(steps)
    contract = GraphDeltaContract(
        supported=True,
        steps=tuple(steps),
        identity_lock_regions=tuple(dict.fromkeys(all_identity_locks)),
        read_only_memory_requirements=tuple(memory_requirements),
        routing_candidates=routing_candidates,
        trace=trace,
    )
    validate_graph_delta_contract(contract)
    return DynamicsHandoffResult(supported=True, graph_delta_contract=contract, routing_candidates=routing_candidates, trace=trace)


def _routing_candidates_from_steps(steps: list[GraphDeltaStep] | tuple[GraphDeltaStep, ...]) -> tuple[RoutingCandidate, ...]:
    candidates: list[RoutingCandidate] = []
    seen: set[tuple[str, str, str, str]] = set()
    for step in steps:
        for delta in step.region_deltas:
            if not delta.requires_routing:
                continue
            key = (delta.region_id, _value(delta.delta_type), _value(delta.action_type), delta.phase_id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                RoutingCandidate(
                    region_id=delta.region_id,
                    canonical_region_id=delta.canonical_region_id,
                    reason="requires_future_region_routing_after_dynamics",
                    delta_type=_value(delta.delta_type),
                    action_type=_value(delta.action_type),
                    phase_id=delta.phase_id,
                    requires_rendering_candidate=bool(delta.requires_rendering_candidate),
                )
            )
    return tuple(candidates)


def validate_region_delta(delta: RegionDelta) -> RegionDelta:
    if not delta.entity_id or not delta.region_id or not delta.canonical_region_id:
        raise DynamicsValidationError("missing_region_provenance", "RegionDelta requires entity_id, region_id, and canonical_region_id")
    if not delta.phase_id:
        raise DynamicsValidationError("missing_phase_id", "RegionDelta requires phase_id")
    if not _value(delta.action_type):
        raise DynamicsValidationError("missing_action_type", "RegionDelta requires action_type")
    if _value(delta.delta_type) not in SUPPORTED_REGION_DELTA_TYPES:
        raise DynamicsValidationError("unknown_delta_type", f"unknown delta_type={delta.delta_type!r}")
    region = delta.canonical_region_id
    known_region = region in ALLOWED_GARMENT_REGIONS or region in ALLOWED_OBJECT_REGIONS or is_canonical_body_region(region) or is_known_canonical_region_type(region)
    if not known_region:
        raise DynamicsValidationError("unknown_region", f"unknown dynamics region={region!r}")
    if _is_private_or_optional(region) or delta.private_or_optional_region:
        raise DynamicsValidationError("private_auto_target_forbidden", f"private/sex-specific region cannot be a dynamics target: {region}")
    if delta.identity_locked and delta.allowed_to_modify_identity:
        raise DynamicsValidationError("identity_modification_forbidden", f"identity-locked region cannot allow identity modification: {region}")
    if delta.rendered_pixels_generated:
        raise DynamicsValidationError("pixel_generation_forbidden", "dynamics cannot claim rendered/generated pixels")
    if delta.observed_evidence_created:
        raise DynamicsValidationError("observed_evidence_forbidden", "dynamics cannot create observed perception evidence")
    if delta.memory_write_performed:
        raise DynamicsValidationError("memory_write_forbidden", "dynamics cannot write memory")
    if delta.region_routing_called:
        raise DynamicsValidationError("region_routing_call_forbidden", "dynamics cannot call region routing")
    if delta.reveal_may_be_required and _is_private_or_optional(region):
        raise DynamicsValidationError("private_reveal_forbidden", "private/sex-specific regions cannot become reveal targets")
    if delta.requires_rendering_candidate and _is_private_or_optional(region):
        raise DynamicsValidationError("private_render_candidate_forbidden", "private/sex-specific regions cannot become render candidates")
    if _value(delta.delta_type) == RegionDeltaType.GARMENT_INTENT_DELTA.value and delta.garment_intent:
        if delta.garment_intent.removes_clothing or delta.garment_intent.reveals_hidden_anatomy or delta.garment_intent.synthesizes_underlying_content:
            raise DynamicsValidationError("garment_reveal_generation_forbidden", "garment_intent_delta cannot remove clothing or reveal/synthesize hidden anatomy")
    return delta


def validate_graph_delta_contract(contract: GraphDeltaContract, *, allow_unsupported: bool = False) -> GraphDeltaContract:
    if not isinstance(contract, GraphDeltaContract):
        raise DynamicsValidationError("invalid_contract", "expected GraphDeltaContract")
    if any((contract.rendered_pixels_generated, contract.observed_evidence_created, contract.memory_write_performed, contract.region_routing_called, contract.scene_graph_mutation_performed, contract.learned_motion_claimed)):
        raise DynamicsValidationError("forbidden_dynamics_claim", "dynamics contract claims a forbidden operation")
    if not contract.supported:
        if not allow_unsupported and not contract.trace.unsupported_planner_code and not contract.trace.unsupported_planner_reasons:
            raise DynamicsValidationError("unsupported_contract", "unsupported dynamics contract requires unsupported planner trace")
        if contract.steps:
            raise DynamicsValidationError("unsupported_contract_has_steps", "unsupported ActionPlan cannot produce normal GraphDelta steps")
        return contract
    if not contract.steps:
        raise DynamicsValidationError("empty_graph_delta", "supported GraphDeltaContract requires steps")
    previous_action = -1
    previous_phase_for_action: dict[int, int] = {}
    for step in contract.steps:
        if not step.phase_id or not _value(step.action_type):
            raise DynamicsValidationError("missing_step_provenance", "GraphDeltaStep requires phase_id and action_type")
        if step.action_order < previous_action:
            raise DynamicsValidationError("non_monotonic_action_order", "GraphDeltaStep action_order must be monotonic")
        previous_action = step.action_order
        prev_phase = previous_phase_for_action.get(step.action_order, -1)
        if step.phase_order <= prev_phase:
            raise DynamicsValidationError("non_monotonic_phase_order", "GraphDeltaStep phase_order must increase within each action")
        previous_phase_for_action[step.action_order] = step.phase_order
        if not step.region_deltas:
            raise DynamicsValidationError("empty_region_deltas", f"GraphDeltaStep {step.step_id} has no RegionDelta entries")
        for delta in step.region_deltas:
            validate_region_delta(delta)
            if _value(delta.action_type) != _value(step.action_type) or delta.phase_id != step.phase_id:
                raise DynamicsValidationError("delta_step_linkage_mismatch", "RegionDelta must preserve action/phase linkage")
    candidate_keys = {(c.region_id, c.delta_type, c.action_type, c.phase_id) for c in contract.routing_candidates}
    expected_keys = {
        (d.region_id, _value(d.delta_type), _value(d.action_type), d.phase_id)
        for step in contract.steps
        for d in step.region_deltas
        if d.requires_routing
    }
    if not expected_keys.issubset(candidate_keys):
        raise DynamicsValidationError("missing_routing_candidate", "requires_routing deltas must be exposed as routing candidates")
    return contract
