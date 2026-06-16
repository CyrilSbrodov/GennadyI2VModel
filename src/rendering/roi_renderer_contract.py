from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from core.body_ontology import BODY_ONTOLOGY, BodyRegionGroup
from runtime.region_routing_contract import RegionRoutingContract, RegionRoutingDecision, RegionRoutingDecisionType


class ROIRenderValidationError(ValueError):
    """Raised when the Sprint-8 ROI renderer contract is violated."""

    def __init__(self, code: str, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details or {}


class ROIRenderMode(str, Enum):
    PRESERVE = "preserve"
    WARP = "warp"
    DEFORM = "deform"
    REFINE = "refine"
    REVEAL_MEMORY = "reveal_memory"
    WEAK_MEMORY_REFINE = "weak_memory_refine"
    EXPRESSION_LOCAL_UPDATE = "expression_local_update"
    POSE_LOCAL_UPDATE = "pose_local_update"
    GARMENT_INTENT_UPDATE = "garment_intent_update"


_IDENTITY_REGIONS = frozenset({"face", "head", "hair", "scalp"})
_FORBIDDEN_OUTPUT_AUTHORITIES = frozenset({"authoritative", "reusable", "observed"})

_ROUTE_TO_MODES: dict[str, tuple[ROIRenderMode, ...]] = {
    RegionRoutingDecisionType.ROUTE_POSE_UPDATE.value: (ROIRenderMode.POSE_LOCAL_UPDATE, ROIRenderMode.DEFORM),
    RegionRoutingDecisionType.ROUTE_EXPRESSION_UPDATE.value: (ROIRenderMode.EXPRESSION_LOCAL_UPDATE,),
    RegionRoutingDecisionType.ROUTE_VISIBILITY_UPDATE.value: (ROIRenderMode.REFINE, ROIRenderMode.DEFORM),
    RegionRoutingDecisionType.ROUTE_INTERACTION_UPDATE.value: (ROIRenderMode.DEFORM, ROIRenderMode.REFINE),
    RegionRoutingDecisionType.ROUTE_GARMENT_INTENT.value: (ROIRenderMode.GARMENT_INTENT_UPDATE,),
    RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value: (ROIRenderMode.REVEAL_MEMORY,),
    RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value: (ROIRenderMode.WEAK_MEMORY_REFINE,),
    RegionRoutingDecisionType.ROUTE_PRESERVE_VISIBLE.value: (ROIRenderMode.PRESERVE, ROIRenderMode.REFINE),
    RegionRoutingDecisionType.ROUTE_MEMORY_ASSISTED_IDENTITY_LOCKED.value: (ROIRenderMode.REFINE, ROIRenderMode.PRESERVE),
}


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _value(value: object) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _private_or_optional(region: str, explicit: bool = False) -> bool:
    if explicit:
        return True
    meta = BODY_ONTOLOGY.get(str(region or ""))
    return bool(meta and (meta.memory_family == "private" or meta.group in {BodyRegionGroup.OPTIONAL_PRIVATE, BodyRegionGroup.OPTIONAL_SEX_SPECIFIC}))


def _identity_region(region: str, decision: RegionRoutingDecision | None = None) -> bool:
    return region in _IDENTITY_REGIONS or bool(decision and decision.identity_locked)


@dataclass(frozen=True, slots=True)
class ROIRenderSafetyPolicy:
    renderer_must_not_create_observed_evidence: bool = True
    renderer_must_not_create_identity_memory: bool = True
    memory_write_allowed: bool = False
    observed_evidence_claim_allowed: bool = False
    scene_graph_mutation_allowed: bool = False
    private_region_rendering_allowed: bool = False
    hidden_anatomy_generation_allowed: bool = False
    clothing_removal_allowed: bool = False

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ROIRenderIdentityPolicy:
    identity_locked: bool
    renderer_must_preserve_identity: bool
    allowed_to_modify_identity: bool = False
    identity_memory_write_allowed: bool = False
    identity_modified: bool = False
    identity_memory_created: bool = False

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ROIRenderMemoryUsage:
    memory_reference_kind: str = "none"
    memory_authority: str = "none"
    source_memory_authority: str = "none"
    memory_family: str = "none"
    weak_memory_candidate: bool = False
    memory_write_allowed: bool = False
    output_reference_authority: str = "none"

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ROIRenderReferenceUsage:
    reference_payload_summary: dict[str, object] = field(default_factory=dict)
    output_reference_authority: str = "none"
    reference_used: bool = False

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ROIRenderTrace:
    contract_version: str = "roi_renderer_contract_v1"
    provenance: tuple[str, ...] = ("region_routing_v2_to_roi_renderer",)
    route_validation_reasons: tuple[str, ...] = ()
    forbidden_operation_flags: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ROIRenderDecision:
    region_id: str
    canonical_region_id: str
    route_decision_type: str
    render_mode: ROIRenderMode | str
    output_authority: str

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ROIRenderRequest:
    region_id: str
    entity_id: str
    canonical_region_id: str
    route_decision_type: str
    source_delta_type: str | None
    source_reveal_decision_type: str | None
    action_type: str
    phase_id: str
    action_order: int
    phase_order: int
    roi_bbox: tuple[float, float, float, float] | None
    current_frame_shape: tuple[int, ...]
    input_frame_ref: str | None
    frame_index: int | None
    render_mode: ROIRenderMode | str
    identity_policy: ROIRenderIdentityPolicy
    memory_usage: ROIRenderMemoryUsage
    reference_usage: ROIRenderReferenceUsage
    safety_policy: ROIRenderSafetyPolicy
    route_validation_reasons: tuple[str, ...]
    source_route_decision: dict[str, object]
    trace: ROIRenderTrace

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]
    def to_training_metadata(self) -> dict[str, object]:
        return {
            "route_decision_type": self.route_decision_type,
            "render_mode": _value(self.render_mode),
            "canonical_region_id": self.canonical_region_id,
            "memory_family": self.memory_usage.memory_family,
            "memory_authority": self.memory_usage.memory_authority,
            "source_memory_authority": self.memory_usage.source_memory_authority,
            "output_authority": self.memory_usage.output_reference_authority,
            "identity_locked": self.identity_policy.identity_locked,
            "weak_memory_candidate": self.memory_usage.weak_memory_candidate,
            "source_delta_type": self.source_delta_type,
            "source_reveal_decision_type": self.source_reveal_decision_type,
            "action_type": self.action_type,
            "phase_id": self.phase_id,
            "safety_flags": self.safety_policy.as_dict(),
            "forbidden_operation_flags": self.trace.forbidden_operation_flags,
        }


@dataclass(frozen=True, slots=True)
class ROIRenderOutput:
    region_id: str
    canonical_region_id: str
    render_mode: ROIRenderMode | str
    route_decision_type: str
    patch_ref: str | None
    patch_shape: tuple[int, ...]
    alpha_mask_ref: str | None = None
    mask_info: dict[str, object] = field(default_factory=dict)
    output_authority: str = "generated"
    memory_write_allowed: bool = False
    observed_evidence_created: bool = False
    identity_memory_created: bool = False
    identity_modified: bool = False
    scene_graph_mutated: bool = False
    private_region_rendered: bool = False
    hidden_anatomy_generated: bool = False
    clothing_removed: bool = False
    source_route_decision: dict[str, object] = field(default_factory=dict)
    reference_usage: ROIRenderReferenceUsage = field(default_factory=ROIRenderReferenceUsage)
    identity_policy: ROIRenderIdentityPolicy | None = None
    safety_policy: ROIRenderSafetyPolicy = field(default_factory=ROIRenderSafetyPolicy)
    trace: ROIRenderTrace = field(default_factory=ROIRenderTrace)

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ROIRendererContract:
    contract_version: str = "roi_renderer_contract_v1"
    requests: tuple[ROIRenderRequest, ...] = ()
    outputs: tuple[ROIRenderOutput, ...] = ()
    renderable_region_ids: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]: return _jsonable(asdict(self))  # type: ignore[return-value]


def allowed_modes_for_route(route_decision_type: str) -> tuple[ROIRenderMode, ...]:
    return _ROUTE_TO_MODES.get(route_decision_type, ())


def default_mode_for_route(route_decision_type: str) -> ROIRenderMode:
    modes = allowed_modes_for_route(route_decision_type)
    if not modes:
        raise ROIRenderValidationError("unsupported_route_decision", "No ROI render mode is allowed for this route decision", details={"route_decision_type": route_decision_type})
    return modes[0]


def validate_roi_render_request(request: ROIRenderRequest, *, route_decision: RegionRoutingDecision, routing_contract: RegionRoutingContract) -> ROIRenderRequest:
    dt = _value(route_decision.decision_type)
    if route_decision.region_id not in routing_contract.renderable_region_ids:
        raise ROIRenderValidationError("region_not_renderable", "Region is absent from region_routing_contract.renderable_region_ids")
    if not route_decision.route_allowed or not route_decision.render_candidate_allowed or route_decision.blocked or route_decision.diagnostic_only or not route_decision.roi_required:
        raise ROIRenderValidationError("unsafe_route_decision", "Route decision is blocked, diagnostic, non-renderable, or does not require ROI rendering")
    if _private_or_optional(route_decision.canonical_region_id, route_decision.private_or_optional_region):
        raise ROIRenderValidationError("private_optional_region", "Private or optional region cannot reach ROI renderer")
    if _value(request.render_mode) not in {_value(m) for m in allowed_modes_for_route(dt)}:
        raise ROIRenderValidationError("invalid_render_mode", "Renderer mode is not allowed for route decision", details={"mode": _value(request.render_mode), "route": dt})
    if _identity_region(route_decision.canonical_region_id, route_decision):
        if not request.identity_policy.identity_locked or request.identity_policy.allowed_to_modify_identity or not request.identity_policy.renderer_must_preserve_identity:
            raise ROIRenderValidationError("identity_policy_violation", "Identity-sensitive rendering must be locked and preserve identity")
        if dt == RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value:
            raise ROIRenderValidationError("weak_identity_reveal_forbidden", "Weak identity reveal cannot be rendered")
        if dt == RegionRoutingDecisionType.ROUTE_REVEAL_OBSERVED_MEMORY.value and request.memory_usage.memory_authority != "authoritative":
            raise ROIRenderValidationError("identity_authoritative_memory_required", "Observed identity reveal requires authoritative identity memory")
    if request.memory_usage.output_reference_authority in _FORBIDDEN_OUTPUT_AUTHORITIES:
        raise ROIRenderValidationError("renderer_output_authority_forbidden", "ROI renderer output cannot claim observed/reusable/authoritative authority")
    return request


def build_roi_render_request(*, route_decision: RegionRoutingDecision, routing_contract: RegionRoutingContract, roi_bbox: Sequence[float] | None, current_frame_shape: Sequence[int], input_frame_ref: str | None = None, frame_index: int | None = None, reference_payload_summary: Mapping[str, object] | None = None, render_mode: ROIRenderMode | str | None = None) -> ROIRenderRequest:
    dt = _value(route_decision.decision_type)
    mode = render_mode or default_mode_for_route(dt)
    identity = _identity_region(route_decision.canonical_region_id, route_decision)
    output_authority = "weak" if dt == RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value else ("generated_from_memory" if route_decision.requires_reveal_memory or route_decision.memory_authority not in {"", "none"} else "generated")
    safety = ROIRenderSafetyPolicy(
        renderer_must_not_create_observed_evidence=bool(route_decision.renderer_must_not_create_observed_evidence),
        renderer_must_not_create_identity_memory=bool(route_decision.renderer_must_not_create_identity_memory),
        memory_write_allowed=False,
        observed_evidence_claim_allowed=False,
    )
    request = ROIRenderRequest(
        region_id=route_decision.region_id,
        entity_id=route_decision.entity_id,
        canonical_region_id=route_decision.canonical_region_id,
        route_decision_type=dt,
        source_delta_type=route_decision.source_delta_type,
        source_reveal_decision_type=route_decision.source_reveal_decision_type,
        action_type=route_decision.action_type,
        phase_id=route_decision.phase_id,
        action_order=int(route_decision.action_order),
        phase_order=int(route_decision.phase_order),
        roi_bbox=tuple(float(v) for v in roi_bbox) if roi_bbox is not None else None,  # type: ignore[arg-type]
        current_frame_shape=tuple(int(v) for v in current_frame_shape),
        input_frame_ref=input_frame_ref,
        frame_index=frame_index,
        render_mode=mode,
        identity_policy=ROIRenderIdentityPolicy(identity_locked=bool(route_decision.identity_locked or identity), renderer_must_preserve_identity=bool(route_decision.renderer_must_preserve_identity or identity), allowed_to_modify_identity=False),
        memory_usage=ROIRenderMemoryUsage(route_decision.memory_reference_kind, route_decision.memory_authority, route_decision.memory_authority, route_decision.memory_family, bool(route_decision.weak_memory_candidate), False, output_authority),
        reference_usage=ROIRenderReferenceUsage(dict(reference_payload_summary or {}), output_authority, bool(reference_payload_summary)),
        safety_policy=safety,
        route_validation_reasons=tuple(route_decision.validation_reasons),
        source_route_decision=route_decision.as_dict(),
        trace=ROIRenderTrace(route_validation_reasons=tuple(route_decision.validation_reasons), forbidden_operation_flags={"observed_evidence_created": False, "identity_memory_created": False, "memory_write_performed": False, "scene_graph_mutated": False, "private_region_rendered": False, "hidden_anatomy_generated": False, "clothing_removed": False}),
    )
    return validate_roi_render_request(request, route_decision=route_decision, routing_contract=routing_contract)


def validate_roi_render_output(output: ROIRenderOutput, *, request: ROIRenderRequest | None = None) -> ROIRenderOutput:
    violations = {
        "observed_evidence_created": output.observed_evidence_created,
        "identity_memory_created": output.identity_memory_created,
        "memory_write_allowed": output.memory_write_allowed,
        "identity_modified": output.identity_modified,
        "scene_graph_mutated": output.scene_graph_mutated,
        "private_region_rendered": output.private_region_rendered,
        "hidden_anatomy_generated": output.hidden_anatomy_generated,
        "clothing_removed": output.clothing_removed,
    }
    bad = [k for k, v in violations.items() if bool(v)]
    if bad:
        raise ROIRenderValidationError("unsafe_render_output", "ROI render output claims forbidden operation", details={"violations": bad})
    if output.output_authority in _FORBIDDEN_OUTPUT_AUTHORITIES:
        raise ROIRenderValidationError("renderer_output_authority_forbidden", "ROI renderer output cannot claim observed/reusable/authoritative authority")
    if request is not None:
        if output.region_id != request.region_id or output.route_decision_type != request.route_decision_type:
            raise ROIRenderValidationError("output_request_mismatch", "Output does not match ROI render request")
        if _value(output.render_mode) not in {_value(m) for m in allowed_modes_for_route(request.route_decision_type)}:
            raise ROIRenderValidationError("invalid_render_mode", "Output render mode is not allowed for route decision")
        if request.memory_usage.weak_memory_candidate and output.output_authority != "weak":
            raise ROIRenderValidationError("weak_memory_authority_escalation", "Weak memory render output must remain weak")
    return output


def wrap_roi_render_output(*, request: ROIRenderRequest, patch_ref: str | None, patch_shape: Sequence[int], alpha_mask_ref: str | None = None, mask_info: Mapping[str, object] | None = None) -> ROIRenderOutput:
    authority = "weak" if request.route_decision_type == RegionRoutingDecisionType.ROUTE_REVEAL_WEAK_MEMORY.value else ("generated_from_memory" if request.memory_usage.memory_authority not in {"", "none"} else "generated")
    out = ROIRenderOutput(request.region_id, request.canonical_region_id, request.render_mode, request.route_decision_type, patch_ref, tuple(int(v) for v in patch_shape), alpha_mask_ref, dict(mask_info or {}), authority, False, False, False, False, False, False, False, False, request.source_route_decision, request.reference_usage, request.identity_policy, request.safety_policy, request.trace)
    return validate_roi_render_output(out, request=request)
