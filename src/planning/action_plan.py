from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from core.body_ontology import BODY_ONTOLOGY, BodyRegionGroup, is_canonical_body_region
from core.reference_families import reference_family_for_region
from core.schema import SceneGraph


class PlannerValidationError(ValueError):
    """Raised when an intent cannot produce a valid Sprint-4 planner contract."""

    def __init__(self, code: str, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details or {}


class ActionType(str, Enum):
    EXPRESSION_CHANGE = "expression_change"
    GAZE_SHIFT = "gaze_shift"
    HEAD_TURN = "head_turn"
    TORSO_TURN = "torso_turn"
    ARM_RAISE = "arm_raise"
    ARM_LOWER = "arm_lower"
    HAND_REACH = "hand_reach"
    SIT_DOWN = "sit_down"
    STAND_UP = "stand_up"
    STEP_FORWARD = "step_forward"
    GARMENT_ADJUST = "garment_adjust"
    OBJECT_REACH = "object_reach"


class PlannerRegionRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    STABILIZER = "stabilizer"
    IDENTITY_LOCK = "identity_lock"
    GARMENT = "garment"
    OBJECT = "object"
    FUTURE_REVEAL_CONTEXT = "future_reveal_context"


class PlannerVisibilityExpectation(str, Enum):
    NO_VISIBILITY_DECISION = "no_visibility_decision"
    MAY_CHANGE_VISIBILITY = "may_change_visibility"
    PRESERVE_IDENTITY_VISIBILITY = "preserve_identity_visibility"
    FUTURE_REVEAL_MAY_BE_REQUIRED = "future_reveal_may_be_required"


class PlannerOcclusionExpectation(str, Enum):
    NO_OCCLUSION_DECISION = "no_occlusion_decision"
    OCCLUSION_REASONING_REQUIRED = "occlusion_reasoning_required"
    MAY_SELF_OCCLUDE = "may_self_occlude"
    MAY_GARMENT_OCCLUDE = "may_garment_occlude"
    MAY_OBJECT_OCCLUDE = "may_object_occlude"


VALID_MEMORY_FAMILIES = frozenset({"identity", "skin", "body_shape", "soft_tissue", "garment", "accessory"})
VALID_DYNAMICS_HINTS = frozenset({"pose_update", "expression_update", "visibility_update", "occlusion_update", "interaction_intent", "secondary_motion_required"})
VALID_ROUTING_HINTS = frozenset({"route_after_dynamics", "preserve_identity_regions", "garment_route_context", "object_route_context", "no_rendering_decision"})
VALID_GRAPH_DELTA_TYPES = frozenset({"pose_delta", "expression_delta", "visibility_delta", "occlusion_delta", "interaction_delta", "garment_intent_delta"})


def _enum_value(value: object) -> str:
    return value.value if isinstance(value, Enum) else str(value)


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
_ALLOWED_GARMENT_REGIONS = frozenset({"upper_garment", "lower_garment", "outer_garment", "inner_garment", "garments", "sleeves"})
_ALLOWED_OBJECT_REGIONS = frozenset({"reachable_object", "held_object", "support_object"})
_FORBIDDEN_RENDER_KEYS = frozenset({"render_mode", "renderer", "rendered_frame", "rgb_patch", "alpha_mask", "composited_frame", "final_pose_coordinates"})
_FORBIDDEN_EVIDENCE_KEYS = frozenset({"observed_region_created", "observation_status", "mask_evidence_type", "generated_observed_region"})
_SEGMENT_SPLIT_RE = re.compile(r"[.,;]+|(?<!\w)and(?!\w)|(?<!\w)и(?!\w)")
_HARMLESS_UNMATCHED_TOKENS = frozenset({
    "the",
    "a",
    "an",
    "to",
    "on",
    "at",
    "with",
    "then",
    "and",
    "на",
    "в",
    "и",
    "затем",
    "потом",
    "стул",
    "стуле",
})


@dataclass(frozen=True, slots=True)
class PlannerTrace:
    planner_source: str = "controlled_phrase_contract"
    provenance: str = "sprint4_action_decomposition"
    support_level: str = "supported"
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PlannerIntent:
    raw_text: str = ""
    action_type: ActionType | str | None = None
    target_entity_id: str | None = None
    side: str | None = None
    target_object_id: str | None = None
    strict: bool = True


@dataclass(frozen=True, slots=True)
class RegionPlanTarget:
    region_id: str
    role: PlannerRegionRole | str
    entity_id: str | None = None
    canonical_region_id: str | None = None
    memory_family: str = "unknown"


@dataclass(frozen=True, slots=True)
class PlannerMemoryRequirement:
    family: str
    regions: tuple[str, ...]
    required: bool = True
    reason: str = "planner_contract_reference_requirement"


@dataclass(frozen=True, slots=True)
class PlannerDynamicsRequirement:
    expected_graph_delta_types: tuple[str, ...]
    required_dynamics_capabilities: tuple[str, ...]
    target_regions: tuple[str, ...]
    secondary_motion_required: bool = False
    reveal_may_be_required: bool = False
    occlusion_reasoning_required: bool = False
    identity_lock_regions: tuple[str, ...] = ()
    produces_graph_delta: bool = False


@dataclass(frozen=True, slots=True)
class ActionPhase:
    phase_id: str
    phase_type: str
    normalized_start: float
    normalized_end: float
    affected_regions: tuple[RegionPlanTarget, ...]
    expected_motion_role: str
    expected_visibility_changes: tuple[PlannerVisibilityExpectation | str, ...]
    expected_occlusion_changes: tuple[PlannerOcclusionExpectation | str, ...]
    required_memory_families: tuple[str, ...]
    dynamics_hint: str
    routing_hint: str
    confidence: float
    planner_source: str
    support_level: str = "supported"
    provenance: str = "sprint4_action_decomposition"


@dataclass(frozen=True, slots=True)
class PlannedAction:
    action_type: ActionType | str
    target_entity_id: str
    phases: tuple[ActionPhase, ...]
    region_targets: tuple[RegionPlanTarget, ...]
    memory_requirements: tuple[PlannerMemoryRequirement, ...]
    dynamics_requirement: PlannerDynamicsRequirement
    trace: PlannerTrace
    side: str | None = None
    target_object_id: str | None = None


@dataclass(frozen=True, slots=True)
class ActionPlan:
    intent: PlannerIntent
    actions: tuple[PlannedAction, ...] = ()
    supported: bool = True
    unsupported_code: str | None = None
    unsupported_reasons: tuple[str, ...] = ()
    unsupported_fragments: tuple[str, ...] = ()
    trace: PlannerTrace = field(default_factory=PlannerTrace)
    planner_contract_version: str = "planner_action_plan_v1"
    forbidden_outputs: dict[str, object] = field(default_factory=dict)
    scene_graph_mutation_performed: bool = False

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _ActionSpec:
    phases: tuple[str, ...]
    regions: tuple[str, ...]
    role_by_region: dict[str, PlannerRegionRole]
    memory_regions: dict[str, tuple[str, ...]]
    graph_delta_types: tuple[str, ...]
    dynamics_capabilities: tuple[str, ...]
    dynamics_hint: str
    routing_hint: str
    identity_lock_regions: tuple[str, ...] = ()
    secondary_motion_required: bool = False
    reveal_may_be_required: bool = False
    occlusion_reasoning_required: bool = False
    target_objects: bool = False


def _side_regions(side: str | None, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    if side in {"left", "right"}:
        return tuple(f"{side}_{suffix}" for suffix in suffixes)
    return tuple(f"{s}_{suffix}" for s in ("left", "right") for suffix in suffixes)


def _identity_targets() -> tuple[str, ...]:
    return ("head", "face", "hair", "scalp", "neck")


class ActionPlanner:
    """Strict Sprint-4 planner: text/structured intent -> validated action decomposition only."""

    _explicit_phrases: tuple[tuple[tuple[str, ...], ActionType], ...] = (
        (("turn head", "head turn", "поворачивает голову", "повернуть голову"), ActionType.HEAD_TURN),
        (("look left", "look right", "look up", "look down", "shift gaze", "смотрит", "посмотреть"), ActionType.GAZE_SHIFT),
        (("raise left arm", "raise right arm", "raise arm", "поднимает руку", "поднять руку"), ActionType.ARM_RAISE),
        (("lower left arm", "lower right arm", "lower arm", "опускает руку", "опустить руку"), ActionType.ARM_LOWER),
        (("reach with hand", "hand reach", "reach hand", "тянет руку"), ActionType.HAND_REACH),
        (("turn torso", "torso turn", "rotate torso", "поворачивает корпус"), ActionType.TORSO_TURN),
        (("sit down", "sits down", "садится", "сесть"), ActionType.SIT_DOWN),
        (("stand up", "stands up", "встает", "встаёт", "встать"), ActionType.STAND_UP),
        (("step forward", "steps forward", "шаг вперед", "шаг вперёд"), ActionType.STEP_FORWARD),
        (("change expression", "smile", "frown", "улыбается", "улыба"), ActionType.EXPRESSION_CHANGE),
        (("adjust garment", "adjust clothes", "поправляет одежду", "поправ"), ActionType.GARMENT_ADJUST),
        (("reach object", "reach for object", "object reach", "дотянуться до объекта"), ActionType.OBJECT_REACH),
    )

    def plan(self, intent: PlannerIntent | str, scene_graph: SceneGraph | None = None, *, strict: bool | None = None) -> ActionPlan:
        planner_intent = intent if isinstance(intent, PlannerIntent) else PlannerIntent(raw_text=str(intent or ""))
        if strict is not None:
            planner_intent = PlannerIntent(
                raw_text=planner_intent.raw_text,
                action_type=planner_intent.action_type,
                target_entity_id=planner_intent.target_entity_id,
                side=planner_intent.side,
                target_object_id=planner_intent.target_object_id,
                strict=strict,
            )
        action_sequence, parse_reasons, unsupported_fragments = self.resolve_action_sequence(planner_intent)
        if not action_sequence:
            return self._unsupported(planner_intent, "unsupported_action", parse_reasons or ["no controlled phrase or explicit supported action_type matched"])
        if unsupported_fragments and planner_intent.strict:
            raise PlannerValidationError(
                "partial_unsupported_action",
                "; ".join(unsupported_fragments),
                details={"raw_text": planner_intent.raw_text, "unsupported_fragments": list(unsupported_fragments)},
            )
        side = planner_intent.side or self._detect_side(planner_intent.raw_text)
        entity_id = self._resolve_entity(planner_intent, scene_graph)
        entity_reason = "entity_resolved_explicitly" if planner_intent.target_entity_id else "entity_resolved_single_person_default"
        actions: list[PlannedAction] = []
        for action_order, (action_type, action_reason) in enumerate(action_sequence):
            spec = self._spec_for(action_type, side)
            region_targets = tuple(self._make_target(entity_id, region, spec.role_by_region.get(region, PlannerRegionRole.PRIMARY)) for region in spec.regions)
            phases = self._build_phases(action_type, spec, region_targets)
            memory_reqs = tuple(
                PlannerMemoryRequirement(family=family, regions=regions, required=family in {"identity", "garment"}, reason=f"{action_type.value}:{family}_reference")
                for family, regions in spec.memory_regions.items()
                if regions
            )
            dyn = PlannerDynamicsRequirement(
                expected_graph_delta_types=spec.graph_delta_types,
                required_dynamics_capabilities=spec.dynamics_capabilities,
                target_regions=tuple(dict.fromkeys(spec.regions)),
                secondary_motion_required=spec.secondary_motion_required,
                reveal_may_be_required=spec.reveal_may_be_required,
                occlusion_reasoning_required=spec.occlusion_reasoning_required,
                identity_lock_regions=spec.identity_lock_regions,
                produces_graph_delta=False,
            )
            action_trace = PlannerTrace(reasons=[action_reason, f"action_order:{action_order}", entity_reason])
            actions.append(
                PlannedAction(
                    action_type=action_type,
                    target_entity_id=entity_id,
                    phases=phases,
                    region_targets=region_targets,
                    memory_requirements=memory_reqs,
                    dynamics_requirement=dyn,
                    trace=action_trace,
                    side=side,
                    target_object_id=planner_intent.target_object_id,
                )
            )
        unsupported_reasons = [f"unsupported_intent_fragment:{fragment}" for fragment in unsupported_fragments]
        trace = PlannerTrace(reasons=parse_reasons + unsupported_reasons + [entity_reason, f"action_count:{len(actions)}"])
        plan = ActionPlan(
            intent=planner_intent,
            actions=tuple(actions),
            unsupported_fragments=tuple(unsupported_fragments),
            trace=trace,
        )
        validate_action_plan(plan)
        return plan

    def _unsupported(self, intent: PlannerIntent, code: str, reasons: list[str]) -> ActionPlan:
        trace = PlannerTrace(support_level="unsupported", reasons=reasons)
        plan = ActionPlan(intent=intent, actions=(), supported=False, unsupported_code=code, unsupported_reasons=tuple(reasons), trace=trace)
        if intent.strict:
            raise PlannerValidationError(code, "; ".join(reasons), details={"raw_text": intent.raw_text})
        return plan

    def resolve_action_sequence(self, intent: PlannerIntent) -> tuple[list[tuple[ActionType, str]], list[str], tuple[str, ...]]:
        if intent.action_type is not None:
            try:
                action_type = ActionType(_enum_value(intent.action_type))
            except ValueError:
                return [], [f"unknown explicit action_type={intent.action_type!r}"], ()
            return [(action_type, "explicit_action_type")], ["explicit_action_type"], ()

        text = self._normalize(intent.raw_text)
        raw_matches: list[tuple[int, int, ActionType, str]] = []
        for phrases, action in self._explicit_phrases:
            for phrase in phrases:
                start = text.find(phrase)
                while start >= 0:
                    end = start + len(phrase)
                    raw_matches.append((start, end, action, phrase))
                    start = text.find(phrase, start + 1)

        if not raw_matches:
            return [], ["unsupported_action_text"], ()

        raw_matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2].value))
        for idx, current in enumerate(raw_matches):
            c_start, c_end, c_action, c_phrase = current
            for n_start, n_end, n_action, n_phrase in raw_matches[idx + 1 :]:
                if n_start > c_start:
                    break
                if n_start == c_start and n_action != c_action:
                    return [], [f"ambiguous_action:same_span:{c_phrase!r}|{n_phrase!r}"], ()

        ordered: list[tuple[int, int, ActionType, str]] = []
        for start, end, action, phrase in raw_matches:
            if any(existing_action == action and not (end <= existing_start or start >= existing_end) for existing_start, existing_end, existing_action, _ in ordered):
                continue
            ordered.append((start, end, action, phrase))

        ordered.sort(key=lambda item: (item[0], item[1]))
        sequence = [(action, f"controlled_phrase_match:{action.value}:{phrase}@{start}") for start, _end, action, phrase in ordered]
        reasons = [reason for _action, reason in sequence]
        unsupported_fragments = self._unsupported_fragments_from_unmatched_text(text, tuple((start, end) for start, end, _action, _phrase in ordered))
        return sequence, reasons, unsupported_fragments

    def _resolve_action_type(self, intent: PlannerIntent) -> tuple[ActionType | None, list[str]]:
        sequence, reasons, _unsupported_fragments = self.resolve_action_sequence(intent)
        if not sequence:
            return None, reasons
        if len(sequence) > 1:
            return None, [f"ambiguous_action:{[action.value for action, _reason in sequence]}"]
        return sequence[0][0], reasons

    @staticmethod
    def _unsupported_fragments_from_unmatched_text(text: str, matched_spans: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
        if not text or not matched_spans:
            return ()
        trim_chars = " \t\n\r:!?()[]{}'\"-—–"
        chars = list(text)
        for start, end in matched_spans:
            for idx in range(max(0, start), min(len(chars), end)):
                chars[idx] = " "
        unmatched = "".join(chars)
        fragments: list[str] = []
        for segment in _SEGMENT_SPLIT_RE.split(unmatched):
            cleaned = segment.strip(trim_chars)
            if not cleaned:
                continue
            tokens = tuple(token.strip(trim_chars) for token in cleaned.split())
            tokens = tuple(token for token in tokens if token)
            if not tokens:
                continue
            if all(token in _HARMLESS_UNMATCHED_TOKENS for token in tokens):
                continue
            fragments.append(" ".join(tokens))
        return tuple(dict.fromkeys(fragments))

    def _resolve_entity(self, intent: PlannerIntent, scene_graph: SceneGraph | None) -> str:
        if scene_graph is None:
            if intent.target_entity_id:
                return intent.target_entity_id
            raise PlannerValidationError("missing_scene_graph", "entity resolution requires a SceneGraph unless target_entity_id is explicit")
        person_ids = [p.person_id for p in scene_graph.persons]
        if intent.target_entity_id:
            if intent.target_entity_id not in person_ids:
                raise PlannerValidationError("missing_target", f"target person {intent.target_entity_id!r} does not exist", details={"known_person_ids": person_ids})
            return intent.target_entity_id
        if len(person_ids) == 1:
            return person_ids[0]
        if not person_ids:
            raise PlannerValidationError("missing_target", "no person entity is available for planner target resolution")
        raise PlannerValidationError("ambiguous_target", "multi-person scene requires explicit target_entity_id", details={"known_person_ids": person_ids})

    def _build_phases(self, action_type: ActionType, spec: _ActionSpec, targets: tuple[RegionPlanTarget, ...]) -> tuple[ActionPhase, ...]:
        width = 1.0 / len(spec.phases)
        phases: list[ActionPhase] = []
        for idx, phase_type in enumerate(spec.phases):
            start = round(idx * width, 6)
            end = 1.0 if idx == len(spec.phases) - 1 else round((idx + 1) * width, 6)
            phases.append(
                ActionPhase(
                    phase_id=f"{action_type.value}:{idx}:{phase_type}",
                    phase_type=phase_type,
                    normalized_start=start,
                    normalized_end=end,
                    affected_regions=targets,
                    expected_motion_role=spec.dynamics_hint,
                    expected_visibility_changes=(PlannerVisibilityExpectation.MAY_CHANGE_VISIBILITY if "visibility_update" in spec.dynamics_capabilities else PlannerVisibilityExpectation.NO_VISIBILITY_DECISION,),
                    expected_occlusion_changes=(PlannerOcclusionExpectation.OCCLUSION_REASONING_REQUIRED if spec.occlusion_reasoning_required else PlannerOcclusionExpectation.NO_OCCLUSION_DECISION,),
                    required_memory_families=tuple(spec.memory_regions.keys()),
                    dynamics_hint=spec.dynamics_hint,
                    routing_hint=spec.routing_hint,
                    confidence=0.82,
                    planner_source="controlled_phrase_contract",
                )
            )
        return tuple(phases)

    def _make_target(self, entity_id: str, region: str, role: PlannerRegionRole) -> RegionPlanTarget:
        family = reference_family_for_region(region)
        return RegionPlanTarget(region_id=f"{entity_id}:{region}", canonical_region_id=region, entity_id=entity_id, role=role, memory_family=family)

    def _spec_for(self, action: ActionType, side: str | None) -> _ActionSpec:
        id_targets = _identity_targets()
        arm_regions = _side_regions(side, ("arm", "upper_arm", "elbow", "forearm", "wrist", "hand"))
        hand_regions = _side_regions(side, ("arm", "shoulder", "upper_arm", "elbow", "forearm", "wrist", "hand"))
        leg_regions = ("pelvis", "hips", "left_thigh", "right_thigh", "left_knee", "right_knee", "left_calf", "right_calf", "left_foot", "right_foot", "torso")
        specs = {
            ActionType.HEAD_TURN: _ActionSpec(("initial_hold", "rotate_head", "settle"), id_targets, {r: PlannerRegionRole.IDENTITY_LOCK for r in id_targets}, {"identity": ("face", "head", "hair", "scalp")}, ("pose_delta", "visibility_delta", "occlusion_delta"), ("pose_update", "visibility_update", "occlusion_update"), "pose_update", "preserve_identity_regions", identity_lock_regions=("head", "face", "hair", "scalp"), occlusion_reasoning_required=True),
            ActionType.GAZE_SHIFT: _ActionSpec(("initial_gaze", "eye_rotation", "gaze_settle"), ("face", "left_eye", "right_eye"), {"face": PlannerRegionRole.IDENTITY_LOCK, "left_eye": PlannerRegionRole.PRIMARY, "right_eye": PlannerRegionRole.PRIMARY}, {"identity": ("face",)}, ("pose_delta", "expression_delta"), ("pose_update", "expression_update"), "pose_update", "preserve_identity_regions", identity_lock_regions=("face",)),
            ActionType.EXPRESSION_CHANGE: _ActionSpec(("initial_expression", "expression_transition", "expression_hold"), ("face", "mouth", "lips", "jaw", "left_eye", "right_eye"), {"face": PlannerRegionRole.IDENTITY_LOCK}, {"identity": ("face",)}, ("expression_delta",), ("expression_update",), "expression_update", "preserve_identity_regions", identity_lock_regions=("face",)),
            ActionType.TORSO_TURN: _ActionSpec(("initial_hold", "torso_rotate", "pelvis_stabilize", "settle"), ("torso", "upper_torso", "lower_torso", "chest", "abdomen", "left_shoulder", "right_shoulder", "pelvis"), {"pelvis": PlannerRegionRole.STABILIZER}, {"body_shape": ("torso", "upper_torso", "lower_torso", "chest", "abdomen", "pelvis")}, ("pose_delta", "visibility_delta", "occlusion_delta"), ("pose_update", "visibility_update", "occlusion_update"), "pose_update", "route_after_dynamics", occlusion_reasoning_required=True),
            ActionType.ARM_RAISE: _ActionSpec(("initial_hold", "shoulder_lift", "elbow_adjust", "hand_settle"), arm_regions, {}, {"body_shape": tuple(r for r in arm_regions if "hand" not in r), "skin": tuple(r for r in arm_regions if "hand" in r or "wrist" in r)}, ("pose_delta", "visibility_delta", "occlusion_delta"), ("pose_update", "visibility_update", "occlusion_update"), "pose_update", "route_after_dynamics", occlusion_reasoning_required=True),
            ActionType.ARM_LOWER: _ActionSpec(("initial_hold", "shoulder_lower", "elbow_adjust", "hand_settle"), arm_regions, {}, {"body_shape": tuple(r for r in arm_regions if "hand" not in r), "skin": tuple(r for r in arm_regions if "hand" in r or "wrist" in r)}, ("pose_delta", "visibility_delta", "occlusion_delta"), ("pose_update", "visibility_update", "occlusion_update"), "pose_update", "route_after_dynamics", occlusion_reasoning_required=True),
            ActionType.HAND_REACH: _ActionSpec(("initial_hold", "shoulder_extend", "elbow_extend", "hand_target_approach", "settle"), hand_regions, {}, {"body_shape": tuple(r for r in hand_regions if "hand" not in r), "skin": tuple(r for r in hand_regions if "hand" in r or "wrist" in r)}, ("pose_delta", "interaction_delta", "occlusion_delta"), ("pose_update", "interaction_intent", "occlusion_update"), "pose_update", "route_after_dynamics", occlusion_reasoning_required=True),
            ActionType.SIT_DOWN: _ActionSpec(("initial_stand", "knees_bend", "pelvis_lower", "torso_adjust", "seated_settle"), leg_regions, {"torso": PlannerRegionRole.STABILIZER}, {"body_shape": ("pelvis", "left_thigh", "right_thigh", "left_knee", "right_knee", "left_foot", "right_foot", "torso"), "soft_tissue": ("hips",)}, ("pose_delta", "visibility_delta", "occlusion_delta"), ("pose_update", "visibility_update", "occlusion_update", "secondary_motion_required"), "pose_update", "route_after_dynamics", secondary_motion_required=True, occlusion_reasoning_required=True),
            ActionType.STAND_UP: _ActionSpec(("seated_start", "torso_lean", "pelvis_raise", "knees_extend", "standing_settle"), leg_regions, {"torso": PlannerRegionRole.STABILIZER}, {"body_shape": ("pelvis", "left_thigh", "right_thigh", "left_knee", "right_knee", "left_foot", "right_foot", "torso"), "soft_tissue": ("hips",)}, ("pose_delta", "visibility_delta", "occlusion_delta"), ("pose_update", "visibility_update", "occlusion_update", "secondary_motion_required"), "pose_update", "route_after_dynamics", secondary_motion_required=True, occlusion_reasoning_required=True),
            ActionType.STEP_FORWARD: _ActionSpec(("initial_stand", "weight_shift", "foot_advance", "weight_transfer", "step_settle"), leg_regions, {"torso": PlannerRegionRole.STABILIZER}, {"body_shape": ("pelvis", "left_thigh", "right_thigh", "left_knee", "right_knee", "left_foot", "right_foot", "torso")}, ("pose_delta", "visibility_delta", "occlusion_delta"), ("pose_update", "visibility_update", "occlusion_update"), "pose_update", "route_after_dynamics", occlusion_reasoning_required=True),
            ActionType.GARMENT_ADJUST: _ActionSpec(("initial_hold", "hand_to_garment", "garment_contact_intent", "settle"), ("upper_garment", "outer_garment", "sleeves") + _side_regions(side, ("hand",)), {"upper_garment": PlannerRegionRole.GARMENT, "outer_garment": PlannerRegionRole.GARMENT, "sleeves": PlannerRegionRole.GARMENT}, {"garment": ("upper_garment", "outer_garment", "sleeves"), "skin": tuple(_side_regions(side, ("hand",)))}, ("interaction_delta", "garment_intent_delta", "visibility_delta", "occlusion_delta"), ("interaction_intent", "visibility_update", "occlusion_update"), "interaction_intent", "garment_route_context", reveal_may_be_required=True, occlusion_reasoning_required=True),
            ActionType.OBJECT_REACH: _ActionSpec(("initial_hold", "hand_extend", "object_target_approach", "settle"), _side_regions(side, ("arm", "forearm", "hand")) + ("reachable_object",), {"reachable_object": PlannerRegionRole.OBJECT}, {"body_shape": tuple(_side_regions(side, ("arm", "forearm"))), "skin": tuple(_side_regions(side, ("hand",)))}, ("pose_delta", "interaction_delta", "occlusion_delta"), ("pose_update", "interaction_intent", "occlusion_update"), "interaction_intent", "object_route_context", occlusion_reasoning_required=True, target_objects=True),
        }
        return specs[action]

    def _detect_side(self, text: str) -> str | None:
        normalized = self._normalize(text)
        if any(token in normalized for token in ("left", "левой", "левую", "слева")):
            return "left"
        if any(token in normalized for token in ("right", "правой", "правую", "справа")):
            return "right"
        if "both" in normalized or "обе" in normalized:
            return "both"
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").lower().replace("ё", "е").split())


def _region_name(target: RegionPlanTarget | str) -> str:
    if isinstance(target, RegionPlanTarget):
        return target.canonical_region_id or target.region_id.rsplit(":", 1)[-1]
    return str(target).rsplit(":", 1)[-1]


def _is_private_or_optional(region: str) -> bool:
    meta = BODY_ONTOLOGY.get(region)
    if meta is None:
        return False
    return meta.memory_family == "private" or meta.group in {BodyRegionGroup.OPTIONAL_PRIVATE, BodyRegionGroup.OPTIONAL_SEX_SPECIFIC}


def validate_action_plan(plan: ActionPlan) -> ActionPlan:
    if not plan.supported:
        if not plan.unsupported_code or not plan.unsupported_reasons:
            raise PlannerValidationError("invalid_unsupported_plan", "unsupported plan must include code and reasons")
        return plan
    if plan.scene_graph_mutation_performed:
        raise PlannerValidationError("scene_graph_mutation_forbidden", "planner contract cannot mutate SceneGraph")
    forbidden_keys = set(plan.forbidden_outputs) & (_FORBIDDEN_RENDER_KEYS | _FORBIDDEN_EVIDENCE_KEYS)
    if forbidden_keys:
        raise PlannerValidationError("forbidden_planner_output", f"planner emitted forbidden output keys: {sorted(forbidden_keys)}")
    if not plan.actions:
        raise PlannerValidationError("empty_action_plan", "supported planner output must include at least one action")
    for action in plan.actions:
        try:
            ActionType(_enum_value(action.action_type))
        except ValueError as exc:
            raise PlannerValidationError("unknown_action", f"unknown action_type={action.action_type!r}") from exc
        if not action.target_entity_id:
            raise PlannerValidationError("missing_target", "planned action requires target_entity_id")
        if not action.phases:
            raise PlannerValidationError("empty_phases", f"{action.action_type} has no phases")
        region_names = [_region_name(target) for target in action.region_targets]
        if not region_names:
            raise PlannerValidationError("empty_regions", f"{action.action_type} has no region targets")
        for region in region_names:
            if region not in _ALLOWED_GARMENT_REGIONS and region not in _ALLOWED_OBJECT_REGIONS and not is_canonical_body_region(region):
                raise PlannerValidationError("unknown_region", f"unknown planner region={region!r}")
            if _is_private_or_optional(region):
                raise PlannerValidationError("private_auto_target_forbidden", f"private/sex-specific region cannot be auto-targeted: {region}")
        previous_end = -1.0
        for phase in action.phases:
            if not (0.0 <= phase.normalized_start < phase.normalized_end <= 1.0):
                raise PlannerValidationError("invalid_phase_timing", f"phase {phase.phase_id} timing outside 0..1")
            if phase.normalized_start < previous_end:
                raise PlannerValidationError("non_monotonic_phase_timing", f"phase {phase.phase_id} starts before previous phase ends")
            previous_end = phase.normalized_end
            if not phase.affected_regions:
                raise PlannerValidationError("empty_phase_regions", f"phase {phase.phase_id} has no affected regions")
            for target in phase.affected_regions:
                region = _region_name(target)
                if region not in _ALLOWED_GARMENT_REGIONS and region not in _ALLOWED_OBJECT_REGIONS and not is_canonical_body_region(region):
                    raise PlannerValidationError("unknown_region", f"unknown phase region={region!r}")
                if _is_private_or_optional(region):
                    raise PlannerValidationError("private_auto_target_forbidden", f"private/sex-specific phase region cannot be auto-targeted: {region}")
            for family in phase.required_memory_families:
                if family not in VALID_MEMORY_FAMILIES:
                    raise PlannerValidationError("unknown_memory_family", f"unknown memory family={family!r}")
            if phase.dynamics_hint not in VALID_DYNAMICS_HINTS:
                raise PlannerValidationError("unknown_dynamics_hint", f"unknown dynamics hint={phase.dynamics_hint!r}")
            if phase.routing_hint not in VALID_ROUTING_HINTS:
                raise PlannerValidationError("unknown_routing_hint", f"unknown routing hint={phase.routing_hint!r}")
        for req in action.memory_requirements:
            if req.family not in VALID_MEMORY_FAMILIES:
                raise PlannerValidationError("unknown_memory_family", f"unknown memory requirement family={req.family!r}")
            if req.family == "private":
                raise PlannerValidationError("private_memory_requirement_forbidden", "private regions are no-reference by default")
            for region in req.regions:
                if _is_private_or_optional(region):
                    raise PlannerValidationError("private_memory_requirement_forbidden", f"private/sex-specific memory region forbidden: {region}")
        dyn = action.dynamics_requirement
        if dyn.produces_graph_delta:
            raise PlannerValidationError("graph_delta_generation_forbidden", "planner must not generate GraphDelta")
        for hint in dyn.required_dynamics_capabilities:
            if hint not in VALID_DYNAMICS_HINTS:
                raise PlannerValidationError("unknown_dynamics_hint", f"unknown dynamics capability={hint!r}")
        for delta_type in dyn.expected_graph_delta_types:
            if delta_type not in VALID_GRAPH_DELTA_TYPES:
                raise PlannerValidationError("unknown_graph_delta_type", f"unknown expected GraphDelta type={delta_type!r}")
    return plan
