from __future__ import annotations

from dataclasses import asdict, dataclass, field

from core.region_ids import make_region_id, parse_region_id
from core.routing_contracts import DecisionKind, RUNTIME_ROUTING_DECISION_KINDS
from core.schema import GraphDelta, RegionRef, RuntimeSemanticTransition, SceneGraph, VideoMemory
from memory.video_memory import MemoryManager
from rendering.roi_renderer import ROISelector


TransitionCategory = str
RevealMode = str
RendererModeHint = str


@dataclass(slots=True)
class RegionTransitionSemantic:
    canonical_region: str
    transition_category: TransitionCategory
    visibility_before: str
    visibility_after: str
    identity_sensitive: bool
    garment_covered: bool
    newly_revealed: bool
    newly_occluded: bool
    memory_reliable_for_reuse: bool
    memory_suitable_for_reveal: bool
    memory_exists: bool
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class RegionRoutingDecision:
    canonical_region: str
    decision: DecisionKind
    priority: int
    reasons: list[str] = field(default_factory=list)
    memory_source_available: bool = False
    memory_support_level: str = "none"
    reveal_mode: RevealMode = "none"
    synthesis_required: bool = False
    renderer_mode_hint: RendererModeHint = "keep"
    confidence: float = 0.0
    should_render: bool = False


@dataclass(slots=True)
class RegionRoutingPlan:
    entity_id: str
    transition_semantics: list[RegionTransitionSemantic]
    decisions: list[RegionRoutingDecision]
    render_regions: list[RegionRef]

    def as_debug_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "transition_semantics": [asdict(item) for item in self.transition_semantics],
            "decisions": [asdict(item) for item in self.decisions],
            "render_regions": [r.region_id for r in self.render_regions],
        }

    def decision_for_region_id(self, region_id: str) -> RegionRoutingDecision | None:
        _, region = parse_region_id(region_id)
        for decision in self.decisions:
            if decision.canonical_region == region:
                return decision
        return None


class CanonicalRegionRouter:
    _CANONICAL_REGIONS = (
        "face",
        "hair",
        "head",
        "neck",
        "torso",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "pelvis",
        "left_leg",
        "right_leg",
        "upper_garment",
        "lower_garment",
        "outer_garment",
        "inner_garment",
        "accessories",
    )
    _IDENTITY_SENSITIVE = {"face", "hair", "head"}
    _DETAIL_SENSITIVE = {"face", "left_hand", "right_hand", "hair"}
    _GARMENT_RELATED = {"upper_garment", "lower_garment", "outer_garment", "inner_garment", "torso", "left_arm", "right_arm"}
    _GARMENT_EXPOSURE_REGIONS = {"torso", "left_arm", "right_arm"}
    _POSE_RELATED = {"torso", "pelvis", "left_leg", "right_leg", "left_arm", "right_arm", "left_hand", "right_hand"}

    def __init__(self, memory_manager: MemoryManager, roi_selector: ROISelector) -> None:
        self.memory_manager = memory_manager
        self.roi_selector = roi_selector

    def build_plan(
        self,
        *,
        scene_graph: SceneGraph,
        delta: GraphDelta,
        memory: VideoMemory,
        semantic_transition: RuntimeSemanticTransition | None,
    ) -> RegionRoutingPlan:
        entity_id = delta.affected_entities[0] if delta.affected_entities else (scene_graph.persons[0].person_id if scene_graph.persons else "scene")
        person = next((p for p in scene_graph.persons if p.person_id == entity_id), scene_graph.persons[0] if scene_graph.persons else None)

        canonical_regions = list(self._CANONICAL_REGIONS)
        if person is not None and person.canonical_regions:
            canonical_regions = sorted(set(canonical_regions).union(person.canonical_regions.keys()))

        semantics: list[RegionTransitionSemantic] = []
        decisions: list[RegionRoutingDecision] = []
        render_regions: list[RegionRef] = []

        newly_revealed_ids = {r.region_id for r in delta.newly_revealed_regions}
        newly_occluded_ids = {r.region_id for r in delta.newly_occluded_regions}
        target_regions = set(delta.affected_regions)
        if delta.expression_deltas:
            target_regions.update({"face", "head"})
        if delta.garment_deltas:
            target_regions.update({"outer_garment", "upper_garment", "inner_garment", "lower_garment"})
            if any(any(token in reason for token in ("open", "remove", "reveal", "exposure")) for reason in delta.semantic_reasons):
                target_regions.update(self._GARMENT_EXPOSURE_REGIONS)
        if delta.pose_deltas:
            target_regions.update(self._POSE_RELATED)
        for region_type, visibility in delta.visibility_deltas.items():
            if visibility in {"visible", "partially_visible", "hidden"}:
                target_regions.add(region_type)

        for region in canonical_regions:
            region_id = make_region_id(entity_id, region)
            memory_entry = self.memory_manager.get_best_region_memory(memory, entity_id, region)
            previous_visibility = str(memory_entry.visibility_state) if memory_entry is not None else "unknown"
            current_visibility = previous_visibility
            if person is not None:
                payload = person.canonical_regions.get(region, {})
                if isinstance(payload, dict):
                    current_visibility = str(payload.get("visibility_state", current_visibility))

            newly_revealed = region_id in newly_revealed_ids or (memory_entry is not None and memory_entry.reveal_lifecycle == "newly_revealed")
            newly_occluded = region_id in newly_occluded_ids or (memory_entry is not None and memory_entry.reveal_lifecycle == "newly_occluded")
            identity_sensitive = region in self._IDENTITY_SENSITIVE
            garment_covered = region in self._GARMENT_RELATED and any("garment" in rel for rel in delta.semantic_reasons)
            category = "needs_temporal_stabilization"
            reasons: list[str] = []

            if newly_revealed:
                category = "newly_revealed"
                reasons.append("visibility_reveal")
            elif newly_occluded:
                category = "newly_occluded"
                reasons.append("visibility_occluded")
            elif region in target_regions and region in self._GARMENT_RELATED and delta.garment_deltas:
                category = "garment_changed"
                reasons.append("garment_delta")
            elif region in target_regions and region in {"face", "head"} and delta.expression_deltas:
                category = "expression_changed"
                reasons.append("expression_delta")
            elif region in target_regions and delta.pose_deltas and region in self._POSE_RELATED:
                category = "pose_changed"
                reasons.append("pose_delta")
            elif region in target_regions and region in delta.visibility_deltas:
                category = "visibility_changed"
                reasons.append("explicit_visibility_delta")
            elif identity_sensitive and region in target_regions:
                category = "identity_sensitive_update"
                reasons.append("identity_sensitive_target")
            elif region in self._DETAIL_SENSITIVE and semantic_transition and semantic_transition.family in {"expression_transition", "interaction_transition"}:
                category = "high_priority_detail_update"
                reasons.append(f"semantic_family={semantic_transition.family}")
            elif memory_entry is not None and memory_entry.reliable_for_reuse and current_visibility in {"visible", "partially_visible"}:
                category = "unchanged"
                reasons.append("reliable_reuse")
            else:
                category = "needs_temporal_stabilization"
                reasons.append("conservative_stabilize")

            memory_support_level, memory_support_trace = self._memory_support(memory_entry)
            reasons.append(f"memory_support={memory_support_level}")
            reasons.extend(memory_support_trace)

            semantic = RegionTransitionSemantic(
                canonical_region=region,
                transition_category=category,
                visibility_before=previous_visibility,
                visibility_after=current_visibility,
                identity_sensitive=identity_sensitive,
                garment_covered=garment_covered,
                newly_revealed=newly_revealed,
                newly_occluded=newly_occluded,
                memory_reliable_for_reuse=bool(memory_entry and memory_entry.reliable_for_reuse),
                memory_suitable_for_reveal=bool(memory_entry and memory_entry.suitable_for_reveal),
                memory_exists=memory_entry is not None,
                reasons=reasons,
                confidence=float(memory_entry.confidence if memory_entry else 0.45),
            )
            decision = self._route_semantic(semantic, memory_support_level=memory_support_level)
            semantics.append(semantic)
            decisions.append(decision)
            if decision.should_render:
                ref = self.roi_selector.semantic_roi_from_graph(scene_graph, entity_id, region)
                if ref is None:
                    ref = self.roi_selector.fallback_roi_from_person_bbox(scene_graph, entity_id, region)
                if ref is not None:
                    render_regions.append(ref)

        decisions.sort(key=lambda d: d.priority, reverse=True)
        by_id = {r.region_id: r for r in render_regions}
        ordered_render = []
        for d in decisions:
            rid = make_region_id(entity_id, d.canonical_region)
            if rid in by_id:
                ordered_render.append(by_id[rid])

        return RegionRoutingPlan(
            entity_id=entity_id,
            transition_semantics=semantics,
            decisions=decisions,
            render_regions=ordered_render,
        )

    def _memory_support(self, memory_entry: object | None) -> tuple[str, list[str]]:
        if memory_entry is None:
            return "none", ["memory_entry=missing"]
        confidence = float(getattr(memory_entry, "confidence", 0.0) or 0.0)
        freshness = int(getattr(memory_entry, "freshness_frames", 99) or 99)
        reliable = bool(getattr(memory_entry, "reliable_for_reuse", False))
        suitable = bool(getattr(memory_entry, "suitable_for_reveal", False))
        generated = bool(getattr(memory_entry, "generated", False))
        inferred = bool(getattr(memory_entry, "inferred", False))
        trace = [
            f"memory_confidence={confidence:.3f}",
            f"memory_freshness={freshness}",
            f"reuse_reliable={str(reliable).lower()}",
            f"reveal_suitable={str(suitable).lower()}",
            f"memory_generated={str(generated).lower()}",
            f"memory_inferred={str(inferred).lower()}",
        ]
        if generated or (inferred and not reliable):
            return "weak", trace + ["support_reason=generated_or_inferred_without_reliability"]
        if suitable and reliable and confidence >= 0.7 and freshness <= 4:
            return "strong", trace + ["support_reason=fresh_reliable_reveal_ready"]
        if confidence >= 0.5 or reliable:
            return "medium", trace + ["support_reason=usable_but_not_strong"]
        return "weak", trace + ["support_reason=low_confidence_or_stale"]

    def _route_semantic(self, semantic: RegionTransitionSemantic, *, memory_support_level: str) -> RegionRoutingDecision:
        decision: DecisionKind = "temporal_stabilize"
        priority = 10
        reveal_mode = "none"
        renderer_mode_hint = "keep"
        synthesis_required = False
        should_render = False

        if semantic.transition_category == "newly_revealed":
            priority = 95 if semantic.identity_sensitive else 88
            should_render = True
            renderer_mode_hint = "reveal"
            if semantic.memory_suitable_for_reveal and memory_support_level in {"strong", "medium"}:
                decision = "reveal_from_memory"
                reveal_mode = "from_memory"
            elif semantic.memory_exists and memory_support_level in {"weak", "medium"}:
                decision = "reveal_partial_memory_assist"
                reveal_mode = "partial_memory_assist"
                synthesis_required = True
            else:
                decision = "reveal_requires_synthesis"
                reveal_mode = "requires_synthesis"
                synthesis_required = True
        elif semantic.transition_category == "newly_occluded":
            decision = "temporal_stabilize"
            renderer_mode_hint = "keep"
            priority = 35
        elif semantic.transition_category == "unchanged":
            decision = "direct_reuse"
            priority = 5
        elif semantic.transition_category in {"garment_changed"}:
            decision = "garment_transition_update"
            should_render = True
            renderer_mode_hint = "warp"
            priority = 82
        elif semantic.transition_category in {"expression_changed", "identity_sensitive_update", "high_priority_detail_update"}:
            decision = "expression_refine"
            should_render = True
            renderer_mode_hint = "refine"
            priority = 90 if semantic.identity_sensitive else 72
        elif semantic.transition_category in {"pose_changed", "visibility_changed"}:
            decision = "pose_exposure_update"
            should_render = True
            renderer_mode_hint = "deform"
            priority = 70
        else:
            decision = "local_deform_or_update"
            should_render = semantic.visibility_after in {"visible", "partially_visible"}
            renderer_mode_hint = "refine"
            priority = 55

        if semantic.transition_category == "needs_temporal_stabilization" and semantic.visibility_after in {"visible", "partially_visible"} and not semantic.memory_reliable_for_reuse:
            decision = "local_deform_or_update"
            should_render = True
            priority = max(priority, 60)
            renderer_mode_hint = "refine"

        return RegionRoutingDecision(
            canonical_region=semantic.canonical_region,
            decision=decision if decision in RUNTIME_ROUTING_DECISION_KINDS else "local_deform_or_update",
            priority=priority,
            reasons=list(semantic.reasons),
            memory_source_available=semantic.memory_exists,
            memory_support_level=memory_support_level,
            reveal_mode=reveal_mode,
            synthesis_required=synthesis_required,
            renderer_mode_hint=renderer_mode_hint,
            confidence=semantic.confidence,
            should_render=should_render,
        )
