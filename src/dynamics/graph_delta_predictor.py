from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.schema import BBox, GraphDelta, RegionRef, RuntimeSemanticTransition, SceneGraph, TransitionTargetProfile, VideoMemory
from core.semantic_roi import SemanticROIHelper
from dynamics.model import DynamicsModel, DynamicsModelContractError, DynamicsModelError, decode_prediction, featurize_runtime
from dynamics.transition_contracts import (
    ExpressionTransitionIntent,
    GarmentTransitionIntent,
    InteractionTransitionIntent,
    MemoryInfluence,
    PlannerTransitionContext,
    PoseTransitionIntent,
    TransitionDiagnostics,
    TransitionIntent,
    VisibilityTransitionIntent,
)
from planning.transition_engine import PlannedState
from representation.scene_graph_queries import SceneGraphQueries


@dataclass(slots=True)
class DynamicsMetrics:
    delta_magnitude: float
    constraint_violations: int
    temporal_smoothness_proxy: float


class GraphDeltaPredictor:
    """Пошаговый движок эволюции scene-state для single-image сценария."""

    def __init__(self, *, strict_mode: bool = False) -> None:
        weights = Path("artifacts/checkpoints/dynamics/dynamics_weights.json")
        self.model = DynamicsModel.load(str(weights)) if weights.exists() else DynamicsModel()
        self.roi = SemanticROIHelper()
        self.legacy_mode = False
        self.strict_mode = strict_mode

    def _serialize_graph(self, scene_graph: SceneGraph) -> str:
        return f"f={scene_graph.frame_index};p={len(scene_graph.persons)};o={len(scene_graph.objects)};" + ";".join(sorted(p.person_id for p in scene_graph.persons))

    def _clamp_pose(self, value: float, lo: float = -45.0, hi: float = 45.0) -> float:
        return max(lo, min(hi, value))

    def _region_from_person(self, scene_graph: SceneGraph, person_id: str, person_bbox: BBox, kind: str, reason: str) -> RegionRef:
        resolved = self.roi.resolve_region(scene_graph, person_id, kind)
        if resolved is not None:
            return RegionRef(region_id=resolved.region_id, bbox=resolved.bbox, reason=reason)
        box = self.roi.fallback_person_bbox(person_bbox, kind)
        return RegionRef(region_id=f"{person_id}:{kind}", bbox=box, reason=reason)

    def predict(self, scene_graph: SceneGraph, target_state: PlannedState, planner_context: dict[str, float] | None = None, memory: VideoMemory | None = None) -> tuple[GraphDelta, DynamicsMetrics]:
        context = planner_context or {}
        labels = list(target_state.labels)
        planner = self._parse_planner_context(target_state=target_state, context=context)
        try:
            model_out = self.model.forward(featurize_runtime(scene_graph, target_state, context, memory))
            delta = decode_prediction(
                model_out,
                scene_graph=scene_graph,
                phase=planner.phase,
                semantic_reasons=labels,
                planner_context=context,
            )
            magnitude = sum(abs(v) for v in delta.pose_deltas.values()) + sum(abs(v) for v in delta.interaction_deltas.values())
            smoothness = 1.0 / (1.0 + magnitude)
            delta.transition_diagnostics = {
                "runtime_path": "learned_primary",
                "delta_magnitude": magnitude,
                "temporal_smoothness_proxy": smoothness,
                "constraint_violations": [],
                "fallback_usage": [],
                "family_contribution": {
                    "pose": sum(abs(v) for v in delta.pose_deltas.values()),
                    "garment": abs(float(delta.garment_deltas.get("attachment_delta", 0.0))) + abs(float(delta.garment_deltas.get("coverage_delta", 0.0))),
                    "expression": abs(float(delta.expression_deltas.get("smile_intensity", 0.0))),
                    "interaction": sum(abs(v) for v in delta.interaction_deltas.values()),
                    "visibility": float(len(delta.visibility_deltas)),
                },
            }
            return (
                delta,
                DynamicsMetrics(
                    delta_magnitude=magnitude,
                    constraint_violations=0,
                    temporal_smoothness_proxy=smoothness,
                ),
            )
        except (DynamicsModelContractError, ValueError) as exc:
            if self.strict_mode:
                raise
            self.legacy_mode = True
            delta, metrics = self._predict_legacy(scene_graph, target_state, planner_context, memory)
            delta.transition_diagnostics["fallback_reason"] = type(exc).__name__
            return delta, metrics
        except DynamicsModelError:
            if self.strict_mode:
                raise
            self.legacy_mode = True
            delta, metrics = self._predict_legacy(scene_graph, target_state, planner_context, memory)
            delta.transition_diagnostics["fallback_reason"] = "DynamicsModelError"
            return delta, metrics

    def _predict_legacy(self, scene_graph: SceneGraph, target_state: PlannedState, planner_context: dict[str, float] | None = None, memory: VideoMemory | None = None) -> tuple[GraphDelta, DynamicsMetrics]:
        context = planner_context or {}
        person = scene_graph.persons[0] if scene_graph.persons else None
        semantic_hints = self._extract_semantic_hints(target_state)
        planner = self._parse_planner_context(target_state=target_state, context=context)
        memory_influence = self._parse_memory_context(memory)

        fallback_scores = {"pose": 0.5, "garment": 0.5, "expression": 0.5, "interaction": 0.5, "visibility": 0.5}
        intent = self._build_transition_intent(
            scene_graph,
            semantic_hints,
            planner,
            memory_influence,
            fallback_scores,
            runtime_transition=target_state.semantic_transition,
        )
        delta = GraphDelta(transition_phase=planner.phase)

        if person:
            delta.affected_entities.append(person.person_id)
            delta.state_before = self._build_state_before(scene_graph, person, person.expression_state.label or "neutral")

        diagnostics = TransitionDiagnostics()
        family_deltas = self._compute_family_deltas(intent, fallback_scores)
        self._merge_family_deltas(delta, intent, family_deltas, diagnostics)
        self._derive_visibility_consequences(scene_graph, person, intent, delta, diagnostics)
        self._derive_region_transition_modes(intent, delta)

        if person:
            delta.state_after = self._derive_state_after(delta, delta.state_before, intent)

        self._compute_diagnostics(delta, intent, diagnostics)
        delta.transition_diagnostics = {
            "runtime_path": "legacy_heuristic_fallback",
            "delta_magnitude": diagnostics.delta_magnitude,
            "family_contribution": diagnostics.family_contribution,
            "transition_smoothness_proxy": diagnostics.transition_smoothness_proxy,
            "constraint_violations": diagnostics.constraint_violations,
            "fallback_usage": diagnostics.fallback_usage + ["legacy_heuristic"],
            "explainability_summary": diagnostics.explainability_summary,
            "semantic_families": intent.active_families,
            "canonical_goals": intent.goals,
            "target_regions": intent.target_regions,
            "phase": intent.planner.phase,
            "family_subphases": {
                "pose": intent.pose.progression,
                "garment": intent.garment.progression_state,
                "interaction": intent.interaction.support_progression,
                "expression": intent.expression.progression,
            },
            "region_modes": delta.region_transition_mode,
            "visibility_consequences": {
                "revealed": [r for r in intent.visibility.reveal_regions if r in delta.predicted_visibility_changes],
                "occluded": [r for r in intent.visibility.occlude_regions if r in delta.predicted_visibility_changes],
            },
            "target_profile": {
                "primary_regions": intent.target_profile.primary_regions,
                "secondary_regions": intent.target_profile.secondary_regions,
                "context_regions": intent.target_profile.context_regions,
                "entity": intent.target_profile.entity,
                "entity_id": intent.target_profile.entity_id,
                "object_role": intent.target_profile.object_role,
                "support_target": intent.target_profile.support_target,
            },
            "region_selection_rationale": self._region_selection_rationale(intent),
            "reveal_occlusion_rationale": self._reveal_rationale(intent),
            "lexical_bootstrap_influence": self._lexical_bootstrap_influence(target_state),
            "phase_sequence": {
                "global": ["prepare", "transition", "contact_or_reveal", "stabilize"],
                "family_subphases": {
                    "pose": intent.pose.phase_sequence,
                    "garment": intent.garment.phase_sequence,
                    "interaction": intent.interaction.phase_sequence,
                    "expression": ["subtle_rise", "forming_expression", "stable_expression", "relaxed_expression"],
                },
            },
        }

        metrics = DynamicsMetrics(
            delta_magnitude=diagnostics.delta_magnitude,
            constraint_violations=len(diagnostics.constraint_violations),
            temporal_smoothness_proxy=diagnostics.transition_smoothness_proxy,
        )
        return delta, metrics

    def _parse_planner_context(self, target_state: PlannedState, context: dict[str, float]) -> PlannerTransitionContext:
        """Нормализует контекст планировщика в фазовый контракт."""
        step_index = int(context.get("step_index", target_state.step_index))
        total_steps = max(1, int(context.get("total_steps", context.get("plan_length", 4))))
        progress = step_index / float(total_steps)
        phase = "prepare" if progress <= 0.25 else ("transition" if progress <= 0.75 else "contact_or_reveal")
        if step_index >= total_steps:
            phase = "stabilize"
        stage = str(context.get("sequencing_stage", phase))
        intensity = float(context.get("intensity", self._extract_intensity(target_state.labels)))
        duration = float(context.get("target_duration", context.get("duration", 1.0)))
        return PlannerTransitionContext(
            step_index=step_index,
            total_steps=total_steps,
            phase=phase,
            sequencing_stage=stage,
            target_duration=max(0.1, duration),
            intensity=max(0.0, min(1.0, intensity)),
        )

    def _parse_memory_context(self, memory: VideoMemory | None) -> MemoryInfluence:
        """Строит priors из memory для visibility/garment ограничений."""
        if memory is None:
            return MemoryInfluence()
        hidden = min(1.0, len(memory.hidden_region_slots) / 8.0)
        garment = min(1.0, len(memory.garment_memory) / 6.0)
        body = min(1.0, len(memory.region_descriptors) / 16.0)
        identity = min(1.0, len(memory.identity_memory) / 4.0)
        safety = max(0.15, min(1.0, 0.4 * hidden + 0.2 * body + 0.4 * identity))
        return MemoryInfluence(
            hidden_reveal_evidence=hidden,
            garment_memory_strength=garment,
            body_region_memory_strength=body,
            identity_continuity=identity,
            visibility_safety=safety,
        )

    def _build_transition_intent(
        self,
        scene_graph: SceneGraph,
        semantic_hints: dict[str, str],
        planner: PlannerTransitionContext,
        memory: MemoryInfluence,
        model_out: dict[str, float],
        runtime_transition: RuntimeSemanticTransition | None = None,
    ) -> TransitionIntent:
        """Собирает transition intent до вычисления GraphDelta."""
        person = scene_graph.persons[0] if scene_graph.persons else None
        target_entity = person.person_id if person else "scene"
        families = self._derive_families(semantic_hints)
        target_profile = self._derive_target_profile(semantic_hints, target_state_semantic=runtime_transition)
        intent = TransitionIntent(
            active_families=families,
            goals={family: semantic_hints.get(family, "unknown") for family in families},
            target_entity=target_entity,
            target_regions=sorted(
                set(
                    target_profile.primary_regions
                    + target_profile.secondary_regions
                    + target_profile.context_regions
                )
            ),
            target_profile=target_profile,
            planner=planner,
            memory=memory,
            pose=self._pose_intent(semantic_hints, planner),
            garment=self._garment_intent(semantic_hints, planner, memory),
            visibility=VisibilityTransitionIntent(),
            interaction=self._interaction_intent(semantic_hints, planner),
            expression=self._expression_intent(semantic_hints, planner, model_out),
        )
        if runtime_transition is not None:
            if runtime_transition.phase.pose_subphase != "steady":
                intent.pose.progression = runtime_transition.phase.pose_subphase
            if runtime_transition.phase.garment_subphase != "stable":
                intent.garment.progression_state = runtime_transition.phase.garment_subphase
            if runtime_transition.phase.interaction_subphase != "free":
                intent.interaction.support_progression = runtime_transition.phase.interaction_subphase
            if runtime_transition.phase.expression_subphase != "neutral":
                intent.expression.progression = runtime_transition.phase.expression_subphase
        intent.visibility = self._visibility_intent(intent)
        return intent

    def _extract_semantic_hints(self, target_state: PlannedState) -> dict[str, str]:
        hints: dict[str, str] = {}
        transition = target_state.semantic_transition
        if transition is not None:
            hints[transition.family] = transition.goal
            if transition.family == "garment_transition":
                hints["visibility_transition"] = "reveal_region"
            if transition.family == "pose_transition" and transition.goal == "seated_pose":
                hints["interaction_transition"] = "support_contact"
            if transition.family == "pose_transition" and transition.goal == "upright_pose":
                hints["interaction_transition"] = "support_release"
            return hints

        joined = " ".join(label for label in target_state.labels if not label.startswith("intensity="))
        coarse_bootstrap = {
            "sit_down": ("pose_transition", "seated_pose"),
            "stand_up": ("pose_transition", "upright_pose"),
            "raise_arm": ("pose_transition", "arm_elevation"),
            "turn_head": ("pose_transition", "head_rotation"),
            "remove_garment": ("garment_transition", "outer_layer_removal"),
            "open_garment": ("garment_transition", "outer_layer_opening"),
            "smile": ("expression_transition", "smile_like"),
        }
        for marker, (family, goal) in coarse_bootstrap.items():
            if marker in joined:
                hints[family] = goal
        if "visibility_transition" not in hints and hints:
            hints["visibility_transition"] = "preserve_identity_region"
        if hints.get("pose_transition") == "seated_pose":
            hints.setdefault("interaction_transition", "support_contact")
        if hints.get("pose_transition") == "upright_pose":
            hints.setdefault("interaction_transition", "support_release")
        return hints

    def _derive_families(self, hints: dict[str, str]) -> list[str]:
        families: list[str] = []
        for family in ("pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition"):
            if family in hints:
                families.append(family)
        return sorted(set(families))

    def _derive_target_profile(
        self,
        hints: dict[str, str],
        target_state_semantic: RuntimeSemanticTransition | None = None,
    ) -> TransitionTargetProfile:
        if target_state_semantic is not None:
            return target_state_semantic.target_profile
        pose_goal = hints.get("pose_transition")
        if pose_goal == "seated_pose":
            return TransitionTargetProfile(primary_regions=["legs", "pelvis"], secondary_regions=["torso"], context_regions=["support_zone"])
        if pose_goal == "arm_elevation":
            return TransitionTargetProfile(primary_regions=["left_arm", "right_arm"], secondary_regions=["upper_torso", "shoulders"])
        if pose_goal == "head_rotation" or hints.get("expression_transition") in {"smile_like", "gaze_shift"}:
            return TransitionTargetProfile(primary_regions=["face", "head", "neck"], secondary_regions=["upper_torso"])
        if hints.get("garment_transition") in {"outer_layer_removal", "outer_layer_opening"}:
            context = ["support_zone"] if "interaction_transition" in hints else []
            return TransitionTargetProfile(primary_regions=["garments", "sleeves"], secondary_regions=["torso", "inner_garment"], context_regions=context)
        return TransitionTargetProfile(primary_regions=["torso"])

    def _pose_intent(self, hints: dict[str, str], planner: PlannerTransitionContext) -> PoseTransitionIntent:
        goal = hints.get("pose_transition")
        if goal == "seated_pose":
            progression = "weight_shift" if planner.phase == "prepare" else ("lowering" if planner.phase == "transition" else "contact_settle")
            return PoseTransitionIntent(goal=goal, progression=progression, phase_sequence=["weight_shift", "lowering", "contact_settle", "seated_stabilization"], weight_shift=0.35 + 0.4 * planner.intensity)
        if goal == "upright_pose":
            progression = "lift_off_support" if planner.phase != "stabilize" else "standing_stabilization"
            return PoseTransitionIntent(goal=goal, progression=progression, phase_sequence=["pre_lift", "lift_off_support", "support_release", "standing_stabilization"], weight_shift=0.25)
        if goal == "arm_elevation":
            return PoseTransitionIntent(goal=goal, progression="shoulder_lift", weight_shift=0.1)
        if goal == "head_rotation":
            return PoseTransitionIntent(goal=goal, progression="head_rotation", weight_shift=0.0)
        return PoseTransitionIntent()

    def _garment_intent(self, hints: dict[str, str], planner: PlannerTransitionContext, memory: MemoryInfluence) -> GarmentTransitionIntent:
        goal = hints.get("garment_transition")
        if goal not in {"outer_layer_removal", "outer_layer_opening"}:
            return GarmentTransitionIntent()
        progression = {
            "prepare": "tensioned",
            "transition": "opening",
            "contact_or_reveal": "partially_detached",
            "stabilize": "half_removed",
        }.get(planner.phase, "opening")
        if planner.phase == "stabilize" and planner.intensity > 0.65:
            progression = "removed"
        attachment_delta = -0.15 - 0.35 * planner.intensity
        reveal_bias = 0.4 * memory.hidden_reveal_evidence + 0.6 * memory.garment_memory_strength
        return GarmentTransitionIntent(goal=goal, progression_state=progression, attachment_delta=attachment_delta, reveal_bias=reveal_bias)

    def _interaction_intent(self, hints: dict[str, str], planner: PlannerTransitionContext) -> InteractionTransitionIntent:
        goal = hints.get("interaction_transition")
        if goal == "support_contact":
            stage = {
                "prepare": "near_support",
                "transition": "approach_contact",
                "contact_or_reveal": "weight_transfer",
                "stabilize": "stabilized_seated",
            }.get(planner.phase, "contact_established")
            return InteractionTransitionIntent(goal=goal, support_progression=stage, support_target="chair", contact_bias=0.3 + 0.6 * planner.intensity)
        if goal == "support_release":
            return InteractionTransitionIntent(goal=goal, support_progression="support_release", support_target="chair", contact_bias=0.2)
        return InteractionTransitionIntent()

    def _expression_intent(self, hints: dict[str, str], planner: PlannerTransitionContext, model_out: dict[str, float]) -> ExpressionTransitionIntent:
        goal = hints.get("expression_transition")
        if goal != "smile_like":
            return ExpressionTransitionIntent()
        progression = {
            "prepare": "subtle_rise",
            "transition": "forming_expression",
            "contact_or_reveal": "stable_expression",
            "stabilize": "relaxed_expression",
        }.get(planner.phase, "forming_expression")
        return ExpressionTransitionIntent(goal=goal, expression_label="smile", progression=progression, intensity_delta=0.1 + 0.35 * model_out["expression"])

    def _visibility_intent(self, intent: TransitionIntent) -> VisibilityTransitionIntent:
        reveal: list[str] = []
        occlude: list[str] = []
        stable: list[str] = ["torso", "preserve_identity_region"]
        if intent.pose.goal == "arm_elevation":
            reveal.extend(["left_arm", "right_arm", "sleeves"])
        if intent.garment.goal in {"outer_layer_removal", "outer_layer_opening"}:
            reveal.extend(["torso", "inner_garment"])
            occlude.append("outer_garment")
        if intent.pose.goal == "seated_pose":
            occlude.append("legs")
            stable.append("face")
        if intent.expression.goal == "smile_like":
            reveal.append("face")
        return VisibilityTransitionIntent(
            goal="reveal_region" if reveal else "preserve_identity_region",
            reveal_regions=sorted(set(reveal)),
            occlude_regions=sorted(set(occlude)),
            stable_regions=sorted(set(stable)),
        )

    def _compute_family_deltas(self, intent: TransitionIntent, model_out: dict[str, float]) -> dict[str, dict[str, float | str]]:
        """Вычисляет sub-deltas по семействам переходов."""
        out: dict[str, dict[str, float | str]] = {"pose": {}, "garment": {}, "expression": {}, "interaction": {}, "visibility": {}}

        if "pose_transition" in intent.active_families:
            if intent.pose.goal == "seated_pose":
                out["pose"] = {
                    "left_knee": self._clamp_pose(20.0 * model_out["pose"]),
                    "right_knee": self._clamp_pose(20.0 * model_out["pose"]),
                    "torso_pitch": self._clamp_pose(10.0 * model_out["pose"]),
                }
            elif intent.pose.goal == "arm_elevation":
                out["pose"] = {
                    "left_shoulder": self._clamp_pose(18.0 * model_out["pose"]),
                    "left_elbow": self._clamp_pose(13.0 * model_out["pose"]),
                }
            elif intent.pose.goal == "head_rotation":
                out["pose"] = {"head_yaw": self._clamp_pose(14.0 * model_out["pose"])}
            elif intent.pose.goal == "upright_pose":
                out["pose"] = {"torso_pitch": self._clamp_pose(-8.0 * model_out["pose"])}

        if "garment_transition" in intent.active_families:
            out["garment"] = {
                "outer_attachment": intent.garment.attachment_delta,
                "garment_progression": intent.garment.progression_state,
                "coverage_expectation": max(0.0, min(1.0, 0.8 - intent.garment.reveal_bias)),
            }

        if "expression_transition" in intent.active_families:
            out["expression"] = {
                "smile_intensity": intent.expression.intensity_delta,
                "expression_progression": intent.expression.progression,
                "expression_label": intent.expression.expression_label,
            }

        if "interaction_transition" in intent.active_families:
            out["interaction"] = {
                "support_contact": min(1.0, max(0.0, intent.interaction.contact_bias * (0.8 + 0.4 * model_out["interaction"]))),
                "weight_transfer": intent.pose.weight_shift,
            }

        return out

    def _merge_family_deltas(
        self,
        delta: GraphDelta,
        intent: TransitionIntent,
        family_deltas: dict[str, dict[str, float | str]],
        diagnostics: TransitionDiagnostics,
    ) -> None:
        """Сливает sub-deltas в финальный GraphDelta с учетом вкладов."""
        delta.pose_deltas.update({k: float(v) for k, v in family_deltas["pose"].items()})
        delta.garment_deltas.update(family_deltas["garment"])
        delta.expression_deltas.update(family_deltas["expression"])
        delta.interaction_deltas.update({k: float(v) for k, v in family_deltas["interaction"].items()})

        delta.semantic_reasons.extend(
            reason for reason, payload in (
                ("pose_transition", delta.pose_deltas),
                ("garment_transition", delta.garment_deltas),
                ("expression_transition", delta.expression_deltas),
                ("interaction_transition", delta.interaction_deltas),
                ("visibility_transition", delta.visibility_deltas),
            )
            if payload
        )
        profile_regions = (
            intent.target_profile.primary_regions
            + intent.target_profile.secondary_regions
            + intent.target_profile.context_regions
        )
        delta.affected_regions = sorted(set(delta.affected_regions + list(delta.visibility_deltas.keys()) + profile_regions))

        diagnostics.family_contribution = {
            "pose": float(len(delta.pose_deltas)),
            "garment": float(len(delta.garment_deltas)),
            "expression": float(len(delta.expression_deltas)),
            "interaction": float(len(delta.interaction_deltas)),
            "visibility": float(len(delta.visibility_deltas)),
        }

    def _derive_visibility_consequences(
            self,
            scene_graph: SceneGraph,
            person,
            intent: TransitionIntent,
            delta: GraphDelta,
            diagnostics: TransitionDiagnostics,
    ) -> None:
        """Отдельный шаг вычисления visibility/occlusion последствий."""

        reveal_state = "visible" if intent.memory.visibility_safety >= 0.45 else "partially_visible"

        for region in intent.visibility.reveal_regions:
            delta.predicted_visibility_changes[region] = reveal_state
            delta.visibility_deltas[region] = reveal_state

        for region in intent.visibility.occlude_regions:
            delta.predicted_visibility_changes[region] = "partially_visible"
            delta.visibility_deltas[region] = "partially_visible"

        for region in intent.visibility.stable_regions:
            delta.predicted_visibility_changes.setdefault(region, "visible")
            delta.visibility_deltas.setdefault(region, "visible")

        if delta.visibility_deltas and "visibility_transition" not in delta.semantic_reasons:
            delta.semantic_reasons.append("visibility_transition")

        delta.affected_regions = sorted(set(delta.affected_regions + list(delta.visibility_deltas.keys())))
        diagnostics.family_contribution["visibility"] = float(len(delta.visibility_deltas))

        if person:
            for region in intent.visibility.reveal_regions:
                delta.newly_revealed_regions.append(
                    self._region_from_person(
                        scene_graph,
                        person.person_id,
                        person.bbox,
                        region,
                        f"visibility_reveal:{intent.planner.phase}",
                    )
                )
            for region in intent.visibility.occlude_regions:
                delta.newly_occluded_regions.append(
                    self._region_from_person(
                        scene_graph,
                        person.person_id,
                        person.bbox,
                        region,
                        f"visibility_occlude:{intent.planner.phase}",
                    )
                )

    def _derive_region_transition_modes(self, intent: TransitionIntent, delta: GraphDelta) -> None:
        """Формирует режимы переходов регионов по семействам."""
        for region in intent.target_profile.primary_regions + intent.target_profile.secondary_regions + intent.target_profile.context_regions:
            delta.region_transition_mode.setdefault(region, "stable")
        if "garment_transition" in intent.active_families:
            for region in {"garments", "sleeves"} & set(intent.target_profile.primary_regions):
                delta.region_transition_mode[region] = "garment_surface"
        for region in intent.visibility.reveal_regions:
            mode = "visibility_reveal"
            if "garment_transition" in intent.active_families and region in {"torso", "inner_garment"}:
                mode = "garment_reveal"
            elif "pose_transition" in intent.active_families and "arm" in region:
                mode = "pose_exposure"
            delta.region_transition_mode[region] = mode
        for region in intent.visibility.occlude_regions:
            delta.region_transition_mode[region] = "visibility_occlusion"
        if "expression_transition" in intent.active_families:
            delta.region_transition_mode["face"] = "expression_refine"

    def _build_state_before(self, scene_graph: SceneGraph, person, expression_label: str) -> dict[str, str]:
        support_targets = SceneGraphQueries.get_supported_by(scene_graph, person.person_id)
        visible = SceneGraphQueries.get_visible_regions(scene_graph, person.person_id)
        occluded = SceneGraphQueries.get_occluded_regions(scene_graph, person.person_id)
        attached_garments = SceneGraphQueries.get_attached_garments(scene_graph, person.person_id)

        pose_state = person.pose_state.coarse_pose or "standing"

        return {
            "pose_state": pose_state,
            "garment_state": "worn" if attached_garments else "removed",
            "visibility_state": "hidden" if len(occluded) > len(visible) else "visible",
            "interaction_state": "support" if support_targets else "free_contact",
            "expression_state": expression_label,
            "phase_state": "prepare",
        }

    def _derive_state_after(self, delta: GraphDelta, before: dict[str, str], intent: TransitionIntent) -> dict[
        str, str]:
        """Агрегирует state_after из intent + sub-deltas."""

        pose_state = before.get("pose_state", "stable")
        if intent.pose.goal in {"seated_pose", "upright_pose", "arm_elevation", "head_rotation"}:
            pose_state = intent.pose.goal

        garment_state = str(delta.garment_deltas.get("garment_progression", before.get("garment_state", "worn")))
        if garment_state == "half_removed" and intent.planner.phase == "stabilize":
            garment_state = "stabilized_removed"

        visibility_state = before.get("visibility_state", "visible")
        if delta.newly_revealed_regions:
            visibility_state = "revealing"
        elif delta.newly_occluded_regions:
            visibility_state = "occluding"

        support_contact = float(delta.interaction_deltas.get("support_contact", 0.0))
        if support_contact >= 0.85:
            interaction_state = "weight_transfer"
        elif support_contact >= 0.65:
            interaction_state = "contact_established"
        elif support_contact >= 0.35:
            interaction_state = "approach_contact"
        else:
            interaction_state = (
                intent.interaction.support_progression
                if intent.interaction.support_progression != "free"
                else before.get("interaction_state", "free_contact")
            )

        expression_state = str(
            delta.expression_deltas.get("expression_progression", before.get("expression_state", "neutral")))

        return {
            "pose_state": pose_state,
            "garment_state": garment_state,
            "visibility_state": visibility_state,
            "interaction_state": interaction_state,
            "expression_state": expression_state,
            "transition_phase": intent.planner.phase,
            "phase_state": intent.planner.phase,
            "pose_subphase": intent.pose.progression,
            "garment_subphase": intent.garment.progression_state,
            "interaction_subphase": intent.interaction.support_progression,
            "expression_subphase": intent.expression.progression,
        }

    def _compute_diagnostics(self, delta: GraphDelta, intent: TransitionIntent, diagnostics: TransitionDiagnostics) -> None:
        """Считает полезную диагностику для explainability/debug."""
        magnitude = sum(abs(v) for v in delta.pose_deltas.values())
        magnitude += sum(abs(v) for v in delta.interaction_deltas.values())
        diagnostics.delta_magnitude = magnitude
        diagnostics.transition_smoothness_proxy = 1.0 / (1.0 + magnitude)

        if "interaction_transition" in intent.active_families and delta.interaction_deltas.get("support_contact", 0.0) < 0.2:
            diagnostics.constraint_violations.append("weak_support_contact")
        if "garment_transition" in intent.active_families and float(delta.garment_deltas.get("coverage_expectation", 1.0)) > 0.85:
            diagnostics.constraint_violations.append("garment_progress_low")

        diagnostics.visibility_uncertainty = 1.0 - intent.memory.visibility_safety
        diagnostics.garment_uncertainty = max(0.0, 1.0 - intent.memory.garment_memory_strength)
        diagnostics.interaction_uncertainty = 0.1 if "interaction_transition" in intent.active_families else 0.0
        diagnostics.hidden_transition_uncertainty = max(0.0, 1.0 - intent.memory.hidden_reveal_evidence)

        if not delta.pose_deltas:
            diagnostics.fallback_usage.append("pose_fallback_stable")
        if not delta.visibility_deltas:
            diagnostics.fallback_usage.append("visibility_fallback_stable")

        dominant = max(diagnostics.family_contribution.items(), key=lambda item: item[1])[0] if diagnostics.family_contribution else "none"
        diagnostics.explainability_summary = (
            f"phase={intent.planner.phase};dominant_family={dominant};"
            f"visibility_safety={intent.memory.visibility_safety:.2f};"
            f"families={','.join(intent.active_families)};"
            f"goals={intent.goals}"
        )

    def _region_selection_rationale(self, intent: TransitionIntent) -> dict[str, str]:
        rationale: dict[str, str] = {}
        for region in intent.target_profile.primary_regions:
            rationale[region] = "primary_goal_region"
        for region in intent.target_profile.secondary_regions:
            rationale[region] = "secondary_influence_region"
        for region in intent.target_profile.context_regions:
            rationale[region] = "context_support_region"
        return rationale

    def _reveal_rationale(self, intent: TransitionIntent) -> str:
        if intent.garment.goal in {"outer_layer_removal", "outer_layer_opening"}:
            return "garment_transition_drives_reveal"
        if intent.pose.goal == "arm_elevation":
            return "pose_exposure_drives_reveal"
        if intent.expression.goal == "smile_like":
            return "expression_focus_face_reveal"
        return "identity_preservation"

    def _lexical_bootstrap_influence(self, target_state: PlannedState) -> dict[str, float | bool]:
        if target_state.semantic_transition is not None:
            return {
                "used_runtime_semantic_contract": True,
                "semantic_origin": "runtime_typed_contract",
                "bootstrap_score": float(target_state.semantic_transition.lexical_bootstrap_score),
            }
        lexical_tokens = [label for label in target_state.labels if not label.startswith("intensity=")]
        token_density = min(1.0, len(lexical_tokens) / 4.0)
        return {"used_runtime_semantic_contract": False, "semantic_origin": "coarse_lexical_fallback", "bootstrap_score": token_density}

    def _extract_intensity(self, labels: list[str]) -> float:
        for label in labels:
            if label.startswith("intensity="):
                try:
                    return float(label.split("=", 1)[1])
                except ValueError:
                    return 0.5
        return 0.5
