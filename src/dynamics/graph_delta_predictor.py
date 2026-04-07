from __future__ import annotations

from dataclasses import dataclass

from core.schema import BBox, GraphDelta, RegionRef, SceneGraph, VideoMemory
from core.semantic_roi import SemanticROIHelper
from dynamics.model import DynamicsInputs, DynamicsModel
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

    def __init__(self) -> None:
        self.model = DynamicsModel()
        self.roi = SemanticROIHelper()

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
        person = scene_graph.persons[0] if scene_graph.persons else None
        labels = set(target_state.labels)
        planner = self._parse_planner_context(target_state=target_state, context=context)
        memory_influence = self._parse_memory_context(memory)

        model_out = self.model.forward(
            DynamicsInputs(
                serialized_scene_graph=self._serialize_graph(scene_graph),
                action_tokens=list(labels),
                planner_context=[float(planner.step_index), planner.intensity, planner.target_duration],
                memory_features=[
                    memory_influence.hidden_reveal_evidence,
                    memory_influence.garment_memory_strength,
                    memory_influence.identity_continuity,
                    memory_influence.visibility_safety,
                ],
            )
        )

        intent = self._build_transition_intent(scene_graph, labels, planner, memory_influence, model_out)
        delta = GraphDelta(transition_phase=planner.phase)

        if person:
            delta.affected_entities.append(person.person_id)
            delta.state_before = self._build_state_before(scene_graph, person.person_id, person.expression_state.label or "neutral")

        diagnostics = TransitionDiagnostics()
        family_deltas = self._compute_family_deltas(intent, model_out)
        self._merge_family_deltas(delta, family_deltas, diagnostics)
        self._derive_visibility_consequences(scene_graph, person, intent, delta)
        self._derive_region_transition_modes(intent, delta)

        if person:
            delta.state_after = self._derive_state_after(delta, delta.state_before, intent)

        self._compute_diagnostics(delta, intent, diagnostics)
        delta.transition_diagnostics = {
            "delta_magnitude": diagnostics.delta_magnitude,
            "family_contribution": diagnostics.family_contribution,
            "transition_smoothness_proxy": diagnostics.transition_smoothness_proxy,
            "constraint_violations": diagnostics.constraint_violations,
            "visibility_uncertainty": diagnostics.visibility_uncertainty,
            "garment_uncertainty": diagnostics.garment_uncertainty,
            "interaction_uncertainty": diagnostics.interaction_uncertainty,
            "hidden_transition_uncertainty": diagnostics.hidden_transition_uncertainty,
            "fallback_usage": diagnostics.fallback_usage,
            "explainability_summary": diagnostics.explainability_summary,
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
        phase = "early" if progress <= 0.25 else ("mid" if progress <= 0.75 else "late")
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
        labels: set[str],
        planner: PlannerTransitionContext,
        memory: MemoryInfluence,
        model_out: dict[str, float],
    ) -> TransitionIntent:
        """Собирает transition intent до вычисления GraphDelta."""
        person = scene_graph.persons[0] if scene_graph.persons else None
        target_entity = person.person_id if person else "scene"
        families = self._derive_families(labels)
        intent = TransitionIntent(
            action_families=families,
            target_entity=target_entity,
            target_regions=self._derive_target_regions(families),
            planner=planner,
            memory=memory,
            pose=self._pose_intent(labels, planner),
            garment=self._garment_intent(labels, planner, memory),
            visibility=VisibilityTransitionIntent(),
            interaction=self._interaction_intent(labels, planner),
            expression=self._expression_intent(labels, planner, model_out),
        )
        intent.visibility = self._visibility_intent(intent, labels)
        return intent

    def _derive_families(self, labels: set[str]) -> list[str]:
        families: list[str] = []
        if labels & {"raise_arm", "turn_head", "sit_down", "stand_up", "weight_shift", "posture_change"}:
            families.append("pose")
        if labels & {"remove_garment", "open_garment"}:
            families.append("garment")
        if labels & {"smile", "expression_change"}:
            families.append("expression")
        if labels & {"sit_down", "stand_up", "support", "touch"}:
            families.append("interaction")
        if families:
            families.append("visibility")
        return sorted(set(families))

    def _derive_target_regions(self, families: list[str]) -> list[str]:
        region_map = {
            "pose": ["torso", "arms", "legs"],
            "garment": ["garments", "torso"],
            "expression": ["face"],
            "interaction": ["pelvis", "legs", "support_zone"],
            "visibility": ["face", "torso", "arms", "legs", "garments"],
        }
        out: list[str] = []
        for family in families:
            out.extend(region_map.get(family, []))
        return sorted(set(out))

    def _pose_intent(self, labels: set[str], planner: PlannerTransitionContext) -> PoseTransitionIntent:
        if "sit_down" in labels:
            progression = "weight_shift" if planner.phase == "early" else ("lowering" if planner.phase == "mid" else "seated_stabilization")
            return PoseTransitionIntent(target_pose="sitting", progression=progression, weight_shift=0.35 + 0.4 * planner.intensity)
        if "stand_up" in labels:
            progression = "lift_off_support" if planner.phase != "stabilize" else "standing_stabilization"
            return PoseTransitionIntent(target_pose="standing", progression=progression, weight_shift=0.25)
        if "raise_arm" in labels:
            return PoseTransitionIntent(target_pose="arm_raised", progression="shoulder_lift", weight_shift=0.1)
        if "turn_head" in labels:
            return PoseTransitionIntent(target_pose="head_turned", progression="head_rotation", weight_shift=0.0)
        return PoseTransitionIntent()

    def _garment_intent(self, labels: set[str], planner: PlannerTransitionContext, memory: MemoryInfluence) -> GarmentTransitionIntent:
        if "remove_garment" not in labels and "open_garment" not in labels:
            return GarmentTransitionIntent()
        progression = {
            "early": "tensioned",
            "mid": "opening",
            "late": "partially_detached",
            "stabilize": "half_removed",
        }.get(planner.phase, "opening")
        if planner.phase == "stabilize" and planner.intensity > 0.65:
            progression = "removed"
        attachment_delta = -0.15 - 0.35 * planner.intensity
        reveal_bias = 0.4 * memory.hidden_reveal_evidence + 0.6 * memory.garment_memory_strength
        return GarmentTransitionIntent(progression_state=progression, attachment_delta=attachment_delta, reveal_bias=reveal_bias)

    def _interaction_intent(self, labels: set[str], planner: PlannerTransitionContext) -> InteractionTransitionIntent:
        if "sit_down" in labels:
            stage = {
                "early": "near_support",
                "mid": "approach_contact",
                "late": "weight_transfer",
                "stabilize": "stabilized_seated",
            }.get(planner.phase, "contact_established")
            return InteractionTransitionIntent(support_progression=stage, support_target="chair", contact_bias=0.3 + 0.6 * planner.intensity)
        if "stand_up" in labels:
            return InteractionTransitionIntent(support_progression="support_release", support_target="chair", contact_bias=0.2)
        return InteractionTransitionIntent()

    def _expression_intent(self, labels: set[str], planner: PlannerTransitionContext, model_out: dict[str, float]) -> ExpressionTransitionIntent:
        if "smile" not in labels:
            return ExpressionTransitionIntent()
        progression = {
            "early": "subtle_rise",
            "mid": "forming_expression",
            "late": "stable_expression",
            "stabilize": "relaxed_expression",
        }.get(planner.phase, "forming_expression")
        return ExpressionTransitionIntent(expression_label="smile", progression=progression, intensity_delta=0.1 + 0.35 * model_out["expression"])

    def _visibility_intent(self, intent: TransitionIntent, labels: set[str]) -> VisibilityTransitionIntent:
        reveal: list[str] = []
        occlude: list[str] = []
        stable: list[str] = ["torso"]
        if "raise_arm" in labels:
            reveal.extend(["left_arm", "sleeves"])
        if "remove_garment" in labels:
            reveal.extend(["torso", "inner_garment"])
            occlude.append("outer_garment")
        if "sit_down" in labels:
            occlude.append("legs")
            stable.append("face")
        if "smile" in labels:
            reveal.append("face")
        return VisibilityTransitionIntent(reveal_regions=sorted(set(reveal)), occlude_regions=sorted(set(occlude)), stable_regions=sorted(set(stable)))

    def _compute_family_deltas(self, intent: TransitionIntent, model_out: dict[str, float]) -> dict[str, dict[str, float | str]]:
        """Вычисляет sub-deltas по семействам переходов."""
        out: dict[str, dict[str, float | str]] = {"pose": {}, "garment": {}, "expression": {}, "interaction": {}, "visibility": {}}

        if "pose" in intent.action_families:
            if intent.pose.target_pose == "sitting":
                out["pose"] = {
                    "left_knee": self._clamp_pose(20.0 * model_out["pose"]),
                    "right_knee": self._clamp_pose(20.0 * model_out["pose"]),
                    "torso_pitch": self._clamp_pose(10.0 * model_out["pose"]),
                }
            elif intent.pose.target_pose == "arm_raised":
                out["pose"] = {
                    "left_shoulder": self._clamp_pose(18.0 * model_out["pose"]),
                    "left_elbow": self._clamp_pose(13.0 * model_out["pose"]),
                }
            elif intent.pose.target_pose == "head_turned":
                out["pose"] = {"head_yaw": self._clamp_pose(14.0 * model_out["pose"])}

        if "garment" in intent.action_families:
            out["garment"] = {
                "outer_attachment": intent.garment.attachment_delta,
                "garment_progression": intent.garment.progression_state,
                "coverage_expectation": max(0.0, min(1.0, 0.8 - intent.garment.reveal_bias)),
            }

        if "expression" in intent.action_families:
            out["expression"] = {
                "smile_intensity": intent.expression.intensity_delta,
                "expression_progression": intent.expression.progression,
                "expression_label": intent.expression.expression_label,
            }

        if "interaction" in intent.action_families:
            out["interaction"] = {
                "support_contact": min(1.0, max(0.0, intent.interaction.contact_bias * (0.8 + 0.4 * model_out["interaction"]))),
                "weight_transfer": intent.pose.weight_shift,
            }

        if "visibility" in intent.action_families:
            out["visibility"] = {
                region: "visible" for region in intent.visibility.reveal_regions
            }
            for region in intent.visibility.occlude_regions:
                out["visibility"][region] = "partially_visible"
        return out

    def _merge_family_deltas(self, delta: GraphDelta, family_deltas: dict[str, dict[str, float | str]], diagnostics: TransitionDiagnostics) -> None:
        """Сливает sub-deltas в финальный GraphDelta с учетом вкладов."""
        delta.pose_deltas.update({k: float(v) for k, v in family_deltas["pose"].items()})
        delta.garment_deltas.update(family_deltas["garment"])
        delta.expression_deltas.update(family_deltas["expression"])
        delta.interaction_deltas.update({k: float(v) for k, v in family_deltas["interaction"].items()})
        delta.visibility_deltas.update(family_deltas["visibility"])

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
        delta.affected_regions = sorted(set(delta.affected_regions + list(delta.visibility_deltas.keys())))

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
    ) -> None:
        """Отдельный шаг вычисления visibility/occlusion последствий."""
        reveal_state = "visible" if intent.memory.visibility_safety >= 0.45 else "partially_visible"
        for region in intent.visibility.reveal_regions:
            delta.predicted_visibility_changes[region] = reveal_state
        for region in intent.visibility.occlude_regions:
            delta.predicted_visibility_changes[region] = "partially_visible"
        for region in intent.visibility.stable_regions:
            delta.predicted_visibility_changes.setdefault(region, "visible")

        if person:
            for region in intent.visibility.reveal_regions:
                delta.newly_revealed_regions.append(
                    self._region_from_person(scene_graph, person.person_id, person.bbox, region, f"visibility_reveal:{intent.planner.phase}")
                )
            for region in intent.visibility.occlude_regions:
                delta.newly_occluded_regions.append(
                    self._region_from_person(scene_graph, person.person_id, person.bbox, region, f"visibility_occlude:{intent.planner.phase}")
                )

    def _derive_region_transition_modes(self, intent: TransitionIntent, delta: GraphDelta) -> None:
        """Формирует режимы переходов регионов по семействам."""
        for region in intent.visibility.reveal_regions:
            mode = "visibility_reveal"
            if "garment" in intent.action_families and region in {"torso", "inner_garment"}:
                mode = "garment_reveal"
            elif "pose" in intent.action_families and "arm" in region:
                mode = "pose_exposure"
            delta.region_transition_mode[region] = mode
        for region in intent.visibility.occlude_regions:
            delta.region_transition_mode[region] = "visibility_occlusion"
        if "expression" in intent.action_families:
            delta.region_transition_mode["face"] = "expression_refine"

    def _build_state_before(self, scene_graph: SceneGraph, person_id: str, expression_label: str) -> dict[str, str]:
        support_targets = SceneGraphQueries.get_supported_by(scene_graph, person_id)
        visible = SceneGraphQueries.get_visible_regions(scene_graph, person_id)
        occluded = SceneGraphQueries.get_occluded_regions(scene_graph, person_id)
        attached_garments = SceneGraphQueries.get_attached_garments(scene_graph, person_id)
        return {
            "pose_state": "standing" if support_targets else "free",
            "garment_state": "worn" if attached_garments else "removed",
            "visibility_state": "hidden" if len(occluded) > len(visible) else "visible",
            "interaction_state": "support" if support_targets else "free_contact",
            "expression_state": expression_label,
        }

    def _derive_state_after(self, delta: GraphDelta, before: dict[str, str], intent: TransitionIntent) -> dict[str, str]:
        """Агрегирует state_after из intent + sub-deltas."""
        pose_state = before.get("pose_state", "stable")
        if intent.pose.target_pose in {"sitting", "standing", "arm_raised", "head_turned"}:
            pose_state = intent.pose.target_pose

        garment_state = str(delta.garment_deltas.get("garment_progression", before.get("garment_state", "worn")))
        if garment_state == "half_removed" and intent.planner.phase == "stabilize":
            garment_state = "stabilized_removed"

        visibility_state = "revealing" if delta.newly_revealed_regions else before.get("visibility_state", "visible")
        if delta.newly_occluded_regions and not delta.newly_revealed_regions:
            visibility_state = "occluding"

        interaction_state = intent.interaction.support_progression if intent.interaction.support_progression != "free" else before.get("interaction_state", "free_contact")
        expression_state = str(delta.expression_deltas.get("expression_progression", before.get("expression_state", "neutral")))

        return {
            "pose_state": pose_state,
            "garment_state": garment_state,
            "visibility_state": visibility_state,
            "interaction_state": interaction_state,
            "expression_state": expression_state,
            "transition_phase": intent.planner.phase,
        }

    def _compute_diagnostics(self, delta: GraphDelta, intent: TransitionIntent, diagnostics: TransitionDiagnostics) -> None:
        """Считает полезную диагностику для explainability/debug."""
        magnitude = sum(abs(v) for v in delta.pose_deltas.values())
        magnitude += sum(abs(v) for v in delta.interaction_deltas.values())
        diagnostics.delta_magnitude = magnitude
        diagnostics.transition_smoothness_proxy = 1.0 / (1.0 + magnitude)

        if "interaction" in intent.action_families and delta.interaction_deltas.get("support_contact", 0.0) < 0.2:
            diagnostics.constraint_violations.append("weak_support_contact")
        if "garment" in intent.action_families and float(delta.garment_deltas.get("coverage_expectation", 1.0)) > 0.85:
            diagnostics.constraint_violations.append("garment_progress_low")

        diagnostics.visibility_uncertainty = 1.0 - intent.memory.visibility_safety
        diagnostics.garment_uncertainty = max(0.0, 1.0 - intent.memory.garment_memory_strength)
        diagnostics.interaction_uncertainty = 0.1 if "interaction" in intent.action_families else 0.0
        diagnostics.hidden_transition_uncertainty = max(0.0, 1.0 - intent.memory.hidden_reveal_evidence)

        if not delta.pose_deltas:
            diagnostics.fallback_usage.append("pose_fallback_stable")
        if not delta.visibility_deltas:
            diagnostics.fallback_usage.append("visibility_fallback_stable")

        dominant = max(diagnostics.family_contribution.items(), key=lambda item: item[1])[0] if diagnostics.family_contribution else "none"
        diagnostics.explainability_summary = (
            f"phase={intent.planner.phase};dominant_family={dominant};"
            f"visibility_safety={intent.memory.visibility_safety:.2f};"
            f"families={','.join(intent.action_families)}"
        )

    def _extract_intensity(self, labels: list[str]) -> float:
        for label in labels:
            if label.startswith("intensity="):
                try:
                    return float(label.split("=", 1)[1])
                except ValueError:
                    return 0.5
        return 0.5
