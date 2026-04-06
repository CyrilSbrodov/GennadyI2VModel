from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import ActionPlan, BBox, PlannerDiagnostics, SceneGraph, SceneObjectNode


@dataclass(slots=True)
class PlannedState:
    step_index: int
    labels: list[str] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0


@dataclass(slots=True)
class StatePlan:
    steps: list[PlannedState] = field(default_factory=list)
    diagnostics: PlannerDiagnostics = field(default_factory=PlannerDiagnostics)


class TransitionPlanner:
    _templates = {
        "remove_garment": ["hand_to_garment", "garment_opening", "garment_half_removed", "garment_removed"],
        "sit_down": ["weight_shift", "knees_bending", "chair_contact", "seated"],
        "smile": ["mouth_corner_raise", "smile_stable"],
        "raise_arm": ["shoulder_lift", "elbow_bend", "arm_raised"],
        "turn_head": ["head_rotation_start", "head_rotation_mid", "head_rotation_end"],
        "stand_up": ["forward_lean", "lift_off_support", "standing"],
        "open_garment": ["zipper_grasp", "zipper_opening", "garment_open"],
        "hold_pose": ["steady"],
    }

    def expand(
        self,
        scene_graph: SceneGraph,
        action_plan: ActionPlan,
        scene_constraints: dict[str, bool] | None = None,
        runtime_profile: dict[str, int | float | bool] | None = None,
        policy: str = "fail",
        fps: int | None = None,
        target_duration_sec: float | None = None,
    ) -> StatePlan:
        constraints = scene_constraints or self._constraints_from_scene(scene_graph)
        profile = runtime_profile or {}
        effective_fps = fps or int(profile.get("fps", scene_graph.global_context.fps or 16))
        max_steps = int(profile.get("max_transition_steps", 999))
        diagnostics = PlannerDiagnostics()

        steps: list[PlannedState] = [PlannedState(step_index=0, labels=["initial_state"], start_frame=0, end_frame=0)]
        cursor = 1
        current_frame = 1

        for action in action_plan.actions:
            ok, reason = self._precheck_action(action.type, constraints)
            if not ok:
                if policy == "insert":
                    diagnostics.inserted_objects.append(reason)
                    diagnostics.policy_decisions.append(f"auto-insert::{reason}")
                    constraints[reason] = True
                    if reason == "chair":
                        scene_graph.objects.append(
                            SceneObjectNode(
                                object_id="chair_auto",
                                object_type="chair",
                                bbox=BBox(0.6, 0.5, 0.3, 0.4),
                                confidence=0.4,
                                source="planner:auto_insert",
                            )
                        )
                elif policy == "use_existing":
                    diagnostics.skipped_actions.append(action.type)
                    diagnostics.policy_decisions.append(f"soft-skip::{action.type}")
                    continue
                else:
                    diagnostics.skipped_actions.append(action.type)
                    diagnostics.constraint_warnings.append(f"hard-fail::{reason}")
                    diagnostics.policy_decisions.append(f"hard-fail::{action.type}:{reason}")
                    continue

            labels = self._templates.get(action.type, [action.type])
            intensity = float(action.intensity if action.intensity is not None else 0.5)
            step_duration = self._step_duration_frames(
                num_labels=len(labels),
                action_duration_sec=action.duration_sec,
                fps=effective_fps,
                target_duration_sec=target_duration_sec,
                action_count=max(1, len(action_plan.actions)),
                intensity=intensity,
            )
            for label in labels:
                if cursor >= max_steps:
                    diagnostics.constraint_warnings.append("max_transition_steps_reached")
                    break
                steps.append(
                    PlannedState(
                        step_index=cursor,
                        labels=[action.type, label, f"intensity={intensity:.2f}"],
                        start_frame=current_frame,
                        end_frame=current_frame + step_duration,
                    )
                )
                current_frame += step_duration
                cursor += 1

        return StatePlan(steps=steps, diagnostics=diagnostics)

    def _constraints_from_scene(self, scene_graph: SceneGraph) -> dict[str, bool]:
        object_types = {obj.object_type for obj in scene_graph.objects}
        has_outer = any(
            garment.garment_type in {"coat", "jacket", "hoodie"}
            for person in scene_graph.persons
            for garment in person.garments
        )
        return {
            "chair": "chair" in object_types,
            "outer_garment": has_outer,
            "anatomy": any(person.body_parts for person in scene_graph.persons),
        }

    def _precheck_action(self, action_type: str, constraints: dict[str, bool]) -> tuple[bool, str]:
        if action_type == "sit_down" and not constraints.get("chair", False):
            return False, "chair"
        if action_type == "remove_garment" and not constraints.get("outer_garment", False):
            return False, "outer_garment"
        if action_type in {"raise_arm", "turn_head"} and not constraints.get("anatomy", False):
            return False, "anatomy"
        return True, ""

    def _step_duration_frames(
        self,
        num_labels: int,
        action_duration_sec: float | None,
        fps: int,
        target_duration_sec: float | None,
        action_count: int,
        intensity: float,
    ) -> int:
        intensity_scale = max(0.6, min(1.4, 1.2 - intensity * 0.4))
        if action_duration_sec is not None:
            return max(1, int(((action_duration_sec * intensity_scale) * fps) / max(1, num_labels)))
        if target_duration_sec is not None:
            return max(1, int((target_duration_sec * fps) / max(1, action_count * num_labels)))
        return 1
