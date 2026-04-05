from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import ActionPlan, SceneGraph


@dataclass(slots=True)
class PlannedState:
    step_index: int
    labels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StatePlan:
    steps: list[PlannedState] = field(default_factory=list)


class TransitionPlanner:
    """Template-based planner that expands each action to plausible intermediates."""

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

    def expand(self, scene_graph: SceneGraph, action_plan: ActionPlan) -> StatePlan:
        _ = scene_graph
        steps: list[PlannedState] = [PlannedState(step_index=0, labels=["initial_state"])]
        cursor = 1
        for action in action_plan.actions:
            for label in self._templates.get(action.type, [action.type]):
                steps.append(PlannedState(step_index=cursor, labels=[action.type, label]))
                cursor += 1
        return StatePlan(steps=steps)
