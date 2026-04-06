from __future__ import annotations

from dataclasses import dataclass

from core.schema import BBox, GraphDelta, RegionRef, SceneGraph, VideoMemory
from dynamics.model import DynamicsInputs, DynamicsModel
from planning.transition_engine import PlannedState


@dataclass(slots=True)
class DynamicsMetrics:
    delta_magnitude: float
    constraint_violations: int
    temporal_smoothness_proxy: float


class GraphDeltaPredictor:
    def __init__(self) -> None:
        self.model = DynamicsModel()

    def _serialize_graph(self, scene_graph: SceneGraph) -> str:
        return (
            f"f={scene_graph.frame_index};"
            f"p={len(scene_graph.persons)};o={len(scene_graph.objects)};"
            + ";".join(sorted(p.person_id for p in scene_graph.persons))
        )

    def _clamp_pose(self, value: float, lo: float = -45.0, hi: float = 45.0) -> float:
        return max(lo, min(hi, value))

    def _apply_action_rules(self, labels: set[str], delta: GraphDelta, pose_scale: float, garment_scale: float, expr_scale: float, interact_scale: float, vis_scale: float) -> None:
        if "sit_down" in labels:
            delta.pose_deltas.update(
                {
                    "left_knee": self._clamp_pose(25.0 * pose_scale),
                    "right_knee": self._clamp_pose(25.0 * pose_scale),
                    "torso_pitch": self._clamp_pose(8.0 * pose_scale),
                }
            )
            delta.interaction_deltas["chair_contact"] = min(1.0, 0.2 + interact_scale)
            delta.newly_revealed_regions.append(RegionRef("torso_reveal", BBox(0.4, 0.4, 0.2, 0.18), "posture_change"))

        if "stand_up" in labels:
            delta.pose_deltas.update({"left_knee": -6.0 * pose_scale, "right_knee": -6.0 * pose_scale, "torso_pitch": -4.0 * pose_scale})
            delta.interaction_deltas["chair_contact"] = max(0.0, 0.5 - interact_scale)

        if "remove_garment" in labels:
            delta.garment_deltas.update(
                {
                    "coat_attachment_torso": -0.3 * garment_scale,
                    "coat_state": "half_removed" if garment_scale < 0.8 else "removed",
                }
            )
            delta.visibility_deltas["shirt"] = "partially_visible" if vis_scale > 0.2 else "hidden"
            delta.newly_revealed_regions.append(RegionRef("shirt_reveal", BBox(0.38, 0.3, 0.24, 0.28), "garment_opening"))

        if "smile" in labels:
            delta.expression_deltas.update({"smile_intensity": 0.2 * expr_scale, "mouth_state": "smile"})
            delta.newly_revealed_regions.append(RegionRef("face_expression", BBox(0.43, 0.14, 0.14, 0.11), "facial_change"))

        if "raise_arm" in labels:
            delta.pose_deltas.update({"left_shoulder": 10.0 * pose_scale})
            delta.newly_revealed_regions.append(RegionRef("arm_motion", BBox(0.2, 0.25, 0.2, 0.2), "arm_raise"))

        if "turn_head" in labels:
            delta.pose_deltas.update({"head_yaw": 12.0 * pose_scale})

        if "walk_step" in labels or "shift_weight" in labels:
            delta.pose_deltas.update({"pelvis_shift": 3.0 * pose_scale, "torso_pitch": 1.5 * pose_scale})
            delta.newly_revealed_regions.append(RegionRef("lower_body", BBox(0.38, 0.45, 0.24, 0.3), "weight_shift"))

    def predict(
        self,
        scene_graph: SceneGraph,
        target_state: PlannedState,
        planner_context: dict[str, float] | None = None,
        memory: VideoMemory | None = None,
    ) -> tuple[GraphDelta, DynamicsMetrics]:
        labels = set(target_state.labels)
        delta = GraphDelta()
        context = planner_context or {}

        model_out = self.model.forward(
            DynamicsInputs(
                serialized_scene_graph=self._serialize_graph(scene_graph),
                action_tokens=list(labels),
                planner_context=[float(context.get("step_index", target_state.step_index))],
                memory_features=[float(len(memory.region_descriptors)) / 10.0] if memory else [0.0],
            )
        )

        self._apply_action_rules(
            labels=labels,
            delta=delta,
            pose_scale=model_out["pose"],
            garment_scale=model_out["garment"],
            expr_scale=model_out["expression"],
            interact_scale=model_out["interaction"],
            vis_scale=model_out["visibility"],
        )

        if not delta.newly_revealed_regions:
            delta.newly_revealed_regions.append(RegionRef("micro_adjust", BBox(0.45, 0.22, 0.1, 0.1), "stabilization"))

        magnitude = sum(abs(v) for v in delta.pose_deltas.values()) + sum(abs(v) for v in delta.interaction_deltas.values())
        metrics = DynamicsMetrics(
            delta_magnitude=magnitude,
            constraint_violations=1 if delta.interaction_deltas.get("chair_contact", 0.0) < 0 and "sit_down" in labels else 0,
            temporal_smoothness_proxy=1.0 / (1.0 + magnitude),
        )
        return delta, metrics
