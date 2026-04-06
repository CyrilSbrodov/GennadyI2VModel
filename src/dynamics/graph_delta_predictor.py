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
                scene_graph_features=[float(scene_graph.frame_index)],
                action_embedding=[0.4 if "sit_down" in labels else 0.2],
                planner_context=[float(context.get("step_index", target_state.step_index))],
                memory_features=[float(len(memory.region_descriptors)) / 10.0] if memory else [0.0],
            )
        )
        delta_scale = max(0.05, abs(model_out.get("delta_scale", 0.1)))

        if "sit_down" in labels:
            delta.pose_deltas.update({"left_knee": 12.0 * delta_scale, "right_knee": 12.0 * delta_scale, "torso_pitch": 3.5})
            delta.interaction_deltas["chair_contact"] = min(1.0, 0.25 + delta_scale)
            delta.newly_revealed_regions.append(RegionRef("torso_reveal", BBox(0.4, 0.4, 0.2, 0.18), "posture_change"))

        if "remove_garment" in labels:
            delta.garment_deltas.update({"coat_attachment_torso": -0.25 * delta_scale, "coat_state": "half_removed"})
            delta.visibility_deltas["shirt"] = "partially_visible"
            delta.newly_revealed_regions.append(RegionRef("shirt_reveal", BBox(0.38, 0.3, 0.24, 0.28), "garment_opening"))

        if "smile" in labels:
            delta.expression_deltas.update({"smile_intensity": 0.12 * delta_scale, "mouth_state": "smile"})
            delta.newly_revealed_regions.append(RegionRef("face_expression", BBox(0.43, 0.14, 0.14, 0.11), "facial_change"))

        if not delta.newly_revealed_regions:
            delta.newly_revealed_regions.append(RegionRef("micro_adjust", BBox(0.45, 0.22, 0.1, 0.1), "stabilization"))

        magnitude = sum(abs(v) for v in delta.pose_deltas.values()) + sum(
            abs(v) for v in delta.interaction_deltas.values()
        )
        metrics = DynamicsMetrics(
            delta_magnitude=magnitude,
            constraint_violations=1 if delta.interaction_deltas.get("chair_contact", 0.0) < 0 and "sit_down" in labels else 0,
            temporal_smoothness_proxy=1.0 / (1.0 + magnitude),
        )
        return delta, metrics
