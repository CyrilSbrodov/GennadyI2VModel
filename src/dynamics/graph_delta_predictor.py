from __future__ import annotations

from core.schema import BBox, GraphDelta, RegionRef, SceneGraph
from planning.transition_engine import PlannedState


class GraphDeltaPredictor:
    def predict(self, scene_graph: SceneGraph, target_state: PlannedState) -> GraphDelta:
        labels = set(target_state.labels)
        delta = GraphDelta()

        if "sit_down" in labels:
            delta.pose_deltas.update({"left_knee": 12.0, "right_knee": 12.0, "torso_pitch": 3.5})
            delta.interaction_deltas["chair_contact"] = 0.25
            delta.newly_revealed_regions.append(RegionRef("torso_reveal", BBox(0.4, 0.4, 0.2, 0.18), "posture_change"))

        if "remove_garment" in labels:
            delta.garment_deltas.update({"coat_attachment_torso": -0.25, "coat_state": "half_removed"})
            delta.visibility_deltas["shirt"] = "partially_visible"
            delta.newly_revealed_regions.append(RegionRef("shirt_reveal", BBox(0.38, 0.3, 0.24, 0.28), "garment_opening"))

        if "smile" in labels:
            delta.expression_deltas.update({"smile_intensity": 0.12, "mouth_state": "smile"})
            delta.newly_revealed_regions.append(RegionRef("face_expression", BBox(0.43, 0.14, 0.14, 0.11), "facial_change"))

        if not delta.newly_revealed_regions:
            delta.newly_revealed_regions.append(RegionRef("micro_adjust", BBox(0.45, 0.22, 0.1, 0.1), "stabilization"))

        _ = scene_graph
        return delta


def apply_delta(scene_graph: SceneGraph, delta: GraphDelta) -> SceneGraph:
    if scene_graph.persons:
        person = scene_graph.persons[0]
        if "smile_intensity" in delta.expression_deltas:
            person.expression_state.smile_intensity = min(
                1.0, person.expression_state.smile_intensity + float(delta.expression_deltas["smile_intensity"])
            )
            person.expression_state.label = "smile"
        if "left_knee" in delta.pose_deltas:
            person.pose_state.coarse_pose = "transition"
            if delta.interaction_deltas.get("chair_contact", 0.0) > 0.5:
                person.pose_state.coarse_pose = "seated"
        for garment in person.garments:
            if garment.garment_type == "coat" and "coat_state" in delta.garment_deltas:
                garment.garment_state = str(delta.garment_deltas["coat_state"])

    scene_graph.frame_index += 1
    return scene_graph
