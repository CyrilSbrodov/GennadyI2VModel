from __future__ import annotations

from core.schema import GraphDelta, SceneGraph


def apply_delta(scene_graph: SceneGraph, delta: GraphDelta) -> SceneGraph:
    if scene_graph.persons:
        person = scene_graph.persons[0]

        if "smile_intensity" in delta.expression_deltas:
            person.expression_state.smile_intensity = min(
                1.0,
                max(0.0, person.expression_state.smile_intensity + float(delta.expression_deltas["smile_intensity"])),
            )
            person.expression_state.label = str(delta.expression_deltas.get("mouth_state", "smile"))

        if delta.pose_deltas:
            person.pose_state.coarse_pose = "transition"
            if delta.interaction_deltas.get("chair_contact", 0.0) > 0.5:
                person.pose_state.coarse_pose = "seated"
            elif delta.interaction_deltas.get("chair_contact", 1.0) < 0.3:
                person.pose_state.coarse_pose = "standing"

        for garment in person.garments:
            if garment.garment_type == "coat" and "coat_state" in delta.garment_deltas:
                garment.garment_state = str(delta.garment_deltas["coat_state"])
            if garment.garment_type in delta.visibility_deltas:
                garment.visibility = delta.visibility_deltas[garment.garment_type]
            elif garment.garment_id in delta.visibility_deltas:
                garment.visibility = delta.visibility_deltas[garment.garment_id]

    scene_graph.frame_index += 1
    return scene_graph
