from __future__ import annotations

from dataclasses import dataclass

from core.schema import BBox, GraphDelta, RegionRef, SceneGraph, VideoMemory
from core.semantic_roi import SemanticROIHelper
from dynamics.model import DynamicsInputs, DynamicsModel
from planning.transition_engine import PlannedState
from representation.scene_graph_queries import SceneGraphQueries


@dataclass(slots=True)
class DynamicsMetrics:
    delta_magnitude: float
    constraint_violations: int
    temporal_smoothness_proxy: float


class GraphDeltaPredictor:
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
        labels = set(target_state.labels)
        delta = GraphDelta()
        context = planner_context or {}
        person = scene_graph.persons[0] if scene_graph.persons else None

        model_out = self.model.forward(
            DynamicsInputs(
                serialized_scene_graph=self._serialize_graph(scene_graph),
                action_tokens=list(labels),
                planner_context=[float(context.get("step_index", target_state.step_index))],
                memory_features=[float(len(memory.region_descriptors)) / 10.0] if memory else [0.0],
            )
        )

        if person:
            delta.affected_entities.append(person.person_id)
            supports = any(r.relation == "supports" and r.source == person.person_id for r in scene_graph.relations)
            visible = SceneGraphQueries.get_visible_regions(scene_graph, person.person_id)
            occluded = SceneGraphQueries.get_occluded_regions(scene_graph, person.person_id)
            delta.state_before = {
                "pose_state": person.pose_state.coarse_pose or "standing",
                "garment_state": "removed" if any(g.garment_state == "removed" for g in person.garments) else "worn",
                "visibility_state": "hidden" if len(occluded) > len(visible) else "visible",
                "interaction_state": "support" if supports else "contact",
            }

        if "sit_down" in labels:
            delta.semantic_reasons.append("sit_down")
            phase = int(context.get("step_index", target_state.step_index)) % 4
            phase_name = ["bend_knees", "lower_pelvis", "contact_chair", "stabilize_pose"][phase]
            delta.transition_phase = phase_name
            delta.pose_deltas.update({"left_knee": self._clamp_pose(25.0 * model_out["pose"]), "right_knee": self._clamp_pose(25.0 * model_out["pose"]), "torso_pitch": self._clamp_pose(8.0 * model_out["pose"])})
            base_contact = min(1.0, 0.2 + model_out["interaction"])
            if phase_name == "contact_chair":
                base_contact = max(base_contact, 0.7)
            if phase_name == "stabilize_pose":
                base_contact = max(base_contact, 0.9)
            delta.interaction_deltas["chair_contact"] = base_contact
            delta.affected_regions.extend(["pelvis", "legs"])
            delta.predicted_visibility_changes.update({"legs": "partially_visible", "torso": "partially_visible"})
            if person:
                delta.newly_revealed_regions.append(self._region_from_person(scene_graph, person.person_id, person.bbox, "pelvis", f"sit_down:{phase_name}"))

        if "remove_garment" in labels:
            delta.semantic_reasons.append("garment_change")
            delta.garment_deltas.update({"coat_attachment_torso": -0.3 * model_out["garment"], "coat_state": "half_removed" if model_out["garment"] < 0.8 else "removed"})
            delta.visibility_deltas["shirt"] = "partially_visible" if model_out["visibility"] > 0.2 else "hidden"
            delta.affected_regions.append("garments")
            delta.predicted_visibility_changes.update({"garments": "partially_visible", "torso": "visible"})
            if person:
                delta.newly_revealed_regions.append(self._region_from_person(scene_graph, person.person_id, person.bbox, "garments", "garment_opening"))

        if "smile" in labels:
            delta.semantic_reasons.append("expression_change")
            delta.expression_deltas.update({"smile_intensity": 0.2 * model_out["expression"], "mouth_state": "smile"})
            delta.affected_regions.append("face")
            delta.predicted_visibility_changes["face"] = "visible"
            if person:
                delta.newly_revealed_regions.append(self._region_from_person(scene_graph, person.person_id, person.bbox, "face", "facial_change"))

        if "raise_arm" in labels:
            delta.semantic_reasons.append("raise_arm")
            delta.pose_deltas.update({"left_shoulder": 10.0 * model_out["pose"], "left_elbow": 8.0 * model_out["pose"]})
            delta.affected_regions.extend(["left_arm", "sleeves"])
            delta.predicted_visibility_changes["left_arm"] = "visible"
            if person:
                delta.newly_revealed_regions.append(self._region_from_person(scene_graph, person.person_id, person.bbox, "left_arm", "arm_raise"))

        if "turn_head" in labels:
            delta.semantic_reasons.append("turn_head")
            delta.pose_deltas.update({"head_yaw": 12.0 * model_out["pose"]})
            delta.affected_regions.append("face")

        if not delta.newly_revealed_regions and person:
            delta.newly_revealed_regions.append(self._region_from_person(scene_graph, person.person_id, person.bbox, "torso", "stabilization"))
            delta.semantic_reasons.append("micro_adjust")

        if person:
            pose_after = person.pose_state.coarse_pose or "standing"
            if "sit_down" in labels:
                pose_after = "transitioning" if delta.transition_phase != "stabilize_pose" else "sitting"
            garment_after = delta.garment_deltas.get("coat_state", "worn")
            visible_after = "revealed" if any(v == "visible" for v in delta.predicted_visibility_changes.values()) else delta.state_before.get("visibility_state", "visible")
            interaction_after = "support" if delta.interaction_deltas.get("chair_contact", 0.0) >= 0.65 else delta.state_before.get("interaction_state", "contact")
            delta.state_after = {
                "pose_state": str(pose_after),
                "garment_state": str(garment_after),
                "visibility_state": str(visible_after),
                "interaction_state": str(interaction_after),
            }

        magnitude = sum(abs(v) for v in delta.pose_deltas.values()) + sum(abs(v) for v in delta.interaction_deltas.values())
        metrics = DynamicsMetrics(
            delta_magnitude=magnitude,
            constraint_violations=1 if delta.interaction_deltas.get("chair_contact", 0.0) < 0 and "sit_down" in labels else 0,
            temporal_smoothness_proxy=1.0 / (1.0 + magnitude),
        )
        return delta, metrics
