from __future__ import annotations

from dataclasses import dataclass

from core.region_ids import make_region_id
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
        return f"f={scene_graph.frame_index};p={len(scene_graph.persons)};o={len(scene_graph.objects)};" + ";".join(sorted(p.person_id for p in scene_graph.persons))

    def _clamp_pose(self, value: float, lo: float = -45.0, hi: float = 45.0) -> float:
        return max(lo, min(hi, value))

    def _region_from_person(self, person_id: str, person_bbox: BBox, kind: str, reason: str) -> RegionRef:
        if kind == "face":
            box = BBox(person_bbox.x + person_bbox.w * 0.3, person_bbox.y, person_bbox.w * 0.4, person_bbox.h * 0.2)
        elif kind in {"left_arm", "right_arm"}:
            x = person_bbox.x if kind == "left_arm" else person_bbox.x + person_bbox.w * 0.5
            box = BBox(x, person_bbox.y + person_bbox.h * 0.2, person_bbox.w * 0.5, person_bbox.h * 0.32)
        elif kind == "pelvis":
            box = BBox(person_bbox.x + person_bbox.w * 0.3, person_bbox.y + person_bbox.h * 0.5, person_bbox.w * 0.4, person_bbox.h * 0.2)
        elif kind == "legs":
            box = BBox(person_bbox.x + person_bbox.w * 0.18, person_bbox.y + person_bbox.h * 0.58, person_bbox.w * 0.64, person_bbox.h * 0.4)
        else:
            box = BBox(person_bbox.x + person_bbox.w * 0.1, person_bbox.y + person_bbox.h * 0.2, person_bbox.w * 0.8, person_bbox.h * 0.55)
        return RegionRef(region_id=make_region_id(person_id, kind), bbox=box, reason=reason)

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

        if "sit_down" in labels:
            delta.semantic_reasons.append("sit_down")
            delta.pose_deltas.update({"left_knee": self._clamp_pose(25.0 * model_out["pose"]), "right_knee": self._clamp_pose(25.0 * model_out["pose"]), "torso_pitch": self._clamp_pose(8.0 * model_out["pose"])})
            delta.interaction_deltas["chair_contact"] = min(1.0, 0.2 + model_out["interaction"])
            delta.affected_regions.extend(["pelvis", "legs"])
            delta.predicted_visibility_changes.update({"legs": "partially_visible", "torso": "partially_visible"})
            if person:
                delta.newly_revealed_regions.append(self._region_from_person(person.person_id, person.bbox, "pelvis", "sit_down"))

        if "remove_garment" in labels:
            delta.semantic_reasons.append("garment_change")
            delta.garment_deltas.update({"coat_attachment_torso": -0.3 * model_out["garment"], "coat_state": "half_removed" if model_out["garment"] < 0.8 else "removed"})
            delta.visibility_deltas["shirt"] = "partially_visible" if model_out["visibility"] > 0.2 else "hidden"
            delta.affected_regions.append("garments")
            delta.predicted_visibility_changes.update({"garments": "partially_visible", "torso": "visible"})
            if person:
                delta.newly_revealed_regions.append(self._region_from_person(person.person_id, person.bbox, "garments", "garment_opening"))

        if "smile" in labels:
            delta.semantic_reasons.append("expression_change")
            delta.expression_deltas.update({"smile_intensity": 0.2 * model_out["expression"], "mouth_state": "smile"})
            delta.affected_regions.append("face")
            delta.predicted_visibility_changes["face"] = "visible"
            if person:
                delta.newly_revealed_regions.append(self._region_from_person(person.person_id, person.bbox, "face", "facial_change"))

        if "raise_arm" in labels:
            delta.semantic_reasons.append("raise_arm")
            delta.pose_deltas.update({"left_shoulder": 10.0 * model_out["pose"], "left_elbow": 8.0 * model_out["pose"]})
            delta.affected_regions.extend(["left_arm", "sleeves"])
            delta.predicted_visibility_changes["left_arm"] = "visible"
            if person:
                delta.newly_revealed_regions.append(self._region_from_person(person.person_id, person.bbox, "left_arm", "arm_raise"))

        if "turn_head" in labels:
            delta.semantic_reasons.append("turn_head")
            delta.pose_deltas.update({"head_yaw": 12.0 * model_out["pose"]})
            delta.affected_regions.append("face")

        if not delta.newly_revealed_regions and person:
            delta.newly_revealed_regions.append(self._region_from_person(person.person_id, person.bbox, "torso", "stabilization"))
            delta.semantic_reasons.append("micro_adjust")

        magnitude = sum(abs(v) for v in delta.pose_deltas.values()) + sum(abs(v) for v in delta.interaction_deltas.values())
        metrics = DynamicsMetrics(
            delta_magnitude=magnitude,
            constraint_violations=1 if delta.interaction_deltas.get("chair_contact", 0.0) < 0 and "sit_down" in labels else 0,
            temporal_smoothness_proxy=1.0 / (1.0 + magnitude),
        )
        return delta, metrics
