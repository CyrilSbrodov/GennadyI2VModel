from core.schema import (
    BBox,
    BodyPartNode,
    ExpressionState,
    GarmentNode,
    PersonNode,
    PoseState,
    SceneGraph,
    VideoMemory,
)
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from planning.transition_engine import PlannedState


def _scene() -> SceneGraph:
    person = PersonNode(
        person_id="p1",
        track_id="t1",
        bbox=BBox(0.2, 0.1, 0.5, 0.8),
        mask_ref=None,
        pose_state=PoseState(coarse_pose="standing"),
        expression_state=ExpressionState(label="neutral"),
        body_parts=[BodyPartNode(part_id="bp1", part_type="arm", visibility="visible")],
        garments=[GarmentNode(garment_id="g1", garment_type="coat", garment_state="worn")],
    )
    return SceneGraph(frame_index=1, persons=[person])


def test_transition_intent_pipeline_produces_structured_delta_and_diagnostics() -> None:
    predictor = GraphDeltaPredictor()
    planned = PlannedState(step_index=2, labels=["sit_down", "remove_garment", "smile", "intensity=0.7"])
    memory = VideoMemory(identity_memory={"p1": object()}, garment_memory={"g1": object()}, hidden_region_slots={"h1": object()})

    delta, metrics = predictor.predict(
        scene_graph=_scene(),
        target_state=planned,
        planner_context={"step_index": 2.0, "total_steps": 4.0, "target_duration": 2.0},
        memory=memory,
    )

    assert delta.transition_phase in {"mid", "late", "stabilize", "early"}
    assert delta.pose_deltas
    assert delta.garment_deltas.get("garment_progression")
    assert delta.expression_deltas.get("expression_progression")
    assert delta.interaction_deltas.get("support_contact", 0.0) >= 0.0
    assert delta.visibility_deltas
    assert delta.transition_diagnostics.get("family_contribution")
    assert metrics.delta_magnitude > 0
