from core.schema import (
    BBox,
    BodyPartNode,
    ExpressionState,
    GarmentNode,
    PersonNode,
    PoseState,
    RuntimeSemanticTransition,
    SceneGraph,
    VideoMemory,
)
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from planning.transition_engine import PlannedState, TransitionPlanner
from text.intent_parser import IntentParser


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

    assert delta.transition_phase in {"prepare", "transition", "contact_or_reveal", "stabilize"}
    assert delta.pose_deltas
    assert delta.garment_deltas.get("garment_progression")
    assert delta.expression_deltas.get("expression_progression")
    assert delta.interaction_deltas.get("support_contact", 0.0) >= 0.0
    assert delta.visibility_deltas
    assert delta.transition_diagnostics.get("family_contribution")
    assert metrics.delta_magnitude > 0
    assert delta.transition_diagnostics.get("semantic_families")
    assert "canonical_goals" in delta.transition_diagnostics
    assert "phase_sequence" in delta.transition_diagnostics


def test_arm_elevation_targets_arm_regions_not_full_body() -> None:
    predictor = GraphDeltaPredictor(strict_mode=True)
    planned = PlannedState(step_index=1, labels=["raise_arm", "shoulder_lift", "intensity=0.5"])

    delta, _ = predictor._predict_legacy(_scene(), planned, {"step_index": 1.0, "total_steps": 4.0}, VideoMemory())

    assert delta.pose_deltas.get("left_shoulder", 0.0) > 0.0
    assert "left_arm" in delta.transition_diagnostics["target_profile"]["primary_regions"]
    assert delta.region_transition_mode.get("sleeves") == "pose_exposure"


def test_head_rotation_and_smile_focus_face_head_neck() -> None:
    predictor = GraphDeltaPredictor(strict_mode=True)
    planned = PlannedState(step_index=2, labels=["turn_head", "head_rotation_mid", "smile", "intensity=0.6"])

    delta, _ = predictor._predict_legacy(_scene(), planned, {"step_index": 2.0, "total_steps": 4.0}, VideoMemory())

    assert "face" in delta.transition_diagnostics["target_regions"]
    assert delta.region_transition_mode.get("face") == "expression_refine"
    assert delta.expression_deltas.get("expression_label") == "smile"


def test_garment_opening_visibility_reveal_progression() -> None:
    predictor = GraphDeltaPredictor(strict_mode=True)
    planned = PlannedState(step_index=3, labels=["open_garment", "garment_opening", "intensity=0.8"])
    memory = VideoMemory(hidden_region_slots={"h1": object()}, garment_memory={"g1": object()})

    delta, _ = predictor._predict_legacy(_scene(), planned, {"step_index": 3.0, "total_steps": 4.0}, memory)

    assert delta.garment_deltas.get("garment_progression") in {"partially_detached", "half_removed", "removed", "opening"}
    assert "inner_garment" in delta.visibility_deltas
    assert delta.region_transition_mode.get("inner_garment") == "garment_reveal"


def test_state_contract_consistent_with_phase_and_interaction() -> None:
    predictor = GraphDeltaPredictor(strict_mode=True)
    planned = PlannedState(step_index=4, labels=["sit_down", "chair_contact", "intensity=0.7"])

    delta, _ = predictor._predict_legacy(_scene(), planned, {"step_index": 4.0, "total_steps": 4.0}, VideoMemory())

    assert delta.state_before["phase_state"] == "prepare"
    assert delta.state_after["transition_phase"] == "stabilize"
    assert delta.state_after["phase_state"] == "stabilize"
    assert delta.state_after["pose_state"] == "seated_pose"
    assert delta.state_after["pose_subphase"]
    assert delta.state_after["interaction_subphase"]
    assert delta.state_after["interaction_state"] in {"weight_transfer", "contact_established", "approach_contact", "stabilized_seated"}


def test_planner_passes_first_class_semantic_transition_to_dynamics() -> None:
    parser = IntentParser()
    action_plan = parser.parse("Садится на стул")
    state_plan = TransitionPlanner().expand(_scene(), action_plan)
    target_step = next(step for step in state_plan.steps if step.step_index > 0)
    assert isinstance(target_step.semantic_transition, RuntimeSemanticTransition)

    predictor = GraphDeltaPredictor(strict_mode=True)
    delta, _ = predictor._predict_legacy(_scene(), target_step, {"step_index": float(target_step.step_index), "total_steps": 4.0}, VideoMemory())
    assert delta.transition_diagnostics["lexical_bootstrap_influence"]["used_runtime_semantic_contract"] is True


def test_region_role_and_transition_mode_are_separated() -> None:
    predictor = GraphDeltaPredictor(strict_mode=True)
    planned = PlannedState(step_index=3, labels=["open_garment", "intensity=0.8"])
    delta, _ = predictor._predict_legacy(_scene(), planned, {"step_index": 3.0, "total_steps": 4.0}, VideoMemory())

    assert "primary_goal_region" in delta.transition_diagnostics["region_selection_rationale"].values()
    assert "primary_transition" not in delta.region_transition_mode.values()
    assert "secondary_influence" not in delta.region_transition_mode.values()


def test_diagnostics_separate_global_phase_sequence_from_family_subphases() -> None:
    predictor = GraphDeltaPredictor(strict_mode=True)
    planned = PlannedState(step_index=2, labels=["sit_down", "intensity=0.6"])
    delta, _ = predictor._predict_legacy(_scene(), planned, {"step_index": 2.0, "total_steps": 4.0}, VideoMemory())

    phase_seq = delta.transition_diagnostics["phase_sequence"]
    assert phase_seq["global"] == ["prepare", "transition", "contact_or_reveal", "stabilize"]
    assert "pose" in phase_seq["family_subphases"]
    assert "lowering" in phase_seq["family_subphases"]["pose"] or "weight_shift" in phase_seq["family_subphases"]["pose"]
