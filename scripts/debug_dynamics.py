import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

scene = SceneGraph(
    frame_index=1,
    persons=[
        PersonNode(
            person_id="p1",
            track_id="t1",
            bbox=BBox(0.2, 0.1, 0.5, 0.8),
            mask_ref=None,
            pose_state=PoseState(coarse_pose="standing"),
            expression_state=ExpressionState(label="neutral"),
            body_parts=[BodyPartNode(part_id="bp1", part_type="arm", visibility="visible")],
            garments=[GarmentNode(garment_id="g1", garment_type="coat", garment_state="worn")],
        )
    ],
)

planned = PlannedState(step_index=2, labels=["sit_down", "remove_garment", "smile", "intensity=0.7"])
memory = VideoMemory()

predictor = GraphDeltaPredictor()
delta, metrics = predictor.predict(
    scene_graph=scene,
    target_state=planned,
    planner_context={"step_index": 2.0, "total_steps": 4.0, "target_duration": 2.0},
    memory=memory,
)

print("TRANSITION PHASE:", delta.transition_phase)
print("POSE DELTAS:", delta.pose_deltas)
print("GARMENT DELTAS:", delta.garment_deltas)
print("EXPRESSION DELTAS:", delta.expression_deltas)
print("INTERACTION DELTAS:", delta.interaction_deltas)
print("VISIBILITY DELTAS:", delta.visibility_deltas)
print("PREDICTED VISIBILITY:", delta.predicted_visibility_changes)
print("REGION MODES:", delta.region_transition_mode)
print("STATE BEFORE:", delta.state_before)
print("STATE AFTER:", delta.state_after)
print("DIAGNOSTICS:", delta.transition_diagnostics)
print("METRICS:", metrics)