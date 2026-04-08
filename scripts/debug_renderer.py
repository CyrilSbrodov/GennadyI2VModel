import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

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
from rendering.roi_renderer import ROISelector, PatchRenderer


def make_scene() -> SceneGraph:
    person = PersonNode(
        person_id="p1",
        track_id="t1",
        bbox=BBox(0.2, 0.1, 0.5, 0.8),
        mask_ref=None,
        pose_state=PoseState(coarse_pose="standing"),
        expression_state=ExpressionState(label="neutral"),
        body_parts=[
            BodyPartNode(part_id="bp_face", part_type="face", visibility="visible"),
            BodyPartNode(part_id="bp_arm", part_type="arm", visibility="visible"),
        ],
        garments=[
            GarmentNode(garment_id="g1", garment_type="coat", garment_state="worn"),
        ],
    )
    return SceneGraph(frame_index=1, persons=[person], objects=[])


def main() -> None:
    scene = make_scene()

    # Пустая память для baseline smoke.
    memory = VideoMemory()

    # Фиктивный кадр.
    frame = np.zeros((512, 512, 3), dtype=np.uint8)

    predictor = GraphDeltaPredictor()
    planned = PlannedState(step_index=2, labels=["sit_down", "remove_garment", "smile", "intensity=0.7"])
    delta, metrics = predictor.predict(
        scene_graph=scene,
        target_state=planned,
        planner_context={"step_index": 2.0, "total_steps": 4.0, "target_duration": 2.0},
        memory=memory,
    )

    print("=== GRAPH DELTA ===")
    print("transition_phase:", delta.transition_phase)
    print("pose_deltas:", delta.pose_deltas)
    print("garment_deltas:", delta.garment_deltas)
    print("expression_deltas:", delta.expression_deltas)
    print("interaction_deltas:", delta.interaction_deltas)
    print("visibility_deltas:", delta.visibility_deltas)
    print("predicted_visibility_changes:", delta.predicted_visibility_changes)
    print("region_transition_mode:", delta.region_transition_mode)
    print("semantic_reasons:", delta.semantic_reasons)
    print("affected_regions:", delta.affected_regions)
    print("state_before:", delta.state_before)
    print("state_after:", delta.state_after)
    print("transition_diagnostics:")
    pprint(delta.transition_diagnostics)

    selector = ROISelector()
    rois = selector.select(scene, delta)

    print("\n=== ROI SELECTION ===")
    print("roi_count:", len(rois))
    for i, roi in enumerate(rois, start=1):
        print(f"[{i}] region_id={roi.region_id} reason={roi.reason} bbox={roi.bbox}")

    renderer = PatchRenderer()
    results = []

    for roi in rois:
        patch = renderer.render(
            current_frame=frame,
            scene_graph=scene,
            delta=delta,
            memory=memory,
            region=roi,
        )
        results.append(patch)

    print("\n=== RENDER RESULTS ===")
    print("patch_count:", len(results))

    for i, patch in enumerate(results, start=1):
        print(f"\n[{i}]")
        print("region_id:", patch.region.region_id)
        print("reason:", patch.region.reason)
        print("bbox:", patch.region.bbox)
        print("confidence:", patch.confidence)
        print("rgb_shape:", (patch.height, patch.width, patch.channels))
        print("alpha_shape:", (patch.height, patch.width))
        print("uncertainty_map:", "yes" if patch.uncertainty_map is not None else "no")

        print("debug_trace:")
        pprint(patch.debug_trace)

        print("execution_trace:")
        pprint(patch.execution_trace)


if __name__ == "__main__":
    main()