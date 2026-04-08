import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from core.schema import HiddenRegionSlot, TexturePatchMemory

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


def make_patch(h: int, w: int, value: float) -> list[list[list[float]]]:
    return [[[value, value, value] for _ in range(w)] for _ in range(h)]


def seed_memory(memory: VideoMemory) -> None:
    memory.patch_cache["patch_face_1"] = make_patch(64, 64, 0.35)
    memory.patch_cache["patch_outer_1"] = make_patch(96, 128, 0.55)
    memory.patch_cache["patch_inner_1"] = make_patch(96, 128, 0.65)
    memory.patch_cache["patch_torso_1"] = make_patch(128, 128, 0.45)

    memory.texture_patches["patch_face_1"] = TexturePatchMemory(
        patch_id="patch_face_1",
        region_type="face",
        entity_id="p1",
        source_frame=0,
        patch_ref="debug://patch_face_1",
        confidence=0.72,
        descriptor={"mean": [0.35, 0.35, 0.35], "std": [0.02, 0.02, 0.02], "edge_density": 0.05, "energy": 0.12},
        evidence_score=0.61,
        semantic_family="face",
        coverage_targets=["face"],
        attachment_targets=["face"],
        suitable_for_reveal=False,
    )

    memory.texture_patches["patch_outer_1"] = TexturePatchMemory(
        patch_id="patch_outer_1",
        region_type="outer_garment",
        entity_id="p1",
        source_frame=0,
        patch_ref="debug://patch_outer_1",
        confidence=0.81,
        descriptor={"mean": [0.55, 0.55, 0.55], "std": [0.03, 0.03, 0.03], "edge_density": 0.08, "energy": 0.21},
        evidence_score=0.74,
        semantic_family="garment",
        coverage_targets=["outer_garment", "torso"],
        attachment_targets=["torso"],
        suitable_for_reveal=False,
    )

    memory.texture_patches["patch_inner_1"] = TexturePatchMemory(
        patch_id="patch_inner_1",
        region_type="inner_garment",
        entity_id="p1",
        source_frame=0,
        patch_ref="debug://patch_inner_1",
        confidence=0.84,
        descriptor={"mean": [0.65, 0.65, 0.65], "std": [0.04, 0.04, 0.04], "edge_density": 0.07, "energy": 0.28},
        evidence_score=0.79,
        semantic_family="garment",
        coverage_targets=["inner_garment", "torso"],
        attachment_targets=["torso"],
        suitable_for_reveal=True,
    )

    memory.texture_patches["patch_torso_1"] = TexturePatchMemory(
        patch_id="patch_torso_1",
        region_type="torso",
        entity_id="p1",
        source_frame=0,
        patch_ref="debug://patch_torso_1",
        confidence=0.77,
        descriptor={"mean": [0.45, 0.45, 0.45], "std": [0.03, 0.03, 0.03], "edge_density": 0.06, "energy": 0.18},
        evidence_score=0.68,
        semantic_family="torso",
        coverage_targets=["torso"],
        attachment_targets=["torso"],
        suitable_for_reveal=True,
    )

    memory.hidden_region_slots["p1:inner_garment"] = HiddenRegionSlot(
        slot_id="p1:inner_garment",
        region_type="inner_garment",
        owner_entity="p1",
        candidate_patch_ids=["patch_inner_1"],
        confidence=0.82,
        hidden_type="known_hidden",
        evidence_score=0.76,
    )

    memory.hidden_region_slots["p1:torso"] = HiddenRegionSlot(
        slot_id="p1:torso",
        region_type="torso",
        owner_entity="p1",
        candidate_patch_ids=["patch_torso_1"],
        confidence=0.51,
        hidden_type="unknown_hidden",
        evidence_score=0.48,
    )

    memory.identity_memory["p1"] = {"identity_strength": 0.9}
    memory.garment_memory["g1"] = {"garment_type": "coat"}


def main() -> None:
    scene = make_scene()
    memory = VideoMemory()
    seed_memory(memory)

    frame = np.zeros((512, 512, 3), dtype=np.float32)

    predictor = GraphDeltaPredictor()
    planned = PlannedState(step_index=2, labels=["sit_down", "remove_garment", "smile", "intensity=0.7"])
    delta, metrics = predictor.predict(
        scene_graph=scene,
        target_state=planned,
        planner_context={"step_index": 2.0, "total_steps": 4.0, "target_duration": 2.0},
        memory=memory,
    )

    print("=== GRAPH DELTA ===")
    print("semantic_reasons:", delta.semantic_reasons)
    print("affected_regions:", delta.affected_regions)
    print("region_transition_mode:", delta.region_transition_mode)
    print("state_before:", delta.state_before)
    print("state_after:", delta.state_after)
    print("transition_diagnostics:")
    pprint(delta.transition_diagnostics)

    selector = ROISelector()
    rois = selector.select(scene, delta)

    print("\n=== ROI SELECTION ===")
    for i, roi in enumerate(rois, start=1):
        print(f"[{i}] {roi.region_id} reason={roi.reason} bbox={roi.bbox}")

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
    for i, patch in enumerate(results, start=1):
        print(f"\n[{i}] region_id={patch.region.region_id}")
        print("confidence:", patch.confidence)
        print("debug_trace:")
        pprint(patch.debug_trace)

        selection = patch.execution_trace.get("selection", {})
        retrieval = patch.execution_trace.get("retrieval", {})
        hidden_state = patch.execution_trace.get("hidden_state", {})
        proposal = patch.execution_trace.get("proposal", {})
        fallback_reason = patch.execution_trace.get("fallback_reason", None)

        print("selection:")
        pprint(selection)
        print("hidden_state:")
        pprint(hidden_state)
        print("retrieval:")
        pprint(retrieval)
        print("proposal:")
        pprint(proposal)
        print("fallback_reason:", fallback_reason)


if __name__ == "__main__":
    main()