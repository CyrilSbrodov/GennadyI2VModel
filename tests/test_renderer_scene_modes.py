from __future__ import annotations

from core.schema import BBox, GraphDelta, PersonNode, RegionRef, SceneGraph, SceneObjectNode
from memory.video_memory import MemoryManager
from rendering.compositor import Compositor
from rendering.roi_renderer import PatchRenderer


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def _scene() -> SceneGraph:
    return SceneGraph(
        frame_index=0,
        persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)],
        objects=[SceneObjectNode(object_id="chair_1", object_type="chair", bbox=BBox(0.55, 0.55, 0.3, 0.35), mask_ref=None)],
    )


def test_existing_region_update_path_is_selected_for_existing_entity() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    renderer = PatchRenderer()
    region = RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.15, 0.25, 0.25), reason="expression")
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["face"], region_transition_mode={"face": "expression_refine"})

    patch = renderer.render(_solid(64, 64, (0.2, 0.3, 0.4)), graph, delta, memory, region)

    assert patch.execution_trace["selection"]["selected_render_mode"] == "refine"
    assert patch.execution_trace["selection"]["entity_lifecycle"] == "already_existing"
    assert patch.execution_trace["selected_render_strategy"] == "EXISTING_REGION_UPDATE"


def test_reveal_region_path_uses_hidden_lifecycle() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    memory = mm.update_from_frame(memory, _solid(64, 64, (0.8, 0.2, 0.2)), graph)
    rid = "p1:garments"
    region = RegionRef(region_id=rid, bbox=BBox(0.22, 0.22, 0.35, 0.45), reason="reveal")
    delta = GraphDelta(newly_revealed_regions=[region], affected_entities=["p1"], affected_regions=["garments"])
    renderer = PatchRenderer()

    patch = renderer.render(_solid(64, 64, (0.25, 0.25, 0.25)), graph, delta, memory, region)

    assert patch.execution_trace["selection"]["selected_render_mode"] == "reveal"
    assert patch.execution_trace["selection"]["entity_lifecycle"] == "previously_hidden_now_revealed"
    assert patch.execution_trace["module_trace"]["module"] == "reveal_synthesizer"


def test_new_entity_insertion_path_is_selected_for_absent_entity() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    renderer = PatchRenderer()
    region = RegionRef(region_id="p2:torso", bbox=BBox(0.58, 0.25, 0.25, 0.45), reason="new_person")
    delta = GraphDelta(
        affected_entities=["p2"],
        affected_regions=["torso"],
        transition_diagnostics={"insertion_context": {"p2": {"entity_type": "person", "initial_pose_role": "standing"}}},
    )

    patch = renderer.render(_solid(64, 64, (0.3, 0.3, 0.3)), graph, delta, memory, region)

    assert patch.execution_trace["selection"]["selected_render_mode"] == "insert_new"
    assert patch.execution_trace["selection"]["entity_lifecycle"] == "newly_inserted"
    assert patch.execution_trace["selected_render_strategy"] == "NEW_ENTITY_INSERTION"
    assert region.region_id in memory.region_descriptors
    assert patch.execution_trace["entity_registration_summary"]["registered_texture_patch"] is True
    assert patch.execution_trace["entity_registration_summary"]["registered_hidden_slot"] is True
    assert patch.execution_trace["learnable_mode_surface"]["insertion_path_contract"]["reusable_artifact_expected"] is True
    assert patch.execution_trace["reusable_output"]["reusable_next_frame"] is True

    # insertion output должен быть содержательным, а не плоской заливкой
    center = patch.rgb_patch[len(patch.rgb_patch) // 2][len(patch.rgb_patch[0]) // 2]
    corner = patch.rgb_patch[0][0]
    assert center != corner
    assert patch.confidence > 0.35


def test_mode_router_does_not_send_existing_region_to_insertion() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    renderer = PatchRenderer()
    region = RegionRef(region_id="p1:torso", bbox=BBox(0.2, 0.25, 0.35, 0.45), reason="stable")
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["torso"], region_transition_mode={"torso": "stable"})

    patch = renderer.render(_solid(64, 64, (0.4, 0.4, 0.4)), graph, delta, memory, region)

    assert patch.execution_trace["selection"]["selected_render_mode"] == "keep"
    assert patch.execution_trace["selected_render_strategy"] == "EXISTING_REGION_UPDATE"


def test_layered_compositing_prefers_inserted_entity_on_top() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    renderer = PatchRenderer()
    frame = _solid(32, 32, (0.1, 0.1, 0.1))

    existing = renderer.render(
        frame,
        graph,
        GraphDelta(affected_entities=["p1"], affected_regions=["torso"], region_transition_mode={"torso": "stable"}),
        memory,
        RegionRef(region_id="p1:torso", bbox=BBox(0.3, 0.3, 0.3, 0.3), reason="existing"),
    )
    inserted = renderer.render(
        frame,
        graph,
        GraphDelta(affected_entities=["pX"], affected_regions=["torso"], transition_diagnostics={"insertion_context": {"pX": {"entity_type": "object"}}}),
        memory,
        RegionRef(region_id="pX:torso", bbox=BBox(0.3, 0.3, 0.3, 0.3), reason="insert"),
    )
    composed = Compositor().compose(frame, [existing, inserted], GraphDelta())

    assert inserted.z_index > existing.z_index
    assert composed[16][16] != frame[16][16]
    assert inserted.execution_trace["layer_priority"] > existing.execution_trace["layer_priority"]


def test_metadata_and_debug_trace_include_router_and_memory_semantics() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    renderer = PatchRenderer()
    region = RegionRef(region_id="p1:left_arm", bbox=BBox(0.15, 0.3, 0.25, 0.3), reason="pose")
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["left_arm"], region_transition_mode={"left_arm": "pose_exposure"})

    patch = renderer.render(_solid(64, 64, (0.22, 0.28, 0.36)), graph, delta, memory, region)

    assert any(msg.startswith("render_mode=") for msg in patch.debug_trace)
    assert any(msg.startswith("lifecycle=") for msg in patch.debug_trace)
    assert "memory_dependency_summary" in patch.execution_trace
    assert "mode_reasons" in patch.execution_trace["selection"]
    assert "learnable_mode_surface" in patch.execution_trace
    surface = patch.execution_trace["learnable_mode_surface"]
    assert "update_path_contract" in surface or "reveal_path_contract" in surface


def test_reveal_path_has_hidden_memory_specific_semantics() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    memory = mm.update_from_frame(memory, _solid(64, 64, (0.7, 0.25, 0.25)), graph)
    renderer = PatchRenderer()
    region = RegionRef(region_id="p1:garments", bbox=BBox(0.22, 0.22, 0.35, 0.45), reason="reveal")
    delta = GraphDelta(
        newly_revealed_regions=[region],
        affected_entities=["p1"],
        affected_regions=["garments"],
        region_transition_mode={"garments": "garment_reveal"},
    )
    patch = renderer.render(_solid(64, 64, (0.25, 0.25, 0.25)), graph, delta, memory, region)

    assert patch.execution_trace["selection"]["selected_render_mode"] == "reveal"
    assert patch.execution_trace["module_trace"]["reveal_confidence_semantics"]
    assert patch.execution_trace["module_trace"]["hidden_region_slots_used"] is True
    assert "reveal_path_contract" in patch.execution_trace["learnable_mode_surface"]


def test_inserted_entity_is_routed_as_existing_on_next_step() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    renderer = PatchRenderer()
    ins_region = RegionRef(region_id="p2:torso", bbox=BBox(0.58, 0.25, 0.25, 0.45), reason="new_person")
    ins_delta = GraphDelta(
        affected_entities=["p2"],
        affected_regions=["torso"],
        transition_diagnostics={"insertion_context": {"p2": {"entity_type": "person", "initial_pose_role": "standing"}}},
    )
    first = renderer.render(_solid(64, 64, (0.3, 0.3, 0.3)), graph, ins_delta, memory, ins_region)
    assert first.execution_trace["selection"]["selected_render_mode"] == "insert_new"

    # на следующем шаге та же сущность должна стать reusable existing path
    upd_delta = GraphDelta(affected_entities=["p2"], affected_regions=["torso"], region_transition_mode={"torso": "stable"})
    second = renderer.render(_solid(64, 64, (0.31, 0.31, 0.31)), graph, upd_delta, memory, ins_region)
    assert second.execution_trace["selection"]["selected_render_mode"] in {"keep", "warp", "refine", "deform"}
    assert second.execution_trace["selection"]["entity_lifecycle"] == "already_existing"
