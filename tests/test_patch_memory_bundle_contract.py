from __future__ import annotations

from core.schema import BBox, GraphDelta, HiddenRegionSlot, PersonNode, RegionRef, SceneGraph
from memory.video_memory import MemoryManager
from rendering.roi_renderer import PatchRenderer


def _scene_and_memory():
    scene = SceneGraph(frame_index=1, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)])
    memory = MemoryManager().initialize(scene)
    return scene, memory


def _frame() -> list:
    return [[[0.2, 0.2, 0.2] for _ in range(64)] for _ in range(64)]


def _region() -> RegionRef:
    return RegionRef("p1:face", BBox(0.2, 0.2, 0.4, 0.4), "test")


def test_patch_request_includes_region_memory_bundle() -> None:
    _, memory = _scene_and_memory()
    manager = MemoryManager()
    bundle = manager.get_region_memory_bundle(memory, "p1", "face")
    transition_context = {
        "region_memory_bundle": bundle,
        "region_memory_support_level": bundle.memory_support_level,
        "region_memory_retrieval_reasons": list(bundle.retrieval_reasons),
    }
    assert "region_memory_bundle" in transition_context
    assert "region_memory_support_level" in transition_context


def test_patch_execution_trace_records_memory_bundle_support() -> None:
    scene, memory = _scene_and_memory()
    manager = MemoryManager()
    bundle = manager.get_region_memory_bundle(memory, "p1", "face")
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["face"], region_transition_mode={"face": "expression_refine"})
    out = PatchRenderer().render(_frame(), scene, delta, memory, _region(), transition_context={"region_memory_bundle": bundle})
    trace = out.execution_trace
    assert trace["memory_bundle_present"] is True
    assert "memory_support_level" in trace
    assert "memory_bundle_has_current_reuse" in trace
    assert "memory_bundle_has_hidden_slot" in trace


def test_renderer_does_not_treat_revealed_history_slot_as_active_hidden_support() -> None:
    scene, memory = _scene_and_memory()
    memory.hidden_region_slots["p1:face"] = HiddenRegionSlot(
        slot_id="p1:face",
        region_type="face",
        owner_entity="p1",
        hidden_type="revealed_history",
        candidate_patch_ids=["patch-1"],
    )
    manager = MemoryManager()
    bundle = manager.get_region_memory_bundle(memory, "p1", "face")
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["face"], region_transition_mode={"face": "expression_refine"})

    out = PatchRenderer().render(_frame(), scene, delta, memory, _region(), transition_context={"region_memory_bundle": bundle})
    trace = out.execution_trace
    assert trace["memory_bundle_hidden_type"] == "revealed_history"
    assert trace["memory_bundle_hidden_support_active"] is False


def test_legacy_patch_path_marks_memory_bundle_absent() -> None:
    scene, memory = _scene_and_memory()
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["face"], region_transition_mode={"face": "expression_refine"})
    out = PatchRenderer().render(_frame(), scene, delta, memory, _region())
    assert out.execution_trace["memory_bundle_present"] is False
