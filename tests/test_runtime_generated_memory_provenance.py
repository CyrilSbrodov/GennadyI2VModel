from core.schema import GraphDelta
from memory.video_memory import MemoryManager
from runtime.orchestrator import GennadyEngine


def _delta() -> GraphDelta:
    return GraphDelta(
        transition_phase="motion",
        state_after={
            "visibility_phase": "revealing",
            "garment_phase": "opening",
            "pose_phase": "turning",
        },
        affected_regions=["face"],
        region_transition_mode={"face": "expression_refine"},
    )


def test_generated_memory_update_context_contains_source_flags() -> None:
    context = GennadyEngine._memory_update_context_for_generated_frame(
        _delta(),
        temporal_refinement_enabled=True,
    )

    assert context["generated"] is True
    assert context["is_generated"] is True
    assert "generated" in str(context["update_source"])
    assert "renderer" in str(context["update_source"])
    assert str(context["frame_source"])
    assert "runtime" in str(context["frame_source"])
    assert "generated" in str(context["frame_source"])
    assert context["transition_phase"] == "motion"
    assert context["visibility_phase"] == "revealing"
    assert context["garment_phase"] == "opening"
    assert context["pose_phase"] == "turning"
    assert context["region_transition_mode"] == "expression_refine"


def test_composited_output_memory_update_context_is_generated() -> None:
    context = GennadyEngine._memory_update_context_for_generated_frame(
        _delta(),
        temporal_refinement_enabled=False,
    )

    assert context["generated"] is True
    assert context["is_generated"] is True
    update_source = str(context["update_source"])
    assert "renderer" in update_source
    assert "composited" in update_source
    assert "generated" in update_source
    assert context["frame_source"] == "runtime_generated_stable_frame"


def test_runtime_generated_context_is_compatible_with_memory_generated_detection() -> None:
    context = GennadyEngine._memory_update_context_for_generated_frame(
        _delta(),
        temporal_refinement_enabled=True,
    )
    manager = MemoryManager()

    assert manager._source_indicates_generated(
        context["frame_source"],
        context["update_source"],
        context["generated"],
        context["is_generated"],
    )
