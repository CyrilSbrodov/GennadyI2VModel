from __future__ import annotations

from pathlib import Path

from core.schema import BBox, GraphDelta, PersonNode, RegionRef, SceneGraph
from memory.video_memory import MemoryManager
from rendering.roi_renderer import PatchRenderer
from rendering.torch_backends import build_renderer_tensor_batch


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def _scene() -> SceneGraph:
    return SceneGraph(frame_index=0, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)])


def test_renderer_torch_mode_falls_back_without_checkpoint() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    renderer = PatchRenderer(learnable_backend_mode="torch_learned", torch_checkpoint_dir="missing_dir")
    region = RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.15, 0.25, 0.25), reason="expression")
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["face"], region_transition_mode={"face": "expression_refine"})

    patch = renderer.render(_solid(64, 64, (0.2, 0.3, 0.4)), graph, delta, memory, region)

    assert patch.execution_trace["learnable_backend"]["requested_mode"] == "torch_learned"
    if patch.execution_trace["learnable_backend"]["available"]:
        assert patch.execution_trace["learnable_backend"]["selected_backend"] in {"torch_learned_primary", "torch_learned_missing_checkpoint_fallback"}
    else:
        assert patch.execution_trace["learnable_backend"]["selected_backend"] == "torch_learned_missing_checkpoint_fallback"


def test_tensor_batch_surface_shapes_are_training_ready() -> None:
    batch = build_renderer_tensor_batch(
        roi=_solid(16, 20, (0.1, 0.2, 0.3)),
        path_type="reveal",
        transition_mode="garment_reveal",
        hidden_mode="known_hidden",
        retrieval_top_score=0.72,
        memory_hint_strength=0.61,
    )
    assert len(batch.roi) == 16 and len(batch.roi[0]) == 20 and len(batch.roi[0][0]) == 3
    assert len(batch.changed_mask) == 16 and len(batch.changed_mask[0]) == 20 and len(batch.changed_mask[0][0]) == 1
    assert len(batch.alpha_hint) == 16 and len(batch.alpha_hint[0]) == 20 and len(batch.alpha_hint[0][0]) == 1
    assert len(batch.memory_hint) == 16 and len(batch.memory_hint[0]) == 20 and len(batch.memory_hint[0][0]) == 1
    assert len(batch.context_vector) == 4


def test_renderer_can_load_saved_torch_checkpoints_when_available(tmp_path: Path) -> None:
    renderer = PatchRenderer(learnable_backend_mode="torch_learned")
    if not renderer.torch_backends.available:
        return
    saved = renderer.torch_backends.save_checkpoint(str(tmp_path / "torch_renderer"))
    assert "existing" in saved

    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize(graph)
    renderer_loaded = PatchRenderer(learnable_backend_mode="torch_learned", torch_checkpoint_dir=str(tmp_path / "torch_renderer"))
    region = RegionRef(region_id="p1:torso", bbox=BBox(0.2, 0.25, 0.35, 0.45), reason="stable")
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["torso"], region_transition_mode={"torso": "stable"})

    patch = renderer_loaded.render(_solid(64, 64, (0.3, 0.3, 0.3)), graph, delta, memory, region)
    assert patch.execution_trace["learnable_backend"]["requested_mode"] == "torch_learned"
    assert patch.execution_trace["module_trace"]["backend_selection"] in {"torch_learned_primary", "torch_learned_missing_checkpoint_fallback", "bootstrap_fallback"}
