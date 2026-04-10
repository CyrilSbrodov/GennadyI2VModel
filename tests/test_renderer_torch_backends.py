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
    backend = patch.execution_trace["learnable_backend"]
    if backend["available"]:
        assert backend["usable_for_inference"] is False
        assert backend["checkpoint_status"] in {"checkpoint_directory_missing", "checkpoint_file_missing", "bootstrap_only"}
    assert backend["selected_backend"] != "torch_learned_primary"


def test_tensor_batch_surface_shapes_are_training_ready() -> None:
    batch = build_renderer_tensor_batch(
        roi=_solid(16, 20, (0.1, 0.2, 0.3)),
        path_type="reveal",
        transition_mode="garment_reveal",
        hidden_mode="known_hidden",
        retrieval_top_score=0.72,
        memory_hint_strength=0.61,
        lifecycle="previously_hidden_now_revealed",
        region_role="primary",
        region_type="torso",
        entity_type="person",
        reveal_type="garment_change_reveal",
        retrieval_evidence=0.71,
        reveal_memory_strength=0.75,
        transition_strength=0.62,
    )
    assert len(batch.roi) == 16 and len(batch.roi[0]) == 20 and len(batch.roi[0][0]) == 3
    assert len(batch.changed_mask) == 16 and len(batch.changed_mask[0]) == 20 and len(batch.changed_mask[0][0]) == 1
    assert len(batch.alpha_hint) == 16 and len(batch.alpha_hint[0]) == 20 and len(batch.alpha_hint[0][0]) == 1
    assert len(batch.memory_hint) == 16 and len(batch.memory_hint[0]) == 20 and len(batch.memory_hint[0][0]) == 1
    assert len(batch.conditioning_map) == 16 and len(batch.conditioning_map[0]) == 20 and len(batch.conditioning_map[0][0]) == 1
    assert len(batch.reveal_signal) == 16 and len(batch.reveal_signal[0]) == 20 and len(batch.reveal_signal[0][0]) == 1
    assert len(batch.insertion_signal) == 16 and len(batch.insertion_signal[0]) == 20 and len(batch.insertion_signal[0][0]) == 1
    assert len(batch.context_vector) == 16
    assert batch.conditioning_summary["region_type"] == "torso"
    assert batch.conditioning_summary["reveal_type"] == "garment_change_reveal"


def test_tensor_batch_mode_specific_features_are_distinct() -> None:
    reveal = build_renderer_tensor_batch(
        roi=_solid(8, 8, (0.1, 0.2, 0.3)),
        path_type="reveal",
        transition_mode="garment_reveal",
        hidden_mode="known_hidden",
        retrieval_top_score=0.8,
        memory_hint_strength=0.7,
        lifecycle="previously_hidden_now_revealed",
        reveal_memory_strength=0.9,
        reveal_type="garment_change_reveal",
    )
    insertion = build_renderer_tensor_batch(
        roi=_solid(8, 8, (0.1, 0.2, 0.3)),
        path_type="insertion",
        transition_mode="stable",
        hidden_mode="not_hidden",
        retrieval_top_score=0.2,
        memory_hint_strength=0.2,
        lifecycle="newly_inserted",
        insertion_type="new_entity",
        insertion_context_strength=0.88,
        entity_type="person",
    )
    assert reveal.context_vector[2] != insertion.context_vector[2]
    assert reveal.context_vector[11] > insertion.context_vector[11]
    assert insertion.context_vector[12] > reveal.context_vector[12]


def test_backend_usability_policy_blocks_random_weights_by_default() -> None:
    renderer = PatchRenderer(learnable_backend_mode="torch_learned")
    if not renderer.torch_backends.available:
        return
    status = renderer.torch_backends.backend_runtime_status("existing_update")
    assert status["checkpoint_status"] == "bootstrap_only"
    assert status["usable_for_inference"] is False


def test_backend_usability_policy_can_allow_random_for_dev() -> None:
    renderer = PatchRenderer(learnable_backend_mode="torch_learned", torch_allow_random_init_for_dev=True)
    if not renderer.torch_backends.available:
        return
    status = renderer.torch_backends.backend_runtime_status("existing_update")
    assert status["checkpoint_status"] == "bootstrap_only"
    assert status["usable_for_inference"] is True


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
    backend_trace = patch.execution_trace["learnable_backend"]
    assert backend_trace["checkpoint_trace"]["loaded"]
    assert backend_trace["checkpoint_status"] in {"checkpoint_loaded", "checkpoint_file_missing"}
    assert patch.execution_trace["module_trace"]["backend_selection"] in {"torch_learned_primary", "torch_learned_unusable_fallback", "bootstrap_fallback"}
