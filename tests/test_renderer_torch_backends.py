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


def test_torch_local_patch_generator_upgrades_v2_9_channel_stem_checkpoint(tmp_path: Path) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from rendering.patch_conditioning_contract import GLOBAL_COND_DIM
    from rendering.torch_local_patch_generator import BASE_LOCAL_INPUT_CHANNELS, LOCAL_INPUT_CHANNELS, TorchLocalPatchGenerator, TorchLocalPatchGeneratorNet

    legacy = TorchLocalPatchGeneratorNet(global_dim=GLOBAL_COND_DIM, input_channels=BASE_LOCAL_INPUT_CHANNELS)
    with torch.no_grad():
        legacy.stem.net[0].weight.fill_(0.25)
    checkpoint = tmp_path / "legacy_9_channel.pt"
    torch.save({"state_dict": legacy.state_dict(), "local_input_channels": BASE_LOCAL_INPUT_CHANNELS}, str(checkpoint))

    loaded = TorchLocalPatchGenerator.load(str(checkpoint))
    weight = loaded.net.stem.net[0].weight.detach().cpu()

    assert loaded.net.input_channels == LOCAL_INPUT_CHANNELS
    assert torch.allclose(weight[:, :BASE_LOCAL_INPUT_CHANNELS], legacy.stem.net[0].weight.detach().cpu())
    assert torch.count_nonzero(weight[:, BASE_LOCAL_INPUT_CHANNELS:]) == 0
    assert loaded.checkpoint_compatibility["checkpoint_reference_channel_upgrade"] is True


def test_torch_local_patch_generator_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    model = TorchLocalPatchGenerator()
    state = model.net.state_dict()
    state.pop("stem.net.2.weight")
    checkpoint = tmp_path / "missing_key.pt"
    torch.save({"state_dict": state}, str(checkpoint))

    with pytest.raises(ValueError, match="key mismatch"):
        TorchLocalPatchGenerator.load(str(checkpoint))


def test_torch_local_patch_generator_rejects_unexpected_checkpoint_key(tmp_path: Path) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    model = TorchLocalPatchGenerator()
    state = model.net.state_dict()
    state["unexpected.weight"] = torch.zeros(1)
    checkpoint = tmp_path / "unexpected_key.pt"
    torch.save({"state_dict": state}, str(checkpoint))

    with pytest.raises(ValueError, match="key mismatch"):
        TorchLocalPatchGenerator.load(str(checkpoint))


def test_torch_local_patch_generator_rejects_unsupported_shape_mismatch(tmp_path: Path) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    model = TorchLocalPatchGenerator()
    state = model.net.state_dict()
    state["stem.net.2.weight"] = state["stem.net.2.weight"][:, :1, :, :]
    checkpoint = tmp_path / "bad_shape.pt"
    torch.save({"state_dict": state}, str(checkpoint))

    with pytest.raises(ValueError, match="tensor shape mismatch"):
        TorchLocalPatchGenerator.load(str(checkpoint))


def _torch_batch_with_reference(h: int = 16, w: int = 16, *, validity: float = 1.0, mask: float = 1.0, preservation: float = 0.0):
    import numpy as np
    from core.schema import BBox, SceneGraph, RegionRef
    from learned.interfaces import PatchSynthesisRequest
    from rendering.trainable_patch_renderer import build_patch_batch

    request = PatchSynthesisRequest(
        region=RegionRef(region_id="p1:face", bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="test"),
        scene_state=SceneGraph(frame_index=0),
        memory_summary={},
        transition_context={},
        retrieval_summary={},
        current_frame=np.full((h, w, 3), 0.2, dtype=np.float32).tolist(),
    )
    roi = np.full((h, w, 3), 0.2, dtype=np.float32)
    batch = build_patch_batch(request, roi)
    batch.reference_rgb = np.zeros((h, w, 3), dtype=np.float32)
    batch.reference_mask = np.full((h, w, 1), mask, dtype=np.float32)
    batch.reference_validity = np.full((h, w, 1), validity, dtype=np.float32)
    batch.changed_mask = np.ones((h, w, 1), dtype=np.float32)
    batch.blend_hint = np.ones((h, w, 1), dtype=np.float32)
    batch.preservation_mask = np.full((h, w, 1), preservation, dtype=np.float32)
    return batch


def test_torch_local_patch_generator_local_tensor_layout_contract() -> None:
    import numpy as np
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import LOCAL_INPUT_CHANNELS, TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator()
    batch = _torch_batch_with_reference(16, 16, validity=1.0, mask=1.0)
    tb = gen._to_torch_batch(batch)
    assert tb.local_maps.shape[1] == LOCAL_INPUT_CHANNELS == 14
    assert np.allclose(tb.local_maps[0, 9:12].cpu().numpy(), 0.0)
    assert float(tb.local_maps[0, 13:14].mean().item()) > 0.0
    assert batch.conditioning_summary["reference_tensor_input_used"] is True
    assert batch.conditioning_summary["reference_tensor_zero_fallback"] is False


def test_torch_local_patch_generator_material_gate_cold_start_and_caps() -> None:
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import MATERIAL_GATE_MAX, TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=1)
    batch = _torch_batch_with_reference(16, 16, validity=1.0, mask=1.0)
    pred = gen._predict_tensors(batch)
    assert pred["material_gate_mean"] < 0.2
    assert pred["material_gate_max"] <= MATERIAL_GATE_MAX + 1e-6


def test_torch_local_patch_generator_reference_validity_zero_disables_gate() -> None:
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=2)
    batch = _torch_batch_with_reference(16, 16, validity=0.0, mask=1.0)
    pred = gen._predict_tensors(batch)
    assert pred["material_gate_max"] == 0.0
    assert pred["reference_tensor_zero_fallback"] is True


def test_torch_local_patch_generator_reference_mask_zero_disables_gate() -> None:
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=6)
    batch = _torch_batch_with_reference(16, 16, validity=1.0, mask=0.0)
    pred = gen._predict_tensors(batch)
    assert pred["material_gate_max"] == 0.0


def test_torch_local_patch_generator_preservation_suppresses_gate_and_drift() -> None:
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=7)
    low_pres = _torch_batch_with_reference(16, 16, validity=1.0, mask=1.0, preservation=0.0)
    hi_pres = _torch_batch_with_reference(16, 16, validity=1.0, mask=1.0, preservation=1.0)
    low_pred = gen._predict_tensors(low_pres)
    hi_pred = gen._predict_tensors(hi_pres)
    low_inf = gen.infer(low_pres)
    hi_inf = gen.infer(hi_pres)
    assert hi_pred["material_gate_mean"] == 0.0
    assert hi_pred["material_gate_suppressed_by_preservation"] > 0.0
    assert abs(low_pred["material_gate_suppressed_by_preservation"]) <= 1e-6
    assert hi_pred["material_gate_mean"] <= low_pred["material_gate_mean"]
    assert hi_inf["preservation_drift"] <= low_inf["preservation_drift"] + 1e-6


def test_torch_local_patch_generator_valid_reference_enables_bounded_gate() -> None:
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import MATERIAL_GATE_MAX, TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=8)
    batch = _torch_batch_with_reference(16, 16, validity=1.0, mask=1.0, preservation=0.0)
    pred = gen._predict_tensors(batch)
    assert pred["material_gate_mean"] > 0.0
    assert pred["material_gate_max"] <= MATERIAL_GATE_MAX + 1e-6


def test_torch_local_patch_generator_legacy_tiny_encoder_checkpoint_fails_fast(tmp_path: Path) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    checkpoint = tmp_path / "legacy_encoder_key.pt"
    torch.save({"state_dict": {"encoder.0.weight": torch.zeros((32, 9, 3, 3))}}, str(checkpoint))
    with pytest.raises(ValueError, match="legacy tiny encoder checkpoint is not compatible"):
        TorchLocalPatchGenerator.load(str(checkpoint))


def test_torch_local_patch_generator_material_consistency_weighted_no_nan() -> None:
    import math
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=3)
    valid_batch = _torch_batch_with_reference(16, 16, validity=1.0, mask=1.0)
    zero_batch = _torch_batch_with_reference(16, 16, validity=0.0, mask=1.0)
    valid_metrics = gen.train_step(valid_batch)
    zero_metrics = gen.train_step(zero_batch)

    assert math.isfinite(valid_metrics["material_consistency_loss"])
    assert math.isfinite(zero_metrics["material_consistency_loss"])
    assert abs(zero_metrics["material_consistency_loss"]) <= 1e-6
    assert abs(zero_metrics["material_gate_area_penalty"]) <= 1e-6


def test_torch_local_patch_generator_eval_no_nan() -> None:
    import math
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=9)
    metrics = gen.eval_step(_torch_batch_with_reference(16, 16, validity=1.0, mask=1.0))
    assert math.isfinite(metrics["total_loss"])
    assert math.isfinite(metrics["material_consistency_loss"])


def test_torch_local_patch_generator_supports_standard_roi_sizes() -> None:
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=4)
    for size in (16, 32, 64):
        pred = gen.infer(_torch_batch_with_reference(size, size, validity=1.0, mask=1.0))
        assert pred["material_gate_cap"] > 0.0


def test_torch_local_patch_generator_rejects_too_small_roi() -> None:
    import pytest

    pytest.importorskip("torch")
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    gen = TorchLocalPatchGenerator(seed=5)
    with pytest.raises(ValueError, match="ROI >= 4x4"):
        gen.infer(_torch_batch_with_reference(2, 2, validity=1.0, mask=1.0))
