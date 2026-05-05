from __future__ import annotations

import math
import pytest

from core.schema import BBox, GlobalSceneContext, GraphDelta, HiddenRegionSlot, PersonNode, SceneGraph
from learned.interfaces import PatchSynthesisRequest


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def _request(region_id: str, *, mode: str | None = None, role: str = "primary") -> PatchSynthesisRequest:
    from core.schema import RegionRef
    from memory.video_memory import MemoryManager

    graph = SceneGraph(
        frame_index=2,
        persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)],
        global_context=GlobalSceneContext(frame_size=(64, 64), fps=16, source_type="single_image"),
    )
    region_type = region_id.split(":", 1)[1]
    rr = RegionRef(region_id=region_id, bbox=BBox(0.2, 0.2, 0.4, 0.4), reason="test")
    resolved_mode = mode or ("garment_reveal" if region_type == "torso" else "expression_refine")
    target_profile = {"primary_regions": [region_type] if role == "primary" else [], "secondary_regions": [region_type] if role == "secondary" else [], "context_regions": [region_type] if role == "context" else []}
    delta = GraphDelta(affected_entities=["p1"], affected_regions=[region_type], region_transition_mode={region_type: resolved_mode}, transition_phase="motion")
    memory = MemoryManager().initialize(graph)
    return PatchSynthesisRequest(region=rr, scene_state=graph, memory_summary={"hidden_region_count": 1, "texture_patch_count": 3}, transition_context={"graph_delta": delta, "video_memory": memory, "transition_phase": "motion", "step_index": 2, "target_profile": target_profile, "region_selection_rationale": {region_type: "primary_goal_region"}, "semantic_families": ["garment_transition"] if "garment" in resolved_mode else ["expression_transition"]}, retrieval_summary={"backend": "learned_primary", "profile": "rich", "top_score": 0.7}, current_frame=_solid(64, 64, (0.3, 0.32, 0.35)), memory_channels={"hidden_regions": {"slot": 1}, "garments": {"coat": 1}, "identity": {"p1": 1}}, identity_embedding=[0.1] * 12)


def _roi_from_request(request: PatchSynthesisRequest):
    import numpy as np

    frame = np.asarray(request.current_frame, dtype=np.float32)
    h, w, _ = frame.shape
    x0, y0 = int(request.region.bbox.x * w), int(request.region.bbox.y * h)
    x1, y1 = int((request.region.bbox.x + request.region.bbox.w) * w), int((request.region.bbox.y + request.region.bbox.h) * h)
    return frame[y0:y1, x0:x1]


def test_patch_conditioning_contract_constants_import_without_numpy_dependency() -> None:
    from rendering.patch_conditioning_contract import GLOBAL_COND_DIM

    assert GLOBAL_COND_DIM > 0


def test_import_learned_bridge_without_instantiating_torch_backend() -> None:
    pytest.importorskip("numpy")
    import importlib

    mod = importlib.import_module("rendering.learned_bridge")
    assert hasattr(mod, "TrainablePatchSynthesisModel")


def test_torch_backend_global_condition_dim_matches_batch() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from rendering.trainable_patch_renderer import build_patch_batch
    from rendering.torch_local_patch_generator import GLOBAL_COND_DIM, TorchLocalPatchGenerator

    batch = build_patch_batch(_request("p1:torso"), _roi_from_request(_request("p1:torso")))
    model = TorchLocalPatchGenerator()
    assert len(model._global_cond(batch)) == GLOBAL_COND_DIM


def test_torch_generator_train_eval_step_returns_losses() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from rendering.trainable_patch_renderer import build_patch_batch
    from rendering.torch_local_patch_generator import TorchLocalPatchGenerator

    req = _request("p1:face")
    batch = build_patch_batch(req, _roi_from_request(req))
    model = TorchLocalPatchGenerator()
    train, eval_stats = model.train_step(batch), model.eval_step(batch)
    for key in ("total_loss", "reconstruction_loss", "alpha_loss", "uncertainty_calibration_loss", "appearance_preservation_loss", "seam_loss", "drift_penalty"):
        assert key in train and math.isfinite(float(train[key]))
    assert math.isfinite(float(eval_stats["total_loss"]))


def test_torch_unavailable_non_strict_falls_back_with_top_level_trace() -> None:
    pytest.importorskip("numpy")
    from rendering.learned_bridge import TrainablePatchSynthesisModel
    import rendering.learned_bridge as lb
    from rendering.trainable_patch_renderer import RendererInferenceError

    req = _request("p1:face")
    orig = lb.TorchLocalPatchGenerator
    lb.TorchLocalPatchGenerator = lambda *a, **k: (_ for _ in ()).throw(RendererInferenceError("torch unavailable"))
    try:
        out = TrainablePatchSynthesisModel(backend="torch_local", strict_mode=False).synthesize_patch(req)
    finally:
        lb.TorchLocalPatchGenerator = orig
    assert out.execution_trace["torch_backend_used"] is False
    assert out.execution_trace["fallback_used"] is True
    assert out.execution_trace["fallback_reason"] == "torch_unavailable"


def test_numpy_backend_trace_is_not_torch_fallback() -> None:
    pytest.importorskip("numpy")
    from rendering.learned_bridge import TrainablePatchSynthesisModel

    out = TrainablePatchSynthesisModel(backend="numpy_local", strict_mode=False).synthesize_patch(_request("p1:face"))
    assert out.execution_trace["torch_backend_used"] is False
    assert out.execution_trace["fallback_used"] is False
    assert out.execution_trace["model_family"] == "numpy_linear_patch_generator"
