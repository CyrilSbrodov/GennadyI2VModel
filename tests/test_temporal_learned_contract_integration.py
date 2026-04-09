from __future__ import annotations

from pathlib import Path

import numpy as np

from core.schema import BBox, GlobalSceneContext, PersonNode, PoseState, ExpressionState, OrientationState, RegionRef, SceneGraph
from dynamics.learned_bridge import LearnedDynamicsTransitionModel
from dynamics.temporal_contract_alignment import compute_temporal_contract_alignment
from dynamics.temporal_transition_encoder import TemporalTransitionEncoder
from dynamics.transition_contracts import LearnedTemporalTransitionContract
from learned.interfaces import DynamicsTransitionRequest, PatchSynthesisRequest
from runtime.orchestrator import GennadyEngine
from text.learned_bridge import BaselineTextEncoderAdapter
from rendering.trainable_patch_renderer import build_patch_batch


def _scene() -> SceneGraph:
    return SceneGraph(
        frame_index=1,
        persons=[
            PersonNode(
                person_id="p1",
                track_id="t1",
                bbox=BBox(0.1, 0.1, 0.7, 0.8),
                mask_ref=None,
                pose_state=PoseState(coarse_pose="standing"),
                expression_state=ExpressionState(label="neutral", smile_intensity=0.15),
                orientation=OrientationState(),
            )
        ],
        global_context=GlobalSceneContext(frame_size=(64, 64), fps=16, source_type="single_image"),
    )


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def test_temporal_transition_typed_contract_smoke() -> None:
    enc = TemporalTransitionEncoder()
    pred = enc.forward(np.zeros((128,), dtype=np.float64))
    typed = enc.to_typed_contract(pred)
    serialized = enc.to_contract(pred)
    assert isinstance(typed, LearnedTemporalTransitionContract)
    assert typed.is_learned_primary is True
    assert serialized["is_learned_primary"] is True
    assert "target_profile" in serialized


def test_temporal_transition_contract_is_primary_for_dynamics_conditioning() -> None:
    req = DynamicsTransitionRequest(
        graph_state=_scene(),
        memory_summary={},
        text_action_summary=BaselineTextEncoderAdapter().encode("sit and smile"),
        memory_channels={},
        step_context={"step_index": 2},
    )
    out = LearnedDynamicsTransitionModel(strict_mode=True).predict_transition(req)
    contract = LearnedTemporalTransitionContract.from_metadata(out.metadata.get("temporal_transition_contract"))
    assert contract is not None
    assert out.delta.transition_phase == contract.predicted_phase
    assert out.delta.transition_diagnostics.get("learned_temporal_primary") is True


def test_temporal_transition_contract_is_primary_for_renderer_conditioning() -> None:
    scene = _scene()
    region = RegionRef(region_id="p1:torso", bbox=BBox(0.2, 0.2, 0.4, 0.4), reason="test")
    request = PatchSynthesisRequest(
        region=region,
        scene_state=scene,
        memory_summary={},
        transition_context={
            "transition_phase": "transition",
            "step_index": 2,
            "target_profile": {"primary_regions": [], "secondary_regions": ["torso"], "context_regions": []},
            "learned_temporal_contract": {
                "predicted_family": "visibility_transition",
                "predicted_phase": "contact_or_reveal",
                "target_profile": {"primary_regions": ["torso"], "secondary_regions": [], "context_regions": []},
                "reveal_score": 0.2,
                "occlusion_score": 0.9,
                "support_contact_score": 0.1,
                "transition_embedding": [0.1, 0.2],
                "confidence": 0.8,
                "teacher_source": "weak_manifest_bootstrap",
                "is_learned_primary": True,
            },
        },
        retrieval_summary={},
        current_frame=_solid(64, 64, (0.3, 0.3, 0.3)),
    )
    roi = np.asarray(_solid(26, 26, (0.3, 0.3, 0.3)), dtype=np.float32)
    batch = build_patch_batch(request, roi)
    assert batch.transition_mode == "visibility_occlusion"
    assert batch.profile_role == "primary"


def test_weak_manifest_contract_becomes_fallback_not_primary_when_learned_contract_present() -> None:
    req = DynamicsTransitionRequest(
        graph_state=_scene(),
        memory_summary={},
        text_action_summary=BaselineTextEncoderAdapter().encode("open coat"),
        memory_channels={},
        step_context={"step_index": 1},
    )
    out = LearnedDynamicsTransitionModel(strict_mode=True).predict_transition(req)
    fallback = out.delta.transition_diagnostics.get("weak_manifest_fallback", {})
    assert isinstance(fallback, dict)
    assert "phase" in fallback
    assert out.delta.transition_diagnostics.get("learned_temporal_primary") is True


def test_runtime_orchestrator_passes_learned_temporal_contract_downstream(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    pixels = "\n".join(" ".join(map(str, (120, 100, 70))) for _ in range(32 * 32))
    img.write_text(f"P3\n32 32\n255\n{pixels}\n", encoding="utf-8")
    engine = GennadyEngine()
    captured: dict[str, object] = {}
    original = engine.backends.patch_backend.synthesize_patch

    def _spy(request: PatchSynthesisRequest):
        captured["ctx"] = request.transition_context
        return original(request)

    engine.backends.patch_backend.synthesize_patch = _spy  # type: ignore[method-assign]
    engine.run([str(img)], "Снимает пальто и садится на стул.", quality_profile="debug")
    ctx = captured.get("ctx", {})
    assert isinstance(ctx, dict)
    assert isinstance(ctx.get("learned_temporal_contract"), dict)


def test_temporal_contract_alignment_metrics_smoke() -> None:
    contract = LearnedTemporalTransitionContract(
        predicted_family="garment_transition",
        predicted_phase="transition",
        reveal_score=0.7,
        occlusion_score=0.2,
        support_contact_score=0.6,
    )
    req = DynamicsTransitionRequest(
        graph_state=_scene(),
        memory_summary={},
        text_action_summary=BaselineTextEncoderAdapter().encode("sit"),
        memory_channels={},
        step_context={"step_index": 1},
    )
    out = LearnedDynamicsTransitionModel(strict_mode=True).predict_transition(req)
    metrics = compute_temporal_contract_alignment(contract, out.delta, renderer_transition_mode="garment_reveal")
    assert "learned_contract_to_dynamics_alignment" in metrics
    assert "learned_contract_to_renderer_alignment" in metrics
    assert "target_profile_consistency" in metrics
    assert "reveal_occlusion_support_agreement" in metrics


def test_renderer_and_dynamics_remain_runtime_compatible_after_learned_contract_wiring(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    pixels = "\n".join(" ".join(map(str, (90, 70, 150))) for _ in range(24 * 24))
    img.write_text(f"P3\n24 24\n255\n{pixels}\n", encoding="utf-8")
    artifacts = GennadyEngine().run([str(img)], "Улыбается и садится.", quality_profile="debug")
    assert artifacts.frames
    first_step = artifacts.debug["step_execution"][0]
    dyn = first_step["dynamics"]["diagnostics_summary"]
    assert isinstance(dyn.get("temporal_contract_alignment", {}), dict)
