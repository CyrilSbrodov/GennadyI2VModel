from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from core.schema import BBox, PersonNode, RegionRef, SceneGraph
import numpy as np
from learned.interfaces import PatchSynthesisRequest
from rendering.trainable_patch_renderer import build_patch_batch, output_from_prediction
from runtime.i2v_frame_planner import I2VFramePlanEntry, plan_i2v_frames
from runtime.orchestrator import GennadyEngine


def _write_ppm(path: Path, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_frame_planner_slots_dataclass_serializes_with_asdict() -> None:
    entry = I2VFramePlanEntry(
        frame_index=0,
        action_phase="head_turn",
        affected_entities=["p1"],
        affected_regions=["p1:face"],
        region_transition_mode={"p1:face": "head_turn"},
        expected_reference_families={"p1:face": "identity"},
        use_input_frame_visual_anchors=True,
    )
    d = asdict(entry)
    assert d["action_phase"] == "head_turn"
    assert d["use_input_frame_visual_anchors"] is True


def test_single_image_generates_multi_frame_and_keeps_input_anchor_contract(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    _write_ppm(img, 32, 32, (64, 120, 200))
    engine = GennadyEngine()
    artifacts = engine.run([str(img)], "head turn smile and raise arm", fps=6, duration=1.0, quality_profile="debug")

    assert len(artifacts.frames) >= 2
    assert artifacts.frames[0] != artifacts.frames[-1]
    assert artifacts.debug.get("i2v_frame_plan")

    steps = artifacts.debug.get("step_execution", [])
    assert steps
    any_patch = any(bool(step.get("patch")) for step in steps)
    assert any_patch
    assert any(step.get("frame_plan_applied") is True or len(step.get("frame_plan_added_regions", [])) > 0 for step in steps)

    traces = [p.get("execution_trace", {}) for step in steps for p in step.get("patch", [])]
    assert traces
    assert any("reference_patch_material_missing_reason" in t for t in traces if isinstance(t, dict))
    planned_regions = {rid for step in steps for rid in step.get("frame_plan", {}).get("affected_regions", [])}
    rendered_regions = {rid for step in steps for rid in step.get("region_render_order", [])}
    assert planned_regions.intersection(rendered_regions)
    patched_regions = {p.get("region_id", "") for step in steps for p in step.get("patch", [])}
    assert planned_regions.intersection(patched_regions)
    assert any(
        p.get("i2v_action_phase") and p.get("i2v_region_transition_mode")
        for step in steps
        for p in step.get("patch", [])
    )

    summary = artifacts.debug.get("reference_coverage_summary", {})
    assert "overall_expected_reference_coverage_ratio" in summary
    assert "overall_expected_material_coverage_ratio" in summary
    assert "overall_input_frame_material_coverage_ratio" in summary
    assert summary.get("input_frame_material_coverage_source") == "observed_directly_not_generated_v1"

    for step in steps:
        provenance = step.get("memory_update_provenance", {})
        assert provenance.get("generated") is True
        assert "generated" in str(provenance.get("frame_source", ""))


def test_planner_covers_requested_actions() -> None:
    entries = plan_i2v_frames("head turn smile torso shift raise arm adjust coat", frame_count=8, entity_id="p1")
    phases = {e.action_phase for e in entries}
    assert {"stable_idle", "head_turn", "expression_smile", "torso_shift", "arm_raise", "garment_reveal_or_adjust"}.issubset(phases)


def test_debug_script_writes_frames_and_report_with_bootstrap(tmp_path: Path, monkeypatch) -> None:
    import scripts.debug_i2v_generation as script

    class _Artifacts:
        def __init__(self):
            self.frames = [[[[0.1, 0.2, 0.3] for _ in range(4)] for _ in range(4)] for _ in range(3)]
            self.state_plan = type("P", (), {"steps": [0, 1, 2]})()
            self.debug = {"i2v_frame_plan": [{"frame_index": 0}], "step_execution": [{"patch": [{"execution_trace": {"selected_render_strategy": "x"}}]}], "reference_coverage_summary": {"overall_input_frame_material_coverage_ratio": 1.0}, "video_export": ""}

    class _Engine:
        def run(self, *args, **kwargs):
            return _Artifacts()

    monkeypatch.setattr(script, "GennadyEngine", _Engine)
    img = tmp_path / "i.ppm"
    _write_ppm(img, 4, 4, (50, 50, 50))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["debug_i2v_generation.py", "--image", str(img), "--prompt", "head turn", "--frames", "3", "--fps", "4", "--out", str(out)])
    script.main()

    assert (out / "frame_0000.png").exists()
    assert (out / "debug_report.json").exists()
    report = json.loads((out / "debug_report.json").read_text())
    assert report["frames_generated"] == 3
    assert report["frames_saved"] == 3
    assert "i2v_frame_plan" in report


def test_resolve_planned_region_uses_deterministic_fallback_when_selector_methods_absent() -> None:
    scene = SceneGraph(
        frame_index=0,
        persons=[PersonNode(person_id="p1", track_id="p1", bbox=BBox(0.1, 0.1, 0.6, 0.8), mask_ref=None)],
    )

    class _FakeROISelector:
        pass

    resolved = GennadyEngine._resolve_planned_region(scene, _FakeROISelector(), "p1:face")
    assert resolved is not None
    assert resolved.region_id == "p1:face"


def _i2v_request(region_id: str, phase: str) -> PatchSynthesisRequest:
    return PatchSynthesisRequest(
        region=RegionRef(region_id=region_id, bbox=BBox(0.1, 0.1, 0.4, 0.4), reason="i2v"),
        scene_state=SceneGraph(frame_index=0, persons=[PersonNode(person_id="p1", track_id="p1", bbox=BBox(0.1, 0.1, 0.6, 0.8), mask_ref=None)]),
        memory_summary={},
        transition_context={"i2v_action_phase": phase, "i2v_region_transition_mode": {region_id: phase}},
        retrieval_summary={},
        current_frame=np.full((16, 16, 3), 0.3, dtype=np.float32).tolist(),
    )


def test_i2v_head_turn_face_batch_exposes_phase_and_nonzero_masks() -> None:
    req = _i2v_request("p1:face", "head_turn")
    batch = build_patch_batch(req, np.full((16, 16, 3), 0.3, dtype=np.float32))
    assert batch.conditioning_summary["i2v_action_phase"] == "head_turn"
    assert float(np.mean(batch.changed_mask)) > 0.0
    assert float(np.mean(batch.blend_hint)) > 0.0


def test_i2v_build_patch_batch_request_has_no_name_error() -> None:
    reqs = (
        _i2v_request("p1:face", "head_turn"),
        _i2v_request("p1:left_arm", "arm_raise"),
        _i2v_request("p1:face", "arm_raise"),
    )
    for req in reqs:
        batch = build_patch_batch(req, np.full((16, 16, 3), 0.3, dtype=np.float32))
        assert batch is not None


def test_i2v_expression_smile_changes_face_target() -> None:
    req = _i2v_request("p1:face", "expression_smile")
    before = np.full((16, 16, 3), 0.3, dtype=np.float32)
    batch = build_patch_batch(req, before)
    assert not np.allclose(batch.roi_after, before)


def test_i2v_arm_raise_stronger_arm_mask_than_idle() -> None:
    idle = build_patch_batch(_i2v_request("p1:left_arm", "stable_idle"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    raised = build_patch_batch(_i2v_request("p1:left_arm", "arm_raise"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    assert float(np.mean(raised.changed_mask)) > float(np.mean(idle.changed_mask))


def test_i2v_arm_raise_face_is_not_arm_targeted() -> None:
    idle = build_patch_batch(_i2v_request("p1:face", "stable_idle"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    face_arm_raise = build_patch_batch(_i2v_request("p1:face", "arm_raise"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    assert float(np.mean(face_arm_raise.changed_mask)) <= float(np.mean(idle.changed_mask)) + 0.02
    idle_delta = float(np.mean(np.abs(idle.roi_after - idle.roi_before)))
    arm_delta = float(np.mean(np.abs(face_arm_raise.roi_after - face_arm_raise.roi_before)))
    assert arm_delta <= idle_delta + 0.01


def test_i2v_arm_raise_left_arm_stronger_mask_and_delta_than_idle() -> None:
    idle = build_patch_batch(_i2v_request("p1:left_arm", "stable_idle"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    raised = build_patch_batch(_i2v_request("p1:left_arm", "arm_raise"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    idle_delta = float(np.mean(np.abs(idle.roi_after - idle.roi_before)))
    raised_delta = float(np.mean(np.abs(raised.roi_after - raised.roi_before)))
    assert float(np.mean(raised.changed_mask)) > float(np.mean(idle.changed_mask))
    assert raised_delta > idle_delta


def test_i2v_edit_strength_boost_only_for_targeted_regions() -> None:
    arm_idle = build_patch_batch(_i2v_request("p1:left_arm", "stable_idle"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    arm_raise = build_patch_batch(_i2v_request("p1:left_arm", "arm_raise"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    face_idle = build_patch_batch(_i2v_request("p1:face", "stable_idle"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    face_arm_raise = build_patch_batch(_i2v_request("p1:face", "arm_raise"), np.full((16, 16, 3), 0.3, dtype=np.float32))
    face_head_turn = build_patch_batch(_i2v_request("p1:face", "head_turn"), np.full((16, 16, 3), 0.3, dtype=np.float32))

    assert float(arm_raise.conditioning_summary["edit_strength"]) > float(arm_idle.conditioning_summary["edit_strength"])
    assert abs(float(face_arm_raise.conditioning_summary["edit_strength"]) - float(face_idle.conditioning_summary["edit_strength"])) <= 0.02
    assert float(face_head_turn.conditioning_summary["edit_strength"]) > float(face_idle.conditioning_summary["edit_strength"])


def test_i2v_garment_phase_traces_output_and_reference_fallback() -> None:
    req = _i2v_request("p1:upper_clothes", "garment_reveal_or_adjust")
    batch = build_patch_batch(req, np.full((16, 16, 3), 0.3, dtype=np.float32))
    pred = {"rgb": batch.roi_after, "alpha": batch.alpha_target[..., 0], "uncertainty": batch.uncertainty_target[..., 0], "confidence": 0.8}
    out = output_from_prediction(req, pred, "learned_primary", {}, batch)
    expected_fields = {
        "i2v_action_phase",
        "i2v_action_active",
        "i2v_region_action_mode",
        "i2v_action_strength",
        "i2v_motion_direction_x",
        "i2v_motion_direction_y",
        "i2v_expression_bias",
        "i2v_pose_bias",
        "i2v_garment_bias",
    }
    assert out.execution_trace["i2v_action_phase"] == "garment_reveal_or_adjust"
    assert out.metadata["i2v_action_phase"] == "garment_reveal_or_adjust"
    assert expected_fields.issubset(set(out.execution_trace.keys()))
    assert expected_fields.issubset(set(out.metadata.keys()))
    assert "reference_tensor_zero_fallback" in out.execution_trace
