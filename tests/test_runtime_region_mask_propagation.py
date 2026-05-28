from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.schema import BBox, BodyPartNode, FrameRegionObservation, MemoryEntry, PersonNode, SceneGraph
from memory.video_memory import MemoryManager
from perception.mask_store import InMemoryMaskStore
from runtime.orchestrator import GennadyEngine
from runtime.region_mask_propagation import HIGH_CONFIDENCE_THRESHOLD, propagate_region_masks_for_frame


def _scene() -> SceneGraph:
    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.7, 0.8), mask_ref=None, body_parts=[BodyPartNode(part_id="f", part_type="face", bbox=BBox(0.2, 0.15, 0.2, 0.2), visibility="visible", confidence=0.9), BodyPartNode(part_id="t", part_type="torso", bbox=BBox(0.2, 0.35, 0.4, 0.35), visibility="visible", confidence=0.9)], confidence=0.9)
    return SceneGraph(frame_index=0, persons=[person], objects=[])


def test_patch_alpha_creates_generated_observation() -> None:
    scene = _scene()
    patch = SimpleNamespace(region=SimpleNamespace(region_id="p1:face", bbox=BBox(0.2, 0.15, 0.2, 0.2)), alpha_mask=np.ones((8, 8), dtype=np.float32))
    result = propagate_region_masks_for_frame(1, np.zeros((16, 16, 3), dtype=np.float32), [], scene, ["p1:face"], [patch], InMemoryMaskStore(), strict=False)
    face = next(o for o in result.observations if o.region_id == "p1:face")
    assert face.observation_source == "patch_alpha_update"
    assert face.mask_provenance == "generated_patch_alpha"
    assert face.is_generated_evidence is True
    assert face.confidence <= HIGH_CONFIDENCE_THRESHOLD


def test_carry_forward_for_unchanged_region() -> None:
    scene = _scene()
    prev = [FrameRegionObservation(frame_index=0, region_id="p1:torso", bbox=BBox(0.2, 0.35, 0.4, 0.35), mask_ref="mask://old", mask_kind="alpha", mask_provenance="input_frame_observed", observation_source="input_perception", confidence=0.7, evidence_strength_score=0.7, metadata_completeness_score=1.0, drift_score=0.2)]
    result = propagate_region_masks_for_frame(1, np.zeros((16, 16, 3), dtype=np.float32), prev, scene, [], [], InMemoryMaskStore(), strict=False)
    torso = next(o for o in result.observations if o.region_id == "p1:torso")
    assert torso.observation_source == "carry_forward_previous_frame"
    assert torso.is_carry_forward is True
    assert torso.stale_frame_count == 1
    assert torso.mask_provenance != "input_frame_observed"


def test_stale_carry_forward_status_and_counter() -> None:
    scene = _scene()
    prev = [FrameRegionObservation(frame_index=2, region_id="p1:torso", bbox=BBox(0.0, 0.0, 0.2, 0.2), mask_ref="mask://old", mask_kind="alpha", mask_provenance="propagated_from_previous_frame", observation_source="carry_forward_previous_frame", confidence=0.6, evidence_strength_score=0.6, metadata_completeness_score=0.9, drift_score=0.5, stale_frame_count=2, is_carry_forward=True)]
    result = propagate_region_masks_for_frame(3, np.zeros((16, 16, 3), dtype=np.float32), prev, scene, [], [], InMemoryMaskStore(), strict=False)
    torso = next(o for o in result.observations if o.region_id == "p1:torso")
    assert torso.stale_frame_count == 3
    assert result.region_drift_summary["regions_by_status"]["stale_carry_forward"] == ["p1:torso"]


def test_reconciliation_iou_statuses() -> None:
    scene = _scene()
    prev = [
        FrameRegionObservation(frame_index=0, region_id="p1:face", bbox=BBox(0.2, 0.15, 0.2, 0.2), mask_ref="m1", mask_kind="alpha", mask_provenance="input_frame_observed", observation_source="input_perception", confidence=0.8, evidence_strength_score=0.8, metadata_completeness_score=1.0, drift_score=0.2),
        FrameRegionObservation(frame_index=0, region_id="p1:torso", bbox=BBox(0.4, 0.35, 0.4, 0.35), mask_ref="m2", mask_kind="alpha", mask_provenance="input_frame_observed", observation_source="input_perception", confidence=0.8, evidence_strength_score=0.8, metadata_completeness_score=1.0, drift_score=0.2),
    ]
    result = propagate_region_masks_for_frame(1, np.zeros((16, 16, 3), dtype=np.float32), prev, scene, [], [], InMemoryMaskStore(), strict=False)
    assert "p1:face" in result.region_drift_summary["regions_by_status"]["aligned"]
    assert "p1:torso" in result.region_drift_summary["regions_by_status"]["minor_drift"] or "p1:torso" in result.region_drift_summary["regions_by_status"]["major_drift"]


def test_changed_region_without_fresh_mask_reports_violation() -> None:
    scene = _scene()
    prev = [FrameRegionObservation(frame_index=0, region_id="p1:face", bbox=BBox(0.2, 0.15, 0.2, 0.2), mask_ref="old", mask_kind="alpha", mask_provenance="input_frame_observed", observation_source="input_perception", confidence=0.9, evidence_strength_score=0.9, metadata_completeness_score=1.0, drift_score=0.1)]
    result = propagate_region_masks_for_frame(1, np.zeros((16, 16, 3), dtype=np.float32), prev, scene, ["p1:face"], [], InMemoryMaskStore(), strict=True)
    assert any(v in {"changed_region_without_fresh_mask_evidence", "missing_changed_region_mask_evidence"} for v in result.diagnostics["violations"])


def test_orchestrator_debug_contains_propagation_contracts(tmp_path) -> None:
    engine = GennadyEngine()
    ppm = tmp_path / "seed.ppm"
    ppm.write_text("P3\n2 2\n255\n255 0 0  0 255 0  0 0 255  255 255 255\n", encoding="utf-8")
    artifacts = engine.run([str(ppm)], "улыбается", quality_profile="debug")
    assert "region_mask_propagation" in artifacts.debug
    assert "graph_mask_reconciliation" in artifacts.debug


def test_memory_generated_observation_does_not_update_identity_anchor() -> None:
    mm = MemoryManager()
    scene = _scene()
    memory = mm.initialize_from_scene(scene)
    before = list(memory.identity_memory["p1"].embedding)
    frame = np.ones((16, 16, 3), dtype=np.float32).tolist()
    mm.update_from_frame(memory, frame, scene, transition_context={"frame_source": "renderer_temporal_refined_generated_output", "generated": True, "source_is_input_frame": False, "immutable_i2v_anchor": False})
    after = list(memory.identity_memory["p1"].embedding)
    assert after == before


def test_memory_fallback_or_carryforward_cannot_raise_identity_confidence() -> None:
    mm = MemoryManager()
    scene = _scene()
    memory = mm.initialize_from_scene(scene)
    memory.identity_memory["p1"] = MemoryEntry(entity_id="p1", entry_type="identity", embedding=[0.1, 0.2], confidence=0.5)
    before_conf = memory.identity_memory["p1"].confidence
    frame = np.ones((16, 16, 3), dtype=np.float32).tolist()
    mm.update_from_frame(memory, frame, scene, transition_context={"frame_source": "generated_runtime_frame", "generated": True, "source_is_input_frame": False})
    assert memory.identity_memory["p1"].confidence <= before_conf
