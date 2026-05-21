from __future__ import annotations

from dataclasses import asdict

import numpy as np

from core.schema import BBox, ReferencePatchPayload, RegionMemoryBundle, RegionRef, SceneGraph, TexturePatchMemory, VideoMemory
from learned.interfaces import PatchSynthesisRequest
from memory.video_memory import MemoryManager
from rendering.trainable_patch_renderer import build_patch_batch
from runtime.orchestrator import _expected_reference_payload_for_region, _serialize_reference_payload_context, _trusted_matching_payloads


def _payload(kind: str, region: str) -> ReferencePatchPayload:
    return ReferencePatchPayload(
        reference_kind=kind,
        region_id=f"p1:{region}",
        canonical_region=region,
        entity_id="p1",
        patch_id=f"patch::p1:{region}:0",
        patch_ref="roi://0,0,8,8",
        confidence=0.8,
        evidence_score=0.8,
        observed_directly=True,
    )


def test_expected_payload_selected_for_face() -> None:
    identity_payload = _payload("identity_reference", "face")
    bundle = RegionMemoryBundle(
        entity_id="p1",
        canonical_region="face",
        region_id="p1:face",
        identity_reference_payload=identity_payload,
        reference_payloads=[identity_payload],
    )

    ctx = _serialize_reference_payload_context("p1:face", bundle)

    assert ctx["expected_reference_payload"]["reference_kind"] == "identity_reference"
    assert ctx["reference_patch_payloads"] == [ctx["expected_reference_payload"]]


def test_expected_payload_selected_for_torso_body_shape() -> None:
    body_payload = _payload("body_shape_reference", "torso")
    bundle = RegionMemoryBundle(
        entity_id="p1",
        canonical_region="torso",
        region_id="p1:torso",
        body_shape_reference_payload=body_payload,
        reference_payloads=[body_payload],
    )

    ctx = _serialize_reference_payload_context("p1:torso", bundle)

    assert ctx["expected_reference_payload"]["reference_kind"] == "body_shape_reference"
    assert ctx["reference_patch_payloads"] == [ctx["expected_reference_payload"]]


def test_expected_payload_selected_for_left_arm_body_shape() -> None:
    body_payload = _payload("body_shape_reference", "left_arm")
    bundle = RegionMemoryBundle(
        entity_id="p1",
        canonical_region="left_arm",
        region_id="p1:left_arm",
        body_shape_reference_payload=body_payload,
        reference_payloads=[body_payload],
    )

    ctx = _serialize_reference_payload_context("p1:left_arm", bundle)

    assert ctx["expected_reference_payload"]["reference_kind"] == "body_shape_reference"
    assert ctx["reference_payload_trace_reasons"] == []


def test_left_arm_skin_payload_is_not_selected_for_body_shape_contract() -> None:
    skin_payload = _payload("skin_reference", "left_arm")
    bundle = RegionMemoryBundle(
        entity_id="p1",
        canonical_region="left_arm",
        region_id="p1:left_arm",
        skin_reference_payload=skin_payload,
        reference_payloads=[skin_payload],
    )

    ctx = _serialize_reference_payload_context("p1:left_arm", bundle)

    assert ctx["expected_reference_payload"] is None
    assert "expected_reference_payload_missing" in ctx["reference_payload_trace_reasons"]


def test_expected_payload_can_fallback_to_single_matching_aggregate_payload() -> None:
    body_payload = _payload("body_shape_reference", "left_arm")
    bundle = RegionMemoryBundle(
        entity_id="p1",
        canonical_region="left_arm",
        region_id="p1:left_arm",
        reference_payloads=[body_payload],
    )

    ctx = _serialize_reference_payload_context("p1:left_arm", bundle)

    assert ctx["expected_reference_payload"]["reference_kind"] == "body_shape_reference"
    assert "expected_reference_payload_resolved_from_matching_payload" in ctx["reference_payload_trace_reasons"]


def test_matching_aggregate_payload_is_synced_before_material_consumption() -> None:
    payload = _payload("body_shape_reference", "left_arm")
    bundle = RegionMemoryBundle(
        entity_id="p1",
        canonical_region="left_arm",
        region_id="p1:left_arm",
        reference_payloads=[payload],
    )
    context = _serialize_reference_payload_context("p1:left_arm", bundle)
    expected = _expected_reference_payload_for_region("p1:left_arm", bundle)
    if expected is None:
        matches = _trusted_matching_payloads("p1:left_arm", bundle)
        if len(matches) == 1:
            expected = matches[0]
    assert expected is payload
    context = dict(context)
    context["expected_reference_payload"] = asdict(expected)
    reasons = list(context.get("reference_payload_trace_reasons", []))
    if "expected_reference_payload_resolved_from_matching_payload" not in reasons:
        reasons.append("expected_reference_payload_resolved_from_matching_payload")
    context["reference_payload_trace_reasons"] = reasons

    patch_id = payload.patch_id
    memory = VideoMemory(
        texture_patches={
            patch_id: TexturePatchMemory(
                patch_id=patch_id,
                region_type="left_arm",
                entity_id="p1",
                source_frame=0,
                patch_ref="roi://0,0,2,2",
                confidence=0.9,
                evidence_score=0.9,
                rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist(),
            )
        }
    )
    material = MemoryManager().build_reference_patch_material(memory, expected)
    context["expected_reference_patch_material"] = asdict(material)

    request = PatchSynthesisRequest(
        region=RegionRef(region_id="p1:left_arm", bbox=BBox(0, 0, 1, 1), reason="aggregate payload material sync"),
        scene_state=SceneGraph(frame_index=0),
        memory_summary={},
        transition_context=context,
        retrieval_summary={},
        current_frame=np.zeros((3, 3, 3), dtype=np.float32).tolist(),
    )
    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))

    assert request.transition_context["expected_reference_payload"]["patch_id"] == payload.patch_id
    assert batch.conditioning_summary["reference_patch_material_used"] is True
    assert "expected_reference_payload_resolved_from_matching_payload" in batch.conditioning_summary["reference_payload_trace_reasons"]
