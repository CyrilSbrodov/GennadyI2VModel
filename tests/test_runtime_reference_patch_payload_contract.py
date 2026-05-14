from __future__ import annotations

from core.schema import ReferencePatchPayload, RegionMemoryBundle
from runtime.orchestrator import _serialize_reference_payload_context


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
