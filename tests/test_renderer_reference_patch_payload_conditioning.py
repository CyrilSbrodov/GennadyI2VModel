from __future__ import annotations

import numpy as np

from core.schema import BBox, RegionRef, SceneGraph
from learned.interfaces import PatchSynthesisRequest
from rendering.trainable_patch_renderer import (
    build_patch_batch,
    extract_reference_payload_conditioning,
    output_from_prediction,
)


def _request(payload: dict[str, object] | None = None, *, region_id: str = "p1:face") -> PatchSynthesisRequest:
    ctx: dict[str, object] = {}
    if payload is not None:
        ctx["expected_reference_payload"] = payload
        ctx["reference_patch_payloads"] = [payload]
    return PatchSynthesisRequest(
        region=RegionRef(region_id, BBox(0.0, 0.0, 1.0, 1.0), "unit_test"),
        scene_state=SceneGraph(frame_index=0, persons=[], objects=[]),
        memory_summary={},
        transition_context=ctx,
        retrieval_summary={},
        current_frame=np.zeros((8, 8, 3), dtype=np.float32).tolist(),
    )


def _trusted_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "reference_kind": "identity_reference",
        "region_id": "p1:face",
        "canonical_region": "face",
        "entity_id": "p1",
        "patch_id": "patch::p1:face:0",
        "patch_ref": "roi://0,0,8,8",
        "confidence": 0.8,
        "evidence_score": 0.8,
        "observed_directly": True,
        "generated": False,
        "inferred": False,
        "descriptor": {"mean": [0.1, 0.2, 0.3], "std": [0.01, 0.02, 0.03]},
    }
    payload.update(overrides)
    return payload


def test_renderer_parses_trusted_identity_payload_and_conditions_vectors() -> None:
    payload = _trusted_payload()
    request = _request(payload)
    baseline = build_patch_batch(_request(), np.zeros((8, 8, 3), dtype=np.float32))
    batch = build_patch_batch(request, np.zeros((8, 8, 3), dtype=np.float32))
    cond = extract_reference_payload_conditioning(request.transition_context)

    assert cond["reference_payload_trusted"] is True
    assert cond["expected_reference_payload_present"] is True
    assert cond["expected_reference_payload_patch_id_present"] is True
    assert cond["expected_reference_payload_descriptor_present"] is True
    assert batch.memory_cond[8] > baseline.memory_cond[8]
    assert batch.appearance_cond[6] > baseline.appearance_cond[6]

    pred = {
        "rgb": np.zeros((8, 8, 3), dtype=np.float32),
        "alpha": np.ones((8, 8), dtype=np.float32) * 0.5,
        "uncertainty": np.ones((8, 8), dtype=np.float32) * 0.5,
        "confidence": 0.7,
    }
    output = output_from_prediction(request, pred, "learned_primary", {}, batch)
    assert output.execution_trace["reference_payload_trusted"] is True
    assert output.execution_trace["expected_reference_payload_descriptor_keys"] == ["mean", "std"]
    assert output.metadata["reference_payload_trusted"] is True


def test_untrusted_generated_payload_does_not_boost_reference_lane() -> None:
    request = _request(_trusted_payload(generated=True, inferred=True))
    baseline = build_patch_batch(_request(), np.zeros((8, 8, 3), dtype=np.float32))
    batch = build_patch_batch(request, np.zeros((8, 8, 3), dtype=np.float32))

    assert batch.conditioning_summary["reference_payload_trusted"] is False
    assert batch.conditioning_summary["reference_payload_untrusted_reason"] == "generated"
    assert batch.memory_cond[8] == baseline.memory_cond[8]
    assert batch.appearance_cond[7] >= baseline.appearance_cond[7]


def test_no_payload_keeps_backward_compatibility() -> None:
    request = _request()
    batch = build_patch_batch(request, np.zeros((8, 8, 3), dtype=np.float32))

    assert batch.conditioning_summary["reference_payload_present"] is False
    assert batch.conditioning_summary["expected_reference_payload_present"] is False


def test_payload_without_patch_cache_is_present_but_untrusted() -> None:
    payload: dict[str, object] = {
        "reference_kind": "body_shape_reference",
        "region_id": "p1:torso",
        "canonical_region": "torso",
        "entity_id": "p1",
        "patch_id": None,
        "patch_ref": None,
        "confidence": 0.9,
        "evidence_score": 0.9,
        "observed_directly": True,
        "generated": False,
        "inferred": False,
        "descriptor": {},
    }
    request = _request(payload, region_id="p1:torso")
    baseline = build_patch_batch(_request(region_id="p1:torso"), np.zeros((8, 8, 3), dtype=np.float32))
    batch = build_patch_batch(request, np.zeros((8, 8, 3), dtype=np.float32))
    cond = extract_reference_payload_conditioning(request.transition_context)

    assert cond["reference_payload_present"] is True
    assert cond["expected_reference_payload_present"] is True
    assert cond["expected_reference_payload_patch_id_present"] is False
    assert cond["expected_reference_payload_descriptor_present"] is False
    assert cond["reference_payload_trusted"] is False
    assert cond["reference_payload_untrusted_reason"] == "missing_patch_cache"
    assert batch.conditioning_summary["reference_payload_trusted"] is False
    assert batch.conditioning_summary["reference_payload_untrusted_reason"] == "missing_patch_cache"
    assert batch.memory_cond[8] == baseline.memory_cond[8]
    assert batch.appearance_cond[7] >= baseline.appearance_cond[7]
