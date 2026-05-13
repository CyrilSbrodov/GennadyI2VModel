from __future__ import annotations

import numpy as np

from core.schema import BBox, RegionRef, SceneGraph
from learned.interfaces import PatchSynthesisRequest
from rendering.trainable_patch_renderer import (
    apply_memory_bundle_conditioning_to_vectors,
    extract_memory_bundle_conditioning,
)


def _request(
    region_id: str,
    reasons: list[str],
    support: str = "strong",
    bundle_fields: dict[str, object] | None = None,
) -> PatchSynthesisRequest:
    bundle = {
        "memory_bundle_present": True,
        "memory_support_level": support,
        "retrieval_reasons": reasons,
    }
    if bundle_fields:
        bundle.update(bundle_fields)
    return PatchSynthesisRequest(
        region=RegionRef(region_id=region_id, bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="reference family test"),
        scene_state=SceneGraph(frame_index=1),
        memory_summary={},
        transition_context={"region_memory_bundle_serialized": bundle},
        retrieval_summary={},
        current_frame=[[[0.1, 0.2, 0.3]]],
    )


def _conditioned_vectors(request: PatchSynthesisRequest) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cond = extract_memory_bundle_conditioning(request)
    mem = np.zeros(10, dtype=np.float32)
    app = np.full(8, 0.2, dtype=np.float32)
    new_mem, new_app = apply_memory_bundle_conditioning_to_vectors(mem, app, cond, region_id=request.region.region_id)
    return cond, mem, app, new_mem, new_app


def test_body_shape_reference_affects_torso_conditioning_without_identity() -> None:
    request = _request("p1:torso", ["body_shape_reference_available", "body_shape_reference_observed_strong"])

    cond, mem, app, new_mem, new_app = _conditioned_vectors(request)

    assert cond["body_shape_reference_used"] is True
    assert cond["body_shape_reference_strength"] >= 0.9
    assert cond["identity_reference_used"] is False
    assert not np.array_equal(new_mem, mem)
    assert not np.array_equal(new_app, app)


def test_generated_body_reference_is_blocked_without_strong_boost() -> None:
    request = _request("p1:torso", ["body_shape_reference_blocked_generated"], support="weak")

    cond, mem, app, new_mem, new_app = _conditioned_vectors(request)

    assert cond["body_shape_reference_used"] is False
    assert cond["body_shape_reference_blocked"] is True
    assert cond["body_shape_reference_strength"] == 0.0
    assert new_mem[5] == mem[5]
    assert new_app[7] > app[7]


def test_garment_reference_affects_outer_garment_conditioning_not_identity() -> None:
    request = _request("p1:outer_garment", ["garment_reference_available", "garment_reference_observed_strong"])

    cond, mem, app, new_mem, new_app = _conditioned_vectors(request)

    assert cond["garment_reference_used"] is True
    assert cond["garment_reference_strength"] >= 0.9
    assert cond["identity_reference_used"] is False
    assert not np.array_equal(new_mem, mem)
    assert not np.array_equal(new_app, app)


def test_blocked_body_shape_reference_with_has_flag_does_not_set_usable_marker() -> None:
    request = _request(
        "p1:torso",
        ["body_shape_reference_blocked_generated"],
        support="strong",
        bundle_fields={"has_body_shape_reference": True},
    )

    cond, mem, _app, new_mem, _new_app = _conditioned_vectors(request)

    assert cond["body_shape_reference_blocked"] is True
    assert cond["body_shape_reference_used"] is False
    assert cond["body_shape_reference_strength"] == 0.0
    assert new_mem[5] == mem[5]
    assert new_mem[8] != 1.0


def test_blocked_garment_reference_with_has_flag_does_not_set_usable_marker() -> None:
    request = _request(
        "p1:outer_garment",
        ["garment_reference_blocked_generated"],
        support="strong",
        bundle_fields={"has_garment_reference": True},
    )

    cond, _mem, _app, new_mem, _new_app = _conditioned_vectors(request)

    assert cond["garment_reference_blocked"] is True
    assert cond["garment_reference_used"] is False
    assert cond["garment_reference_strength"] == 0.0
    assert new_mem[8] != 1.0
