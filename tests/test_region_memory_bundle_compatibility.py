from __future__ import annotations

from dataclasses import asdict

from core.schema import BBox, RegionMemoryBundle, RegionRef, SceneGraph
from learned.interfaces import PatchSynthesisRequest
from rendering.trainable_patch_renderer import extract_memory_bundle_conditioning


def _request_with_context(transition_context: dict[str, object]) -> PatchSynthesisRequest:
    return PatchSynthesisRequest(
        region=RegionRef(region_id="p1:torso", bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="bundle compatibility"),
        scene_state=SceneGraph(frame_index=1),
        memory_summary={},
        transition_context=transition_context,
        retrieval_summary={},
        current_frame=[[[0.1, 0.2, 0.3]]],
    )


def _body_shape_bundle() -> RegionMemoryBundle:
    return RegionMemoryBundle(
        entity_id="p1",
        canonical_region="torso",
        region_id="p1:torso",
        memory_support_level="strong",
        retrieval_reasons=["body_shape_reference_available", "body_shape_reference_observed_strong"],
        has_body_shape_reference=True,
    )


def test_minimal_region_memory_bundle_constructor_defaults_new_reference_fields() -> None:
    bundle = RegionMemoryBundle(entity_id="p1", canonical_region="face", region_id="p1:face")

    assert bundle.has_skin_reference is False
    assert bundle.has_body_shape_reference is False
    assert bundle.has_accessory_reference is False
    assert bundle.skin_reference is None
    assert bundle.body_shape_reference is None
    assert bundle.accessory_reference is None


def test_region_memory_bundle_asdict_includes_new_reference_fields() -> None:
    bundle = RegionMemoryBundle(entity_id="p1", canonical_region="face", region_id="p1:face")

    payload = asdict(bundle)

    assert "skin_reference" in payload
    assert "body_shape_reference" in payload
    assert "accessory_reference" in payload
    assert "has_skin_reference" in payload
    assert "has_body_shape_reference" in payload
    assert "has_accessory_reference" in payload


def test_renderer_parses_object_region_memory_bundle_body_shape_fields() -> None:
    bundle = _body_shape_bundle()
    request = _request_with_context({"region_memory_bundle": bundle})

    cond = extract_memory_bundle_conditioning(request)

    assert cond["body_shape_reference_used"] is True
    assert cond["body_shape_reference_strength"] >= 0.9
    assert cond["identity_reference_used"] is False


def test_renderer_parses_serialized_region_memory_bundle_body_shape_fields() -> None:
    bundle = _body_shape_bundle()
    request = _request_with_context({"region_memory_bundle_serialized": asdict(bundle)})

    cond = extract_memory_bundle_conditioning(request)

    assert cond["body_shape_reference_used"] is True
    assert cond["body_shape_reference_strength"] >= 0.9
    assert cond["identity_reference_used"] is False
