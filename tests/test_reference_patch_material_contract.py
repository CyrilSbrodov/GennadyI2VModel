from __future__ import annotations

from dataclasses import asdict

import numpy as np

from core.schema import BBox, ReferencePatchPayload, RegionRef, SceneGraph, TexturePatchMemory, VideoMemory
from learned.interfaces import PatchSynthesisRequest
from memory.video_memory import MemoryManager
from rendering.trainable_patch_renderer import build_patch_batch, output_from_prediction, summarize_reference_material_trace, validate_reference_material_for_request
from runtime.orchestrator import GennadyEngine
from training.datasets import RendererDataset
from training.renderer_manifest_exporter import RendererManifestRecordExporter, _execution_trace_summary
from training.renderer_trainer import RendererBatchAdapter


def _payload(kind: str = "identity_reference", region: str = "face", entity: str = "p1", patch_id: str = "patch::p1:face:0") -> ReferencePatchPayload:
    return ReferencePatchPayload(
        reference_kind=kind,
        region_id=f"{entity}:{region}",
        canonical_region=region,
        entity_id=entity,
        patch_id=patch_id,
        patch_ref="roi://0,0,4,4",
        confidence=0.82,
        evidence_score=0.82,
        observed_directly=True,
        generated=False,
        inferred=False,
        provenance="parser",
        descriptor={"mean": [0.2, 0.3, 0.4]},
    )


def _patch(patch_id: str = "patch::p1:face:0", region: str = "face", entity: str = "p1", rgb_patch: list | None = None) -> TexturePatchMemory:
    return TexturePatchMemory(
        patch_id=patch_id,
        region_type=region,
        entity_id=entity,
        source_frame=0,
        patch_ref="roi://0,0,4,4",
        confidence=0.9,
        evidence_score=0.9,
        descriptor={"std": [0.1, 0.1, 0.1]},
        rgb_patch=rgb_patch,
    )


def _request(ctx: dict[str, object], region_id: str = "p1:face") -> PatchSynthesisRequest:
    return PatchSynthesisRequest(
        region=RegionRef(region_id=region_id, bbox=BBox(0, 0, 1, 1), reason="material contract"),
        scene_state=SceneGraph(frame_index=0),
        memory_summary={},
        transition_context=ctx,
        retrieval_summary={},
        current_frame=np.full((6, 8, 3), 0.1, dtype=np.float32).tolist(),
    )


def test_trusted_face_payload_resolves_to_trusted_material_with_bounded_rgb() -> None:
    memory = VideoMemory()
    rgb = np.full((96, 80, 3), [0.2, 0.4, 0.6], dtype=np.float32).tolist()
    memory.texture_patches["patch::p1:face:0"] = _patch(rgb_patch=rgb)

    material = MemoryManager().build_reference_patch_material(memory, _payload())

    assert material is not None
    assert material.material_trusted is True
    assert material.material_missing_reason == ""
    assert material.reference_kind == "identity_reference"
    assert np.asarray(material.rgb_patch).shape == (64, 53, 3)


def test_trusted_torso_payload_resolves_to_body_shape_material() -> None:
    memory = VideoMemory()
    patch_id = "patch::p1:torso:0"
    memory.texture_patches[patch_id] = _patch(patch_id=patch_id, region="torso", rgb_patch=np.ones((4, 4, 3), dtype=np.float32).tolist())

    material = MemoryManager().build_reference_patch_material(memory, _payload("body_shape_reference", "torso", patch_id=patch_id))

    assert material is not None
    assert material.material_trusted is True
    assert material.reference_kind == "body_shape_reference"
    assert material.canonical_region == "torso"


def test_generated_or_missing_tensor_or_mismatch_rejects_material() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(rgb_patch=None)

    generated = _payload()
    generated.generated = True
    assert manager.build_reference_patch_material(memory, generated).material_missing_reason == "generated_or_inferred"

    assert manager.build_reference_patch_material(memory, _payload()).material_missing_reason == "patch_tensor_missing"

    memory.texture_patches["patch::p1:face:0"] = _patch(entity="p2", rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())
    assert manager.build_reference_patch_material(memory, _payload()).material_missing_reason == "entity_mismatch"

    memory.texture_patches["patch::p1:face:0"] = _patch(region="torso", rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())
    assert manager.build_reference_patch_material(memory, _payload()).material_missing_reason == "region_mismatch"


def test_inferred_and_not_observed_payload_are_rejected() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())

    inferred = _payload()
    inferred.inferred = True
    assert manager.build_reference_patch_material(memory, inferred).material_missing_reason == "generated_or_inferred"

    unobserved = _payload()
    unobserved.observed_directly = False
    assert manager.build_reference_patch_material(memory, unobserved).material_missing_reason == "payload_untrusted"


def test_generated_texture_patch_cannot_be_trusted_material() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(
        rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist(),
    )
    memory.texture_patches["patch::p1:face:0"].generated = True
    memory.texture_patches["patch::p1:face:0"].observed_directly = False
    memory.texture_patches["patch::p1:face:0"].provenance = "runtime_generated_frame"

    material = manager.build_reference_patch_material(memory, _payload())

    assert material is not None
    assert material.material_trusted is False
    assert material.material_missing_reason == "patch_generated"


def test_runtime_generated_provenance_rejected_even_if_generated_flag_false() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())
    memory.texture_patches["patch::p1:face:0"].generated = False
    memory.texture_patches["patch::p1:face:0"].inferred = False
    memory.texture_patches["patch::p1:face:0"].observed_directly = True
    memory.texture_patches["patch::p1:face:0"].provenance = "runtime_generated_frame"

    material = manager.build_reference_patch_material(memory, _payload())

    assert material is not None
    assert material.material_trusted is False
    assert material.material_missing_reason == "patch_generated"


def test_generated_flag_rejected_even_if_provenance_claims_observed() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())
    memory.texture_patches["patch::p1:face:0"].generated = True
    memory.texture_patches["patch::p1:face:0"].inferred = False
    memory.texture_patches["patch::p1:face:0"].observed_directly = True
    memory.texture_patches["patch::p1:face:0"].provenance = "observed_input_frame"

    material = manager.build_reference_patch_material(memory, _payload())

    assert material is not None
    assert material.material_trusted is False
    assert material.material_missing_reason == "patch_generated"


def test_legacy_unknown_provenance_patch_remains_compatible_when_payload_trusted() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())
    material = manager.build_reference_patch_material(memory, _payload())

    assert material is not None
    assert material.material_trusted is True
    assert material.material_missing_reason == ""
    assert material.descriptor.get("source_patch_trust_note") == "patch_provenance_unknown"

def test_observed_patch_preferred_over_newer_generated_patch_for_reference_payload() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(
        patch_id="patch::p1:face:0",
        rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist(),
    )
    memory.texture_patches["patch::p1:face:0"].source_frame = 0
    memory.texture_patches["patch::p1:face:0"].observed_directly = True
    memory.texture_patches["patch::p1:face:0"].generated = False
    memory.texture_patches["patch::p1:face:0"].provenance = "observed_input_frame"
    memory.texture_patches["patch::p1:face:5"] = _patch(
        patch_id="patch::p1:face:5",
        rgb_patch=np.full((2, 2, 3), 0.5, dtype=np.float32).tolist(),
    )
    memory.texture_patches["patch::p1:face:5"].source_frame = 5
    memory.texture_patches["patch::p1:face:5"].observed_directly = False
    memory.texture_patches["patch::p1:face:5"].generated = True
    memory.texture_patches["patch::p1:face:5"].inferred = True
    memory.texture_patches["patch::p1:face:5"].provenance = "runtime_generated_frame"

    payload = _payload(patch_id="patch::p1:face:5")
    payload.source_frame = 5
    material = manager.build_reference_patch_material(memory, payload)
    assert material is not None
    assert material.material_trusted is False
    assert material.material_missing_reason == "patch_generated"

    from core.schema import CanonicalRegionMemoryEntry
    entry = CanonicalRegionMemoryEntry(
        record_id="p1:face",
        entity_id="p1",
        canonical_region="face",
        memory_kind="reference",
        source_frame=5,
        confidence=0.9,
        evidence_score=0.9,
        observed_directly=True,
        generated=False,
        inferred=False,
        reliable_as_reference=True,
        reference_kind="identity_reference",
    )
    chosen = manager._find_reference_texture_patch(memory, entry)
    assert chosen is not None
    assert chosen.patch_id == "patch::p1:face:0"


def test_renderer_maps_trusted_material_to_batch_tensors_and_trace() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 3, 3), [0.7, 0.2, 0.1], dtype=np.float32).tolist())}),
        _payload(),
    ))
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "expected_reference_patch_material": material})

    batch = build_patch_batch(request, np.zeros((5, 7, 3), dtype=np.float32))

    assert batch.reference_rgb.shape == (5, 7, 3)
    assert batch.reference_validity.shape == (5, 7, 1)
    assert float(np.sum(batch.reference_rgb)) > 0.0
    assert batch.conditioning_summary["reference_patch_material_used"] is True
    assert batch.conditioning_summary["reference_patch_material_shape"] == [2, 3, 3]


def test_renderer_missing_material_is_backward_compatible_and_not_used() -> None:
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "reference_patch_material_trace_reasons": ["patch_tensor_missing"]})

    batch = build_patch_batch(request, np.zeros((4, 4, 3), dtype=np.float32))

    assert np.allclose(batch.reference_rgb, 0.0)
    assert np.allclose(batch.reference_validity, 0.0)
    assert batch.conditioning_summary["reference_patch_material_used"] is False
    assert batch.conditioning_summary["reference_patch_material_missing_reason"] == "material_missing"
    assert "patch_tensor_missing" in batch.conditioning_summary["reference_patch_material_trace_reasons"]
    assert batch.conditioning_summary["reference_tensor_zero_fallback"] is True


def test_untrusted_material_is_present_but_not_used_and_raises_ambiguity() -> None:
    material = asdict(MemoryManager()._material_from_payload(_payload(), reason="generated_or_inferred", rgb_patch=np.ones((2, 2, 3)).tolist()))
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "expected_reference_patch_material": material})

    trace = summarize_reference_material_trace(request.transition_context, request=request)
    batch = build_patch_batch(request, np.zeros((4, 4, 3), dtype=np.float32))

    assert trace["reference_patch_material_present"] is True
    assert trace["reference_patch_material_used"] is False
    assert np.allclose(batch.reference_rgb, 0.0)
    assert batch.appearance_cond[7] > 0.0
    assert batch.conditioning_summary["reference_tensor_zero_fallback"] is True


def test_output_trace_and_manifest_include_reference_material_metadata() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.5, dtype=np.float32).tolist())}),
        _payload(),
    ))
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "expected_reference_patch_material": material})
    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))
    pred = {"rgb": batch.roi_after, "alpha": batch.alpha_target[..., 0], "uncertainty": batch.uncertainty_target[..., 0], "confidence": 0.9}

    output = output_from_prediction(request, pred, "learned_primary", {}, batch)
    record = RendererManifestRecordExporter().build_record(request=request, output=output, roi_before=batch.roi_before, roi_after=batch.roi_after, step_index=0, frame_index=0)

    assert output.execution_trace["reference_patch_material_used"] is True
    assert output.metadata["reference_patch_material_used"] is True
    assert record["execution_trace_summary"]["reference_patch_material_present"] is True
    assert record["execution_trace_summary"]["reference_patch_material_trusted"] is True
    assert record["renderer_batch_contract"]["expected_reference_patch_material"]["source_patch_id"] == "patch::p1:face:0"
    assert record["reference_tensor_input_used"] is True


def test_renderer_defensive_validation_rejects_forged_material() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    request = _request({"expected_reference_payload": asdict(_payload()), "expected_reference_patch_material": material})

    wrong_entity = dict(material, entity_id="p2", material_trusted=True)
    assert validate_reference_material_for_request(request, wrong_entity)["reason"] == "entity_mismatch"
    assert build_patch_batch(_request({"expected_reference_payload": asdict(_payload()), "expected_reference_patch_material": wrong_entity}), np.zeros((3, 3, 3), dtype=np.float32)).conditioning_summary["reference_patch_material_missing_reason"] == "entity_mismatch"

    wrong_kind = dict(material, reference_kind="body_shape_reference", material_trusted=True)
    assert validate_reference_material_for_request(request, wrong_kind)["reason"] == "kind_mismatch"

    generated = dict(material, generated=True, material_trusted=True)
    assert validate_reference_material_for_request(request, generated)["reason"] == "generated_or_inferred"

    invalid_rgb = dict(material, rgb_patch=[[1.0, 0.0, 0.0]], material_trusted=True)
    assert validate_reference_material_for_request(request, invalid_rgb)["reason"] == "invalid_rgb_shape"


def test_untrusted_runtime_material_object_remains_in_context_for_renderer_rejection() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())
    generated_payload = _payload()
    generated_payload.generated = True

    material = manager.build_reference_patch_material(memory, generated_payload)
    request = _request({"expected_reference_payload": asdict(generated_payload), "reference_patch_payloads": [asdict(generated_payload)], "expected_reference_patch_material": asdict(material), "reference_patch_material_trace_reasons": [material.material_missing_reason]})
    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))

    assert request.transition_context["expected_reference_patch_material"]["material_trusted"] is False
    assert request.transition_context["reference_patch_material_trace_reasons"] == ["generated_or_inferred"]
    assert batch.conditioning_summary["reference_patch_material_used"] is False
    assert batch.conditioning_summary["reference_patch_material_missing_reason"] == "generated_or_inferred"


def test_manifest_loader_and_training_adapter_restore_reference_tensors() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.6, dtype=np.float32).tolist())}),
        _payload(),
    ))
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "expected_reference_patch_material": material})
    batch = build_patch_batch(request, np.zeros((4, 4, 3), dtype=np.float32))
    pred = {"rgb": batch.roi_after, "alpha": batch.alpha_target[..., 0], "uncertainty": batch.uncertainty_target[..., 0], "confidence": 0.9}
    output = output_from_prediction(request, pred, "learned_primary", {}, batch)
    record = RendererManifestRecordExporter().build_record(request=request, output=output, roi_before=batch.roi_before, roi_after=batch.roi_after, step_index=0, frame_index=0)
    record["roi_pairs"] = [[record["roi_before"], record["roi_after"]]]

    loaded_request = RendererDataset.sample_to_patch_request(record)
    loaded_batch = build_patch_batch(loaded_request, np.asarray(record["roi_before"], dtype=np.float32))
    adapted = RendererBatchAdapter().adapt(record)

    assert loaded_batch.conditioning_summary["reference_patch_material_used"] is True
    assert float(np.sum(adapted.reference_rgb)) > 0.0
    assert adapted.conditioning_summary["reference_tensor_input_used"] is True


def test_missing_material_manifest_restores_zero_reference_tensors() -> None:
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "reference_patch_material_trace_reasons": ["patch_tensor_missing"]})
    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))
    pred = {"rgb": batch.roi_after, "alpha": batch.alpha_target[..., 0], "uncertainty": batch.uncertainty_target[..., 0], "confidence": 0.9}
    output = output_from_prediction(request, pred, "learned_primary", {}, batch)
    record = RendererManifestRecordExporter().build_record(request=request, output=output, roi_before=batch.roi_before, roi_after=batch.roi_after, step_index=0, frame_index=0)
    record["roi_pairs"] = [[record["roi_before"], record["roi_after"]]]

    adapted = RendererBatchAdapter().adapt(record)

    assert np.allclose(adapted.reference_rgb, 0.0)
    assert adapted.conditioning_summary["reference_tensor_zero_fallback"] is True


def test_reference_material_coverage_is_separate_from_family_coverage() -> None:
    coverage = GennadyEngine._summarize_step_reference_coverage([
        {
            "region_id": "p1:face",
            "execution_trace": {
                "identity_reference_used": True,
                "identity_reference_strength": 0.95,
                "reference_patch_material_used": False,
                "reference_patch_material_missing_reason": "material_missing",
            },
        },
        {
            "region_id": "p1:torso",
            "execution_trace": {
                "body_shape_reference_used": True,
                "body_shape_reference_strength": 0.95,
                "reference_patch_material_used": True,
            },
        },
    ])
    agg = GennadyEngine._aggregate_reference_coverage([coverage])

    assert agg["identity_reference_coverage_ratio"] == 1.0
    assert agg["identity_material_coverage_ratio"] == 0.0
    assert agg["body_shape_material_coverage_ratio"] == 1.0
    assert agg["material_missing_reasons"]["identity"]["material_missing"] == 1


def test_renderer_zero_fallback_when_only_generated_material_exists() -> None:
    manager = MemoryManager()
    memory = VideoMemory()
    memory.texture_patches["patch::p1:face:0"] = _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())
    memory.texture_patches["patch::p1:face:0"].generated = True
    memory.texture_patches["patch::p1:face:0"].observed_directly = False
    payload = _payload()
    material = asdict(manager.build_reference_patch_material(memory, payload))
    request = _request({"expected_reference_payload": asdict(payload), "reference_patch_payloads": [asdict(payload)], "expected_reference_patch_material": material})

    batch = build_patch_batch(request, np.zeros((4, 4, 3), dtype=np.float32))

    assert batch.conditioning_summary["reference_patch_material_used"] is False
    assert batch.conditioning_summary["reference_patch_material_trusted"] is False
    assert batch.conditioning_summary["reference_tensor_zero_fallback"] is True
    assert batch.conditioning_summary["reference_patch_material_missing_reason"] == "patch_generated"
    assert np.allclose(batch.reference_validity, 0.0)


def test_generated_material_rejection_appears_in_runtime_coverage_warnings() -> None:
    coverage = GennadyEngine._summarize_step_reference_coverage([
        {
            "region_id": "p1:face",
            "execution_trace": {
                "identity_reference_used": True,
                "identity_reference_strength": 0.9,
                "reference_patch_material_present": True,
                "reference_patch_material_trusted": False,
                "reference_patch_material_used": False,
                "reference_patch_material_missing_reason": "patch_generated",
            },
        }
    ])
    assert coverage["material_missing_reasons"]["identity"]["patch_generated"] == 1
    assert any("patch_generated" in warning for warning in coverage["critical_warnings"])


def test_black_reference_material_counts_as_tensor_input_used() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.zeros((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "expected_reference_patch_material": material})

    batch = build_patch_batch(request, np.full((3, 3, 3), 0.2, dtype=np.float32))

    assert np.allclose(batch.reference_rgb, 0.0)
    assert float(np.sum(batch.reference_validity)) > 0.0
    assert batch.conditioning_summary["reference_tensor_input_used"] is True
    assert batch.conditioning_summary["reference_tensor_zero_fallback"] is False


def test_payload_material_mismatch_rejects_material() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    material["source_patch_id"] = "patch::p1:face:1"
    request = _request({"expected_reference_payload": asdict(_payload()), "expected_reference_patch_material": material})

    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))

    assert batch.conditioning_summary["reference_patch_material_used"] is False
    assert batch.conditioning_summary["reference_patch_material_missing_reason"] == "payload_material_mismatch"
    assert np.allclose(batch.reference_validity, 0.0)


def test_non_request_material_trace_requires_trusted_payload() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    trace = summarize_reference_material_trace({"expected_reference_patch_material": material}, request=None)

    assert trace["reference_patch_material_present"] is True
    assert trace["reference_patch_material_validated"] is True
    assert trace["reference_patch_material_trusted"] is False
    assert trace["reference_patch_material_used"] is False
    assert trace["reference_patch_material_missing_reason"] == "expected_reference_payload_missing"


def test_valid_material_without_expected_payload_is_not_trusted_or_used() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    request = _request({"expected_reference_patch_material": material})

    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))
    output = output_from_prediction(
        request,
        {"rgb": batch.roi_after, "alpha": batch.alpha_target[..., 0], "uncertainty": batch.uncertainty_target[..., 0], "confidence": 0.9},
        "learned_primary",
        {},
        batch,
    )
    record = RendererManifestRecordExporter().build_record(request=request, output=output, roi_before=batch.roi_before, roi_after=batch.roi_after, step_index=0, frame_index=0)

    assert batch.conditioning_summary["reference_patch_material_used"] is False
    assert batch.conditioning_summary["reference_patch_material_validated"] is True
    assert batch.conditioning_summary["reference_patch_material_trusted"] is False
    assert batch.conditioning_summary["reference_patch_material_missing_reason"] == "expected_reference_payload_missing"
    assert output.execution_trace["reference_patch_material_used"] is False
    assert output.execution_trace["reference_patch_material_trusted"] is False
    assert output.execution_trace["reference_patch_material_missing_reason"] == "expected_reference_payload_missing"
    assert record["execution_trace_summary"]["reference_patch_material_used"] is False
    assert record["execution_trace_summary"]["reference_patch_material_trusted"] is False
    assert record["execution_trace_summary"]["reference_patch_material_missing_reason"] == "expected_reference_payload_missing"
    assert np.allclose(batch.reference_validity, 0.0)


def test_identity_subregions_accept_face_identity_material() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    payload = asdict(_payload())
    request = _request({"expected_reference_payload": payload, "reference_patch_payloads": [payload], "expected_reference_patch_material": material}, region_id="p1:eyes")

    result = validate_reference_material_for_request(request, material)
    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))

    assert result["valid"] is True
    assert batch.conditioning_summary["reference_patch_material_used"] is True
    assert batch.conditioning_summary["reference_tensor_input_used"] is True


def test_identity_subregion_rejects_payload_material_patch_mismatch() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    material["source_patch_id"] = "patch::p1:face:1"
    payload = asdict(_payload())
    request = _request({"expected_reference_payload": payload, "reference_patch_payloads": [payload], "expected_reference_patch_material": material}, region_id="p1:eyes")

    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))

    assert batch.conditioning_summary["reference_patch_material_used"] is False
    assert batch.conditioning_summary["reference_patch_material_missing_reason"] == "payload_material_mismatch"
    assert np.allclose(batch.reference_validity, 0.0)


def test_identity_subregion_rejects_material_from_other_entity() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    material["entity_id"] = "p2"
    payload = asdict(_payload())
    request = _request({"expected_reference_payload": payload, "reference_patch_payloads": [payload], "expected_reference_patch_material": material}, region_id="p1:eyes")

    result = validate_reference_material_for_request(request, material)

    assert result["valid"] is False
    assert result["reason"] == "entity_mismatch"


def test_manifest_compacts_transition_summary_but_loader_restores_full_material() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.25, dtype=np.float32).tolist())}),
        _payload(),
    ))
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "expected_reference_patch_material": material})
    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))
    output = output_from_prediction(request, {"rgb": batch.roi_after, "alpha": batch.alpha_target[..., 0], "uncertainty": batch.uncertainty_target[..., 0], "confidence": 0.9}, "learned_primary", {}, batch)

    record = RendererManifestRecordExporter().build_record(request=request, output=output, roi_before=batch.roi_before, roi_after=batch.roi_after, step_index=0, frame_index=0)
    loaded_request = RendererDataset.sample_to_patch_request({**record, "roi_pairs": [[record["roi_before"], record["roi_after"]]]})

    assert "rgb_patch" in record["renderer_batch_contract"]["expected_reference_patch_material"]
    compact = record["transition_context_summary"]["expected_reference_patch_material"]
    assert "rgb_patch" not in compact
    assert compact["rgb_patch_shape"] == [2, 2, 3]
    assert loaded_request.transition_context["expected_reference_patch_material"]["rgb_patch"] == material["rgb_patch"]


def test_manifest_loader_restores_expected_payload_from_renderer_batch_contract() -> None:
    payload = asdict(_payload())
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.35, dtype=np.float32).tolist())}),
        _payload(),
    ))
    record = {
        "region_id": "p1:face",
        "roi_pairs": [[np.zeros((3, 3, 3), dtype=np.float32).tolist(), np.zeros((3, 3, 3), dtype=np.float32).tolist()]],
        "transition_context_summary": {
            "expected_reference_patch_material": {"rgb_patch_shape": [2, 2, 3]},
        },
        "renderer_batch_contract": {
            "expected_reference_payload": payload,
            "expected_reference_patch_material": material,
            "reference_patch_material_trace_reasons": [],
        },
    }

    loaded_request = RendererDataset.sample_to_patch_request(record)
    loaded_batch = build_patch_batch(loaded_request, np.zeros((3, 3, 3), dtype=np.float32))

    assert loaded_request.transition_context["expected_reference_payload"] == payload
    assert loaded_request.transition_context["expected_reference_patch_material"]["rgb_patch"] == material["rgb_patch"]
    assert loaded_batch.conditioning_summary["reference_patch_material_used"] is True
    assert loaded_batch.conditioning_summary["reference_patch_material_trusted"] is True




def test_manifest_loader_restores_top_level_expected_payload_without_renderer_contract() -> None:
    payload = asdict(_payload())
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.45, dtype=np.float32).tolist())}),
        _payload(),
    ))
    record = {
        "region_id": "p1:face",
        "roi_pairs": [[np.zeros((3, 3, 3), dtype=np.float32).tolist(), np.zeros((3, 3, 3), dtype=np.float32).tolist()]],
        "expected_reference_payload": payload,
        "expected_reference_patch_material": material,
    }

    loaded_request = RendererDataset.sample_to_patch_request(record)
    loaded_batch = build_patch_batch(loaded_request, np.zeros((3, 3, 3), dtype=np.float32))

    assert loaded_request.transition_context["expected_reference_payload"] == payload
    assert loaded_request.transition_context["expected_reference_patch_material"]["rgb_patch"] == material["rgb_patch"]
    assert loaded_batch.conditioning_summary["reference_patch_material_used"] is True

def test_training_adapter_revalidates_manifest_material_and_overwrites_stale_summary() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    material["entity_id"] = "p2"
    sample = {
        "region_id": "p1:face",
        "roi_pairs": [[np.zeros((3, 3, 3), dtype=np.float32).tolist(), np.zeros((3, 3, 3), dtype=np.float32).tolist()]],
        "renderer_batch_contract": {
            "expected_reference_patch_material": material,
            "reference_patch_material_used": True,
            "conditioning_summary": {"reference_patch_material_used": True, "reference_tensor_input_used": True},
        },
    }

    adapted = RendererBatchAdapter().adapt(sample)

    assert np.allclose(adapted.reference_rgb, 0.0)
    assert adapted.conditioning_summary["reference_patch_material_used"] is False
    assert adapted.conditioning_summary["reference_tensor_input_used"] is False
    assert adapted.conditioning_summary["reference_tensor_zero_fallback"] is True
    assert adapted.conditioning_summary["reference_patch_material_missing_reason"] == "entity_mismatch"


def test_training_adapter_rejects_material_without_trusted_expected_payload() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    sample = {
        "region_id": "p1:face",
        "roi_pairs": [[np.zeros((3, 3, 3), dtype=np.float32).tolist(), np.zeros((3, 3, 3), dtype=np.float32).tolist()]],
        "renderer_batch_contract": {
            "expected_reference_patch_material": material,
            "reference_patch_material_used": True,
        },
    }

    adapted = RendererBatchAdapter().adapt(sample)

    assert np.allclose(adapted.reference_rgb, 0.0)
    assert adapted.conditioning_summary["reference_patch_material_validated"] is True
    assert adapted.conditioning_summary["reference_patch_material_trusted"] is False
    assert adapted.conditioning_summary["reference_patch_material_used"] is False
    assert adapted.conditioning_summary["reference_tensor_input_used"] is False
    assert adapted.conditioning_summary["reference_tensor_zero_fallback"] is True
    assert adapted.conditioning_summary["reference_patch_material_missing_reason"] == "expected_reference_payload_missing"


def test_training_adapter_uses_material_with_matching_trusted_expected_payload() -> None:
    payload = asdict(_payload())
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.zeros((2, 2, 3), dtype=np.float32).tolist())}),
        _payload(),
    ))
    sample = {
        "region_id": "p1:face",
        "roi_pairs": [[np.full((3, 3, 3), 0.2, dtype=np.float32).tolist(), np.full((3, 3, 3), 0.2, dtype=np.float32).tolist()]],
        "renderer_batch_contract": {
            "expected_reference_payload": payload,
            "expected_reference_patch_material": material,
            "reference_patch_material_used": True,
        },
    }

    adapted = RendererBatchAdapter().adapt(sample)

    assert np.allclose(adapted.reference_rgb, 0.0)
    assert float(np.sum(adapted.reference_validity)) > 0.0
    assert adapted.conditioning_summary["reference_patch_material_used"] is True
    assert adapted.conditioning_summary["reference_tensor_input_used"] is True
    assert adapted.conditioning_summary["reference_tensor_zero_fallback"] is False


def test_adapter_payload_diagnostics_are_truthful_when_material_is_used() -> None:
    payload = asdict(_payload())
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.55, dtype=np.float32).tolist())}),
        _payload(),
    ))
    sample = {
        "region_id": "p1:face",
        "roi_pairs": [[np.full((3, 3, 3), 0.2, dtype=np.float32).tolist(), np.full((3, 3, 3), 0.2, dtype=np.float32).tolist()]],
        "renderer_batch_contract": {
            "expected_reference_payload": payload,
            "expected_reference_patch_material": material,
        },
    }

    adapted = RendererBatchAdapter().adapt(sample)

    assert adapted.conditioning_summary["reference_patch_material_used"] is True
    assert adapted.conditioning_summary["reference_payload_trusted"] is True
    assert adapted.conditioning_summary["expected_reference_payload_present"] is True
    assert adapted.conditioning_summary["expected_reference_payload_kind"] == "identity_reference"


def test_adapter_supports_top_level_reference_payload_and_material_without_contract() -> None:
    payload = asdict(_payload())
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.65, dtype=np.float32).tolist())}),
        _payload(),
    ))
    sample = {
        "region_id": "p1:face",
        "roi_pairs": [[np.full((3, 3, 3), 0.25, dtype=np.float32).tolist(), np.full((3, 3, 3), 0.25, dtype=np.float32).tolist()]],
        "expected_reference_payload": payload,
        "expected_reference_patch_material": material,
    }

    adapted = RendererBatchAdapter().adapt(sample)

    assert adapted.conditioning_summary["reference_patch_material_used"] is True
    assert adapted.conditioning_summary["reference_tensor_input_used"] is True
    assert float(np.sum(adapted.reference_validity)) > 0.0


def test_manifest_execution_trace_fallback_never_trusts_material_without_payload() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.35, dtype=np.float32).tolist())}),
        _payload(),
    ))
    summary = _execution_trace_summary(
        trace={},
        ctx={"expected_reference_patch_material": material},
        memory_bundle={},
        selected_render_strategy="learned_primary",
        renderer_path="learned_primary",
        synthesis_mode="learned_primary",
    )

    assert summary["reference_patch_material_validated"] is True
    assert summary["reference_patch_material_trusted"] is False
    assert summary["reference_patch_material_used"] is False
    assert summary["reference_patch_material_missing_reason"] == "expected_reference_payload_missing"


def test_runtime_debug_trace_exports_reference_patch_material_validated() -> None:
    material = asdict(MemoryManager().build_reference_patch_material(
        VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.full((2, 2, 3), 0.45, dtype=np.float32).tolist())}),
        _payload(),
    ))
    request = _request({"expected_reference_payload": asdict(_payload()), "reference_patch_payloads": [asdict(_payload())], "expected_reference_patch_material": material})
    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))
    output = output_from_prediction(request, {"rgb": batch.roi_after, "alpha": batch.alpha_target[..., 0], "uncertainty": batch.uncertainty_target[..., 0], "confidence": 0.9}, "learned_primary", {}, batch)

    assert output.execution_trace["reference_patch_material_validated"] is True
    assert output.metadata["reference_patch_material_validated"] is True


def test_output_from_prediction_propagates_material_gate_and_local_tensor_channels() -> None:
    import importlib

    importlib.import_module("rendering.torch_local_patch_generator")
    importlib.import_module("rendering.trainable_patch_renderer")
    importlib.import_module("training.renderer_manifest_exporter")

    material = asdict(
        MemoryManager().build_reference_patch_material(
            VideoMemory(texture_patches={"patch::p1:face:0": _patch(rgb_patch=np.ones((2, 2, 3), dtype=np.float32).tolist())}),
            _payload(),
        )
    )
    request = _request(
        {
            "expected_reference_payload": asdict(_payload()),
            "reference_patch_payloads": [asdict(_payload())],
            "expected_reference_patch_material": material,
        }
    )
    batch = build_patch_batch(request, np.zeros((3, 3, 3), dtype=np.float32))
    batch.conditioning_summary["local_tensor_input_channels"] = 14
    pred = {
        "rgb": batch.roi_after,
        "alpha": batch.alpha_target[..., 0],
        "uncertainty": batch.uncertainty_target[..., 0],
        "confidence": 0.9,
        "material_gate_mean": 0.12,
        "material_gate_max": 0.22,
        "material_gate_cap": 0.35,
    }
    output = output_from_prediction(request, pred, "learned_primary", {}, batch)
    record = RendererManifestRecordExporter().build_record(
        request=request,
        output=output,
        roi_before=batch.roi_before,
        roi_after=batch.roi_after,
        step_index=0,
        frame_index=0,
    )
    assert output.execution_trace["material_gate_mean"] == 0.12
    assert output.metadata["material_gate_mean"] == 0.12
    assert any("mat_gate_mean=" in item for item in output.debug_trace)
    assert output.execution_trace["local_tensor_input_channels"] == 14
    assert output.metadata["local_tensor_input_channels"] == 14
    assert record["execution_trace_summary"]["local_tensor_input_channels"] == 14
