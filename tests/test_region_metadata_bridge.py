from __future__ import annotations

import numpy as np

from core.schema import BBox, BodyPartNode, CanonicalRegionMemoryEntry, GarmentNode, GraphDelta, PersonNode, RegionRef, SceneGraph, VideoMemory
from learned.interfaces import DynamicsTransitionOutput, PatchSynthesisOutput, PatchSynthesisRequest, TemporalRefinementOutput
from learned.parity import build_parity_result, patch_io_to_contract
from memory.video_memory import MemoryManager
from representation.learned_bridge import BaselineGraphEncoder, BaselineIdentityAppearanceEncoder
from runtime.orchestrator import GennadyEngine
from text.learned_bridge import BaselineTextEncoderAdapter
from types import SimpleNamespace
from pathlib import Path
from perception.mask_store import DEFAULT_MASK_STORE
from rendering.trainable_patch_renderer import TrainableLocalPatchModel, build_patch_batch, extract_region_metadata_conditioning, output_from_prediction
from runtime.region_metadata import build_region_metadata
from runtime.region_routing import RegionRoutingDecision


def _mask(kind: str = "body_part_mask", *, parser_class: str = "face", class_id: int = 1) -> str:
    return DEFAULT_MASK_STORE.put(
        [[1, 1], [1, 0]],
        0.91,
        "parser:test",
        "region-metadata-test",
        mask_kind=kind,
        backend="unit-parser",
        frame_size=(64, 64),
        tags=["parser", parser_class],
        extra={"pixel_count": 3, "bbox_xyxy": (0.2, 0.2, 0.5, 0.55), "parser_class_name": parser_class, "class_id": class_id},
    )


def _graph(mask_ref: str | None = None, *, with_garment: bool = False) -> SceneGraph:
    canonical = {
        "face": {
            "canonical_name": "face",
            "raw_sources": ["face"],
            "source_regions": ["face"],
            "mask_ref": mask_ref,
            "confidence": 0.88 if mask_ref else 0.2,
            "visibility_state": "visible",
            "provenance": "parser:test" if mask_ref else "fallback_template",
            "attachment_hints": ["head"],
            "ownership_hints": ["person"],
            "coverage_hints": [],
        },
        "outer_garment": {
            "canonical_name": "outer_garment",
            "raw_sources": ["coat"],
            "source_regions": ["coat"],
            "mask_ref": mask_ref if with_garment else None,
            "confidence": 0.86 if with_garment else 0.1,
            "visibility_state": "partially_visible",
            "provenance": "parser:test" if with_garment else "canonical_reasoner",
            "attachment_hints": ["torso"],
            "ownership_hints": ["person"],
            "coverage_hints": ["torso", "left_arm"],
        },
    }
    person = PersonNode(
        person_id="p1",
        track_id="t1",
        bbox=BBox(0.1, 0.1, 0.8, 0.8),
        mask_ref=None,
        body_parts=[BodyPartNode(part_id="p1_face", part_type="face", mask_ref=mask_ref, visibility="visible", confidence=0.9, source="parser:test")],
        garments=[GarmentNode(garment_id="p1_outer", garment_type="outer_garment", mask_ref=mask_ref if with_garment else None, confidence=0.87, source="parser:test")],
        canonical_regions=canonical,
        confidence=0.92,
    )
    return SceneGraph(frame_index=0, persons=[person], objects=[])


def _decision(region: str = "face") -> RegionRoutingDecision:
    return RegionRoutingDecision(
        canonical_region=region,
        decision="expression_refine" if region == "face" else "garment_transition_update",
        priority=90,
        reasons=["unit_test"],
        memory_source_available=True,
        memory_support_level="strong",
        reveal_mode="none",
        synthesis_required=True,
        renderer_mode_hint="expression" if region == "face" else "garment_surface",
        confidence=0.82,
    )


def _memory(mask_ref: str | None = None) -> VideoMemory:
    memory = VideoMemory()
    memory.canonical_region_memory["p1:face"] = CanonicalRegionMemoryEntry(
        record_id="p1:face",
        entity_id="p1",
        canonical_region="face",
        memory_kind="observed",
        mask_ref=mask_ref,
        confidence=0.8,
        visibility_state="visible",
        provenance="unit",
        evidence_score=0.8,
        observed_directly=True,
        reliable_for_reuse=True,
        suitable_for_reveal=True,
    )
    return memory


def test_builder_preserves_parser_mask_graph_and_route_metadata() -> None:
    DEFAULT_MASK_STORE.clear()
    mask_ref = _mask("body_part_mask", parser_class="face", class_id=7)
    graph = _graph(mask_ref)
    region = RegionRef("p1:face", BBox(0.18, 0.18, 0.36, 0.4), f"graph_semantic:body_part_mask;roi_source=parser_mask_bbox;mask_ref={mask_ref};source=parser:test;confidence=0.9")

    metadata = build_region_metadata(scene_graph=graph, memory=_memory(mask_ref), region=region, route_decision=_decision("face"), delta=GraphDelta(affected_regions=["face"], region_transition_mode={"face": "expression_refine"}))

    assert metadata["region_id"] == "p1:face"
    assert metadata["entity_id"] == "p1"
    assert metadata["canonical_region"] == "face"
    assert metadata["roi_source"] == "parser_mask_bbox"
    assert metadata["mask_ref"] == mask_ref
    assert metadata["mask_kind"] == "body_part_mask"
    assert metadata["source_node_type"] == "body_part"
    assert metadata["parser_class_name"] == "face"
    assert metadata["parser_class_id"] == 7
    assert float(metadata["metadata_completeness_score"]) > 0.75
    assert float(metadata["evidence_strength_score"]) > 0.7
    assert "region_id" not in metadata["missing_fields"]
    assert "mask_ref" not in metadata["missing_fields"]


def test_builder_marks_person_bbox_fallback_as_lower_completeness_with_missing_mask_fields() -> None:
    DEFAULT_MASK_STORE.clear()
    graph = _graph(None)
    region = RegionRef("p1:face", BBox(0.2, 0.15, 0.25, 0.25), "roi_source=person_bbox_fallback;fallback:person_bbox_template")

    metadata = build_region_metadata(scene_graph=graph, memory=VideoMemory(), region=region, route_decision=None, delta=None)

    assert metadata["roi_source"] == "person_bbox_fallback"
    assert metadata["source_node_type"] in {"fallback", "canonical_region", "body_part", "face_region"}
    assert not metadata.get("mask_ref")
    assert 0.0 < float(metadata["metadata_completeness_score"]) <= 0.75
    assert float(metadata["evidence_strength_score"]) < 0.2
    assert "mask_ref" in metadata["missing_fields"]
    assert "mask_kind" in metadata["missing_fields"]


def test_renderer_conditioning_uses_region_metadata_without_shape_regression() -> None:
    DEFAULT_MASK_STORE.clear()
    mask_ref = _mask("garment_mask", parser_class="coat", class_id=14)
    graph = _graph(mask_ref, with_garment=True)
    region = RegionRef("p1:outer_garment", BBox(0.15, 0.2, 0.45, 0.5), f"graph_semantic:garment_mask;roi_source=parser_mask_bbox;mask_ref={mask_ref}")
    metadata = build_region_metadata(scene_graph=graph, memory=VideoMemory(), region=region, route_decision=_decision("outer_garment"), delta=GraphDelta(newly_revealed_regions=[region]))
    base = PatchSynthesisRequest(region=region, scene_state=graph, memory_summary={}, transition_context={"graph_delta": GraphDelta()}, retrieval_summary={}, current_frame=[[[0.1, 0.1, 0.1] for _ in range(4)] for _ in range(4)])
    enriched = PatchSynthesisRequest(region=region, scene_state=graph, memory_summary={}, transition_context={"graph_delta": GraphDelta()}, retrieval_summary={}, current_frame=base.current_frame, region_metadata=metadata)
    roi = np.asarray(base.current_frame, dtype=np.float32)

    without = build_patch_batch(base, roi)
    with_metadata = build_patch_batch(enriched, roi)

    assert without.memory_cond.shape == with_metadata.memory_cond.shape
    assert without.appearance_cond.shape == with_metadata.appearance_cond.shape
    assert without.conditioning_summary["region_metadata_used"] is False
    assert with_metadata.conditioning_summary["region_metadata_used"] is True
    assert with_metadata.conditioning_summary["roi_source"] == "parser_mask_bbox"
    assert with_metadata.conditioning_summary["source_node_type"] == "garment"
    assert with_metadata.conditioning_summary["mask_ref_present"] is True
    assert np.any(without.memory_cond != with_metadata.memory_cond) or np.any(without.appearance_cond != with_metadata.appearance_cond)
    prediction = TrainableLocalPatchModel().forward(with_metadata)
    output = output_from_prediction(enriched, prediction, "learned_primary", {"torch_backend_used": False}, batch=with_metadata)
    assert output.execution_trace["region_metadata_used"] is True
    assert output.execution_trace["region_metadata_evidence_strength_score"] > 0.0
    assert output.execution_trace["region_metadata_roi_source"] == "parser_mask_bbox"
    assert output.execution_trace["region_metadata_source_node_type"] == "garment"
    assert output.execution_trace["region_metadata_mask_kind"] == "garment_mask"
    assert output.execution_trace["region_metadata_mask_ref_present"] is True
    assert "has_parser_mask" in output.execution_trace["region_metadata_feature_keys"]


def test_mask_area_ratio_uses_frame_size_instead_of_magic_pixel_saturation() -> None:
    metadata = {
        "mask_ref": "mask://large",
        "mask_pixel_count": "50000",
        "mask_frame_size": (np.float32(1000), "1000"),
        "bbox_xywh": {"x": 0.0, "y": 0.0, "w": 0.9, "h": 0.9},
        "roi_source": "parser_mask_bbox",
        "source_node_type": "body_part",
        "metadata_completeness_score": 1.0,
    }
    request = PatchSynthesisRequest(
        region=RegionRef("p1:torso", BBox(0.0, 0.0, 0.9, 0.9), "test"),
        scene_state=_graph(None),
        memory_summary={},
        transition_context={"graph_delta": GraphDelta()},
        retrieval_summary={},
        current_frame=[[[0.1, 0.1, 0.1] for _ in range(4)] for _ in range(4)],
        region_metadata=metadata,
    )

    metadata_features = extract_region_metadata_conditioning(request)["feature_values"]

    assert metadata_features["has_parser_mask"] == 1.0
    assert metadata_features["mask_area_ratio"] == 0.05  # 50k / 1M, not saturated by a magic pixel-count divisor


def test_patch_contract_reports_low_completeness_for_missing_metadata_and_preserves_enriched_metadata() -> None:
    graph = _graph(None)
    region = RegionRef("p1:face", BBox(0.1, 0.1, 0.2, 0.2), "test")
    out = PatchSynthesisOutput(region=region, rgb_patch=[[[0.0, 0.0, 0.0]]], alpha_mask=[[1.0]], height=1, width=1, channels=3, confidence=0.9, execution_trace={"selected_render_strategy": "LEARNED_TEST"})
    request = PatchSynthesisRequest(region=region, scene_state=graph, memory_summary={}, transition_context={}, retrieval_summary={}, current_frame=[[[0.0, 0.0, 0.0]]])

    contract = patch_io_to_contract(request, out)
    parity = build_parity_result(contract=contract, required_fields=["roi_before", "roi_after", "region_metadata", "selected_render_strategy", "transition_context"], stage="patch", request=request, output=out)

    assert contract["region_metadata"]["missing_fields"] == ["region_metadata"]
    assert "region_metadata_low_completeness" in parity["warnings"]

    enriched_metadata = {"region_id": "p1:face", "roi_source": "parser_mask_bbox", "source_node_type": "body_part", "metadata_completeness_score": 0.9, "missing_fields": []}
    enriched = PatchSynthesisRequest(region=region, scene_state=graph, memory_summary={}, transition_context={}, retrieval_summary={}, current_frame=request.current_frame, region_metadata=enriched_metadata)
    enriched_contract = patch_io_to_contract(enriched, out)

    assert enriched_contract["region_metadata"] == enriched_metadata


def _write_ppm(path: Path, w: int = 16, h: int = 16) -> None:
    pixels = "\n".join("40 80 120" for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n")


def test_runtime_patch_request_and_debug_include_region_metadata(tmp_path: Path) -> None:
    class FakeDynamicsBackend:
        def predict_transition(self, request):
            delta = GraphDelta(
                affected_entities=[request.graph_state.persons[0].person_id],
                affected_regions=["face"],
                expression_deltas={"smile_intensity": 0.4},
                region_transition_mode={"face": "expression_refine"},
                transition_phase="motion",
                semantic_reasons=["expression_delta"],
            )
            return DynamicsTransitionOutput(delta=delta, confidence=0.9, metadata={})

    class CapturingPatchBackend:
        def __init__(self) -> None:
            self.requests = []

        def synthesize_patch(self, request):
            self.requests.append(request)
            return PatchSynthesisOutput(
                region=request.region,
                rgb_patch=[[[0.2, 0.2, 0.25] for _ in range(2)] for _ in range(2)],
                alpha_mask=[[0.5, 0.5], [0.5, 0.5]],
                height=2,
                width=2,
                channels=3,
                confidence=0.8,
                execution_trace={
                    "renderer_path": "learned_primary",
                    "selected_render_strategy": "LEARNED_EXPRESSION_REFINE_PRIMARY",
                    "synthesis_mode": "learned_expression_micro_edit",
                },
                metadata={"renderer_path": "learned_primary"},
            )

    class FakeTemporalBackend:
        def refine_temporal(self, request):
            return TemporalRefinementOutput(
                refined_frame=request.current_composed_frame,
                region_consistency_scores={r.region_id: 0.8 for r in request.changed_regions},
                metadata={"temporal_path": "learned_primary"},
            )

    patch_backend = CapturingPatchBackend()
    bundle = SimpleNamespace(
        graph_encoder=BaselineGraphEncoder(),
        identity_encoder=BaselineIdentityAppearanceEncoder(),
        text_encoder=BaselineTextEncoderAdapter(),
        dynamics_backend=FakeDynamicsBackend(),
        patch_backend=patch_backend,
        temporal_backend=FakeTemporalBackend(),
        backend_names={},
    )
    img = tmp_path / "ref.ppm"
    _write_ppm(img)

    artifacts = GennadyEngine(backend_bundle=bundle).run([str(img)], "Улыбается.", quality_profile="debug")

    assert patch_backend.requests
    request = patch_backend.requests[0]
    assert request.region_metadata
    assert request.region_metadata["region_id"] == request.region.region_id
    patch_debug = artifacts.debug["step_execution"][0]["patch"][0]
    assert "region_metadata_completeness_score" in patch_debug
    assert "region_metadata_evidence_strength_score" in patch_debug
    assert "region_metadata_roi_source" in patch_debug
    assert patch_debug["parity"]["missing_fields"] == []
