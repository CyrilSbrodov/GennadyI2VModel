from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.schema import GraphDelta
from learned.interfaces import DynamicsTransitionOutput, PatchSynthesisOutput, TemporalRefinementOutput
from representation.learned_bridge import BaselineGraphEncoder, BaselineIdentityAppearanceEncoder
from runtime.orchestrator import GennadyEngine
from text.learned_bridge import BaselineTextEncoderAdapter
from training.datasets import RendererDataset


REQUIRED_V2_FIELDS = {
    "record_id",
    "frame_index",
    "step_index",
    "region_id",
    "canonical_region",
    "entity_id",
    "roi_before",
    "roi_after",
    "alpha_mask",
    "region_metadata",
    "transition_context_summary",
    "selected_render_strategy",
    "synthesis_mode",
    "execution_trace_summary",
    "metadata_completeness_score",
    "evidence_strength_score",
    "roi_source",
    "source_node_type",
    "mask_kind",
    "mask_ref_present",
}


def _write_ppm(path: Path, w: int = 16, h: int = 16, rgb: tuple[int, int, int] = (80, 90, 120)) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n", encoding="utf-8")


class _DynamicsBackend:
    def __init__(self, region: str) -> None:
        self.region = region

    def predict_transition(self, request):
        entity_id = request.graph_state.persons[0].person_id
        delta = GraphDelta(
            affected_entities=[entity_id],
            affected_regions=[self.region],
            expression_deltas={"smile_intensity": 0.4} if self.region == "face" else {},
            garment_deltas={"outer_garment": "opening"} if self.region == "outer_garment" else {},
            region_transition_mode={self.region: "expression_refine" if self.region == "face" else "local_update"},
            transition_phase="motion",
            semantic_reasons=["expression_delta" if self.region == "face" else "pose_transition"],
        )
        return DynamicsTransitionOutput(
            delta=delta,
            confidence=0.9,
            metadata={
                "semantic_families": ["face_expression" if self.region == "face" else "sleeve_arm_transition"],
                "target_profile": {"primary_regions": [self.region]},
            },
        )


class _PatchBackend:
    def synthesize_patch(self, request):
        return PatchSynthesisOutput(
            region=request.region,
            rgb_patch=[[[0.2, 0.25, 0.3] for _ in range(2)] for _ in range(2)],
            alpha_mask=[[0.5, 0.5], [0.5, 0.5]],
            height=2,
            width=2,
            channels=3,
            confidence=0.8,
            execution_trace={
                "renderer_path": "learned_primary",
                "selected_render_strategy": "LEARNED_EXPRESSION_REFINE_PRIMARY",
                "synthesis_mode": "learned_expression_micro_edit",
                "metadata_used": bool(request.region_metadata),
            },
            metadata={"renderer_path": "learned_primary"},
        )


class _TemporalBackend:
    def refine_temporal(self, request):
        return TemporalRefinementOutput(
            refined_frame=request.current_composed_frame,
            region_consistency_scores={r.region_id: 0.8 for r in request.changed_regions},
            metadata={"temporal_path": "learned_primary"},
        )


def _engine_for_region(region: str) -> GennadyEngine:
    bundle = SimpleNamespace(
        graph_encoder=BaselineGraphEncoder(),
        identity_encoder=BaselineIdentityAppearanceEncoder(),
        text_encoder=BaselineTextEncoderAdapter(),
        dynamics_backend=_DynamicsBackend(region),
        patch_backend=_PatchBackend(),
        temporal_backend=_TemporalBackend(),
        backend_names={"patch_strict_mode": False},
    )
    return GennadyEngine(backend_bundle=bundle)


def test_runtime_run_exports_renderer_manifest_v2_and_dataset_roundtrips_region_metadata(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    manifest_path = tmp_path / "renderer_manifest.json"
    _write_ppm(img)

    artifacts = _engine_for_region("face").run(
        [str(img)],
        "Улыбается.",
        quality_profile="debug",
        export_renderer_manifest_path=str(manifest_path),
    )

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "renderer_patch_manifest_v2"
    assert manifest["record_count"] == len(manifest["records"])
    assert artifacts.debug["renderer_manifest_export_enabled"] is True
    assert artifacts.debug["renderer_manifest_export_path"] == str(manifest_path)
    assert artifacts.debug["renderer_manifest_export_record_count"] == manifest["record_count"]
    assert artifacts.debug["renderer_manifest_export_contract_version"] == "renderer_patch_manifest_v2"
    assert artifacts.debug["renderer_manifest_export_error"] is None

    assert manifest["record_count"] > 0
    first = manifest["records"][0]
    assert REQUIRED_V2_FIELDS.issubset(first.keys())
    assert isinstance(first["region_metadata"], dict)
    assert first["selected_render_strategy"] == "LEARNED_EXPRESSION_REFINE_PRIMARY"
    assert first["synthesis_mode"] == "learned_expression_micro_edit"

    dataset = RendererDataset.from_renderer_manifest(str(manifest_path), strict=True)
    request = RendererDataset.sample_to_patch_request(dataset.samples[0])
    assert request.region_metadata == dataset.samples[0]["region_metadata"]
    assert request.region_metadata["region_id"] == first["region_id"]


def test_runtime_export_fallback_person_bbox_record_remains_valid_and_diagnosed(tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "ref.ppm"
    manifest_path = tmp_path / "fallback_manifest.json"
    _write_ppm(img, rgb=(40, 80, 120))

    def _fallback_metadata(**_kwargs):
        return {
            "region_id": "person_1:left_arm",
            "entity_id": "person_1",
            "canonical_region": "left_arm",
            "bbox_xywh": [0.1, 0.1, 0.8, 0.8],
            "roi_source": "person_bbox_fallback",
            "source_node_type": "fallback",
            "mask_kind": "",
            "metadata_completeness_score": 0.45,
            "evidence_strength_score": 0.08,
            "missing_fields": ["mask_ref", "mask_kind"],
        }

    monkeypatch.setattr("runtime.orchestrator.build_region_metadata", _fallback_metadata)

    artifacts = _engine_for_region("left_arm").run(
        [str(img)],
        "Поднимает руку.",
        quality_profile="debug",
        export_renderer_manifest_path=str(manifest_path),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "renderer_patch_manifest_v2"
    assert manifest["record_count"] == len(manifest["records"])
    assert artifacts.debug["renderer_manifest_export_error"] is None

    fallback_records = [r for r in manifest["records"] if r["roi_source"] == "person_bbox_fallback"]
    assert fallback_records
    record = fallback_records[0]
    assert record["mask_ref_present"] is False
    assert record["metadata_completeness_score"] <= 0.45
    assert record["evidence_strength_score"] <= 0.1

    dataset = RendererDataset.from_renderer_manifest(str(manifest_path), strict=True)
    assert dataset.diagnostics["fallback_person_bbox_record_count"] >= 1
