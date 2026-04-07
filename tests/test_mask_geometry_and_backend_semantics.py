from pathlib import Path

from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.detector import PersonDetection
from perception.mask_projection import project_mask_to_frame
from perception.mask_store import DEFAULT_MASK_STORE
from perception.parser import (
    FashnHumanParserAdapter,
    ParserBackendConfig,
    ParserFusionEngine,
    SCHPATRParserAdapter,
    SCHPPascalPartParserAdapter,
)
from perception.parser_debug import export_parser_debug_artifacts
from perception.pipeline import PerceptionOutput, PersonFacts


def test_fusion_stores_roi_bbox_and_frame_size_in_mask_store() -> None:
    person = PersonDetection("p1", BBox(0.25, 0.25, 0.5, 0.5), 0.9, "det")
    pred = ParserFusionEngine().fuse(
        person,
        FashnHumanParserAdapter(ParserBackendConfig(backend="fashn"), infer_fn=lambda _: {"label_map": [[1, 3], [16, 12]]}).parse_patch(None),
        SCHPPascalPartParserAdapter(ParserBackendConfig(backend="schp_pascal"), infer_fn=lambda _: {"Torso": [[1, 1], [1, 1]]}).parse_patch(None),
        SCHPATRParserAdapter(ParserBackendConfig(backend="schp_atr"), infer_fn=lambda _: {"Upper-clothes": [[1, 1], [1, 1]]}).parse_patch(None),
        SCHPATRParserAdapter(ParserBackendConfig(backend="schp_atr"), infer_fn=lambda _: {}).parse_patch(None),
        frame_size=(8, 8),
    )
    refs = [pred.mask_ref] + list(pred.enriched.body_part_masks.values()) + list(pred.enriched.garment_masks.values())
    refs = [r for r in refs if r]
    assert refs
    stored = DEFAULT_MASK_STORE.get(refs[0])
    assert stored is not None
    assert stored.roi_bbox == (0.25, 0.25, 0.5, 0.5)
    assert stored.frame_size == (8, 8)
    assert any(t.startswith("person:") for t in stored.tags)


def test_project_mask_to_frame_expands_roi_local_geometry() -> None:
    ref = DEFAULT_MASK_STORE.put(
        payload=[[1, 1], [1, 1]],
        confidence=0.9,
        source="parser:test",
        prefix="test",
        roi_bbox=(0.5, 0.5, 0.5, 0.5),
        frame_size=(4, 4),
    )
    stored = DEFAULT_MASK_STORE.get(ref)
    assert stored is not None
    full, geom = project_mask_to_frame(stored)
    assert geom == "projected"
    assert len(full) == 4 and len(full[0]) == 4
    assert full[3][3] == 1 and full[0][0] == 0


def test_debug_export_handles_roi_local_masks(tmp_path: Path) -> None:
    ref = DEFAULT_MASK_STORE.put(
        payload=[[1, 1], [1, 1]],
        confidence=0.9,
        source="parser:fashn",
        prefix="garment",
        roi_bbox=(0.5, 0.5, 0.5, 0.5),
        frame_size=(4, 4),
    )
    facts = PersonFacts(
        bbox=BBox(0.0, 0.0, 1.0, 1.0),
        bbox_confidence=1.0,
        bbox_source="det",
        mask_ref=None,
        mask_confidence=0.0,
        mask_source="test",
        pose=PoseState(),
        pose_confidence=0.0,
        pose_source="test",
        expression=ExpressionState(),
        expression_confidence=0.0,
        expression_source="test",
        orientation=OrientationState(),
        orientation_confidence=0.0,
        orientation_source="test",
        garment_masks={"top": ref},
        provenance_by_region={"garment:top": "parser:fashn"},
    )
    summary = export_parser_debug_artifacts([[[10, 10, 10] for _ in range(4)] for _ in range(4)], PerceptionOutput(persons=[facts]), tmp_path)
    assert (tmp_path / "person_0" / "fused_masks_overlay.png").exists()
    assert summary["persons"][0].get("mask_geometry", {}).get("top") in {"projected", "full"}


def test_schp_backend_names_are_honest_without_specific_runtime() -> None:
    adapter = SCHPPascalPartParserAdapter(ParserBackendConfig(backend="schp_pascal", variant="some-model"))
    try:
        adapter.parse_patch([[[0, 0, 0]]])
        assert False, "must fail without schp-specific runtime"
    except RuntimeError as exc:
        assert "generic_hf_structural" in str(exc)
