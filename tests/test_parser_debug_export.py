from pathlib import Path

from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.detector import PersonDetection
from perception.parser import (
    FacerFaceParserAdapter,
    FashnHumanParserAdapter,
    ParserBackendConfig,
    ParserFusionEngine,
    SCHPATRParserAdapter,
    SCHPPascalPartParserAdapter,
)
from perception.parser_debug import export_parser_debug_artifacts
from perception.pipeline import PerceptionOutput, PersonFacts


def _solid(h: int, w: int):
    return [[[0.4, 0.4, 0.4] for _ in range(w)] for _ in range(h)]


def test_export_parser_debug_artifacts_writes_outputs(tmp_path: Path) -> None:
    person = PersonDetection("p1", BBox(0.1, 0.1, 0.7, 0.8), 0.9, "det")
    pred = ParserFusionEngine().fuse(
        person,
        FashnHumanParserAdapter(ParserBackendConfig(backend="fashn"), infer_fn=lambda _: {"label_map": [[1, 3], [16, 12]]}).parse_patch(None),
        SCHPPascalPartParserAdapter(ParserBackendConfig(backend="schp_pascal"), infer_fn=lambda _: {"Torso": [[1, 1], [1, 1]]}).parse_patch(None),
        SCHPATRParserAdapter(ParserBackendConfig(backend="schp_atr"), infer_fn=lambda _: {"Upper-clothes": [[1, 1], [1, 1]]}).parse_patch(None),
        FacerFaceParserAdapter(ParserBackendConfig(backend="facer"), infer_fn=lambda _: {"Skin": [[1, 1], [1, 1]]}).parse_patch(None),
    )

    facts = PersonFacts(
        bbox=person.bbox,
        bbox_confidence=person.confidence,
        bbox_source=person.source,
        mask_ref=pred.mask_ref,
        mask_confidence=pred.mask_confidence,
        mask_source=pred.source,
        pose=PoseState(),
        pose_confidence=0.0,
        pose_source="test",
        expression=ExpressionState(),
        expression_confidence=0.0,
        expression_source="test",
        orientation=OrientationState(),
        orientation_confidence=0.0,
        orientation_source="test",
        garment_masks=pred.enriched.garment_masks,
        body_part_masks=pred.enriched.body_part_masks,
        face_region_masks=pred.enriched.face_region_masks,
        accessory_masks=pred.enriched.accessory_masks,
        provenance_by_region=pred.enriched.provenance_by_region,
    )

    out = PerceptionOutput(persons=[facts])
    summary = export_parser_debug_artifacts(frame_rgb=[[[120, 120, 120], [120, 120, 120]], [[120, 120, 120], [120, 120, 120]]], output=out, out_dir=tmp_path)

    assert (tmp_path / "frame.png").exists()
    assert (tmp_path / "person_0" / "fused_masks_overlay.png").exists()
    assert (tmp_path / "fused_summary.json").exists()
    assert summary["persons"]
