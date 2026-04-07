from core.schema import BBox
from perception.detector import DetectorOutput, PersonDetection
from perception.mask_store import DEFAULT_MASK_STORE
from perception.parser import (
    FacerFaceParserAdapter,
    FashnHumanParserAdapter,
    ParserBackendConfig,
    ParserFusionEngine,
    ParserStackConfig,
    SCHPATRParserAdapter,
    SCHPPascalPartParserAdapter,
    SegFormerHumanParserAdapter,
)
from perception.pipeline import PerceptionPipeline
from representation.graph_builder import SceneGraphBuilder


class _SimpleDetector:
    def detect(self, frame) -> DetectorOutput:
        return DetectorOutput(
            persons=[PersonDetection(detection_id="p1", bbox=BBox(0.1, 0.1, 0.7, 0.8), confidence=0.9, source="detector:test")],
            frame_size=(64, 64),
        )


def _solid(h: int, w: int, rgb=(0.5, 0.5, 0.5)):
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def test_fashn_adapter_emits_expected_semantic_masks() -> None:
    label_map = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 16],
    ]
    adapter = FashnHumanParserAdapter(ParserBackendConfig(backend="fashn"), infer_fn=lambda _: {"label_map": label_map})
    out = adapter.parse_patch([[[0, 0, 0] for _ in range(4)] for _ in range(4)])
    assert {"torso", "arms", "legs", "top", "dress", "pants"}.issubset(out.masks.keys())


def test_schp_pascal_adapter_maps_structural_parts_without_fake_sides() -> None:
    adapter = SCHPPascalPartParserAdapter(
        ParserBackendConfig(backend="schp_pascal"),
        infer_fn=lambda _: {
            "Head": [[1, 1], [1, 1]],
            "Upper Arms": [[1, 1], [1, 1]],
            "Lower Legs": [[1, 1], [1, 1]],
        },
    )
    out = adapter.parse_patch([[[0, 0, 0] for _ in range(2)] for _ in range(2)])
    assert "head" in out.masks and "upper_arm" in out.masks and "lower_leg" in out.masks
    assert "left_upper_arm" not in out.masks and "right_upper_arm" not in out.masks


def test_schp_atr_adapter_maps_only_supported_garment_body_labels() -> None:
    adapter = SCHPATRParserAdapter(
        ParserBackendConfig(backend="schp_atr"),
        infer_fn=lambda _: {
            "Upper-clothes": [[1, 1], [1, 1]],
            "Pants": [[1, 1], [1, 1]],
            "Tree": [[1, 1], [1, 1]],
        },
    )
    out = adapter.parse_patch([[[0, 0, 0] for _ in range(2)] for _ in range(2)])
    assert "upper_clothes" in out.masks and "pants" in out.masks
    assert "tree" not in out.masks


def test_facer_adapter_returns_real_face_regions() -> None:
    adapter = FacerFaceParserAdapter(
        ParserBackendConfig(backend="facer", variant="farl/lapa/448"),
        infer_fn=lambda _: {
            "Skin": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "Eyes": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "Mouth": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "Upper-lip": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "Lower-lip": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        },
    )
    out = adapter.parse_patch([[[0, 0, 0] for _ in range(3)] for _ in range(3)])
    assert {"face_skin", "eyes", "mouth", "upper_lip", "lower_lip"}.issubset(out.masks.keys())


def test_fusion_priority_respects_facer_and_pascal_and_fashn() -> None:
    fusion = ParserFusionEngine()
    person = PersonDetection("p1", BBox(0.1, 0.1, 0.7, 0.8), 0.9, "det")
    fashn = FashnHumanParserAdapter(ParserBackendConfig(backend="fashn"), infer_fn=lambda _: {"label_map": [[1, 3], [16, 12]]}).parse_patch(None)
    pascal = SCHPPascalPartParserAdapter(
        ParserBackendConfig(backend="schp_pascal"), infer_fn=lambda _: {"Torso": [[1, 1], [1, 1]]}
    ).parse_patch(None)
    atr = SCHPATRParserAdapter(
        ParserBackendConfig(backend="schp_atr"), infer_fn=lambda _: {"Upper-clothes": [[1, 1], [1, 1]]}
    ).parse_patch(None)
    facer = FacerFaceParserAdapter(
        ParserBackendConfig(backend="facer"), infer_fn=lambda _: {"Skin": [[1, 1], [1, 1]]}
    ).parse_patch(None)

    pred = fusion.fuse(person, fashn, pascal, atr, facer)
    assert pred.enriched.face_region_masks.get("face_skin") is not None
    assert any(bp.part_type == "torso" and bp.source == "parser:schp_pascal" for bp in pred.body_parts)
    assert any(g.garment_type == "top" and g.source == "parser:fashn" for g in pred.garments)


def test_pipeline_and_graph_builder_accept_enriched_contract() -> None:
    person = PersonDetection(detection_id="p1", bbox=BBox(0.1, 0.1, 0.7, 0.8), confidence=0.9, source="detector:test")
    fused = ParserFusionEngine().fuse(
        person,
        FashnHumanParserAdapter(ParserBackendConfig(backend="fashn"), infer_fn=lambda _: {"label_map": [[1, 3], [16, 12]]}).parse_patch(None),
        SCHPPascalPartParserAdapter(ParserBackendConfig(backend="schp_pascal"), infer_fn=lambda _: {"Torso": [[1, 1], [1, 1]]}).parse_patch(None),
        SCHPATRParserAdapter(ParserBackendConfig(backend="schp_atr"), infer_fn=lambda _: {"Upper-clothes": [[1, 1], [1, 1]]}).parse_patch(None),
        FacerFaceParserAdapter(ParserBackendConfig(backend="facer"), infer_fn=lambda _: {"Skin": [[1, 1], [1, 1]], "Mouth": [[1, 1], [1, 1]]}).parse_patch(None),
    )
    assert fused.mask_ref is not None and DEFAULT_MASK_STORE.get(fused.mask_ref) is not None

    pipe = PerceptionPipeline(detector=_SimpleDetector())
    out = pipe.analyze(_solid(32, 32))
    out.persons[0].mask_ref = fused.mask_ref
    out.persons[0].garments = [
        {"type": g.garment_type, "state": g.state, "confidence": g.confidence, "source": g.source, "mask_ref": g.mask_ref}
        for g in fused.garments
    ]
    out.persons[0].body_parts = [{"part_type": b.part_type, "mask_ref": b.mask_ref, "confidence": b.confidence, "source": b.source} for b in fused.body_parts]
    out.persons[0].provenance_by_region = fused.enriched.provenance_by_region
    graph = SceneGraphBuilder().build(out, frame_index=2)
    assert graph.persons[0].garments
    assert any("provenance:" in alt for alt in graph.persons[0].garments[0].alternatives)
