from core.schema import BBox
from perception.detector import DetectorOutput, PersonDetection
from perception.mask_store import DEFAULT_MASK_STORE
from perception.parser import (
    BodyPartMaskPrediction,
    FaceRegionPrediction,
    GarmentPrediction,
    ParsingPrediction,
    ParserBackendConfig,
    ParserStackConfig,
    SegFormerHumanParserAdapter,
    _canonical_garment_label,
)
from perception.pipeline import PerceptionPipeline
from representation.graph_builder import SceneGraphBuilder


class _SimpleDetector:
    def detect(self, frame) -> DetectorOutput:
        return DetectorOutput(
            persons=[PersonDetection(detection_id="p1", bbox=BBox(0.1, 0.1, 0.6, 0.8), confidence=0.9, source="detector:test")],
            frame_size=(64, 64),
        )


class _RealSuccessParser:
    def __init__(self) -> None:
        self.config = ParserStackConfig(person_segmentation=ParserBackendConfig(backend="custom", model_id="x"))

    def is_builtin_backend(self) -> bool:
        return False

    def parse(self, frame, persons):
        ref = DEFAULT_MASK_STORE.put([[1, 1], [1, 0]], 0.9, "parser:test", "person")
        return {
            persons[0].detection_id: ParsingPrediction(
                mask_ref=ref,
                mask_confidence=0.9,
                source="parser:test:real",
                garments=[GarmentPrediction("jacket", "visible", 0.8, "parser:test:real")],
                occlusion_hints=["torso_visible"],
                body_parts=[BodyPartMaskPrediction("torso", ref, 0.8, "visible", "parser:test:real")],
                face_regions=[FaceRegionPrediction("face", ref, 0.7, "parser:test:real")],
            )
        }


class _RealFailParser:
    def __init__(self) -> None:
        self.config = ParserStackConfig(person_segmentation=ParserBackendConfig(backend="custom", model_id="x"))

    def is_builtin_backend(self) -> bool:
        return False

    def parse(self, frame, persons):
        raise RuntimeError("real parser failed")


def _solid(h: int, w: int, rgb=(0.5, 0.5, 0.5)):
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def test_parser_module_fallback_semantics() -> None:
    builtin_pipe = PerceptionPipeline(detector=_SimpleDetector())
    out_builtin = builtin_pipe.analyze(_solid(32, 32))
    assert out_builtin.module_fallbacks["parser"] == "builtin"

    native_pipe = PerceptionPipeline(detector=_SimpleDetector(), parser=_RealSuccessParser())
    out_native = native_pipe.analyze(_solid(32, 32))
    assert out_native.module_fallbacks["parser"] == "native"

    fallback_pipe = PerceptionPipeline(detector=_SimpleDetector(), parser=_RealFailParser())
    out_fallback = fallback_pipe.analyze(_solid(32, 32))
    assert out_fallback.module_fallbacks["parser"] == "fallback"


def test_parser_does_not_emit_unknown_scene_labels_as_garments() -> None:
    assert _canonical_garment_label("wall") is None
    assert _canonical_garment_label("tree") is None
    assert _canonical_garment_label("jacket") == "jacket"


def test_real_parsing_payload_and_graph_builder_integration() -> None:
    pipe = PerceptionPipeline(detector=_SimpleDetector(), parser=_RealSuccessParser())
    out = pipe.analyze(_solid(32, 32))
    person = out.persons[0]
    assert person.mask_ref is not None
    assert DEFAULT_MASK_STORE.get(person.mask_ref) is not None
    assert person.garments
    assert person.occlusion_hints == ["torso_visible"]

    graph = SceneGraphBuilder().build(out, frame_index=1)
    assert any(bp.source.startswith("parser:") for bp in graph.persons[0].body_parts)
