from core.schema import BBox
from perception.detector import DetectorOutput, PersonDetection
from perception.pipeline import PerceptionPipeline


class FailingParser:
    def parse(self, image_ref: str, persons: list[PersonDetection]) -> dict:
        raise RuntimeError("parser offline")


class SimpleDetector:
    def detect(self, image_ref: str) -> DetectorOutput:
        return DetectorOutput(
            persons=[
                PersonDetection(
                    detection_id="person_1",
                    bbox=BBox(0.1, 0.2, 0.3, 0.4),
                    confidence=0.99,
                    source="detector:test",
                )
            ],
            frame_size=(640, 480),
        )


def test_pipeline_emits_real_predictions_with_sources_and_confidences() -> None:
    pipeline = PerceptionPipeline()

    output = pipeline.analyze("frame://1")

    assert output.persons
    person = output.persons[0]
    assert person.bbox_confidence > 0
    assert person.bbox_source.startswith("detector:")
    assert person.pose_confidence > 0
    assert person.pose_source.startswith("pose:")
    assert person.mask_ref is not None
    assert person.garments and "source" in person.garments[0]

    assert output.objects
    assert output.objects[0].source.startswith("objects:")


def test_pipeline_fallback_when_module_unavailable() -> None:
    pipeline = PerceptionPipeline(detector=SimpleDetector(), parser=FailingParser())

    output = pipeline.analyze("frame://1")

    assert output.persons
    person = output.persons[0]
    assert person.mask_ref is None
    assert person.mask_source == "fallback"
    assert person.garments == []
    assert any(w.startswith("parser_unavailable") for w in output.warnings)
