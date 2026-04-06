from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from perception.backend import BackendInferenceEngine, image_ref_to_features
from perception.detector import BackendConfig, PersonDetection


@dataclass(slots=True)
class GarmentPrediction:
    garment_type: str
    state: str
    confidence: float
    source: str


@dataclass(slots=True)
class ParsingPrediction:
    mask_ref: str | None
    mask_confidence: float
    source: str
    garments: list[GarmentPrediction] = field(default_factory=list)
    occlusion_hints: list[str] = field(default_factory=list)


class HumanParser(Protocol):
    def parse(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        ...


class SegFormerHumanParserAdapter:
    source_name = "parser:segformer"

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(checkpoint="checkpoints/segformer.torch")
        self.engine = BackendInferenceEngine(self.source_name, self.config.backend, self.config.checkpoint)

    def parse(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        result: dict[str, ParsingPrediction] = {}
        feats = image_ref_to_features(image_ref)
        if self.config.backend in {"torch", "onnx"}:
            feats = self.engine.infer(feats)
        for person in persons:
            coat_conf = min(0.95, max(0.1, feats[0]))
            shirt_conf = min(0.95, max(0.1, feats[1]))
            result[person.detection_id] = ParsingPrediction(
                mask_ref=f"mask::{image_ref}::{person.detection_id}",
                mask_confidence=min(0.99, max(0.2, feats[2])),
                source=f"{self.source_name}:{self.config.backend}",
                garments=[
                    GarmentPrediction("coat", "worn" if coat_conf > 0.5 else "removed", coat_conf, self.source_name),
                    GarmentPrediction("shirt", "visible" if shirt_conf > 0.4 else "covered", shirt_conf, self.source_name),
                ],
                occlusion_hints=["torso_occluded" if feats[3] > 0.6 else "clear_torso"],
            )
        return result
