from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from perception.detector import PersonDetection


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


class HumanParser(Protocol):
    def parse(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        ...


class SegFormerHumanParserAdapter:
    source_name = "parser:segformer"

    def parse(self, image_ref: str, persons: list[PersonDetection]) -> dict[str, ParsingPrediction]:
        result: dict[str, ParsingPrediction] = {}
        for person in persons:
            result[person.detection_id] = ParsingPrediction(
                mask_ref=f"mask::{image_ref}::{person.detection_id}",
                mask_confidence=0.87,
                source=self.source_name,
                garments=[
                    GarmentPrediction("coat", "worn", 0.88, self.source_name),
                    GarmentPrediction("shirt", "covered", 0.61, self.source_name),
                ],
            )
        return result
