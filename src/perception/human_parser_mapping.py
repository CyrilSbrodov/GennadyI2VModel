from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HumanParserClassMapping:
    original_class_name: str
    canonical_region_type: str
    category: str
    garment_type: str | None = None
    preserve_unknown: bool = True


def _norm(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


_BODY = {
    "face": "face",
    "skin": "torso",
    "hair": "hair",
    "head": "head",
    "neck": "neck",
    "torso": "torso",
    "chest": "torso",
    "arms": "arms",
    "arm": "arms",
    "left_arm": "left_arm",
    "right_arm": "right_arm",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "hands": "hands",
    "legs": "legs",
    "left_leg": "left_leg",
    "right_leg": "right_leg",
    "feet": "feet",
    "foot": "feet",
}

_GARMENTS = {
    "dress": ("dress", "dress"),
    "top": ("upper_garment", "top"),
    "upper_clothes": ("upper_garment", "upper_clothes"),
    "upper_clothing": ("upper_garment", "upper_clothes"),
    "shirt": ("upper_garment", "shirt"),
    "blouse": ("upper_garment", "blouse"),
    "coat": ("outer_garment", "coat"),
    "jacket": ("outer_garment", "jacket"),
    "blazer": ("outer_garment", "blazer"),
    "hoodie": ("outer_garment", "hoodie"),
    "sweater": ("outer_garment", "sweater"),
    "pants": ("lower_garment", "pants"),
    "trousers": ("lower_garment", "pants"),
    "jeans": ("lower_garment", "jeans"),
    "skirt": ("lower_garment", "skirt"),
    "lower_clothes": ("lower_garment", "lower_clothes"),
    "shorts": ("lower_garment", "shorts"),
    "shoes": ("accessory", "shoes"),
    "shoe": ("accessory", "shoes"),
    "belt": ("accessory", "belt"),
    "bag": ("accessory", "bag"),
    "hat": ("accessory", "hat"),
    "scarf": ("accessory", "scarf"),
    "glasses": ("accessory", "glasses"),
    "jewelry": ("accessory", "jewelry"),
}


def map_human_parser_class(class_name: str) -> HumanParserClassMapping:
    key = _norm(class_name)
    original = str(class_name)
    if key in {"background", "bg"}:
        return HumanParserClassMapping(original, "background", "background")
    if key in _BODY:
        canonical = _BODY[key]
        category = "face_hair" if canonical in {"face", "hair"} else "body_part"
        return HumanParserClassMapping(original, canonical, category)
    if key in _GARMENTS:
        canonical, garment_type = _GARMENTS[key]
        return HumanParserClassMapping(original, canonical, "garment" if canonical != "accessory" else "accessory", garment_type)
    return HumanParserClassMapping(original, key or "unknown", "unknown")
