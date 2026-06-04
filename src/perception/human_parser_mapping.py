from __future__ import annotations

from dataclasses import dataclass

from core.body_ontology import is_canonical_body_region


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
    "hair": "hair",
    "head": "head",
    "scalp": "scalp",
    "neck": "neck",
    "torso": "torso",
    "upper_body": "upper_body",
    "lower_body": "lower_body",
    "upper_torso": "upper_torso",
    "lower_torso": "lower_torso",
    "chest": "chest",
    "left_chest": "left_chest",
    "right_chest": "right_chest",
    "abdomen": "abdomen",
    "waist": "waist",
    "back": "back",
    "left_arm": "left_arm",
    "right_arm": "right_arm",
    "left_upper_arm": "left_upper_arm",
    "right_upper_arm": "right_upper_arm",
    "left_lower_arm": "left_forearm",
    "right_lower_arm": "right_forearm",
    "left_forearm": "left_forearm",
    "right_forearm": "right_forearm",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "hands": "hands",
    "arms": "arms",
    "arm": "arms",
    "pelvis": "pelvis",
    "hips": "hips",
    "left_leg": "left_leg",
    "right_leg": "right_leg",
    "left_upper_leg": "left_thigh",
    "right_upper_leg": "right_thigh",
    "left_lower_leg": "left_calf",
    "right_lower_leg": "right_calf",
    "left_thigh": "left_thigh",
    "right_thigh": "right_thigh",
    "left_calf": "left_calf",
    "right_calf": "right_calf",
    "legs": "legs",
    "left_foot": "left_foot",
    "right_foot": "right_foot",
    "feet": "feet",
    "foot": "feet",
    "external_genital_region": "external_genital_region",
    "male_external_genital_region": "male_external_genital_region",
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
        category = "face_hair" if canonical in {"face", "hair", "scalp"} else "body_part"
        return HumanParserClassMapping(original, canonical, category)
    if key in _GARMENTS:
        canonical, garment_type = _GARMENTS[key]
        return HumanParserClassMapping(original, canonical, "garment" if canonical != "accessory" else "accessory", garment_type)
    if is_canonical_body_region(key):
        return HumanParserClassMapping(original, key, "body_part")
    return HumanParserClassMapping(original, key or "unknown", "unknown")
