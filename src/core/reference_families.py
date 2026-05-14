from __future__ import annotations

CORE_IDENTITY_REGIONS = {"face", "head", "hair"}
SKIN_REFERENCE_REGIONS = {"neck", "left_hand", "right_hand", "hands"}
BODY_SHAPE_REFERENCE_REGIONS = {
    "torso",
    "upper_body",
    "lower_body",
    "pelvis",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "legs",
}
GARMENT_REFERENCE_REGIONS = {
    "upper_garment",
    "lower_garment",
    "outer_garment",
    "inner_garment",
    "garments",
    "sleeves",
}
ACCESSORY_REFERENCE_REGIONS = {"accessories"}

REFERENCE_KIND_BY_FAMILY = {
    "identity": "identity_reference",
    "skin": "skin_reference",
    "body_shape": "body_shape_reference",
    "garment": "garment_reference",
    "accessory": "accessory_reference",
}


def reference_family_for_region(region_type: str) -> str:
    canonical = str(region_type or "").strip()
    if canonical in CORE_IDENTITY_REGIONS:
        return "identity"
    if canonical in SKIN_REFERENCE_REGIONS:
        return "skin"
    if canonical in BODY_SHAPE_REFERENCE_REGIONS:
        return "body_shape"
    if canonical in GARMENT_REFERENCE_REGIONS:
        return "garment"
    if canonical in ACCESSORY_REFERENCE_REGIONS:
        return "accessory"
    return "unknown"


def reference_kind_for_region(region_type: str) -> str:
    family = reference_family_for_region(region_type)
    return REFERENCE_KIND_BY_FAMILY.get(family, "none")
