from __future__ import annotations

from core.reference_families import reference_family_for_region, reference_kind_for_region


def test_canonical_reference_family_mapping() -> None:
    for region in ("face", "head", "hair"):
        assert reference_family_for_region(region) == "identity"
        assert reference_kind_for_region(region) == "identity_reference"

    for region in ("neck", "left_hand", "right_hand", "hands"):
        assert reference_family_for_region(region) == "skin"
        assert reference_kind_for_region(region) == "skin_reference"

    for region in ("torso", "pelvis", "left_arm", "right_arm", "left_leg", "right_leg", "legs"):
        assert reference_family_for_region(region) == "body_shape"
        assert reference_kind_for_region(region) == "body_shape_reference"

    for region in ("outer_garment", "inner_garment", "garments", "sleeves"):
        assert reference_family_for_region(region) == "garment"
        assert reference_kind_for_region(region) == "garment_reference"

    assert reference_family_for_region("accessories") == "accessory"
    assert reference_kind_for_region("accessories") == "accessory_reference"
    assert reference_family_for_region("unknown_region") == "unknown"
    assert reference_kind_for_region("unknown_region") == "none"
