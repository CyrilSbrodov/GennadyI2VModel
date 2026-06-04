from __future__ import annotations

from core.body_ontology import (
    BODY_ONTOLOGY,
    CANONICAL_BODY_REGION_ORDER,
    BodyRegionApplicability,
    BodyRegionEvidenceStatus,
    BodyRegionVisibility,
    is_canonical_body_region,
)
from core.region_ids import is_canonical_region_id, is_known_canonical_region_id, make_region_id
from core.reference_families import reference_family_for_region, reference_kind_for_region
from perception.human_parser_mapping import map_human_parser_class
from representation.canonical_human_state import CanonicalHumanNormalizer, canonical_state_to_dict
from core.schema import BBox, ExpressionState, OrientationState, PoseState
from perception.pipeline import PersonFacts


REQUIRED_GENERIC = {
    "head", "scalp", "hair", "face", "forehead", "left_eye", "right_eye", "nose", "mouth", "lips",
    "left_ear", "right_ear", "jaw", "chin", "neck", "torso", "upper_torso", "lower_torso", "chest",
    "left_chest", "right_chest", "abdomen", "waist", "back", "upper_back", "lower_back", "left_shoulder",
    "right_shoulder", "left_shoulder_blade", "right_shoulder_blade", "left_arm", "right_arm", "left_upper_arm",
    "right_upper_arm", "left_elbow", "right_elbow", "left_forearm", "right_forearm", "left_wrist", "right_wrist",
    "left_hand", "right_hand", "left_palm", "right_palm", "left_fingers", "right_fingers", "pelvis", "hips",
    "left_hip", "right_hip", "buttocks", "left_buttock", "right_buttock", "groin_region", "left_leg",
    "right_leg", "left_thigh", "right_thigh", "left_knee", "right_knee", "left_calf", "right_calf",
    "left_ankle", "right_ankle", "left_foot", "right_foot", "left_toes", "right_toes",
}
REQUIRED_COMPOSITE = {"upper_body", "lower_body", "arms", "hands", "legs", "feet"}
REQUIRED_FEMALE_OPTIONAL = {"left_breast", "right_breast", "breast_region", "female_pelvic_region"}
REQUIRED_MALE_OPTIONAL = {"male_chest", "male_pelvic_region", "male_external_genital_region"}
REQUIRED_PRIVATE = {"external_genital_region"}


def _person(**kwargs) -> PersonFacts:
    values = dict(
        bbox=BBox(0.1, 0.1, 0.7, 0.8),
        bbox_confidence=0.9,
        bbox_source="detector:test",
        mask_ref=None,
        mask_confidence=0.0,
        mask_source="unknown",
        pose=PoseState(),
        pose_confidence=0.0,
        pose_source="unknown",
        expression=ExpressionState(),
        expression_confidence=0.0,
        expression_source="unknown",
        orientation=OrientationState(),
        orientation_confidence=0.0,
        orientation_source="unknown",
        person_id="person_1",
    )
    values.update(kwargs)
    return PersonFacts(**values)


def test_complete_adult_body_ontology_contains_required_regions_and_metadata() -> None:
    required = REQUIRED_GENERIC | REQUIRED_COMPOSITE | REQUIRED_FEMALE_OPTIONAL | REQUIRED_MALE_OPTIONAL | REQUIRED_PRIVATE
    assert required <= set(BODY_ONTOLOGY)
    assert required <= set(CANONICAL_BODY_REGION_ORDER)
    for region in required:
        meta = BODY_ONTOLOGY[region]
        assert meta.canonical_name == region
        assert meta.group.value
        assert meta.side.value
        assert meta.applicability_policy.value
        assert meta.motion_role
        assert meta.memory_family
        assert meta.parser_support_level
        if region.startswith("left_"):
            assert meta.side == "left"
        if region.startswith("right_"):
            assert meta.side == "right"


def test_applicability_visibility_and_evidence_status_are_distinct_concepts() -> None:
    assert BodyRegionApplicability.UNKNOWN_APPLICABILITY != BodyRegionApplicability.NOT_APPLICABLE
    assert BodyRegionApplicability.UNSUPPORTED_BY_CURRENT_PARSER != BodyRegionEvidenceStatus.MISSING
    assert BodyRegionVisibility.HIDDEN_BY_GARMENT != BodyRegionApplicability.NOT_APPLICABLE
    assert BodyRegionVisibility.OUT_OF_FRAME != BodyRegionApplicability.NOT_APPLICABLE

    hidden_applicable = {
        "applicability": BodyRegionApplicability.APPLICABLE.value,
        "visibility_state": BodyRegionVisibility.HIDDEN_BY_GARMENT.value,
        "observation_status": BodyRegionEvidenceStatus.UNKNOWN.value,
    }
    assert hidden_applicable["applicability"] == "applicable"
    assert hidden_applicable["visibility_state"] == "hidden_by_garment"
    assert hidden_applicable["observation_status"] == "unknown"


def test_parser_mapping_preserves_anatomy_and_garments_are_not_body() -> None:
    assert map_human_parser_class("face").canonical_region_type == "face"
    assert map_human_parser_class("hair").canonical_region_type == "hair"
    assert map_human_parser_class("left_lower_arm").canonical_region_type == "left_forearm"
    assert map_human_parser_class("right_upper_leg").canonical_region_type == "right_thigh"
    assert map_human_parser_class("chest").canonical_region_type == "chest"
    assert map_human_parser_class("upper_clothes").category == "garment"
    assert map_human_parser_class("upper_clothes").canonical_region_type == "upper_garment"


def test_canonical_human_normalizer_preserves_ontology_without_fake_observation() -> None:
    state = CanonicalHumanNormalizer().normalize(
        _person(face_regions=[{"region_type": "face", "mask_ref": "mask://face", "confidence": 0.91, "source": "face:test", "observation_status": "observed", "mask_evidence_type": "parser_mask"}]),
        person_id="person_1",
    )
    payload = canonical_state_to_dict(state)["regions"]
    assert set(REQUIRED_FEMALE_OPTIONAL | REQUIRED_MALE_OPTIONAL | REQUIRED_PRIVATE) <= set(payload)
    assert payload["face"]["observation_status"] == "observed"
    assert payload["face"]["mask_ref"] == "mask://face"
    assert payload["left_breast"]["observation_status"] == "unknown"
    assert payload["left_breast"]["applicability"] == "unknown_applicability"
    assert payload["male_external_genital_region"]["observation_status"] == "unknown"
    assert payload["male_external_genital_region"]["mask_ref"] is None
    assert payload["chest"]["canonical_name"] == "chest"
    assert payload["torso"]["canonical_name"] == "torso"


def test_composite_parser_labels_do_not_create_observed_left_right_children() -> None:
    state = CanonicalHumanNormalizer().normalize(
        _person(
            body_parts=[
                {"part_type": "arms", "mask_ref": "mask://arms", "confidence": 0.91, "source": "parser:test", "visibility": "visible", "observation_status": "observed", "mask_evidence_type": "parser_mask"},
                {"part_type": "legs", "mask_ref": "mask://legs", "confidence": 0.92, "source": "parser:test", "visibility": "visible", "observation_status": "observed", "mask_evidence_type": "parser_mask"},
                {"part_type": "hands", "mask_ref": "mask://hands", "confidence": 0.93, "source": "parser:test", "visibility": "visible", "observation_status": "observed", "mask_evidence_type": "parser_mask"},
                {"part_type": "feet", "mask_ref": "mask://feet", "confidence": 0.94, "source": "parser:test", "visibility": "visible", "observation_status": "observed", "mask_evidence_type": "parser_mask"},
            ]
        ),
        person_id="person_1",
    )
    payload = canonical_state_to_dict(state)["regions"]
    for composite in ("arms", "legs", "hands", "feet"):
        assert payload[composite]["observation_status"] == "observed"
        assert payload[composite]["mask_ref"] == f"mask://{composite}"
    for child in ("left_arm", "right_arm", "left_leg", "right_leg", "left_hand", "right_hand", "left_foot", "right_foot"):
        assert payload[child]["observation_status"] != "observed"
        assert payload[child]["mask_ref"] is None


def test_generic_skin_parser_label_does_not_mark_torso_observed() -> None:
    state = CanonicalHumanNormalizer().normalize(
        _person(body_parts=[{"part_type": "skin", "mask_ref": "mask://skin", "confidence": 0.9, "source": "parser:test", "visibility": "visible", "observation_status": "observed", "mask_evidence_type": "parser_mask"}]),
        person_id="person_1",
    )
    payload = canonical_state_to_dict(state)["regions"]
    assert payload["torso"]["observation_status"] != "observed"
    assert payload["torso"]["mask_ref"] is None


def test_region_routing_addresses_new_canonical_regions_without_validating_unknown_names() -> None:
    assert make_region_id("person_1", "left_breast") == "person_1:left_breast"
    assert make_region_id("person_1", "right_hand") == "person_1:right_hand"
    assert make_region_id("person_1", "left_knee") == "person_1:left_knee"
    assert make_region_id("person_1", "male_external_genital_region") == "person_1:male_external_genital_region"
    assert is_canonical_region_id("person_1:left_breast")
    assert is_canonical_region_id("person_1:male_external_genital_region")
    assert is_canonical_region_id("person_1:not_a_body_region")
    assert is_known_canonical_region_id("person_1:left_breast")
    assert is_known_canonical_region_id("person_1:male_external_genital_region")
    assert not is_known_canonical_region_id("person_1:not_a_body_region")
    assert is_canonical_body_region("left_knee")


def test_memory_family_policy_keeps_private_regions_out_of_identity_memory() -> None:
    assert reference_family_for_region("face") == "identity"
    assert reference_kind_for_region("face") == "identity_reference"
    assert reference_family_for_region("left_breast") == "soft_tissue"
    assert reference_kind_for_region("left_breast") == "body_shape_reference"
    assert reference_family_for_region("external_genital_region") == "private"
    assert reference_kind_for_region("external_genital_region") == "none"
    assert reference_kind_for_region("male_external_genital_region") == "none"


def test_generated_private_region_cannot_become_authoritative_reference_memory() -> None:
    from core.schema import CanonicalRegionMemoryEntry
    from memory.video_memory import MemoryManager

    entry = CanonicalRegionMemoryEntry(
        record_id="person_1:male_external_genital_region",
        entity_id="person_1",
        canonical_region="male_external_genital_region",
        memory_kind="private",
        confidence=0.99,
        visibility_state="visible",
        provenance="generated:test",
        evidence_score=0.99,
        observed_directly=False,
        inferred=True,
        generated=True,
        freshness_frames=0,
    )
    manager = MemoryManager()
    manager._refresh_reuse_policy(entry)
    assert entry.reference_kind == "none"
    assert entry.reliable_as_reference is False
