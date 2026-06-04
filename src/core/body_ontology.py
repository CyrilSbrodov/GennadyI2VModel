from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class CanonicalBodyRegion(str, Enum):
    HEAD = "head"
    SCALP = "scalp"
    HAIR = "hair"
    FACE = "face"
    FOREHEAD = "forehead"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE = "nose"
    MOUTH = "mouth"
    LIPS = "lips"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    JAW = "jaw"
    CHIN = "chin"
    NECK = "neck"
    TORSO = "torso"
    UPPER_TORSO = "upper_torso"
    LOWER_TORSO = "lower_torso"
    CHEST = "chest"
    LEFT_CHEST = "left_chest"
    RIGHT_CHEST = "right_chest"
    ABDOMEN = "abdomen"
    WAIST = "waist"
    BACK = "back"
    UPPER_BACK = "upper_back"
    LOWER_BACK = "lower_back"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_SHOULDER_BLADE = "left_shoulder_blade"
    RIGHT_SHOULDER_BLADE = "right_shoulder_blade"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_UPPER_ARM = "left_upper_arm"
    RIGHT_UPPER_ARM = "right_upper_arm"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_FOREARM = "left_forearm"
    RIGHT_FOREARM = "right_forearm"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    LEFT_PALM = "left_palm"
    RIGHT_PALM = "right_palm"
    LEFT_FINGERS = "left_fingers"
    RIGHT_FINGERS = "right_fingers"
    PELVIS = "pelvis"
    HIPS = "hips"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    BUTTOCKS = "buttocks"
    LEFT_BUTTOCK = "left_buttock"
    RIGHT_BUTTOCK = "right_buttock"
    GROIN_REGION = "groin_region"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    LEFT_THIGH = "left_thigh"
    RIGHT_THIGH = "right_thigh"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_CALF = "left_calf"
    RIGHT_CALF = "right_calf"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    LEFT_TOES = "left_toes"
    RIGHT_TOES = "right_toes"
    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"
    ARMS = "arms"
    HANDS = "hands"
    LEGS = "legs"
    FEET = "feet"
    LEFT_BREAST = "left_breast"
    RIGHT_BREAST = "right_breast"
    BREAST_REGION = "breast_region"
    FEMALE_PELVIC_REGION = "female_pelvic_region"
    MALE_CHEST = "male_chest"
    MALE_PELVIC_REGION = "male_pelvic_region"
    MALE_EXTERNAL_GENITAL_REGION = "male_external_genital_region"
    EXTERNAL_GENITAL_REGION = "external_genital_region"


class BodyRegionGroup(str, Enum):
    CORE_IDENTITY = "core_identity"
    TORSO = "torso"
    ARM = "arm"
    PELVIS_LOWER_BODY = "pelvis_lower_body"
    COMPOSITE = "composite"
    OPTIONAL_SEX_SPECIFIC = "optional_sex_specific"
    OPTIONAL_PRIVATE = "optional_private"


class BodyRegionSide(str, Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    MIDLINE = "midline"
    BILATERAL = "bilateral"


class BodyRegionApplicability(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN_APPLICABILITY = "unknown_applicability"
    UNSUPPORTED_BY_CURRENT_PARSER = "unsupported_by_current_parser"


class BodyRegionVisibility(str, Enum):
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    HIDDEN = "hidden"
    HIDDEN_BY_SELF = "hidden_by_self"
    HIDDEN_BY_GARMENT = "hidden_by_garment"
    HIDDEN_BY_OBJECT = "hidden_by_object"
    OUT_OF_FRAME = "out_of_frame"
    UNKNOWN = "unknown"
    UNKNOWN_EXPECTED_REGION = "unknown_expected_region"


class BodyRegionEvidenceStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    FALLBACK = "fallback"
    GENERATED = "generated"
    UNKNOWN = "unknown"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class BodyRegionMetadata:
    region: CanonicalBodyRegion
    canonical_name: str
    parent_region: str | None
    child_regions: tuple[str, ...]
    side: BodyRegionSide
    group: BodyRegionGroup
    applicability_policy: BodyRegionApplicability
    sex_applicability: str = "all"
    motion_role: str = "structural"
    identity_relevance: str = "none"
    soft_tissue_relevance: str = "none"
    clothing_interaction_relevance: str = "standard"
    parser_support_level: str = "unsupported"
    memory_family: str = "body_shape"
    routing_enabled: bool = True
    suitable_for_memory_seeding: bool = False
    provenance_requirement: str = "observed_parser_or_explicit_state"


def _m(
    name: str,
    *,
    parent: str | None,
    side: BodyRegionSide = BodyRegionSide.MIDLINE,
    group: BodyRegionGroup,
    applicability: BodyRegionApplicability = BodyRegionApplicability.APPLICABLE,
    sex: str = "all",
    motion: str = "structural",
    identity: str = "none",
    soft: str = "none",
    clothing: str = "standard",
    parser: str = "unsupported",
    memory: str = "body_shape",
    seeding: bool = False,
    provenance: str = "observed_parser_or_explicit_state",
) -> BodyRegionMetadata:
    return BodyRegionMetadata(
        region=CanonicalBodyRegion(name),
        canonical_name=name,
        parent_region=parent,
        child_regions=(),
        side=side,
        group=group,
        applicability_policy=applicability,
        sex_applicability=sex,
        motion_role=motion,
        identity_relevance=identity,
        soft_tissue_relevance=soft,
        clothing_interaction_relevance=clothing,
        parser_support_level=parser,
        memory_family=memory,
        suitable_for_memory_seeding=seeding,
        provenance_requirement=provenance,
    )


def _with_children(items: dict[str, BodyRegionMetadata]) -> dict[str, BodyRegionMetadata]:
    children: dict[str, list[str]] = {k: [] for k in items}
    for name, meta in items.items():
        if meta.parent_region in children:
            children[meta.parent_region].append(name)
    return {name: replace(meta, child_regions=tuple(dict.fromkeys((*meta.child_regions, *children[name])))) for name, meta in items.items()}


_BASE: dict[str, BodyRegionMetadata] = {}
# Core / identity
for n, p, parser in [
    ("head", None, "direct"), ("scalp", "head", "derived_from_hair_or_head"), ("hair", "head", "direct"),
    ("face", "head", "direct"), ("forehead", "face", "unsupported"), ("left_eye", "face", "unsupported"),
    ("right_eye", "face", "unsupported"), ("nose", "face", "unsupported"), ("mouth", "face", "unsupported"),
    ("lips", "mouth", "unsupported"), ("left_ear", "head", "unsupported"), ("right_ear", "head", "unsupported"),
    ("jaw", "face", "unsupported"), ("chin", "face", "unsupported"), ("neck", "head", "direct"),
]:
    side = BodyRegionSide.LEFT if n.startswith("left_") else BodyRegionSide.RIGHT if n.startswith("right_") else BodyRegionSide.MIDLINE
    _BASE[n] = _m(n, parent=p, side=side, group=BodyRegionGroup.CORE_IDENTITY, motion="identity_anchor", identity="strong", parser=parser, memory="identity", seeding=n in {"head", "hair", "face"})
# Torso
for n, p in [("torso", None), ("upper_torso", "torso"), ("lower_torso", "torso"), ("chest", "upper_torso"), ("left_chest", "chest"), ("right_chest", "chest"), ("abdomen", "lower_torso"), ("waist", "lower_torso"), ("back", "torso"), ("upper_back", "back"), ("lower_back", "back"), ("left_shoulder", "upper_torso"), ("right_shoulder", "upper_torso"), ("left_shoulder_blade", "upper_back"), ("right_shoulder_blade", "upper_back")]:
    side = BodyRegionSide.LEFT if n.startswith("left_") else BodyRegionSide.RIGHT if n.startswith("right_") else BodyRegionSide.MIDLINE
    _BASE[n] = _m(n, parent=p, side=side, group=BodyRegionGroup.TORSO, motion="torso_articulation", soft="medium" if n in {"abdomen", "hips"} else "none", parser="direct" if n in {"torso", "chest"} else "unsupported", memory="body_shape")
# Arms
for n, p in [("left_arm", None), ("right_arm", None), ("left_upper_arm", "left_arm"), ("right_upper_arm", "right_arm"), ("left_elbow", "left_arm"), ("right_elbow", "right_arm"), ("left_forearm", "left_arm"), ("right_forearm", "right_arm"), ("left_wrist", "left_hand"), ("right_wrist", "right_hand"), ("left_hand", "left_arm"), ("right_hand", "right_arm"), ("left_palm", "left_hand"), ("right_palm", "right_hand"), ("left_fingers", "left_hand"), ("right_fingers", "right_hand")]:
    _BASE[n] = _m(n, parent=p, side=BodyRegionSide.LEFT if n.startswith("left_") else BodyRegionSide.RIGHT, group=BodyRegionGroup.ARM, motion="limb_articulation", parser="direct" if n in {"left_arm", "right_arm", "left_hand", "right_hand"} else "unsupported", memory="skin" if "hand" in n or "palm" in n or "fingers" in n else "body_shape", seeding=n in {"left_hand", "right_hand"})
# Pelvis/lower body
for n, p in [("pelvis", None), ("hips", "pelvis"), ("left_hip", "hips"), ("right_hip", "hips"), ("buttocks", "pelvis"), ("left_buttock", "buttocks"), ("right_buttock", "buttocks"), ("groin_region", "pelvis"), ("left_leg", None), ("right_leg", None), ("left_thigh", "left_leg"), ("right_thigh", "right_leg"), ("left_knee", "left_leg"), ("right_knee", "right_leg"), ("left_calf", "left_leg"), ("right_calf", "right_leg"), ("left_ankle", "left_foot"), ("right_ankle", "right_foot"), ("left_foot", "left_leg"), ("right_foot", "right_leg"), ("left_toes", "left_foot"), ("right_toes", "right_foot")]:
    side = BodyRegionSide.LEFT if n.startswith("left_") else BodyRegionSide.RIGHT if n.startswith("right_") else BodyRegionSide.MIDLINE
    _BASE[n] = _m(n, parent=p, side=side, group=BodyRegionGroup.PELVIS_LOWER_BODY, motion="limb_articulation" if "leg" in n or n.startswith(("left_", "right_")) else "pelvic_anchor", soft="medium" if n in {"buttocks", "left_buttock", "right_buttock", "hips"} else "none", parser="direct" if n in {"pelvis", "left_leg", "right_leg", "left_foot", "right_foot"} else "unsupported", memory="soft_tissue" if n in {"buttocks", "left_buttock", "right_buttock", "hips"} else "body_shape")
# Composites
for n, kids in {"upper_body": ("torso", "left_arm", "right_arm"), "lower_body": ("pelvis", "left_leg", "right_leg"), "arms": ("left_arm", "right_arm"), "hands": ("left_hand", "right_hand"), "legs": ("left_leg", "right_leg"), "feet": ("left_foot", "right_foot")}.items():
    _BASE[n] = BodyRegionMetadata(CanonicalBodyRegion(n), n, None, kids, BodyRegionSide.BILATERAL, BodyRegionGroup.COMPOSITE, BodyRegionApplicability.APPLICABLE, motion_role="composite_motion", parser_support_level="composite", memory_family="body_shape")
# Optional sex-specific/private
for n, p, sex, group, soft, memory in [
    ("left_breast", "breast_region", "female", BodyRegionGroup.OPTIONAL_SEX_SPECIFIC, "high", "soft_tissue"),
    ("right_breast", "breast_region", "female", BodyRegionGroup.OPTIONAL_SEX_SPECIFIC, "high", "soft_tissue"),
    ("breast_region", "chest", "female", BodyRegionGroup.OPTIONAL_SEX_SPECIFIC, "high", "soft_tissue"),
    ("female_pelvic_region", "pelvis", "female", BodyRegionGroup.OPTIONAL_SEX_SPECIFIC, "medium", "private"),
    ("male_chest", "chest", "male", BodyRegionGroup.OPTIONAL_SEX_SPECIFIC, "low", "body_shape"),
    ("male_pelvic_region", "pelvis", "male", BodyRegionGroup.OPTIONAL_SEX_SPECIFIC, "medium", "private"),
    ("male_external_genital_region", "external_genital_region", "male", BodyRegionGroup.OPTIONAL_PRIVATE, "medium", "private"),
    ("external_genital_region", "groin_region", "unknown", BodyRegionGroup.OPTIONAL_PRIVATE, "medium", "private"),
]:
    side = BodyRegionSide.LEFT if n.startswith("left_") else BodyRegionSide.RIGHT if n.startswith("right_") else BodyRegionSide.MIDLINE
    _BASE[n] = _m(n, parent=p, side=side, group=group, applicability=BodyRegionApplicability.UNKNOWN_APPLICABILITY, sex=sex, motion="soft_tissue_address" if soft != "low" else "structural", soft=soft, clothing="private_occlusion_sensitive", parser="unsupported", memory=memory, provenance="explicit_parser_evidence_required")

BODY_ONTOLOGY: dict[str, BodyRegionMetadata] = _with_children(_BASE)
CANONICAL_BODY_REGION_ORDER: tuple[str, ...] = tuple(BODY_ONTOLOGY.keys())
CANONICAL_BODY_REGION_IDS: frozenset[str] = frozenset(BODY_ONTOLOGY.keys())


def get_body_region_metadata(region: str) -> BodyRegionMetadata | None:
    return BODY_ONTOLOGY.get(str(region or "").strip())


def is_canonical_body_region(region: str) -> bool:
    return str(region or "").strip() in BODY_ONTOLOGY
