from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import CanonicalRelationPayload, CanonicalRegionPayload, RelationEdge
from perception.mask_projection import project_mask_to_frame
from perception.mask_store import DEFAULT_MASK_STORE
from perception.pipeline import ObjectFacts, PersonFacts

_CANONICAL_REGION_ORDER = [
    "head",
    "face",
    "hair",
    "neck",
    "torso",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "pelvis",
    "left_leg",
    "right_leg",
    "upper_body",
    "lower_body",
    "upper_garment",
    "lower_garment",
    "outer_garment",
    "inner_garment",
    "accessories",
]

_COMPOSITE_CHILDREN = {
    "head": ["face", "hair", "neck"],
    "upper_body": ["torso", "left_arm", "right_arm", "left_hand", "right_hand"],
    "lower_body": ["pelvis", "left_leg", "right_leg"],
}

BODY_PART_MAP = {
    "head": "head",
    "neck": "neck",
    "torso": "torso",
    "pelvis": "pelvis",
    "face": "face",
    "hair": "hair",
    "left_arm": "left_arm",
    "left_upper_arm": "left_arm",
    "left_lower_arm": "left_arm",
    "right_arm": "right_arm",
    "right_upper_arm": "right_arm",
    "right_lower_arm": "right_arm",
    "arms": "left_arm",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "hands": "left_hand",
    "left_leg": "left_leg",
    "left_upper_leg": "left_leg",
    "left_lower_leg": "left_leg",
    "right_leg": "right_leg",
    "right_upper_leg": "right_leg",
    "right_lower_leg": "right_leg",
    "legs": "left_leg",
}

FACE_REGION_MAP = {"face": "face", "hair": "hair", "neck": "neck", "head": "head"}

GARMENT_REGION_MAP = {
    "coat": "outer_garment",
    "jacket": "outer_garment",
    "hoodie": "outer_garment",
    "sweater": "outer_garment",
    "top": "upper_garment",
    "shirt": "upper_garment",
    "blouse": "upper_garment",
    "dress": "upper_garment",
    "pants": "lower_garment",
    "skirt": "lower_garment",
    "shorts": "lower_garment",
    "jeans": "lower_garment",
    "belt": "accessories",
    "hat": "accessories",
    "glasses": "accessories",
    "scarf": "accessories",
    "bag": "accessories",
    "jewelry": "accessories",
}

_PROVENANCE_PRIOR = {
    "parser": 0.96,
    "segformer": 0.95,
    "schp": 0.92,
    "fashn": 0.9,
    "face": 0.86,
    "vitpose": 0.83,
    "yolo": 0.82,
    "heuristic": 0.7,
    "fallback": 0.52,
    "unknown": 0.6,
}

_VISIBILITY_PRIORITY = {
    "visible": 1.0,
    "partially_visible": 0.75,
    "hidden_by_garment": 0.72,
    "hidden_by_object": 0.65,
    "hidden_by_self": 0.6,
    "hidden": 0.45,
    "out_of_frame": 0.4,
    "unknown_expected_region": 0.15,
    "unknown": 0.1,
}


@dataclass(slots=True)
class CanonicalRegion:
    canonical_name: str
    raw_sources: list[str] = field(default_factory=list)
    source_regions: list[str] = field(default_factory=list)
    mask_ref: str | None = None
    confidence: float = 0.0
    visibility_state: str = "unknown_expected_region"
    provenance: str = "unknown"
    attachment_hints: list[str] = field(default_factory=list)
    ownership_hints: list[str] = field(default_factory=list)
    coverage_hints: list[str] = field(default_factory=list)
    evidence_score: float = 0.0
    visibility_score: float = 0.0
    mask_score: float = 0.0


@dataclass(slots=True)
class CanonicalRelation:
    source: str
    relation: str
    target: str
    confidence: float
    provenance: str
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CanonicalPersonState:
    person_id: str
    regions: dict[str, CanonicalRegion]
    relations: list[CanonicalRelation] = field(default_factory=list)


class CanonicalHumanNormalizer:
    def normalize(self, person: PersonFacts, person_id: str) -> CanonicalPersonState:
        regions = {name: CanonicalRegion(canonical_name=name) for name in _CANONICAL_REGION_ORDER}
        self._seed_composites(regions)

        for part in person.body_parts:
            raw = str(part.get("part_type", "")).lower().strip()
            canonical = BODY_PART_MAP.get(raw, raw if raw in regions else "")
            if not canonical:
                continue
            self._merge_region(
                region=regions[canonical],
                raw_name=raw,
                source=str(part.get("source", "unknown")),
                mask_ref=part.get("mask_ref"),
                confidence=float(part.get("confidence", 0.0)),
                visibility_hint=str(part.get("visibility", "")),
                coverage_hints=person.coverage_hints.get(raw, []),
            )

        for raw_name, mask_ref in person.body_part_masks.items():
            canonical = BODY_PART_MAP.get(raw_name, raw_name if raw_name in regions else "")
            if not canonical:
                continue
            self._merge_region(
                region=regions[canonical],
                raw_name=raw_name,
                source=person.provenance_by_region.get(f"body:{raw_name}", person.mask_source),
                mask_ref=mask_ref,
                confidence=person.mask_confidence,
                visibility_hint=person.visibility_hints.get(raw_name, ""),
                coverage_hints=person.coverage_hints.get(raw_name, []),
            )

        for region in person.face_regions:
            raw = str(region.get("region_type", "")).lower().strip()
            canonical = FACE_REGION_MAP.get(raw, raw if raw in regions else "")
            if not canonical:
                continue
            self._merge_region(
                region=regions[canonical],
                raw_name=raw,
                source=str(region.get("source", "unknown")),
                mask_ref=region.get("mask_ref"),
                confidence=float(region.get("confidence", 0.0)),
            )

        for raw_name, mask_ref in person.face_region_masks.items():
            canonical = FACE_REGION_MAP.get(raw_name, raw_name if raw_name in regions else "")
            if not canonical:
                continue
            self._merge_region(
                region=regions[canonical],
                raw_name=raw_name,
                source=person.provenance_by_region.get(f"face:{raw_name}", person.mask_source),
                mask_ref=mask_ref,
                confidence=person.mask_confidence,
                visibility_hint=person.visibility_hints.get(raw_name, ""),
            )

        for garment in person.garments:
            gtype = str(garment.get("type", "unknown")).lower().strip()
            source = str(garment.get("source", "unknown"))
            layer_hint = str(garment.get("layer_hint", "")).lower()
            target_canonical = GARMENT_REGION_MAP.get(gtype, "upper_garment")
            if layer_hint in {"outer", "outerwear", "outside"}:
                target_canonical = "outer_garment"
            elif layer_hint in {"inner", "innerwear", "inside"}:
                target_canonical = "inner_garment"
            elif any(t in {"pelvis", "left_leg", "right_leg", "legs"} for t in garment.get("coverage_targets", [])):
                target_canonical = "lower_garment"
            elif any(t in {"torso", "left_arm", "right_arm", "arms"} for t in garment.get("coverage_targets", [])):
                target_canonical = "upper_garment"

            coverage_targets = [str(v) for v in garment.get("coverage_targets", [])]
            attachment_targets = [str(v) for v in garment.get("attachment_targets", [])]
            self._merge_region(
                region=regions[target_canonical],
                raw_name=f"garment:{gtype}",
                source=source,
                mask_ref=garment.get("mask_ref") or person.garment_masks.get(gtype),
                confidence=float(garment.get("confidence", 0.0)),
                visibility_hint=person.visibility_hints.get(gtype, ""),
                coverage_hints=coverage_targets,
                attachment_hints=attachment_targets,
                ownership_hint="person",
            )

        for raw_name, mask_ref in person.garment_masks.items():
            canonical = GARMENT_REGION_MAP.get(raw_name, "upper_garment")
            self._merge_region(
                region=regions[canonical],
                raw_name=f"garment_mask:{raw_name}",
                source=person.provenance_by_region.get(f"garment:{raw_name}", person.mask_source),
                mask_ref=mask_ref,
                confidence=person.mask_confidence,
                coverage_hints=person.coverage_hints.get(raw_name, []),
                ownership_hint="person",
            )

        for acc, mask_ref in person.accessory_masks.items():
            self._merge_region(
                region=regions["accessories"],
                raw_name=acc,
                source=person.provenance_by_region.get(f"accessory:{acc}", person.mask_source),
                mask_ref=mask_ref,
                confidence=person.mask_confidence,
                ownership_hint="person",
            )

        self._finalize_composites(regions)
        return CanonicalPersonState(person_id=person_id, regions=regions)

    @staticmethod
    def _seed_composites(regions: dict[str, CanonicalRegion]) -> None:
        regions["head"].coverage_hints.extend(["face", "hair"])
        regions["upper_body"].attachment_hints.append("torso")
        regions["lower_body"].attachment_hints.append("pelvis")

    @staticmethod
    def _finalize_composites(regions: dict[str, CanonicalRegion]) -> None:
        for parent, children in _COMPOSITE_CHILDREN.items():
            child_regions = [regions[name] for name in children if name in regions]
            if not child_regions:
                continue
            strong = [r for r in child_regions if r.confidence > 0.05]
            if not strong:
                continue
            weighted = sum(r.confidence * max(0.5, r.evidence_score) for r in strong)
            evidence_sum = sum(max(0.5, r.evidence_score) for r in strong)
            regions[parent].confidence = max(regions[parent].confidence, weighted / max(1e-6, evidence_sum))
            vis = max(strong, key=lambda r: r.visibility_score)
            if vis.visibility_score > regions[parent].visibility_score:
                regions[parent].visibility_state = vis.visibility_state
                regions[parent].visibility_score = vis.visibility_score * 0.95
            if regions[parent].provenance == "unknown":
                regions[parent].provenance = vis.provenance
            for child in strong:
                if child.mask_score > regions[parent].mask_score and child.mask_ref is not None:
                    regions[parent].mask_ref = child.mask_ref
                    regions[parent].mask_score = child.mask_score

    @staticmethod
    def _merge_region(
        region: CanonicalRegion,
        raw_name: str,
        source: str,
        mask_ref: str | None,
        confidence: float,
        visibility_hint: str = "",
        ownership_hint: str = "",
        coverage_hints: list[str] | None = None,
        attachment_hints: list[str] | None = None,
    ) -> None:
        conf = max(0.0, min(1.0, confidence))
        src_weight = _source_weight(source)
        evidence = conf * src_weight

        region.source_regions.append(raw_name)
        region.raw_sources.append(source)

        if evidence >= region.evidence_score:
            region.provenance = source
            region.evidence_score = evidence

        region.confidence = max(region.confidence, conf)

        if mask_ref:
            mask_evidence = evidence + 0.2
            if mask_evidence >= region.mask_score:
                region.mask_ref = mask_ref
                region.mask_score = mask_evidence

        if ownership_hint and ownership_hint not in region.ownership_hints:
            region.ownership_hints.append(ownership_hint)
        if coverage_hints:
            for hint in coverage_hints:
                if hint not in region.coverage_hints:
                    region.coverage_hints.append(hint)
        if attachment_hints:
            for hint in attachment_hints:
                if hint not in region.attachment_hints:
                    region.attachment_hints.append(hint)

        explicit_visible = region.mask_ref is not None and region.confidence >= 0.7
        if explicit_visible:
            region.visibility_state = "visible"
            region.visibility_score = _VISIBILITY_PRIORITY["visible"]
            return

        vis_hint = _normalize_visibility_hint(visibility_hint)
        if vis_hint:
            vis_score = _VISIBILITY_PRIORITY.get(vis_hint, 0.0) * max(0.25, src_weight)
            if vis_score > region.visibility_score:
                region.visibility_state = vis_hint
                region.visibility_score = vis_score


class RelationReasoner:
    _PRIORS: list[tuple[str, str, str, float]] = [
        ("face", "part_of", "head", 0.92),
        ("hair", "adjacent_to", "head", 0.86),
        ("left_hand", "part_of", "left_arm", 0.88),
        ("right_hand", "part_of", "right_arm", 0.88),
        ("left_arm", "part_of", "upper_body", 0.82),
        ("right_arm", "part_of", "upper_body", 0.82),
        ("left_leg", "part_of", "lower_body", 0.82),
        ("right_leg", "part_of", "lower_body", 0.82),
        ("torso", "part_of", "upper_body", 0.9),
        ("pelvis", "part_of", "lower_body", 0.9),
    ]

    def infer(self, state: CanonicalPersonState, frame_size: tuple[int, int]) -> list[CanonicalRelation]:
        relations: list[CanonicalRelation] = []
        regions = state.regions

        for source, relation, target, prior in self._PRIORS:
            src = regions[source]
            tgt = regions[target]
            conf = _relation_confidence(prior=prior, source=src, target=tgt, evidence_bonus=0.0)
            if conf < 0.2:
                continue
            relations.append(
                CanonicalRelation(source=source, relation=relation, target=target, confidence=conf, provenance="canonical_relation_reasoner", evidence=["topology_prior"])
            )

        garment_specs = [
            ("upper_garment", "attached_to", "torso", 0.78),
            ("lower_garment", "attached_to", "pelvis", 0.78),
            ("outer_garment", "attached_to", "upper_body", 0.74),
            ("upper_garment", "covers", "torso", 0.72),
            ("upper_garment", "covers", "left_arm", 0.66),
            ("upper_garment", "covers", "right_arm", 0.66),
            ("lower_garment", "covers", "pelvis", 0.72),
            ("lower_garment", "covers", "left_leg", 0.67),
            ("lower_garment", "covers", "right_leg", 0.67),
        ]
        for source, relation, target, prior in garment_specs:
            src = regions[source]
            tgt = regions[target]
            overlap = _mask_overlap_score(src, tgt, frame_size)
            coverage_bonus = 0.12 if target in src.coverage_hints else 0.0
            attach_bonus = 0.1 if target in src.attachment_hints else 0.0
            evidence_bonus = max(overlap * 0.35, coverage_bonus, attach_bonus)
            conf = _relation_confidence(prior=prior, source=src, target=tgt, evidence_bonus=evidence_bonus)
            if conf < 0.2:
                continue
            evid = ["garment_prior"]
            if overlap > 0.0:
                evid.append(f"mask_overlap:{overlap:.3f}")
            if target in src.coverage_hints:
                evid.append("coverage_hint")
            if target in src.attachment_hints:
                evid.append("attachment_hint")
            relations.append(
                CanonicalRelation(source=source, relation=relation, target=target, confidence=conf, provenance="canonical_relation_reasoner", evidence=evid)
            )

        for garment in ("upper_garment", "lower_garment", "outer_garment"):
            src = regions[garment]
            if src.confidence < 0.2:
                continue
            for body in ("torso", "pelvis", "left_arm", "right_arm", "left_leg", "right_leg"):
                tgt = regions[body]
                overlap = _mask_overlap_score(src, tgt, frame_size)
                if overlap < 0.18:
                    continue
                conf = _relation_confidence(prior=0.58, source=src, target=tgt, evidence_bonus=min(0.3, overlap))
                relations.append(
                    CanonicalRelation(
                        source=garment,
                        relation="overlaps",
                        target=body,
                        confidence=conf,
                        provenance="canonical_relation_reasoner",
                        evidence=[f"mask_overlap:{overlap:.3f}"],
                    )
                )

        state.relations = relations
        return relations


class VisibilityOcclusionReasoner:
    def infer(self, state: CanonicalPersonState, person: PersonFacts, objects: list[ObjectFacts]) -> list[CanonicalRelation]:
        occlusion_relations: list[CanonicalRelation] = []
        covers = {(rel.source, rel.target): rel for rel in state.relations if rel.relation == "covers"}
        overlaps = {(rel.source, rel.target): rel for rel in state.relations if rel.relation == "overlaps"}

        for name, region in state.regions.items():
            if region.mask_ref and region.confidence >= 0.72:
                region.visibility_state = "visible"
                region.visibility_score = _VISIBILITY_PRIORITY["visible"]
                continue
            if region.mask_ref and 0.35 <= region.confidence < 0.72:
                region.visibility_state = "partially_visible"
                region.visibility_score = _VISIBILITY_PRIORITY["partially_visible"]
                continue

            if name in {"torso", "pelvis", "left_arm", "right_arm", "left_leg", "right_leg"}:
                garment_cover = _best_cover_for_target(name, covers)
                if garment_cover:
                    region.visibility_state = "hidden_by_garment"
                    region.visibility_score = _VISIBILITY_PRIORITY["hidden_by_garment"]
                    occlusion_relations.append(
                        CanonicalRelation(
                            source=garment_cover.source,
                            relation="occludes",
                            target=name,
                            confidence=min(0.9, 0.58 + garment_cover.confidence * 0.4),
                            provenance="visibility_reasoner",
                            evidence=garment_cover.evidence + ["garment_cover"],
                        )
                    )
                    continue

            object_support = _object_occlusion_support(person, name, objects, person.occlusion_hints)
            if object_support is not None:
                obj, support = object_support
                if support >= 0.55:
                    region.visibility_state = "hidden_by_object"
                    region.visibility_score = _VISIBILITY_PRIORITY["hidden_by_object"] * support
                    occlusion_relations.append(
                        CanonicalRelation(
                            source=f"object:{obj.object_type}",
                            relation="occludes",
                            target=name,
                            confidence=min(0.9, 0.45 + 0.45 * support),
                            provenance="visibility_reasoner",
                            evidence=[f"object_proximity:{support:.3f}", "hint_match"],
                        )
                    )
                    continue

            if _self_occlusion_support(name, person.occlusion_hints, overlaps) >= 0.62:
                region.visibility_state = "hidden_by_self"
                region.visibility_score = _VISIBILITY_PRIORITY["hidden_by_self"]
                continue

            if region.confidence >= 0.18:
                region.visibility_state = "partially_visible"
                region.visibility_score = max(region.visibility_score, _VISIBILITY_PRIORITY["partially_visible"] * 0.55)
                continue

            if _is_bbox_edge(person, name):
                region.visibility_state = "out_of_frame"
                region.visibility_score = _VISIBILITY_PRIORITY["out_of_frame"]
                continue

            if region.visibility_score <= _VISIBILITY_PRIORITY["unknown_expected_region"]:
                region.visibility_state = "unknown_expected_region"
                region.visibility_score = _VISIBILITY_PRIORITY["unknown_expected_region"]
        return occlusion_relations


def _best_cover_for_target(target: str, covers: dict[tuple[str, str], CanonicalRelation]) -> CanonicalRelation | None:
    candidates = [rel for (_, t), rel in covers.items() if t == target]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.confidence)


def _self_occlusion_support(region_name: str, occlusion_hints: list[str], overlaps: dict[tuple[str, str], CanonicalRelation]) -> float:
    hints = " ".join(occlusion_hints).lower()
    region_token = region_name.replace("_", " ")
    score = 0.0
    if region_token in hints and "self" in hints:
        score += 0.45
    if region_name in {"left_arm", "right_arm", "left_hand", "right_hand"} and any(k in hints for k in ["arm", "hand", "crossed"]):
        score += 0.25
    if region_name in {"left_leg", "right_leg"} and any(k in hints for k in ["leg", "knee", "crossed"]):
        score += 0.2
    overlap_support = max(
        (rel.confidence for (src, tgt), rel in overlaps.items() if tgt == region_name and src in {"left_arm", "right_arm", "left_leg", "right_leg"}),
        default=0.0,
    )
    score += overlap_support * 0.25
    return min(1.0, score)


def _object_occlusion_support(
    person: PersonFacts,
    region_name: str,
    objects: list[ObjectFacts],
    occlusion_hints: list[str],
) -> tuple[ObjectFacts, float] | None:
    if not objects:
        return None
    hints = " ".join(occlusion_hints).lower()
    region_token = region_name.replace("_", " ")
    if region_token not in hints and region_name not in hints and "occlusion" not in hints and "occluded" not in hints:
        return None

    best: tuple[ObjectFacts, float] | None = None
    for obj in objects:
        proximity = _bbox_overlap(person, obj)
        type_match = 0.2 if obj.object_type.lower() in hints else 0.0
        generic_object_hint = 0.15 if "object" in hints else 0.0
        support = min(1.0, proximity + type_match + generic_object_hint)
        if best is None or support > best[1]:
            best = (obj, support)
    return best


def _bbox_overlap(person: PersonFacts, obj: ObjectFacts) -> float:
    ax1, ay1, ax2, ay2 = person.bbox.x, person.bbox.y, person.bbox.x + person.bbox.w, person.bbox.y + person.bbox.h
    bx1, by1, bx2, by2 = obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.w, obj.bbox.y + obj.bbox.h
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, person.bbox.w * person.bbox.h)
    area_b = max(1e-6, obj.bbox.w * obj.bbox.h)
    union = area_a + area_b - inter
    iou = inter / max(1e-6, union)
    contained = inter / area_a
    return min(1.0, max(iou, contained))


def _is_bbox_edge(person: PersonFacts, region_name: str) -> bool:
    bbox = person.bbox
    if region_name in {"head", "face", "hair", "neck"}:
        return bbox.y <= 0.015
    if region_name in {"left_leg", "right_leg", "lower_body"}:
        return bbox.y + bbox.h >= 0.99
    return bbox.x <= 0.01 or bbox.x + bbox.w >= 0.99


def _relation_confidence(prior: float, source: CanonicalRegion, target: CanonicalRegion, evidence_bonus: float) -> float:
    source_strength = max(0.12, source.confidence * _source_weight(source.provenance))
    target_strength = max(0.12, target.confidence * _source_weight(target.provenance))
    confidence = prior * ((source_strength + target_strength) / 2.0) + evidence_bonus
    return round(max(0.0, min(1.0, confidence)), 4)


def _mask_overlap_score(a: CanonicalRegion | None, b: CanonicalRegion | None, frame_size: tuple[int, int]) -> float:
    if not a or not b or not a.mask_ref or not b.mask_ref:
        return 0.0
    ma = DEFAULT_MASK_STORE.get(a.mask_ref)
    mb = DEFAULT_MASK_STORE.get(b.mask_ref)
    if ma is None or mb is None:
        return 0.0
    pa, _ = project_mask_to_frame(ma, frame_size=frame_size)
    pb, _ = project_mask_to_frame(mb, frame_size=frame_size)
    h = min(len(pa), len(pb))
    w = min(len(pa[0]), len(pb[0])) if h else 0
    inter = 0
    union = 0
    for y in range(h):
        row_a = pa[y]
        row_b = pb[y]
        for x in range(w):
            va = int(row_a[x] > 0)
            vb = int(row_b[x] > 0)
            inter += int(va and vb)
            union += int(va or vb)
    if union <= 0:
        return 0.0
    return inter / union


class CanonicalHumanSceneProcessor:
    def __init__(self) -> None:
        self.normalizer = CanonicalHumanNormalizer()
        self.relation_reasoner = RelationReasoner()
        self.visibility_reasoner = VisibilityOcclusionReasoner()

    def process(self, person: PersonFacts, person_id: str, frame_size: tuple[int, int], objects: list[ObjectFacts]) -> CanonicalPersonState:
        state = self.normalizer.normalize(person, person_id=person_id)
        self.relation_reasoner.infer(state, frame_size=frame_size)
        state.relations.extend(self.visibility_reasoner.infer(state, person=person, objects=objects))
        return state


def canonical_state_to_dict(state: CanonicalPersonState) -> dict[str, object]:
    return {
        "person_id": state.person_id,
        "regions": {
            name: CanonicalRegionPayload(
                canonical_name=region.canonical_name,
                raw_sources=region.raw_sources,
                source_regions=region.source_regions,
                mask_ref=region.mask_ref,
                confidence=round(region.confidence, 4),
                visibility_state=region.visibility_state,
                provenance=region.provenance,
                attachment_hints=region.attachment_hints,
                ownership_hints=region.ownership_hints,
                coverage_hints=region.coverage_hints,
            )
            for name, region in state.regions.items()
        },
        "relations": [
            CanonicalRelationPayload(
                source=rel.source,
                relation=rel.relation,
                target=rel.target,
                confidence=round(rel.confidence, 4),
                provenance=rel.provenance,
                evidence=rel.evidence,
            )
            for rel in state.relations
        ],
    }


def canonical_relations_to_edges(state: CanonicalPersonState, frame_index: int) -> list[RelationEdge]:
    def _node_id(name: str) -> str:
        if name.startswith("object:"):
            return name.replace("object:", "")
        if name == "person":
            return state.person_id
        return f"{state.person_id}_{name}"

    out: list[RelationEdge] = []
    for rel in state.relations:
        if rel.relation not in {"part_of", "attached_to", "covers", "occludes", "adjacent_to", "overlaps"}:
            continue
        out.append(
            RelationEdge(
                source=_node_id(rel.source),
                relation=rel.relation,
                target=_node_id(rel.target),
                confidence=round(rel.confidence, 4),
                provenance=rel.provenance,
                frame_index=frame_index,
                alternatives=["canonical_reasoning"],
            )
        )
    return out


def _normalize_visibility_hint(visibility_hint: str) -> str:
    hint = visibility_hint.strip().lower()
    if hint in _VISIBILITY_PRIORITY:
        return hint
    if hint in {"partial", "partly_visible", "partially"}:
        return "partially_visible"
    if hint in {"occluded", "hidden"}:
        return "hidden"
    if hint in {"visible_strong", "clearly_visible"}:
        return "visible"
    return ""


def _source_weight(source: str) -> float:
    key = source.lower().split(":")[0]
    for prefix, value in _PROVENANCE_PRIOR.items():
        if key.startswith(prefix):
            return value
    return _PROVENANCE_PRIOR["unknown"]
