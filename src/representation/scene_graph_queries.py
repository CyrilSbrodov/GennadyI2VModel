from __future__ import annotations

from core.schema import BodyPartNode, GarmentNode, GarmentSemanticProfile, RelationEdge, SceneGraph


class SceneGraphQueries:
    """Утилиты доступа к SceneGraph + нормализация семантики одежды."""

    _LAYER_TOKENS = {
        "outerwear": ("coat", "jacket", "hoodie", "blazer", "cardigan"),
        "inner_upper": ("shirt", "tshirt", "top", "blouse", "sweater"),
        "lower_wear": ("pants", "jeans", "skirt", "shorts", "trousers"),
        "footwear": ("shoe", "boot", "sneaker", "heel"),
        "accessory": ("hat", "cap", "scarf", "belt", "glove"),
    }

    @staticmethod
    def _person(scene_graph: SceneGraph, person_id: str):
        return next((p for p in scene_graph.persons if p.person_id == person_id), None)

    @classmethod
    def _is_garment_like_entity(cls, entity_id: str, relation: RelationEdge | None = None) -> bool:
        if relation and relation.relation in {"covers", "attached_to"}:
            return True
        lowered = entity_id.lower()
        return any(token in lowered for tokens in cls._LAYER_TOKENS.values() for token in tokens)

    @classmethod
    def get_body_part(cls, scene_graph: SceneGraph, person_id: str, part_type: str) -> BodyPartNode | None:
        person = cls._person(scene_graph, person_id)
        if person is None:
            return None
        return next((part for part in person.body_parts if part.part_type == part_type), None)

    @classmethod
    def get_garment(cls, scene_graph: SceneGraph, person_id: str, garment_type: str) -> GarmentNode | None:
        person = cls._person(scene_graph, person_id)
        if person is None:
            return None
        return next((garment for garment in person.garments if garment.garment_type == garment_type), None)

    @classmethod
    def get_visible_regions(cls, scene_graph: SceneGraph, person_id: str) -> list[str]:
        person = cls._person(scene_graph, person_id)
        if person is None:
            return []
        regions = [part.part_type for part in person.body_parts if part.visibility in {"visible", "partially_visible"}]
        regions.extend(garment.garment_type for garment in person.garments if garment.visibility in {"visible", "partially_visible"})
        return regions

    @classmethod
    def get_occluded_regions(cls, scene_graph: SceneGraph, person_id: str) -> list[str]:
        person = cls._person(scene_graph, person_id)
        if person is None:
            return []
        regions = [part.part_type for part in person.body_parts if part.visibility == "hidden"]
        regions.extend(garment.garment_type for garment in person.garments if garment.visibility == "hidden")
        covered = [
            relation.target.split("_", 1)[-1]
            for relation in scene_graph.relations
            if relation.relation == "covers" and cls._is_garment_like_entity(relation.source, relation)
        ]
        return sorted(set(regions + covered))

    @staticmethod
    def relations_for_entity(scene_graph: SceneGraph, entity_id: str) -> list[RelationEdge]:
        return [r for r in scene_graph.relations if r.source == entity_id or r.target == entity_id]

    @staticmethod
    def relation_strength(scene_graph: SceneGraph, relation: str, source: str, target: str) -> float:
        scored = [r.confidence for r in scene_graph.relations if r.relation == relation and r.source == source and r.target == target]
        return max(scored) if scored else 0.0

    @staticmethod
    def get_covering_entities(scene_graph: SceneGraph, entity_id: str, target_region: str | None = None) -> list[str]:
        matches = [r.source for r in scene_graph.relations if r.relation in {"covers", "occludes"} and r.target.startswith(entity_id)]
        if target_region is not None:
            suffix = f"_{target_region}"
            matches = [src for src in matches if any(r.source == src and r.target.endswith(suffix) for r in scene_graph.relations if r.relation in {"covers", "occludes"})]
        return sorted(set(matches))

    @staticmethod
    def get_supported_by(scene_graph: SceneGraph, entity_id: str) -> list[str]:
        return sorted(set(r.target for r in scene_graph.relations if r.relation == "supports" and r.source == entity_id))

    @classmethod
    def get_attached_garments(cls, scene_graph: SceneGraph, person_id: str) -> list[str]:
        person = cls._person(scene_graph, person_id)
        garments = {g.garment_id for g in person.garments} if person else set()
        for r in scene_graph.relations:
            if r.relation == "attached_to" and (r.target == person_id or r.target.startswith(person_id)):
                if cls._is_garment_like_entity(r.source, r):
                    garments.add(r.source)
        return sorted(garments)

    @staticmethod
    def get_regions_affected_by_relation(scene_graph: SceneGraph, source_entity: str, relation_type: str) -> list[str]:
        affected: list[str] = []
        for relation in scene_graph.relations:
            if relation.source != source_entity or relation.relation != relation_type:
                continue
            target = relation.target.split(":", 1)[-1]
            affected.append(target.split("_", 1)[-1] if "_" in target else target)
        return sorted(set(affected))

    @staticmethod
    def is_region_likely_revealed(scene_graph: SceneGraph, entity_id: str, region_type: str, predicted_visibility: dict[str, str] | None = None) -> bool:
        vis = (predicted_visibility or {}).get(region_type, "unknown")
        if vis in {"visible", "partially_visible"}:
            return True
        cover_edges = [r for r in scene_graph.relations if r.relation in {"covers", "occludes"} and r.target.startswith(entity_id) and r.target.endswith(f"_{region_type}")]
        if not cover_edges:
            return vis != "hidden"
        avg_cover = sum(r.confidence for r in cover_edges) / max(1, len(cover_edges))
        return avg_cover < 0.45 or vis == "visible"

    @classmethod
    def normalize_garment_semantics(cls, scene_graph: SceneGraph, person_id: str, region_type: str, raw_label: str | None = None) -> GarmentSemanticProfile:
        """Нормализует сырой garment output в property-driven профиль.

        Важно: token hints здесь только слабый сигнал, основной приоритет у graph/coverage/attachment.
        """
        body_regions = {"face", "head", "torso", "left_arm", "right_arm", "arm", "legs", "pelvis", "sleeves"}
        if region_type in body_regions:
            return GarmentSemanticProfile(
                entity_class="body",
                raw_label=region_type,
                layer_role="unknown",
                coverage_targets=[region_type],
                attachment_targets=[region_type],
                front_openable=None,
                removable=False,
                sleeve_presence="unknown",
                sleeve_length_hint="unknown",
                fit_hint="unknown",
                deformation_mode="body_driven",
                occlusion_priority=0.4,
                exposure_behavior="stable",
                semantic_confidence=0.6,
            )

        person = cls._person(scene_graph, person_id)
        if person is None:
            return GarmentSemanticProfile(raw_label=raw_label or "unknown", semantic_confidence=0.0)

        garment = cls._best_garment_for_region(person.garments, region_type)
        label = (raw_label or (garment.garment_type if garment else "unknown")).lower()
        relation_signals = cls._garment_relation_signals(scene_graph, person_id, garment, region_type)

        coverage_targets = list((garment.coverage_targets if garment else []) or cls._default_coverage(region_type))
        attachment_targets = list((garment.attachment_targets if garment else []) or cls._default_attachment(region_type))

        layer_role = cls._infer_layer_role(label, region_type, coverage_targets, attachment_targets, relation_signals)
        front_openable = cls._infer_front_openable(garment, relation_signals)
        removable = cls._infer_removable(garment, relation_signals)
        sleeve_presence, sleeve_length_hint = cls._infer_sleeve_signals(coverage_targets)
        fit_hint = cls._infer_fit_hint(label, garment, relation_signals)
        deformation_mode = cls._infer_deformation_mode(layer_role, fit_hint)

        exposure_behavior = "revealing" if relation_signals["opening_score"] >= 0.25 else "stable"
        occlusion_priority = cls._infer_occlusion_priority(layer_role, coverage_targets, relation_signals)
        confidence = cls._semantic_confidence(garment, relation_signals, coverage_targets, attachment_targets, layer_role)

        return GarmentSemanticProfile(
            raw_label=label,
            layer_role=layer_role,
            coverage_targets=sorted(set(coverage_targets)),
            attachment_targets=sorted(set(attachment_targets)),
            front_openable=front_openable,
            removable=removable,
            sleeve_presence=sleeve_presence,
            sleeve_length_hint=sleeve_length_hint,
            fit_hint=fit_hint,
            deformation_mode=deformation_mode,
            occlusion_priority=occlusion_priority,
            exposure_behavior=exposure_behavior,
            semantic_confidence=confidence,
        )

    @staticmethod
    def _best_garment_for_region(garments: list[GarmentNode], region_type: str) -> GarmentNode | None:
        region_map = {
            "torso": {"torso", "chest"},
            "sleeves": {"left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "arms"},
            "pelvis": {"pelvis", "hips"},
            "legs": {"legs", "left_leg", "right_leg"},
        }
        expected = region_map.get(region_type, {region_type})
        scored: list[tuple[float, GarmentNode]] = []
        for garment in garments:
            cov = set(garment.coverage_targets)
            att = set(garment.attachment_targets)
            overlap = len(cov.intersection(expected))
            attach_bonus = 0.3 if att.intersection({"torso", "pelvis", "arms", "legs"}) else 0.0
            vis_bonus = 0.15 if garment.visibility in {"visible", "partially_visible"} else 0.0
            score = overlap + attach_bonus + vis_bonus + garment.confidence
            scored.append((score, garment))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored and scored[0][0] > 0 else (garments[0] if garments else None)

    @classmethod
    def _garment_relation_signals(cls, scene_graph: SceneGraph, person_id: str, garment: GarmentNode | None, region_type: str) -> dict[str, float]:
        if garment is None:
            return {"cover_score": 0.0, "attach_score": 0.0, "opening_score": 0.0, "occlusion_score": 0.0}
        targets = {f"{person_id}_{region_type}", f"{person_id}:{region_type}", person_id}
        cover_scores = [r.confidence for r in scene_graph.relations if r.source == garment.garment_id and r.relation in {"covers", "occludes"} and any(t in r.target for t in targets)]
        attach_scores = [r.confidence for r in scene_graph.relations if r.source == garment.garment_id and r.relation == "attached_to" and (r.target == person_id or r.target.startswith(person_id))]
        opening_score = 0.0
        if garment.garment_state in {"opening", "removed", "open"}:
            opening_score += 0.5
        if garment.visibility in {"hidden", "unknown"}:
            opening_score += 0.15
        return {
            "cover_score": sum(cover_scores) / max(1, len(cover_scores)),
            "attach_score": sum(attach_scores) / max(1, len(attach_scores)),
            "opening_score": min(1.0, opening_score),
            "occlusion_score": max(cover_scores) if cover_scores else 0.0,
        }

    @classmethod
    def _infer_layer_role(
        cls,
        label: str,
        region_type: str,
        coverage_targets: list[str],
        attachment_targets: list[str],
        relation_signals: dict[str, float],
    ) -> str:
        coverage = set(coverage_targets)
        attachment = set(attachment_targets)
        if coverage.intersection({"legs", "left_leg", "right_leg"}) and "torso" not in coverage:
            return "lower_wear"
        if coverage.intersection({"foot", "left_foot", "right_foot"}):
            return "footwear"
        if coverage.intersection({"torso", "chest"}) and relation_signals.get("occlusion_score", 0.0) >= 0.45:
            return "outerwear"
        if coverage.intersection({"torso", "chest"}) and attachment.intersection({"torso", "arms"}):
            return "inner_upper"
        if region_type in {"sleeves", "torso"} and relation_signals.get("attach_score", 0.0) > 0.3:
            return "outerwear"
        for role, tokens in cls._LAYER_TOKENS.items():
            if any(token in label for token in tokens):
                return role
        return "unknown"

    @staticmethod
    def _infer_front_openable(garment: GarmentNode | None, relation_signals: dict[str, float]) -> bool | None:
        if garment is None:
            return None
        if garment.garment_state in {"open", "opening", "removed"}:
            return True
        if relation_signals.get("cover_score", 0.0) > 0.35 and relation_signals.get("attach_score", 0.0) > 0.2:
            return True
        return None

    @staticmethod
    def _infer_removable(garment: GarmentNode | None, relation_signals: dict[str, float]) -> bool | None:
        if garment is None:
            return None
        if garment.garment_state in {"removed", "opening", "worn", "open"}:
            return True
        return relation_signals.get("attach_score", 0.0) < 0.85

    @staticmethod
    def _infer_sleeve_signals(coverage_targets: list[str]) -> tuple[str, str]:
        coverage = set(coverage_targets)
        if coverage.intersection({"left_upper_arm", "right_upper_arm", "arms"}):
            length = "long" if coverage.intersection({"left_lower_arm", "right_lower_arm"}) else "short"
            return "present", length
        return "unknown", "unknown"

    @staticmethod
    def _infer_fit_hint(label: str, garment: GarmentNode | None, relation_signals: dict[str, float]) -> str:
        if "tight" in label or "fit" in label:
            return "fitted"
        if "loose" in label or "oversize" in label:
            return "relaxed"
        if garment and garment.garment_state == "opening":
            return "relaxed"
        if relation_signals.get("attach_score", 0.0) > 0.6:
            return "fitted"
        return "unknown"

    @staticmethod
    def _infer_deformation_mode(layer_role: str, fit_hint: str) -> str:
        if layer_role in {"outerwear", "inner_upper", "lower_wear"}:
            return "cloth_dynamic" if fit_hint in {"relaxed", "unknown"} else "cloth_tensioned"
        if layer_role == "accessory":
            return "semi_rigid"
        return "unknown"

    @staticmethod
    def _infer_occlusion_priority(layer_role: str, coverage_targets: list[str], relation_signals: dict[str, float]) -> float:
        base = 0.4
        if layer_role == "outerwear":
            base = 0.82
        elif layer_role in {"inner_upper", "lower_wear"}:
            base = 0.62
        elif layer_role == "accessory":
            base = 0.45
        coverage_bonus = 0.04 * min(3, len(set(coverage_targets)))
        relation_bonus = 0.1 * relation_signals.get("occlusion_score", 0.0)
        return max(0.1, min(0.95, base + coverage_bonus + relation_bonus))

    @staticmethod
    def _semantic_confidence(
        garment: GarmentNode | None,
        relation_signals: dict[str, float],
        coverage_targets: list[str],
        attachment_targets: list[str],
        layer_role: str,
    ) -> float:
        base = garment.confidence if garment else 0.2
        graph_bonus = 0.2 * relation_signals.get("cover_score", 0.0) + 0.2 * relation_signals.get("attach_score", 0.0)
        structure_bonus = 0.08 if coverage_targets else 0.0
        structure_bonus += 0.08 if attachment_targets else 0.0
        role_bonus = 0.08 if layer_role != "unknown" else 0.0
        return max(0.0, min(1.0, base * 0.6 + graph_bonus + structure_bonus + role_bonus))

    @staticmethod
    def _default_coverage(region_type: str) -> list[str]:
        defaults = {
            "torso": ["torso"],
            "sleeves": ["left_upper_arm", "right_upper_arm"],
            "pelvis": ["pelvis"],
            "legs": ["legs"],
        }
        return defaults.get(region_type, [region_type])

    @staticmethod
    def _default_attachment(region_type: str) -> list[str]:
        defaults = {
            "torso": ["torso"],
            "sleeves": ["arms", "torso"],
            "pelvis": ["pelvis"],
            "legs": ["pelvis", "legs"],
        }
        return defaults.get(region_type, ["torso"])
