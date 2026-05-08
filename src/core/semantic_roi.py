from __future__ import annotations

from core.region_ids import make_region_id
from core.schema import BBox, RegionRef, SceneGraph
from perception.mask_store import DEFAULT_MASK_STORE
from representation.scene_graph_queries import SceneGraphQueries


class SemanticROIHelper:
    def _bbox_from_keypoints(self, keypoints: list, margin: float = 0.03) -> BBox | None:
        pts = [(k.x, k.y) for k in keypoints if getattr(k, "confidence", 0.0) > 0.2]
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0.0, min(xs) - margin)
        y0 = max(0.0, min(ys) - margin)
        x1 = min(1.0, max(xs) + margin)
        y1 = min(1.0, max(ys) + margin)
        return self.normalize_bbox(BBox(x0, y0, max(0.05, x1 - x0), max(0.05, y1 - y0)))

    def normalize_bbox(self, bbox: BBox) -> BBox:
        x = max(0.0, min(1.0, bbox.x))
        y = max(0.0, min(1.0, bbox.y))
        w = max(0.02, min(1.0 - x, bbox.w))
        h = max(0.02, min(1.0 - y, bbox.h))
        return BBox(x, y, w, h)

    def expand_with_context(self, bbox: BBox, context: float = 0.04) -> BBox:
        return self.normalize_bbox(BBox(bbox.x - context, bbox.y - context, bbox.w + context * 2.0, bbox.h + context * 2.0))

    def fallback_person_bbox(self, person_bbox: BBox, region_type: str) -> BBox:
        if region_type == "head" or "face" in region_type:
            return BBox(person_bbox.x + person_bbox.w * 0.3, person_bbox.y, person_bbox.w * 0.4, person_bbox.h * 0.22)
        if "upper_arm" in region_type or "lower_arm" in region_type or "hand" in region_type or "arm" in region_type:
            return BBox(person_bbox.x, person_bbox.y + person_bbox.h * 0.2, person_bbox.w, person_bbox.h * 0.3)
        if "pelvis" in region_type:
            return BBox(person_bbox.x + person_bbox.w * 0.3, person_bbox.y + person_bbox.h * 0.5, person_bbox.w * 0.4, person_bbox.h * 0.2)
        if "leg" in region_type or "foot" in region_type:
            return BBox(person_bbox.x + person_bbox.w * 0.15, person_bbox.y + person_bbox.h * 0.55, person_bbox.w * 0.7, person_bbox.h * 0.42)
        return BBox(person_bbox.x + person_bbox.w * 0.1, person_bbox.y + person_bbox.h * 0.18, person_bbox.w * 0.8, person_bbox.h * 0.62)


    @staticmethod
    def _bbox_xyxy_from_payload(payload: object) -> tuple[float, float, float, float] | None:
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        if not isinstance(payload, list) or not payload:
            return None
        ys: list[int] = []
        xs: list[int] = []
        for y, row in enumerate(payload):
            if not isinstance(row, list):
                continue
            for x, value in enumerate(row):
                try:
                    active = float(value[0] if isinstance(value, list) else value) > 0
                except Exception:
                    active = False
                if active:
                    ys.append(y)
                    xs.append(x)
        if not xs or not ys:
            return None
        height = max(1, len(payload))
        width = max(1, len(payload[0]) if isinstance(payload[0], list) else 1)
        return (min(xs) / width, min(ys) / height, (max(xs) + 1) / width, (max(ys) + 1) / height)

    def _bbox_from_mask_ref(self, mask_ref: str | None) -> BBox | None:
        if not mask_ref:
            return None
        stored = DEFAULT_MASK_STORE.get(mask_ref)
        if stored is None:
            return None
        bbox = stored.extra.get("bbox_xyxy") if isinstance(stored.extra, dict) else None
        if not (isinstance(bbox, tuple) or isinstance(bbox, list)) or len(bbox) != 4:
            bbox = self._bbox_xyxy_from_payload(stored.payload)
        if not (isinstance(bbox, tuple) or isinstance(bbox, list)) or len(bbox) != 4:
            roi = stored.roi_bbox
            if roi is None:
                return None
            x, y, w, h = roi
            return self.normalize_bbox(BBox(float(x), float(y), float(w), float(h)))
        x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in bbox]
        return self.normalize_bbox(BBox(x1, y1, max(0.02, x2 - x1), max(0.02, y2 - y1)))

    @staticmethod
    def _mask_reason(prefix: str, node: object) -> str:
        return ";".join(
            [
                prefix,
                "roi_source=parser_mask_bbox",
                f"mask_ref={getattr(node, 'mask_ref', '')}",
                f"source={getattr(node, 'source', '')}",
                f"confidence={getattr(node, 'confidence', 0.0)}",
            ]
        )

    def region_from_graph(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        person = SceneGraphQueries._person(scene_graph, entity_id)
        if person is None:
            return None

        part_aliases = {
            "face": ["face"],
            "head": ["face", "head"],
            "hair": ["hair"],
            "neck": ["neck", "head"],
            "torso": ["torso", "upper_body"],
            "arms": ["left_arm", "right_arm", "arms", "upper_body"],
            "left_arm": ["left_arm", "arms"],
            "right_arm": ["right_arm", "arms"],
            "hands": ["left_hand", "right_hand", "hands"],
            "left_hand": ["left_hand", "hands"],
            "right_hand": ["right_hand", "hands"],
            "legs": ["left_leg", "right_leg", "legs"],
        }
        for alias in part_aliases.get(region_type, [region_type]):
            part_node = SceneGraphQueries.get_body_part(scene_graph, entity_id, alias)
            bbox = self._bbox_from_mask_ref(part_node.mask_ref if part_node else None)
            if bbox is not None and part_node is not None:
                return RegionRef(make_region_id(entity_id, region_type), self.expand_with_context(bbox, 0.015), self._mask_reason("graph_semantic:body_part_mask", part_node))

        garment_aliases = {
            "garments": ["dress", "outer_garment", "upper_garment", "lower_garment", "inner_garment"],
            "dress": ["dress", "upper_garment", "outer_garment"],
            "outer_garment": ["outer_garment", "upper_garment", "dress"],
            "upper_garment": ["upper_garment", "dress"],
            "lower_garment": ["lower_garment"],
            "inner_garment": ["inner_garment", "upper_garment"],
        }
        for wanted in garment_aliases.get(region_type, []):
            garment = next((g for g in person.garments if g.garment_type == wanted and g.mask_ref), None)
            bbox = self._bbox_from_mask_ref(garment.mask_ref if garment else None)
            if bbox is not None and garment is not None:
                return RegionRef(make_region_id(entity_id, region_type), self.expand_with_context(bbox, 0.02), self._mask_reason("graph_semantic:garment_mask", garment))

        if region_type in {"face", "head"}:
            part = SceneGraphQueries.get_body_part(scene_graph, entity_id, "head")
            if part and part.keypoints:
                bbox = self._bbox_from_keypoints(part.keypoints)
                if bbox:
                    return RegionRef(make_region_id(entity_id, region_type), self.expand_with_context(bbox, 0.015), "graph_semantic:body_part_keypoints")

        if region_type in {"arms", "left_arm", "right_arm", "left_upper_arm", "right_upper_arm", "sleeves", "hands"}:
            arm_keys = {"left_arm": {"left_upper_arm", "left_lower_arm", "left_hand"}, "right_arm": {"right_upper_arm", "right_lower_arm", "right_hand"}}
            wanted = arm_keys.get(region_type, {region_type})
            kp = [k for bp in person.body_parts if bp.part_type in wanted for k in bp.keypoints]
            if kp:
                bbox = self._bbox_from_keypoints(kp)
                if bbox:
                    return RegionRef(make_region_id(entity_id, region_type), self.expand_with_context(bbox, 0.02), "graph_semantic:arm_keypoints")

        if region_type in {"pelvis", "legs", "left_lower_leg", "right_lower_leg"}:
            wanted = {"pelvis"} if region_type == "pelvis" else {"left_upper_leg", "left_lower_leg", "right_upper_leg", "right_lower_leg", "left_foot", "right_foot"}
            kp = [k for bp in person.body_parts if bp.part_type in wanted for k in bp.keypoints]
            if kp:
                bbox = self._bbox_from_keypoints(kp)
                if bbox:
                    return RegionRef(make_region_id(entity_id, region_type), self.expand_with_context(bbox, 0.02), "graph_semantic:lower_body_keypoints")

        if region_type in {"garments", "torso"} and person.garments:
            covered = {t for g in person.garments for t in g.coverage_targets}
            torso_related = any("torso" in t or "upper_arm" in t for t in covered)
            if torso_related:
                bbox = BBox(person.bbox.x + person.bbox.w * 0.15, person.bbox.y + person.bbox.h * 0.18, person.bbox.w * 0.7, person.bbox.h * 0.42)
                return RegionRef(make_region_id(entity_id, region_type), self.normalize_bbox(bbox), "graph_semantic:garment_coverage")
        return None

    def resolve_region(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        semantic = self.region_from_graph(scene_graph, entity_id, region_type)
        if semantic:
            return semantic
        person = SceneGraphQueries._person(scene_graph, entity_id)
        if person is None:
            return None
        fallback = self.normalize_bbox(self.fallback_person_bbox(person.bbox, region_type))
        return RegionRef(make_region_id(entity_id, region_type), fallback, "roi_source=person_bbox_fallback;fallback:person_bbox_template")
