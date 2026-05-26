from __future__ import annotations

from dataclasses import asdict
import math
import random

import numpy as np

from core.semantic_roi import SemanticROIHelper
from core.region_ids import make_region_id, parse_region_id
from core.reference_families import (
    ACCESSORY_REFERENCE_REGIONS as SHARED_ACCESSORY_REFERENCE_REGIONS,
    BODY_SHAPE_REFERENCE_REGIONS as SHARED_BODY_SHAPE_REFERENCE_REGIONS,
    CORE_IDENTITY_REGIONS as SHARED_CORE_IDENTITY_REGIONS,
    GARMENT_REFERENCE_REGIONS as SHARED_GARMENT_REFERENCE_REGIONS,
    SKIN_REFERENCE_REGIONS as SHARED_SKIN_REFERENCE_REGIONS,
    reference_kind_for_region,
)
from core.schema import (
    BBox,
    CanonicalRegionMemoryEntry,
    GarmentSemanticProfile,
    HiddenRegionSlot,
    MemoryEntry,
    RegionDescriptor,
    ReferencePatchMaterial,
    ReferencePatchPayload,
    SceneGraph,
    TexturePatchMemory,
    VideoMemory,
    RegionMemoryBundle,
)
from utils_tensor import crop, mean_color, shape


class MemoryManager:
    CORE_IDENTITY_REGIONS = SHARED_CORE_IDENTITY_REGIONS
    IDENTITY_SENSITIVE_REGIONS = SHARED_CORE_IDENTITY_REGIONS
    SKIN_REFERENCE_REGIONS = SHARED_SKIN_REFERENCE_REGIONS
    BODY_SHAPE_REGIONS = SHARED_BODY_SHAPE_REFERENCE_REGIONS
    GARMENT_REFERENCE_REGIONS = SHARED_GARMENT_REFERENCE_REGIONS
    ACCESSORY_REFERENCE_REGIONS = SHARED_ACCESSORY_REFERENCE_REGIONS
    APPEARANCE_SENSITIVE_REGIONS = (
        SHARED_CORE_IDENTITY_REGIONS
        | SHARED_BODY_SHAPE_REFERENCE_REGIONS
        | SHARED_SKIN_REFERENCE_REGIONS
        | SHARED_GARMENT_REFERENCE_REGIONS
        | SHARED_ACCESSORY_REFERENCE_REGIONS
    )
    GARMENT_SENSITIVE_REGIONS = SHARED_GARMENT_REFERENCE_REGIONS
    REFERENCE_THRESHOLDS = {
        "identity_reference": (0.70, 0.65),
        "skin_reference": (0.66, 0.62),
        "body_shape_reference": (0.62, 0.60),
        "garment_reference": (0.60, 0.58),
        "accessory_reference": (0.62, 0.60),
        "appearance_reference": (0.58, 0.56),
    }

    _CANONICAL_MEMORY_REGIONS = (
        "face",
        "hair",
        "head",
        "neck",
        "torso",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "pelvis",
        "left_leg",
        "right_leg",
        "upper_garment",
        "lower_garment",
        "outer_garment",
        "inner_garment",
        "accessories",
    )
    _SEMANTIC_REGIONS = ("face", "torso", "sleeves", "garments", "left_arm", "right_arm", "pelvis", "legs")
    _PROVENANCE_RELIABILITY = {
        "parser": 0.9,
        "segformer": 0.88,
        "schp": 0.84,
        "fashn": 0.8,
        "face": 0.78,
        "canonical_reasoner": 0.72,
        "visibility_reasoner": 0.7,
        "heuristic": 0.62,
        "frame_observation": 0.82,
        "graph_delta_state_update": 0.38,
        "generated": 0.35,
        "unknown": 0.5,
    }

    def __init__(self) -> None:
        self.roi = SemanticROIHelper()

    def _is_core_identity_region(self, region_type: str) -> bool:
        canonical = self._canonical_from_region_type(region_type) or region_type
        return canonical in self.CORE_IDENTITY_REGIONS

    def _is_identity_sensitive_region(self, region_type: str) -> bool:
        return self._is_core_identity_region(region_type)

    def _is_skin_reference_region(self, region_type: str) -> bool:
        canonical = self._canonical_from_region_type(region_type) or region_type
        return canonical in self.SKIN_REFERENCE_REGIONS and canonical not in self.CORE_IDENTITY_REGIONS

    def _is_body_shape_region(self, region_type: str) -> bool:
        canonical = self._canonical_from_region_type(region_type) or region_type
        return canonical in self.BODY_SHAPE_REGIONS

    def _is_garment_reference_region(self, region_type: str) -> bool:
        canonical = self._canonical_from_region_type(region_type) or region_type
        return canonical in self.GARMENT_REFERENCE_REGIONS

    def _is_accessory_reference_region(self, region_type: str) -> bool:
        canonical = self._canonical_from_region_type(region_type) or region_type
        return canonical in self.ACCESSORY_REFERENCE_REGIONS

    def _source_indicates_generated(self, *values: object) -> bool:
        generated_markers = ("generated", "synthetic", "renderer", "rendered", "model_output")
        for value in values:
            if isinstance(value, bool):
                if value:
                    return True
                continue
            text = str(value or "").strip().lower()
            if text and any(marker in text for marker in generated_markers):
                return True
        return False

    def _evidence_source_rank(
        self,
        *,
        observed_directly: bool,
        generated: bool,
        inferred: bool,
        evidence_score: float,
        confidence: float,
    ) -> str:
        if generated:
            return "generated"
        if inferred or not observed_directly:
            return "inferred"
        if evidence_score >= 0.7 and confidence >= 0.65:
            return "strong_observed"
        return "weak_observed"

    def _can_update_identity_reference(
        self,
        *,
        region_type: str,
        observed_directly: bool,
        generated: bool,
        inferred: bool,
        evidence_score: float,
        confidence: float,
        reveal_lifecycle: str = "unknown",
    ) -> bool:
        if not self._is_core_identity_region(region_type):
            return False
        if reveal_lifecycle in {"newly_occluded", "currently_hidden", "expected_unknown"}:
            return False
        if reveal_lifecycle == "newly_revealed" and evidence_score < 0.7:
            return False
        return self._evidence_source_rank(
            observed_directly=observed_directly,
            generated=generated,
            inferred=inferred,
            evidence_score=evidence_score,
            confidence=confidence,
        ) == "strong_observed"

    def _reference_kind_for_region(self, region_type: str) -> str:
        canonical = self._canonical_from_region_type(region_type) or region_type
        kind = reference_kind_for_region(canonical)
        if kind != "none":
            return kind
        if canonical in self.APPEARANCE_SENSITIVE_REGIONS:
            return "appearance_reference"
        return "none"

    def _reference_thresholds(self, reference_kind: str) -> tuple[float, float]:
        return self.REFERENCE_THRESHOLDS.get(reference_kind, self.REFERENCE_THRESHOLDS["appearance_reference"])

    def _hidden_or_occluded_reference_state(self, entry: CanonicalRegionMemoryEntry) -> bool:
        return str(entry.visibility_state) in {
            "hidden",
            "hidden_by_self",
            "hidden_by_garment",
            "hidden_by_object",
            "out_of_frame",
        } or entry.reveal_lifecycle in {"newly_occluded", "currently_hidden", "expected_unknown"}

    def _can_update_strong_reference(self, entry: CanonicalRegionMemoryEntry, reference_kind: str) -> bool:
        if reference_kind == "none":
            return False
        evidence_threshold, confidence_threshold = self._reference_thresholds(reference_kind)
        if self._hidden_or_occluded_reference_state(entry):
            return False
        if entry.reveal_lifecycle == "newly_revealed" and entry.evidence_score < evidence_threshold:
            return False
        return bool(
            entry.observed_directly
            and not entry.generated
            and not entry.inferred
            and entry.evidence_score >= evidence_threshold
            and entry.confidence >= confidence_threshold
        )

    def _reference_block_reasons(self, entry: CanonicalRegionMemoryEntry, reference_family: str) -> list[str]:
        reasons: list[str] = []
        reference_kind = f"{reference_family}_reference" if not reference_family.endswith("_reference") else reference_family
        evidence_threshold, confidence_threshold = self._reference_thresholds(reference_kind)
        prefix = reference_kind
        if entry.generated:
            reasons.append(f"{prefix}_blocked_generated")
        if entry.inferred or not entry.observed_directly:
            reasons.append(f"{prefix}_blocked_inferred")
        low_evidence = (
            entry.evidence_score < evidence_threshold
            or entry.confidence < confidence_threshold
            or (entry.reveal_lifecycle == "newly_revealed" and entry.evidence_score < evidence_threshold)
            or self._hidden_or_occluded_reference_state(entry)
        )
        if low_evidence:
            reasons.append(f"{prefix}_blocked_low_evidence")
        return reasons

    def _is_body_like_region(self, region_type: str) -> bool:
        return region_type in {
            "face",
            "head",
            "torso",
            "pelvis",
            "legs",
            "left_arm",
            "right_arm",
            "arm",
            "left_hand",
            "right_hand",
            "hand",
        }

    def _is_limb_region(self, region_type: str) -> bool:
        return region_type in {
            "legs",
            "pelvis",
            "left_arm",
            "right_arm",
            "arm",
            "left_hand",
            "right_hand",
            "hand",
        }

    def _candidate_body_compatibility(self, query_region: str, candidate_region: str, semantic_family: str) -> float:
        """
        Более жёсткая совместимость для body-like retrieval.
        Если query — ноги/таз/руки, torso допустим только как слабый fallback,
        а garment-патчи должны почти всегда проигрывать.
        """
        if query_region == candidate_region:
            return 1.0

        if query_region in {"legs", "pelvis"}:
            if candidate_region in {"legs", "pelvis"}:
                return 0.9
            if candidate_region == "torso":
                return -0.55
            if semantic_family in {"garment", "outerwear", "innerwear"}:
                return -1.0
            if candidate_region in {"inner_garment", "outer_garment", "garments", "sleeves"}:
                return -1.0
            return -0.35

        if query_region in {"left_arm", "right_arm", "arm", "left_hand", "right_hand", "hand"}:
            if candidate_region in {"left_arm", "right_arm", "arm", "left_hand", "right_hand", "hand"}:
                return 0.9
            if candidate_region == "torso":
                return -0.45
            if semantic_family in {"garment", "outerwear", "innerwear"}:
                return -1.0
            if candidate_region in {"inner_garment", "outer_garment", "garments", "sleeves"}:
                return -0.9
            return -0.3

        if query_region in {"face", "head"}:
            if candidate_region in {"face", "head"}:
                return 0.95
            return -0.8

        if query_region == "torso":
            if candidate_region == "torso":
                return 0.95
            if candidate_region in {"pelvis", "legs"}:
                return -0.35
            if semantic_family in {"garment", "outerwear", "innerwear"}:
                return -0.5
            return -0.2

        if self._is_body_like_region(query_region):
            if semantic_family in {"garment", "outerwear", "innerwear"}:
                return -0.8
            if candidate_region in {"inner_garment", "outer_garment", "garments"}:
                return -0.9

        return -0.15

    def _encode_visual(self, token: str, dim: int = 8) -> list[float]:
        seed = abs(hash(token)) % (2**32)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]

    def _descriptor_to_embedding(self, descriptor: dict[str, float | list[float]]) -> list[float]:
        mean = descriptor.get("mean", [0.0, 0.0, 0.0])
        std = descriptor.get("std", [0.0, 0.0, 0.0])
        edge = float(descriptor.get("edge_density", 0.0))
        energy = float(descriptor.get("energy", 0.0))
        raw = [float(mean[0]), float(mean[1]), float(mean[2]), float(std[0]), float(std[1]), float(std[2]), edge, energy]
        n = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / n for v in raw]

    def _descriptor_similarity(self, left: dict[str, float | list[float]] | None, right: dict[str, float | list[float]] | None) -> float:
        if not left or not right:
            return 0.0
        a = self._descriptor_to_embedding(left)
        b = self._descriptor_to_embedding(right)
        return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))

    def _bbox_to_pixels(self, bbox: BBox, frame: list) -> tuple[int, int, int, int]:
        h, w, _ = shape(frame)
        x0 = max(0, min(w - 1, int(bbox.x * w)))
        y0 = max(0, min(h - 1, int(bbox.y * h)))
        x1 = max(x0 + 1, min(w, int((bbox.x + bbox.w) * w)))
        y1 = max(y0 + 1, min(h, int((bbox.y + bbox.h) * h)))
        return x0, y0, x1, y1

    def _patch_descriptor(self, patch: list[list[list[float]]]) -> dict[str, float | list[float]]:
        h, w, _ = shape(patch)
        if h == 0 or w == 0:
            return {"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0], "hist": [0.0] * 12, "edge_density": 0.0, "energy": 0.0}
        n = float(h * w)
        means = mean_color(patch)
        var = [0.0, 0.0, 0.0]
        hist = [0.0] * 12
        edge = 0.0
        energy = 0.0
        for y in range(h):
            for x in range(w):
                px = patch[y][x]
                for k in range(3):
                    dv = px[k] - means[k]
                    var[k] += dv * dv
                    b = min(3, int(px[k] * 4.0))
                    hist[k * 4 + b] += 1.0
                energy += sum(ch * ch for ch in px) / 3.0
                if x + 1 < w:
                    npx = patch[y][x + 1]
                    edge += sum(abs(px[k] - npx[k]) for k in range(3)) / 3.0
                if y + 1 < h:
                    npx = patch[y + 1][x]
                    edge += sum(abs(px[k] - npx[k]) for k in range(3)) / 3.0
        std = [math.sqrt(v / n) for v in var]
        hist = [v / n for v in hist]
        return {"mean": [float(v) for v in means], "std": [float(v) for v in std], "hist": [float(v) for v in hist], "edge_density": float(edge / n), "energy": float(energy / n)}

    def initialize(self, scene_graph: SceneGraph) -> VideoMemory:
        return self.initialize_from_scene(scene_graph)

    def initialize_from_scene(self, scene_graph: SceneGraph) -> VideoMemory:
        memory = VideoMemory(temporal_history=[scene_graph])
        for person in scene_graph.persons:
            memory.identity_memory[person.person_id] = MemoryEntry(entity_id=person.person_id, entry_type="identity", embedding=self._encode_visual(f"identity:{person.person_id}"), confidence=person.confidence, last_seen_frames=[scene_graph.frame_index])
            self._seed_person_semantic_regions(memory, person.person_id, person.bbox, scene_graph.frame_index)
            self._update_canonical_region_memory(memory, person, scene_graph.frame_index)
            for garment in person.garments:
                memory.garment_memory[garment.garment_id] = MemoryEntry(entity_id=garment.garment_id, entry_type="garment", embedding=self._encode_visual(f"garment:{garment.garment_id}"), confidence=garment.confidence, last_seen_frames=[scene_graph.frame_index])
        return memory

    def update(self, memory: VideoMemory, scene_graph: SceneGraph) -> VideoMemory:
        return self.update_from_graph(memory, scene_graph)

    def update_from_graph(self, memory: VideoMemory, scene_graph: SceneGraph) -> VideoMemory:
        memory.temporal_history.append(scene_graph)
        for record in memory.canonical_region_memory.values():
            record.freshness_frames += 1
            self._refresh_reuse_policy(record)
        observed_entities: set[str] = set()
        visible_regions: set[str] = set()
        for person in scene_graph.persons:
            observed_entities.add(person.person_id)
            identity = memory.identity_memory.get(person.person_id)
            if identity is not None:
                self._refresh_entry(identity, scene_graph.frame_index)
            self._update_canonical_region_memory(memory, person, scene_graph.frame_index)
            for part in person.body_parts:
                region_id = make_region_id(person.person_id, part.part_type)
                if part.visibility in ("visible", "partially_visible"):
                    visible_regions.add(region_id)
                    mask_roi = self.roi.resolve_region(scene_graph, person.person_id, part.part_type)
                    memory.region_descriptors[region_id] = RegionDescriptor(region_id=region_id, entity_id=person.person_id, region_type=part.part_type, bbox=(mask_roi.bbox if mask_roi else person.bbox), visibility=part.visibility, confidence=part.confidence, last_update_frame=scene_graph.frame_index)
                else:
                    self.apply_visibility_event(memory, {"region_id": region_id, "entity": person.person_id}, {}, visibility="hidden", transition_reason="graph_visibility_hidden")
            for garment in person.garments:
                observed_entities.add(garment.garment_id)
                gid = make_region_id(person.person_id, "garments")
                if garment.visibility in ("visible", "partially_visible"):
                    visible_regions.add(gid)
                    mask_roi = self.roi.resolve_region(scene_graph, person.person_id, garment.garment_type or "garments")
                    if mask_roi is not None:
                        specific_gid = make_region_id(person.person_id, garment.garment_type)
                        memory.region_descriptors[specific_gid] = RegionDescriptor(region_id=specific_gid, entity_id=person.person_id, region_type=garment.garment_type, bbox=mask_roi.bbox, visibility=garment.visibility, confidence=garment.confidence, last_update_frame=scene_graph.frame_index)
                else:
                    self.apply_visibility_event(memory, {"region_id": gid, "entity": person.person_id}, {}, visibility="hidden", transition_reason="garment_hidden")

        for entry in list(memory.identity_memory.values()) + list(memory.garment_memory.values()):
            if entry.entity_id not in observed_entities:
                entry.confidence *= 0.96
        for region_id, descriptor in memory.region_descriptors.items():
            if region_id not in visible_regions:
                descriptor.confidence *= 0.95
                if descriptor.confidence < 0.3:
                    descriptor.visibility = "hidden"
        for region_id, slot in memory.hidden_region_slots.items():
            if region_id not in visible_regions:
                slot.stale_frames += 1
            else:
                slot.stale_frames = 0
            self._promote_or_decay_hidden_slot(slot)
        return memory

    def update_from_frame(
        self,
        memory: VideoMemory,
        frame: list,
        scene_graph: SceneGraph,
        transition_context: dict[str, str] | None = None,
    ) -> VideoMemory:
        context = transition_context or memory.last_transition_context or {}
        visibility_phase = context.get("visibility_phase", "stable")
        garment_phase = context.get("garment_phase", "worn")
        frame_generated = self._source_indicates_generated(
            context.get("frame_source"),
            context.get("update_source"),
            context.get("generated"),
            context.get("is_generated"),
        )
        source_frame_kind = str(context.get("source_frame_kind", "generated_runtime_frame" if frame_generated else "observed_input_frame"))
        source_is_input_frame = bool(context.get("source_is_input_frame", not frame_generated))
        immutable_i2v_anchor = bool(context.get("immutable_i2v_anchor", (not frame_generated and scene_graph.frame_index == 0)))
        for person in scene_graph.persons:
            for region_type in self._semantic_region_types(person):
                region_ref = self.roi.resolve_region(scene_graph, person.person_id, region_type)
                if region_ref is None:
                    continue
                bbox = region_ref.bbox
                x0, y0, x1, y1 = self._bbox_to_pixels(bbox, frame)
                patch = crop(frame, x0, y0, x1, y1)
                desc = self._patch_descriptor(patch)
                evidence = 0.45 + 0.3 * float(desc["edge_density"]) + 0.25 * float(sum(desc["std"]) / 3.0)
                rid = make_region_id(person.person_id, region_type)
                patch_id = f"patch::{rid}:{scene_graph.frame_index}"
                confidence_boost = self._memory_update_boost(region_type, context)
                memory.texture_patches[patch_id] = TexturePatchMemory(
                    patch_id=patch_id,
                    region_type=region_type,
                    entity_id=person.person_id,
                    source_frame=scene_graph.frame_index,
                    patch_ref=f"roi://{x0},{y0},{x1},{y1}",
                    confidence=min(1.0, 0.35 + person.confidence * 0.25 + evidence * confidence_boost),
                    descriptor=desc,
                    evidence_score=min(1.0, evidence),
                    semantic_family=self._semantic_family(region_type),
                    coverage_targets=self._default_coverage_targets(region_type),
                    attachment_targets=self._default_attachment_targets(region_type),
                    suitable_for_reveal=region_type in {"torso", "sleeves", "garments", "pelvis"},
                    rgb_patch=self._bounded_rgb_patch(patch),
                    observed_directly=not frame_generated,
                    generated=frame_generated,
                    inferred=frame_generated,
                    provenance="runtime_generated_frame" if frame_generated else "observed_input_frame",
                    source_frame_kind=source_frame_kind,
                    immutable_i2v_anchor=immutable_i2v_anchor,
                    source_is_input_frame=source_is_input_frame,
                )
                memory.patch_cache[patch_id] = patch

                descriptor = memory.region_descriptors.get(rid)
                if descriptor:
                    descriptor.last_update_frame = scene_graph.frame_index
                    descriptor.confidence = min(1.0, descriptor.confidence * (1.0 - 0.2 * confidence_boost) + 0.2 * evidence)
                patch_evidence = min(1.0, evidence)
                patch_confidence = min(1.0, 0.35 + person.confidence * 0.25 + evidence * confidence_boost)
                observed_from_frame = not frame_generated
                self._refresh_canonical_memory_from_descriptor(
                    memory=memory,
                    entity_id=person.person_id,
                    canonical_region=region_type,
                    source_frame=scene_graph.frame_index,
                    evidence_score=patch_evidence,
                    confidence=patch_confidence,
                    observed_directly=observed_from_frame,
                    generated=frame_generated,
                )

                identity = memory.identity_memory.get(person.person_id)
                can_update_identity_embedding = (
                    identity is not None
                    and self._is_core_identity_region(region_type)
                    and self._can_update_identity_reference(
                        region_type=region_type,
                        observed_directly=observed_from_frame,
                        generated=frame_generated,
                        inferred=not observed_from_frame,
                        evidence_score=patch_evidence,
                        confidence=patch_confidence,
                        reveal_lifecycle="currently_visible",
                    )
                )
                if can_update_identity_embedding:
                    emb = self._descriptor_to_embedding(desc)
                    blend = 0.18 if visibility_phase == "revealing" else 0.1
                    identity.embedding = [identity.embedding[i] * (1.0 - blend) + emb[i] * blend for i in range(min(len(identity.embedding), len(emb)))]
                for garment in person.garments:
                    garment_entry = memory.garment_memory.get(garment.garment_id)
                    if garment_entry and region_type in {"garments", "sleeves", "torso"}:
                        emb = self._descriptor_to_embedding(desc)
                        gblend = 0.32 if garment_phase in {"opening", "removed"} else 0.22
                        garment_entry.embedding = [garment_entry.embedding[i] * (1.0 - gblend) + emb[i] * gblend for i in range(min(len(garment_entry.embedding), len(emb)))]

                if region_type in {"sleeves", "garments", "legs"}:
                    slot = memory.hidden_region_slots.setdefault(rid, HiddenRegionSlot(slot_id=rid, region_type=region_type, owner_entity=person.person_id, candidate_patch_ids=[], hidden_type="known_hidden"))
                    slot.candidate_patch_ids = [patch_id] + [cid for cid in slot.candidate_patch_ids if cid != patch_id][:4]
                    slot.confidence = min(1.0, 0.45 + evidence * (0.35 + 0.2 * confidence_boost))
                    slot.stale_frames = 0
                    self.transition_hidden_slot(slot, "known_hidden", reason="visible_refresh")
                    slot.evidence_score = max(slot.evidence_score, min(1.0, evidence))
                    self._promote_or_decay_hidden_slot(slot)
        return memory

    def mark_region_revealed(self, memory: VideoMemory, region_id: str, owner_entity: str) -> None:
        self.apply_visibility_event(memory, {"region_id": region_id, "entity": owner_entity}, {}, visibility="revealed", transition_reason="renderer_reveal")

    def query_hidden_region(self, memory: VideoMemory, region_id: str) -> HiddenRegionSlot | None:
        entity, region_type = parse_region_id(region_id)
        return self.retrieve_hidden_region(memory, make_region_id(entity, region_type))

    def retrieve_for_region(self, memory: VideoMemory, region_type: str, owner_entity: str | None = None, query_descriptor: dict[str, float | list[float]] | None = None) -> list[TexturePatchMemory]:
        patches = [p for p in memory.texture_patches.values() if p.region_type == region_type]
        if owner_entity is not None:
            patches = [p for p in patches if p.entity_id == owner_entity]
        most_recent = max((p.source_frame for p in patches), default=0)
        return sorted(
            patches,
            key=lambda p: (
                1.0 if p.descriptor and "mean" in p.descriptor else 0.0,
                p.confidence + 0.2 * self._descriptor_similarity(query_descriptor, p.descriptor),
                p.evidence_score,
                1.0 / (1.0 + max(0, most_recent - p.source_frame)),
            ),
            reverse=True,
        )

    def retrieve_hidden_region(self, memory: VideoMemory, region_id: str, hidden_type: str | None = None) -> HiddenRegionSlot | None:
        slot = memory.hidden_region_slots.get(region_id)
        if slot is None:
            return None
        self._refresh_hidden_slot_priority(slot)
        if hidden_type is not None and slot.hidden_type != hidden_type:
            return None
        return slot

    def retrieve_by_entity(self, memory: VideoMemory, entity_id: str, query_descriptor: dict[str, float | list[float]] | None = None, top_k: int = 5) -> list[TexturePatchMemory]:
        patches = [p for p in memory.texture_patches.values() if p.entity_id == entity_id]
        return sorted(
            patches,
            key=lambda p: (p.confidence, p.evidence_score, self._descriptor_similarity(query_descriptor, p.descriptor), p.source_frame),
            reverse=True,
        )[:top_k]

    def apply_visibility_event(
        self,
        memory: VideoMemory,
        delta: dict[str, str],
        masks: dict[str, object],
        visibility: str,
        transition_reason: str = "visibility_event",
    ) -> None:
        _ = masks
        entity_id, region_type = parse_region_id(delta.get("region_id", "scene:unknown"))
        region_id = make_region_id(entity_id, region_type)
        owner = delta.get("entity", entity_id)
        slot = memory.hidden_region_slots.get(region_id)
        if slot is None:
            slot = HiddenRegionSlot(slot_id=region_id, region_type=region_type, owner_entity=owner, candidate_patch_ids=[], confidence=0.55, hidden_type="unknown_hidden")
            memory.hidden_region_slots[region_id] = slot
        if visibility == "revealed":
            slot.confidence = min(1.0, slot.confidence + 0.15)
            slot.stale_frames = 0
            target_state = "known_hidden" if slot.candidate_patch_ids else "unknown_hidden"
            self.transition_hidden_slot(slot, target_state, reason=f"revealed:{transition_reason}")
        elif visibility == "hidden":
            slot.confidence = max(0.1, slot.confidence - 0.12)
            slot.stale_frames += 1
            target_state = "unknown_hidden" if not slot.candidate_patch_ids else slot.hidden_type
            self.transition_hidden_slot(slot, target_state, reason=f"hidden:{transition_reason}")
        canonical_region = self._canonical_from_region_type(region_type)
        if canonical_region:
            record = memory.canonical_region_memory.get(make_region_id(owner, canonical_region))
            if record is not None:
                previous = str(record.visibility_state)
                if visibility == "revealed":
                    record.visibility_state = "visible"
                    record.reveal_lifecycle = "newly_revealed" if previous not in {"visible", "partially_visible"} else "visible"
                    record.last_transition = "hidden_to_visible"
                    record.freshness_frames = 0
                elif visibility == "hidden":
                    record.visibility_state = "hidden"
                    record.reveal_lifecycle = "newly_occluded" if previous in {"visible", "partially_visible"} else "currently_hidden"
                    record.last_transition = "visible_to_hidden"
                self._refresh_reuse_policy(record)
        self._promote_or_decay_hidden_slot(slot)

    def retrieve(self, memory: VideoMemory, query_embedding: list[float], bank: str = "texture", top_k: int = 3) -> list[dict[str, object]]:
        qn = math.sqrt(sum(v * v for v in query_embedding)) or 1.0
        q = [v / qn for v in query_embedding]
        if bank == "identity":
            entries = list(memory.identity_memory.values())
        elif bank == "garment":
            entries = list(memory.garment_memory.values())
        else:
            entries = [MemoryEntry(entity_id=v.patch_id, entry_type="texture", embedding=self._descriptor_to_embedding(v.descriptor) if v.descriptor else self._encode_visual(v.patch_id), confidence=v.confidence) for v in memory.texture_patches.values()]

        scored: list[tuple[float, MemoryEntry]] = []
        for e in entries:
            en = math.sqrt(sum(v * v for v in e.embedding)) or 1.0
            emb = [v / en for v in e.embedding]
            sim = sum(a * b for a, b in zip(q, emb))
            scored.append((float(sim), e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"entity_id": e.entity_id, "similarity": round(sim, 4), "confidence": round((sim + e.confidence) / 2.0, 4)} for sim, e in scored[:top_k]]

    def query_by_region(self, memory: VideoMemory, region_type: str) -> list[RegionDescriptor]:
        return [descriptor for descriptor in memory.region_descriptors.values() if descriptor.region_type == region_type]

    def query_by_entity(self, memory: VideoMemory, entity_id: str) -> dict[str, object]:
        regions = [descriptor for descriptor in memory.region_descriptors.values() if descriptor.entity_id == entity_id]
        patches = [patch for patch in memory.texture_patches.values() if patch.entity_id == entity_id]
        return {"identity": memory.identity_memory.get(entity_id), "garment": memory.garment_memory.get(entity_id), "regions": regions, "patches": patches}

    def query_hidden_candidate(self, memory: VideoMemory, region_type: str) -> TexturePatchMemory | None:
        candidates = [p for p in memory.texture_patches.values() if p.region_type == region_type]
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.confidence, c.evidence_score))

    def get_region_memory_entries(
        self,
        memory: VideoMemory,
        entity_id: str | None = None,
        canonical_region: str | None = None,
    ) -> list[CanonicalRegionMemoryEntry]:
        entries = list(memory.canonical_region_memory.values())
        if entity_id is not None:
            entries = [e for e in entries if e.entity_id == entity_id]
        if canonical_region is not None:
            entries = [e for e in entries if e.canonical_region == canonical_region]
        return sorted(entries, key=lambda e: (e.entity_id, e.canonical_region, e.source_frame))

    def get_best_region_memory(
        self,
        memory: VideoMemory,
        entity_id: str,
        canonical_region: str,
        *,
        for_reveal: bool = False,
    ) -> CanonicalRegionMemoryEntry | None:
        rid = make_region_id(entity_id, canonical_region)
        entry = memory.canonical_region_memory.get(rid)
        if entry is None:
            return None
        if for_reveal and not entry.suitable_for_reveal:
            return None
        return entry

    def get_region_memory_bundle(
        self,
        memory: VideoMemory,
        entity_id: str,
        canonical_region: str,
    ) -> RegionMemoryBundle:
        region_id = make_region_id(entity_id, canonical_region)
        entry = memory.canonical_region_memory.get(region_id)
        hidden_slot = memory.hidden_region_slots.get(region_id)

        current_reuse = entry if entry and entry.reliable_for_reuse else None
        identity_reference = entry if entry and entry.reliable_as_reference and entry.reference_kind == "identity_reference" else None
        skin_reference = entry if entry and entry.reliable_as_reference and entry.reference_kind == "skin_reference" else None
        body_shape_reference = entry if entry and entry.reliable_as_reference and entry.reference_kind == "body_shape_reference" else None
        appearance_reference = entry if entry and entry.reliable_as_reference and entry.reference_kind == "appearance_reference" else None
        garment_reference = entry if entry and entry.reliable_as_reference and entry.reference_kind == "garment_reference" else None
        accessory_reference = entry if entry and entry.reliable_as_reference and entry.reference_kind == "accessory_reference" else None
        reasons: list[str] = []
        if current_reuse is not None:
            reasons.append("current_reuse_reliable")
        if identity_reference is not None:
            reasons.append("identity_reference_available")
            if self._can_update_identity_reference(
                region_type=entry.canonical_region,
                observed_directly=entry.observed_directly,
                generated=entry.generated,
                inferred=entry.inferred,
                evidence_score=entry.evidence_score,
                confidence=entry.confidence,
                reveal_lifecycle=entry.reveal_lifecycle,
            ):
                reasons.append("identity_reference_observed_strong")
        if skin_reference is not None:
            reasons.append("skin_reference_available")
            if self._can_update_strong_reference(skin_reference, "skin_reference"):
                reasons.append("skin_reference_observed_strong")
        if body_shape_reference is not None:
            reasons.append("body_shape_reference_available")
            if self._can_update_strong_reference(body_shape_reference, "body_shape_reference"):
                reasons.append("body_shape_reference_observed_strong")
        if appearance_reference is not None:
            reasons.append("appearance_reference_available")
        if garment_reference is not None:
            reasons.append("garment_reference_available")
            if self._can_update_strong_reference(garment_reference, "garment_reference"):
                reasons.append("garment_reference_observed_strong")
        if accessory_reference is not None:
            reasons.append("accessory_reference_available")
            if self._can_update_strong_reference(accessory_reference, "accessory_reference"):
                reasons.append("accessory_reference_observed_strong")

        hidden_slot_is_trace_history = bool(hidden_slot and hidden_slot.hidden_type in {"revealed", "revealed_history"})
        active_hidden_support = bool(
            hidden_slot
            and hidden_slot.hidden_type not in {"revealed", "revealed_history"}
            and bool(hidden_slot.candidate_patch_ids)
        )
        if hidden_slot is not None:
            reasons.append(f"hidden_slot:{hidden_slot.hidden_type}")
            if hidden_slot_is_trace_history:
                reasons.append("revealed_history_trace")
            elif active_hidden_support:
                reasons.append("active_hidden_slot_candidates")
        if entry is not None:
            if self._is_core_identity_region(entry.canonical_region) and identity_reference is None:
                reasons.extend(self._reference_block_reasons(entry, "identity"))
            for family, present in (
                ("skin", skin_reference is not None),
                ("body_shape", body_shape_reference is not None),
                ("garment", garment_reference is not None),
                ("accessory", accessory_reference is not None),
            ):
                expected_kind = f"{family}_reference"
                if entry.reference_kind == expected_kind and not present:
                    reasons.extend(self._reference_block_reasons(entry, family))
            reasons.append(f"lifecycle:{entry.reveal_lifecycle}")
            if entry.reveal_lifecycle == "newly_occluded":
                reasons.append("occluded_state")
            if entry.reliable_as_reference:
                reasons.append("reference_reliable")
            elif entry.reference_kind != "none":
                reasons.append("reference_unreliable")
            if not entry.reliable_for_reuse:
                reasons.append("reuse_unreliable")

        all_references = (
            identity_reference,
            skin_reference,
            body_shape_reference,
            appearance_reference,
            garment_reference,
            accessory_reference,
        )
        strong_reference = next(
            (
                ref
                for ref in all_references
                if ref is not None
                and ref.evidence_score >= self._reference_thresholds(ref.reference_kind)[0]
                and not ref.generated
                and not ref.inferred
            ),
            None,
        )
        strong_current_reuse = current_reuse is not None and current_reuse.evidence_score >= 0.7
        low_evidence_newly_revealed = bool(
            entry and entry.reveal_lifecycle == "newly_revealed" and entry.evidence_score < self._reference_thresholds(entry.reference_kind)[0]
        )
        strong = bool((strong_current_reuse or strong_reference is not None) and not low_evidence_newly_revealed)

        any_reliable_reference = any(ref is not None for ref in all_references)
        medium = bool(any_reliable_reference or current_reuse is not None or active_hidden_support)
        weak = bool(
            (entry is not None and not entry.reliable_for_reuse and not any_reliable_reference)
            or hidden_slot_is_trace_history
        )

        support_level = "none"
        if strong:
            support_level = "strong"
        elif medium:
            support_level = "medium"
        elif weak:
            support_level = "weak"

        identity_payload = self._build_reference_patch_payload(memory, identity_reference, "identity_reference", reasons, support_level)
        skin_payload = self._build_reference_patch_payload(memory, skin_reference, "skin_reference", reasons, support_level)
        body_shape_payload = self._build_reference_patch_payload(memory, body_shape_reference, "body_shape_reference", reasons, support_level)
        garment_payload = self._build_reference_patch_payload(memory, garment_reference, "garment_reference", reasons, support_level)
        accessory_payload = self._build_reference_patch_payload(memory, accessory_reference, "accessory_reference", reasons, support_level)
        appearance_payload = self._build_reference_patch_payload(memory, appearance_reference, "appearance_reference", reasons, support_level)
        for kind, payload in (
            ("identity_reference", identity_payload),
            ("skin_reference", skin_payload),
            ("body_shape_reference", body_shape_payload),
            ("garment_reference", garment_payload),
            ("accessory_reference", accessory_payload),
            ("appearance_reference", appearance_payload),
        ):
            if payload is not None:
                reasons.append(f"{kind}_payload_available")
                payload.retrieval_reasons.append(f"{kind}_payload_available")
                if payload.patch_id is None:
                    reasons.append(f"{kind}_payload_without_patch_cache")
                    payload.retrieval_reasons.append(f"{kind}_payload_without_patch_cache")
        reference_payloads = [
            payload
            for payload in (identity_payload, skin_payload, body_shape_payload, garment_payload, accessory_payload, appearance_payload)
            if payload is not None
        ]

        return RegionMemoryBundle(
            entity_id=entity_id,
            canonical_region=canonical_region,
            region_id=region_id,
            current_reuse=current_reuse,
            identity_reference=identity_reference,
            appearance_reference=appearance_reference,
            skin_reference=skin_reference,
            body_shape_reference=body_shape_reference,
            garment_reference=garment_reference,
            accessory_reference=accessory_reference,
            identity_reference_payload=identity_payload,
            skin_reference_payload=skin_payload,
            body_shape_reference_payload=body_shape_payload,
            garment_reference_payload=garment_payload,
            accessory_reference_payload=accessory_payload,
            appearance_reference_payload=appearance_payload,
            reference_payloads=reference_payloads,
            hidden_slot=hidden_slot,
            reveal_lifecycle=entry.reveal_lifecycle if entry else "unknown",
            memory_support_level=support_level,
            retrieval_reasons=reasons,
            has_current_reuse=current_reuse is not None,
            has_identity_reference=identity_reference is not None,
            has_appearance_reference=appearance_reference is not None,
            has_skin_reference=skin_reference is not None,
            has_body_shape_reference=body_shape_reference is not None,
            has_garment_reference=garment_reference is not None,
            has_accessory_reference=accessory_reference is not None,
            has_hidden_slot=hidden_slot is not None,
        )


    def _compact_patch_descriptor(self, descriptor: dict[str, object] | None) -> dict[str, object]:
        compact: dict[str, object] = {}
        if not isinstance(descriptor, dict):
            return compact
        for key, value in descriptor.items():
            if len(compact) >= 16:
                break
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                compact[str(key)] = float(value)
            elif isinstance(value, (list, tuple)):
                numeric = [float(v) for v in value[:16] if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if numeric:
                    compact[str(key)] = numeric
            elif isinstance(value, dict):
                nested = self._compact_patch_descriptor(value)
                if nested:
                    compact[str(key)] = nested
        return compact

    def _find_reference_texture_patch(
        self,
        memory: VideoMemory,
        entry: CanonicalRegionMemoryEntry,
    ) -> TexturePatchMemory | None:
        candidates: list[TexturePatchMemory] = []
        for patch in memory.texture_patches.values():
            if patch.entity_id != entry.entity_id:
                continue
            patch_region = self._canonical_from_region_type(patch.region_type) or patch.region_type
            entry_region = self._canonical_from_region_type(entry.canonical_region) or entry.canonical_region
            if patch_region != entry_region:
                continue
            candidates.append(patch)
        if not candidates:
            return None
        trusted_candidates = [
            patch
            for patch in candidates
            if self._texture_patch_is_trusted_observed_material(patch).get("trusted", False)
        ]
        if trusted_candidates:
            candidates = trusted_candidates
        return max(
            candidates,
            key=lambda patch: (
                -abs(int(patch.source_frame) - int(entry.source_frame)),
                float(patch.evidence_score),
                float(patch.confidence),
            ),
        )

    def _build_reference_patch_payload(
        self,
        memory: VideoMemory,
        entry: CanonicalRegionMemoryEntry | None,
        reference_kind: str,
        retrieval_reasons: list[str],
        memory_support_level: str,
    ) -> ReferencePatchPayload | None:
        if entry is None:
            return None
        evidence_threshold, confidence_threshold = self._reference_thresholds(reference_kind)
        if not (
            entry.reliable_as_reference
            and entry.reference_kind == reference_kind
            and entry.observed_directly
            and not entry.generated
            and not entry.inferred
            and entry.evidence_score >= evidence_threshold
            and entry.confidence >= confidence_threshold
        ):
            return None

        patch = self._find_reference_texture_patch(memory, entry)
        reasons = list(retrieval_reasons)
        patch_id: str | None = None
        patch_ref: str | None = None
        descriptor: dict[str, object] = {}
        confidence = float(entry.confidence)
        evidence_score = float(entry.evidence_score)
        if patch is not None:
            patch_id = patch.patch_id
            patch_ref = patch.patch_ref
            descriptor = self._compact_patch_descriptor(patch.descriptor)
            confidence = min(confidence, max(0.0, float(patch.confidence)))
            evidence_score = min(evidence_score, max(0.0, float(patch.evidence_score))) if patch.evidence_score > 0 else evidence_score
        else:
            reasons.append("reference_payload_without_patch_cache")
            reasons.append(f"{reference_kind}_payload_without_patch_cache")

        return ReferencePatchPayload(
            reference_kind=reference_kind,
            region_id=make_region_id(entry.entity_id, entry.canonical_region),
            canonical_region=entry.canonical_region,
            entity_id=entry.entity_id,
            patch_id=patch_id,
            patch_ref=patch_ref,
            source_frame=int(entry.source_frame),
            confidence=confidence,
            evidence_score=evidence_score,
            evidence_quality=entry.evidence_quality,
            observed_directly=bool(entry.observed_directly),
            generated=bool(entry.generated),
            inferred=bool(entry.inferred),
            provenance=entry.provenance,
            memory_support_level=memory_support_level,
            descriptor=descriptor,
            retrieval_reasons=reasons,
        )


    def _bounded_rgb_patch(self, patch: object, target_shape: tuple[int, int] | None = None, max_size: int = 64) -> list | None:
        arr = np.asarray(patch, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            return None
        arr = np.clip(arr, 0.0, 1.0)
        h, w, _ = arr.shape
        if target_shape is not None:
            th, tw = int(target_shape[0]), int(target_shape[1])
        else:
            scale = min(1.0, float(max_size) / float(max(h, w)))
            th = max(1, int(round(h * scale)))
            tw = max(1, int(round(w * scale)))
        th = max(1, min(max_size, th))
        tw = max(1, min(max_size, tw))
        if (th, tw) != (h, w):
            ys = np.linspace(0, h - 1, th).round().astype(int)
            xs = np.linspace(0, w - 1, tw).round().astype(int)
            arr = arr[ys][:, xs]
        return arr.astype(np.float32).tolist()

    def _material_from_payload(
        self,
        payload: ReferencePatchPayload,
        *,
        reason: str,
        rgb_patch: list | None = None,
        material_source: str = "unknown",
        material_trusted: bool = False,
    ) -> ReferencePatchMaterial:
        return ReferencePatchMaterial(
            reference_kind=payload.reference_kind,
            region_id=payload.region_id,
            canonical_region=payload.canonical_region,
            entity_id=payload.entity_id,
            source_patch_id=payload.patch_id,
            source_patch_ref=payload.patch_ref,
            rgb_patch=rgb_patch,
            alpha_or_mask=None,
            descriptor=dict(payload.descriptor or {}),
            confidence=float(payload.confidence),
            evidence_score=float(payload.evidence_score),
            observed_directly=bool(payload.observed_directly),
            generated=bool(payload.generated),
            inferred=bool(payload.inferred),
            provenance=payload.provenance,
            material_source=material_source,
            material_trusted=bool(material_trusted),
            material_missing_reason=reason,
            source_frame_kind="unknown",
            source_frame_index=int(payload.source_frame),
            immutable_i2v_anchor=False,
            source_is_input_frame=False,
        )

    def _texture_patch_is_trusted_observed_material(self, patch: TexturePatchMemory) -> dict[str, object]:
        provenance = str(getattr(patch, "provenance", "unknown") or "unknown").strip().lower()
        generated = bool(getattr(patch, "generated", False))
        inferred = bool(getattr(patch, "inferred", False))
        observed_directly = bool(getattr(patch, "observed_directly", True))
        source_frame_kind = str(getattr(patch, "source_frame_kind", "unknown") or "unknown")
        source_is_input_frame = bool(getattr(patch, "source_is_input_frame", False))
        immutable_i2v_anchor = bool(getattr(patch, "immutable_i2v_anchor", False))
        if source_frame_kind == "generated_runtime_frame":
            return {"trusted": False, "reason": "generated_runtime_material_rejected", "provenance": provenance}
        if generated or inferred:
            return {"trusted": False, "reason": "generated_runtime_material_rejected", "provenance": provenance}
        if provenance in {"runtime_generated_frame", "generated", "synthetic", "self_generated_runtime_target"}:
            return {"trusted": False, "reason": "generated_runtime_material_rejected", "provenance": provenance}
        if not source_is_input_frame:
            return {"trusted": False, "reason": "non_input_frame_material_rejected", "provenance": provenance}
        if not immutable_i2v_anchor or source_frame_kind != "observed_input_frame":
            return {"trusted": False, "reason": "missing_i2v_anchor", "provenance": provenance}
        if not observed_directly:
            return {"trusted": False, "reason": "generated_or_unobserved_material_rejected", "provenance": provenance}
        if provenance == "unknown":
            return {"trusted": True, "reason": "patch_provenance_unknown", "provenance": provenance}
        return {"trusted": True, "reason": "", "provenance": provenance}

    def build_reference_patch_material(
        self,
        memory: VideoMemory,
        payload: ReferencePatchPayload | None,
        target_shape: tuple[int, int] | None = None,
    ) -> ReferencePatchMaterial | None:
        if payload is None:
            return None
        expected_kind = reference_kind_for_region(payload.canonical_region)
        if expected_kind != "none" and payload.reference_kind != expected_kind:
            return self._material_from_payload(payload, reason="region_mismatch")
        min_evidence, min_confidence = self._reference_thresholds(payload.reference_kind)
        if payload.generated or payload.inferred:
            return self._material_from_payload(payload, reason="generated_or_inferred")
        if not payload.observed_directly:
            return self._material_from_payload(payload, reason="payload_untrusted")
        if payload.confidence < min_confidence:
            return self._material_from_payload(payload, reason="low_confidence")
        if payload.evidence_score < min_evidence:
            return self._material_from_payload(payload, reason="low_evidence")
        if not payload.patch_id:
            return self._material_from_payload(payload, reason="patch_id_missing")

        patch = memory.texture_patches.get(payload.patch_id)
        if patch is None:
            return self._material_from_payload(payload, reason="patch_not_found")
        if patch.entity_id != payload.entity_id:
            return self._material_from_payload(payload, reason="entity_mismatch")
        patch_region = self._canonical_from_region_type(patch.region_type) or patch.region_type
        payload_region = self._canonical_from_region_type(payload.canonical_region) or payload.canonical_region
        if patch_region != payload_region:
            return self._material_from_payload(payload, reason="region_mismatch")
        patch_kind = reference_kind_for_region(patch_region)
        if patch_kind != "none" and patch_kind != payload.reference_kind:
            return self._material_from_payload(payload, reason="region_mismatch")
        patch_trust = self._texture_patch_is_trusted_observed_material(patch)
        if not bool(patch_trust.get("trusted", False)):
            return self._material_from_payload(payload, reason=str(patch_trust.get("reason", "generated_or_unobserved_material_rejected") or "generated_or_unobserved_material_rejected"))

        source = "texture_patch.rgb_patch"
        raw_patch = getattr(patch, "rgb_patch", None)
        if raw_patch is None:
            raw_patch = memory.patch_cache.get(payload.patch_id)
            source = "memory.patch_cache" if raw_patch is not None else source
        bounded = self._bounded_rgb_patch(raw_patch, target_shape=target_shape) if raw_patch is not None else None
        if bounded is None:
            return self._material_from_payload(payload, reason="patch_tensor_missing", material_source=source)
        descriptor = dict(payload.descriptor or {})
        descriptor.setdefault("source_patch_region", patch.region_type)
        descriptor.setdefault("source_patch_semantic_family", patch.semantic_family)
        material = self._material_from_payload(
            payload,
            reason="",
            rgb_patch=bounded,
            material_source=source,
            material_trusted=True,
        )
        material.descriptor = descriptor
        trust_reason = str(patch_trust.get("reason", "") or "")
        if trust_reason == "patch_provenance_unknown":
            material.descriptor.setdefault("source_patch_trust_note", trust_reason)
        material.source_patch_ref = payload.patch_ref or patch.patch_ref
        material.confidence = min(float(payload.confidence), float(patch.confidence))
        material.evidence_score = min(float(payload.evidence_score), float(patch.evidence_score)) if patch.evidence_score > 0 else float(payload.evidence_score)
        material.source_frame_kind = str(getattr(patch, "source_frame_kind", "unknown") or "unknown")
        material.source_frame_index = int(getattr(patch, "source_frame", payload.source_frame))
        material.immutable_i2v_anchor = bool(getattr(patch, "immutable_i2v_anchor", False))
        material.source_is_input_frame = bool(getattr(patch, "source_is_input_frame", False))
        return material

    def debug_canonical_memory(self, memory: VideoMemory, entity_id: str | None = None) -> dict[str, object]:
        entries = self.get_region_memory_entries(memory, entity_id=entity_id)
        return {
            "regions": [
                {
                    "record_id": e.record_id,
                    "entity_id": e.entity_id,
                    "canonical_region": e.canonical_region,
                    "kind": e.memory_kind,
                    "visibility_state": e.visibility_state,
                    "reveal_lifecycle": e.reveal_lifecycle,
                    "confidence": round(e.confidence, 4),
                    "evidence_score": round(e.evidence_score, 4),
                    "evidence_quality": e.evidence_quality,
                    "observed_directly": e.observed_directly,
                    "inferred": e.inferred,
                    "generated": e.generated,
                    "reliable_for_reuse": e.reliable_for_reuse,
                    "suitable_for_reveal": e.suitable_for_reveal,
                    "freshness_frames": e.freshness_frames,
                    "source_frame": e.source_frame,
                    "last_observed_frame": e.last_observed_frame,
                    "provenance": e.provenance,
                    "last_transition": e.last_transition,
                }
                for e in entries
            ]
        }

    def route_region_retrieval(
            self,
            memory: VideoMemory,
            region_id: str,
            region_type: str,
            entity_id: str,
            query_descriptor: dict[str, float | list[float]] | None = None,
            garment_semantics: GarmentSemanticProfile | None = None,
            transition_context: dict[str, str] | None = None,
            hidden_mode: str = "not_hidden",
    ) -> dict[str, object]:
        slot = self.retrieve_hidden_region(memory, region_id)
        ctx = transition_context or {}
        phase = ctx.get("transition_phase", "single")
        visibility_phase = ctx.get("visibility_phase", "stable")
        region_mode = ctx.get("region_transition_mode", "")
        hint = self._strategy_hint(region_type, visibility_phase, region_mode)

        hidden_bias = 0.24 if hidden_mode == "known_hidden" and slot else (
            -0.10 if hidden_mode == "unknown_hidden" else 0.0)
        recency_weight = 0.18 if phase in {"lower_pelvis", "contact_chair", "garment_opening"} else 0.10
        transition_bias = 0.12 if region_mode.startswith("garment_") or region_mode in {"pose_exposure",
                                                                                        "expression_refine"} else 0.0

        candidate_patches = [p for p in memory.texture_patches.values() if p.entity_id == entity_id]

        # Жёсткая фильтрация для limb/body регионов.
        if self._is_limb_region(region_type):
            strict_limb = [
                p
                for p in candidate_patches
                if
                p.region_type in {"legs", "pelvis", "left_arm", "right_arm", "arm", "left_hand", "right_hand", "hand"}
                or getattr(p, "semantic_family", "") in {"body", "limb", "pose", "limb_pose"}
            ]
            if strict_limb:
                candidate_patches = strict_limb

        elif self._is_body_like_region(region_type):
            strict_body = [
                p
                for p in candidate_patches
                if self._is_body_like_region(p.region_type)
                   or getattr(p, "semantic_family", "") in {"face", "torso", "body", "limb", "pose", "limb_pose"}
            ]
            if strict_body:
                candidate_patches = strict_body

        narrowed = [
            p
            for p in candidate_patches
            if p.region_type == region_type
               or (hint == "visibility_transition_patch" and p.suitable_for_reveal)
        ]
        if narrowed:
            candidate_patches = narrowed

        if not candidate_patches:
            candidate_patches = [p for p in memory.texture_patches.values() if p.entity_id == entity_id]

        most_recent = max((p.source_frame for p in candidate_patches), default=0)
        scored: list[tuple[float, TexturePatchMemory, dict[str, float]]] = []

        for patch in candidate_patches:
            recency = 1.0 / (1.0 + max(0, most_recent - patch.source_frame))
            descriptor_sim = self._descriptor_similarity(query_descriptor, patch.descriptor)

            hidden_slot_contrib = 0.15 if slot and patch.patch_id in slot.candidate_patch_ids else 0.0
            same_entity_bonus = 0.16 if patch.entity_id == entity_id else 0.0
            semantic_family_bonus = 0.14 if patch.semantic_family == self._semantic_family(region_type) else 0.0
            region_compat = 0.12 if patch.region_type == region_type else 0.0

            transition_compat = 0.0
            if region_mode.startswith("garment_") and patch.semantic_family == "garment":
                transition_compat = 0.07
            elif region_mode == "expression_refine" and patch.semantic_family in {"face"}:
                transition_compat = 0.07
            elif region_mode in {"visibility_occlusion", "pose_exposure", "pose_deform"} and patch.semantic_family in {
                "limb_pose", "torso", "body"}:
                transition_compat = 0.06

            reveal_compat = 0.10 if visibility_phase == "revealing" and patch.suitable_for_reveal else 0.0
            garment_cov_compat = self._garment_coverage_compatibility(patch, garment_semantics)
            garment_attach_compat = self._garment_attachment_compatibility(patch, garment_semantics)

            lifecycle_compat = 0.0
            if hidden_mode == "known_hidden":
                lifecycle_compat = 0.08 if hidden_slot_contrib > 0 else -0.03
            elif hidden_mode == "unknown_hidden":
                lifecycle_compat = 0.03 if hidden_slot_contrib > 0 else -0.04
            else:
                lifecycle_compat = 0.03

            body_region_bias = 0.0
            if self._is_body_like_region(region_type):
                body_region_bias = 0.35 * self._candidate_body_compatibility(
                    query_region=region_type,
                    candidate_region=patch.region_type,
                    semantic_family=getattr(patch, "semantic_family", "") or "",
                )

            contributions = {
                "base_confidence": patch.confidence,
                "evidence": 0.2 * patch.evidence_score,
                "recency": recency_weight * recency,
                "similarity": 0.25 * descriptor_sim,
                "hidden_slot": hidden_bias + hidden_slot_contrib,
                "transition_bias": transition_bias,
                "same_entity_bonus": same_entity_bonus,
                "semantic_family_bonus": semantic_family_bonus,
                "region_compatibility": region_compat,
                "transition_compatibility": transition_compat,
                "coverage_compatibility": garment_cov_compat,
                "attachment_compatibility": garment_attach_compat,
                "visibility_lifecycle_compatibility": lifecycle_compat,
                "reveal_compatibility": reveal_compat,
                "body_region_bias": body_region_bias,
            }

            score = sum(contributions.values())

            # Дополнительный жёсткий штраф: limb query не должна выбирать torso как хороший retrieval.
            if region_type in {"legs", "pelvis"} and patch.region_type == "torso":
                score -= 0.35
                contributions["limb_torso_penalty"] = -0.35

            if region_type in {"left_arm", "right_arm", "arm", "left_hand", "right_hand",
                               "hand"} and patch.region_type == "torso":
                score -= 0.25
                contributions["limb_torso_penalty"] = -0.25

            # Garment patch почти запрещён для body/limb query.
            if self._is_body_like_region(region_type) and patch.region_type in {"inner_garment", "outer_garment",
                                                                                "garments", "sleeves"}:
                score -= 0.45
                contributions["body_garment_penalty"] = -0.45

            scored.append((score, patch, contributions))

        scored.sort(key=lambda item: item[0], reverse=True)

        # Отрезаем кандидатов с явно плохим match.
        filtered_scored: list[tuple[float, TexturePatchMemory, dict[str, float]]] = []
        for score, patch, contrib in scored:
            if self._is_limb_region(region_type):
                if patch.region_type not in {"legs", "pelvis", "left_arm", "right_arm", "arm", "left_hand",
                                             "right_hand", "hand"}:
                    continue
            filtered_scored.append((score, patch, contrib))

        if filtered_scored:
            scored = filtered_scored

        candidates = [patch for _, patch, _ in scored[:5]]
        top_contrib = scored[0][2] if scored else {}
        top_patch = scored[0][1].patch_id if scored else "none"

        explanation = {
            "hint": hint,
            "slot_type": slot.hidden_type if slot else "none",
            "slot_transition": slot.last_transition if slot else "none",
            "slot_reason": slot.last_transition_reason if slot else "none",
            "visibility_phase": visibility_phase,
            "phase": phase,
            "region_transition_mode": region_mode,
            "candidate_count": len(candidates),
            "top_candidate": top_patch,
            "top_candidate_why": " + ".join(
                f"{k}:{round(v, 3)}" for k, v in top_contrib.items()) if top_contrib else "none",
            "similarity_contribution": round(top_contrib.get("similarity", 0.0), 4),
            "recency_contribution": round(top_contrib.get("recency", 0.0), 4),
            "hidden_slot_contribution": round(top_contrib.get("hidden_slot", 0.0), 4),
            "transition_bias_contribution": round(top_contrib.get("transition_bias", 0.0), 4),
            "body_region_bias": round(top_contrib.get("body_region_bias", 0.0), 4),
        }

        return {
            "candidates": candidates,
            "strategy_hint": hint,
            "explanation": explanation,
            "top_score": round(scored[0][0], 4) if scored else 0.0,
            "top_semantic_compatibility": round(
                top_contrib.get("semantic_family_bonus", 0.0)
                + top_contrib.get("coverage_compatibility", 0.0)
                + top_contrib.get("attachment_compatibility", 0.0),
                4,
            ),
            "summary": {
                "candidate_count": len(candidates),
                "hidden_mode": hidden_mode,
                "strategy_hint": hint,
            },
            "top_score_breakdown": top_contrib,
            "candidate_summaries": [
                {"patch_id": patch.patch_id, "region_type": patch.region_type, "semantic_family": patch.semantic_family}
                for patch in candidates
            ],
            "fallback_reason": "no_candidates" if not candidates else "none",
        }

    def to_dict(self, memory: VideoMemory) -> dict[str, object]:
        return asdict(memory)

    def from_dict(self, payload: dict[str, object]) -> VideoMemory:
        memory = VideoMemory()
        for key, value in (payload.get("identity_memory") or {}).items():
            memory.identity_memory[key] = MemoryEntry(**value)
        for key, value in (payload.get("garment_memory") or {}).items():
            memory.garment_memory[key] = MemoryEntry(**value)
        memory.patch_cache = dict(payload.get("patch_cache") or {})
        memory.temporal_history = []
        for key, value in (payload.get("texture_patches") or {}).items():
            memory.texture_patches[key] = TexturePatchMemory(**value)
        for key, value in (payload.get("hidden_region_slots") or {}).items():
            memory.hidden_region_slots[key] = HiddenRegionSlot(**value)
        for key, value in (payload.get("canonical_region_memory") or {}).items():
            memory.canonical_region_memory[key] = CanonicalRegionMemoryEntry(**value)
        for key, value in (payload.get("region_descriptors") or {}).items():
            bbox = BBox(**value["bbox"])
            memory.region_descriptors[key] = RegionDescriptor(**{**value, "bbox": bbox})
        memory.last_transition_context = dict(payload.get("last_transition_context") or {})
        return memory

    def _refresh_entry(self, entry: MemoryEntry, frame_index: int) -> None:
        entry.confidence = min(1.0, entry.confidence * 0.9 + 0.1)
        entry.last_seen_frames.append(frame_index)

    def _refresh_hidden_slot_priority(self, slot: HiddenRegionSlot) -> None:
        recency = 1.0 / (1.0 + slot.stale_frames)
        slot.retrieval_priority = max(0.0, min(1.0, slot.confidence * 0.6 + slot.evidence_score * 0.3 + recency * 0.1))

    def transition_hidden_slot(self, slot: HiddenRegionSlot, target_state: str, reason: str) -> str:
        current = slot.hidden_type
        transition = "stable"
        if current == "unknown_hidden" and target_state == "known_hidden":
            transition = "unknown_hidden_to_known_hidden"
        elif current in {"unknown_hidden", "known_hidden"} and target_state == "revealed":
            transition = "known_hidden_to_revealed" if current == "known_hidden" else "unknown_hidden_to_revealed"
        elif current == "revealed" and target_state in {"known_hidden", "unknown_hidden"}:
            transition = "revealed_to_hidden"
        elif current == "known_hidden" and target_state == "decayed_unknown":
            transition = "known_hidden_to_decayed_unknown"
            target_state = "unknown_hidden"
        elif target_state == current:
            transition = "stable"
        slot.hidden_type = target_state
        slot.last_transition = transition
        slot.last_transition_reason = reason
        return transition

    def _promote_or_decay_hidden_slot(self, slot: HiddenRegionSlot) -> None:
        self._refresh_hidden_slot_priority(slot)
        if slot.hidden_type == "unknown_hidden" and slot.evidence_score >= 0.55 and len(slot.candidate_patch_ids) >= 2:
            self.transition_hidden_slot(slot, "known_hidden", reason="evidence_promoted")
        if slot.stale_frames >= 6 and slot.hidden_type == "known_hidden":
            slot.confidence = max(0.1, slot.confidence * 0.9)
            if slot.confidence < 0.35:
                self.transition_hidden_slot(slot, "decayed_unknown", reason="stale_decay")

    def _seed_person_semantic_regions(self, memory: VideoMemory, person_id: str, bbox: BBox, frame_index: int) -> None:
        for offset, region_type in enumerate(self._SEMANTIC_REGIONS):
            region_id = make_region_id(person_id, region_type)
            memory.region_descriptors[region_id] = RegionDescriptor(region_id=region_id, entity_id=person_id, region_type=region_type, bbox=BBox(bbox.x, bbox.y + 0.01 * offset, bbox.w, bbox.h), visibility="visible" if region_type in ("face", "torso") else "partially_visible", confidence=0.75, last_update_frame=frame_index)
            patch_id = f"patch::{region_id}"
            memory.texture_patches[patch_id] = TexturePatchMemory(patch_id=patch_id, region_type=region_type, entity_id=person_id, source_frame=frame_index, patch_ref=f"seed://{region_id}", confidence=0.5, descriptor={}, evidence_score=0.2)
            memory.hidden_region_slots[region_id] = HiddenRegionSlot(slot_id=region_id, region_type=region_type, owner_entity=person_id, candidate_patch_ids=[patch_id], confidence=0.45, hidden_type="known_hidden" if region_type in {"sleeves", "garments", "legs"} else "unknown_hidden", evidence_score=0.2)

    def _canonical_from_region_type(self, region_type: str) -> str | None:
        direct = {
            "face": "face",
            "hair": "hair",
            "head": "head",
            "neck": "neck",
            "torso": "torso",
            "left_arm": "left_arm",
            "right_arm": "right_arm",
            "left_hand": "left_hand",
            "right_hand": "right_hand",
            "pelvis": "pelvis",
            "left_leg": "left_leg",
            "right_leg": "right_leg",
            "upper_garment": "upper_garment",
            "lower_garment": "lower_garment",
            "outer_garment": "outer_garment",
            "inner_garment": "inner_garment",
            "accessories": "accessories",
            "garments": "upper_garment",
            "sleeves": "upper_garment",
            "legs": "left_leg",
        }
        canonical = direct.get(region_type)
        if canonical in self._CANONICAL_MEMORY_REGIONS:
            return canonical
        return None

    def _memory_kind_for_region(self, canonical_region: str) -> str:
        if canonical_region in {"face", "hair", "head", "neck"}:
            return "identity"
        if canonical_region in {"upper_garment", "lower_garment", "outer_garment", "inner_garment"}:
            return "garment"
        if canonical_region == "accessories":
            return "accessory"
        return "body"

    def _visibility_to_reveal_lifecycle(self, visibility: str) -> str:
        if visibility in {"visible", "partially_visible"}:
            return "currently_visible"
        if visibility in {"hidden", "hidden_by_garment", "hidden_by_object", "hidden_by_self", "out_of_frame"}:
            return "currently_hidden"
        if visibility == "unknown_expected_region":
            return "expected_unknown"
        return "unknown"

    def _evidence_quality(self, evidence_score: float, visibility: str, observed_directly: bool) -> str:
        if not observed_directly:
            return "inferred"
        if visibility == "visible" and evidence_score >= 0.72:
            return "strong"
        if visibility in {"visible", "partially_visible"} and evidence_score >= 0.45:
            return "medium"
        return "weak"

    def _provenance_reliability(self, provenance: str) -> float:
        key = provenance.lower().split(":")[0] if provenance else "unknown"
        return float(self._PROVENANCE_RELIABILITY.get(key, self._PROVENANCE_RELIABILITY["unknown"]))

    def _resolve_visibility_state(
        self,
        *,
        existing_visibility: str | None,
        observed_visibility: str,
        evidence_score: float,
        observed_directly: bool,
    ) -> str:
        prev = existing_visibility or "unknown_expected_region"
        vis = observed_visibility
        if observed_directly and vis in {"visible", "partially_visible"}:
            return vis
        if prev in {"hidden", "hidden_by_garment", "hidden_by_object", "hidden_by_self"} and vis in {"visible", "partially_visible"} and evidence_score >= 0.42:
            return vis
        if vis in {"hidden", "hidden_by_garment", "hidden_by_object", "hidden_by_self", "out_of_frame"} and evidence_score >= 0.4:
            return vis
        if vis == "unknown_expected_region" and prev in {"visible", "partially_visible"} and evidence_score < 0.35:
            return prev
        if evidence_score >= 0.5:
            return vis
        return prev

    def _refresh_reuse_policy(self, entry: CanonicalRegionMemoryEntry) -> None:
        # Lifecycle semantics guard:
        # - newly_revealed/newly_occluded are special states that can directly
        #   impact reuse/reveal safety.
        # - visibility_changed is intentionally treated as neutral bookkeeping
        #   (captured via last_transition / last_transition_context), not a
        #   positive reveal suitability signal by itself.
        visibility_ok = str(entry.visibility_state) in {"visible", "partially_visible"}
        fresh_bonus = 1.0 if entry.freshness_frames <= 2 else (0.75 if entry.freshness_frames <= 6 else 0.45)
        stale_penalty = 0.22 if entry.freshness_frames > 8 else 0.0
        inferred_penalty = 0.24 if entry.inferred else 0.0
        generated_penalty = 0.35 if entry.generated else 0.0
        lifecycle_penalty = 0.18 if entry.reveal_lifecycle in {"newly_occluded", "currently_hidden", "expected_unknown"} else 0.0
        if entry.reveal_lifecycle == "newly_revealed":
            lifecycle_penalty += 0.08
        provenance_factor = self._provenance_reliability(entry.provenance)
        reuse_score = (
            0.42 * entry.confidence
            + 0.35 * entry.evidence_score
            + 0.16 * provenance_factor
            + 0.12 * fresh_bonus
            - stale_penalty
            - inferred_penalty
            - generated_penalty
            - lifecycle_penalty
        )
        entry.reliable_for_reuse = bool(
            visibility_ok
            and entry.observed_directly
            and not entry.generated
            and not entry.inferred
            and reuse_score >= 0.66
            and (entry.reveal_lifecycle != "newly_revealed" or entry.evidence_score >= 0.62)
            and entry.reveal_lifecycle not in {"newly_occluded", "currently_hidden"}
        )
        reference_kind = self._reference_kind_for_region(entry.canonical_region)
        strong_direct_reference = False
        if reference_kind == "identity_reference":
            strong_direct_reference = self._can_update_identity_reference(
                region_type=entry.canonical_region,
                observed_directly=entry.observed_directly,
                generated=entry.generated,
                inferred=entry.inferred,
                evidence_score=entry.evidence_score,
                confidence=entry.confidence,
                reveal_lifecycle=entry.reveal_lifecycle,
            )
            if not strong_direct_reference:
                reference_kind = "none"
        elif reference_kind != "none":
            strong_direct_reference = self._can_update_strong_reference(entry, reference_kind)
        entry.reference_kind = reference_kind
        entry.reliable_as_reference = bool(reference_kind != "none" and strong_direct_reference)
        entry.suitable_for_reveal = bool(
            entry.reliable_for_reuse
            and entry.evidence_score >= 0.52
            and entry.freshness_frames <= 8
            and entry.reveal_lifecycle not in {"newly_occluded", "expected_unknown"}
        )

    def _derive_observation_semantics(
        self,
        *,
        canonical_name: str,
        visibility: str,
        confidence: float,
        provenance: str,
        mask_ref: str | None,
        source_regions: list[str],
        source_signals: list[str],
    ) -> tuple[bool, bool, float]:
        provenance_rel = self._provenance_reliability(provenance)
        has_mask = bool(mask_ref)
        has_specific_source_region = any(str(s).strip().lower() not in {"", canonical_name, "unknown", "aggregate"} for s in source_regions)
        has_signal = bool(source_signals)
        visibility_factor = 1.0 if visibility == "visible" else (0.72 if visibility == "partially_visible" else 0.38)
        direct_support = (
            0.34 * confidence
            + 0.24 * provenance_rel
            + (0.2 if has_mask else 0.0)
            + (0.12 if has_specific_source_region else 0.0)
            + (0.08 if has_signal else 0.0)
            + (0.1 if visibility == "visible" else 0.0)
        )
        observed_directly = bool(
            visibility in {"visible", "partially_visible"}
            and has_mask
            and provenance_rel >= 0.6
            and direct_support >= (0.72 if visibility == "visible" else 0.82)
        )
        inferred = not observed_directly
        evidence_score = min(1.0, max(0.0, confidence * visibility_factor * (0.75 + 0.25 * provenance_rel) + (0.12 if has_mask else 0.0)))
        if inferred:
            evidence_score *= 0.78
        return observed_directly, inferred, evidence_score

    def _entry_strength(self, entry: CanonicalRegionMemoryEntry) -> float:
        quality_bonus = {"strong": 0.35, "medium": 0.2, "weak": 0.08, "inferred": 0.02}.get(entry.evidence_quality, 0.0)
        visibility_bonus = {"visible": 0.3, "partially_visible": 0.15, "hidden": -0.15, "unknown_expected_region": -0.25}.get(str(entry.visibility_state), 0.0)
        direct_bonus = 0.24 if entry.observed_directly else -0.08
        generated_penalty = 0.35 if entry.generated else 0.0
        inferred_penalty = 0.22 if entry.inferred else 0.0
        provenance_bonus = 0.2 * self._provenance_reliability(entry.provenance)
        freshness_bonus = 0.16 if entry.freshness_frames <= 1 else (0.08 if entry.freshness_frames <= 4 else -0.04 * min(8, entry.freshness_frames))
        lifecycle_penalty = 0.28 if entry.reveal_lifecycle == "newly_revealed" and entry.evidence_score <= 0.55 else 0.0
        lifecycle_penalty += 0.16 if entry.reveal_lifecycle in {"newly_occluded", "currently_hidden"} else 0.0
        identity_guard = 0.0
        if entry.canonical_region in self.IDENTITY_SENSITIVE_REGIONS:
            identity_guard += 0.18 if (entry.observed_directly and not entry.generated and not entry.inferred) else -0.14
        return float(
            entry.confidence
            + 0.8 * entry.evidence_score
            + quality_bonus
            + visibility_bonus
            + direct_bonus
            + provenance_bonus
            + freshness_bonus
            + identity_guard
            - generated_penalty
            - inferred_penalty
            - lifecycle_penalty
        )

    def _is_protected_identity_memory(self, entry: CanonicalRegionMemoryEntry) -> bool:
        return bool(
            entry.canonical_region in self.IDENTITY_SENSITIVE_REGIONS
            and entry.reliable_as_reference
            and entry.observed_directly
            and not entry.generated
            and not entry.inferred
            and entry.evidence_score >= 0.7
            and entry.confidence >= 0.65
        )

    def _update_canonical_region_memory(self, memory: VideoMemory, person: object, frame_index: int) -> None:
        canonical_regions = getattr(person, "canonical_regions", {}) or {}
        for canonical_name in self._CANONICAL_MEMORY_REGIONS:
            raw = canonical_regions.get(canonical_name, {}) if isinstance(canonical_regions, dict) else {}
            visibility = str(raw.get("visibility_state", "unknown_expected_region"))
            lifecycle_state = str(raw.get("lifecycle_state", "") or "").strip().lower()
            last_transition_mode = str(raw.get("last_transition_mode", "") or "").strip()
            last_transition_phase = str(raw.get("last_transition_phase", "") or "").strip()
            last_semantic_reasons = [str(v) for v in raw.get("last_semantic_reasons", [])] if isinstance(raw.get("last_semantic_reasons", []), list) else []
            last_update_source = str(raw.get("last_update_source", "") or "").strip()
            confidence = float(raw.get("confidence", 0.0))
            provenance = str(raw.get("provenance", "unknown"))
            source_regions = [str(v) for v in raw.get("source_regions", [])] if isinstance(raw.get("source_regions", []), list) else []
            source_signals = [str(v) for v in raw.get("raw_sources", [])] if isinstance(raw.get("raw_sources", []), list) else []
            generated = self._source_indicates_generated(
                raw.get("generated"),
                raw.get("is_generated"),
                raw.get("frame_source"),
                raw.get("update_source"),
                last_update_source,
                provenance,
            )
            observed_directly, inferred, evidence_score = self._derive_observation_semantics(
                canonical_name=canonical_name,
                visibility=visibility,
                confidence=confidence,
                provenance=provenance,
                mask_ref=raw.get("mask_ref"),
                source_regions=source_regions,
                source_signals=source_signals,
            )
            if generated:
                observed_directly = False
                inferred = True
                evidence_score = min(evidence_score, confidence * 0.65)
            existing = memory.canonical_region_memory.get(make_region_id(person.person_id, canonical_name))
            resolved_visibility = self._resolve_visibility_state(
                existing_visibility=(str(existing.visibility_state) if existing else None),
                observed_visibility=visibility,
                evidence_score=evidence_score,
                observed_directly=observed_directly,
            )
            entry = CanonicalRegionMemoryEntry(
                record_id=make_region_id(person.person_id, canonical_name),
                entity_id=person.person_id,
                canonical_region=canonical_name,
                memory_kind=self._memory_kind_for_region(canonical_name),
                mask_ref=raw.get("mask_ref"),
                region_ref=make_region_id(person.person_id, canonical_name),
                confidence=confidence,
                visibility_state=resolved_visibility,
                provenance=provenance,
                source_frame=frame_index,
                evidence_score=evidence_score,
                evidence_quality=self._evidence_quality(evidence_score, resolved_visibility, observed_directly),
                observed_directly=observed_directly,
                inferred=inferred,
                generated=generated,
                reliable_for_reuse=False,
                reliable_as_reference=False,
                reference_kind="none",
                suitable_for_reveal=False,
                freshness_frames=0,
                last_observed_frame=(frame_index if observed_directly else None),
                reveal_lifecycle=lifecycle_state if lifecycle_state in {"newly_revealed", "newly_occluded", "visibility_changed"} else self._visibility_to_reveal_lifecycle(resolved_visibility),
                last_transition="stable",
            )
            transition_parts = [lifecycle_state or "stable"]
            if last_transition_mode:
                transition_parts.append(last_transition_mode)
            if last_transition_phase:
                transition_parts.append(last_transition_phase)
            if last_semantic_reasons:
                transition_parts.append("+".join(last_semantic_reasons))
            if last_update_source:
                transition_parts.append(last_update_source)
            entry.last_transition = "|".join(transition_parts)

            trace_region_id = make_region_id(person.person_id, canonical_name)
            memory.last_transition_context[f"{trace_region_id}:lifecycle_state"] = lifecycle_state or "stable"
            if last_transition_mode:
                memory.last_transition_context[f"{trace_region_id}:transition_mode"] = last_transition_mode
            if last_transition_phase:
                memory.last_transition_context[f"{trace_region_id}:transition_phase"] = last_transition_phase
            if last_semantic_reasons:
                memory.last_transition_context[f"{trace_region_id}:semantic_reasons"] = ",".join(last_semantic_reasons)
            if last_update_source:
                memory.last_transition_context[f"{trace_region_id}:update_source"] = last_update_source
            self._refresh_reuse_policy(entry)
            self._upsert_canonical_memory(memory, entry)
            self._sync_hidden_slot_with_graph_lifecycle(
                memory=memory,
                entity_id=person.person_id,
                canonical_name=canonical_name,
                lifecycle_state=lifecycle_state,
                visibility_state=visibility,
                transition_mode=last_transition_mode,
                transition_phase=last_transition_phase,
                semantic_reasons=last_semantic_reasons,
                update_source=last_update_source,
            )

    def _sync_hidden_slot_with_graph_lifecycle(
        self,
        *,
        memory: VideoMemory,
        entity_id: str,
        canonical_name: str,
        lifecycle_state: str,
        visibility_state: str,
        transition_mode: str,
        transition_phase: str,
        semantic_reasons: list[str],
        update_source: str,
    ) -> None:
        if lifecycle_state not in {"newly_revealed", "newly_occluded"}:
            return
        region_id = make_region_id(entity_id, canonical_name)
        slot = memory.hidden_region_slots.get(region_id)
        if slot is None:
            slot = HiddenRegionSlot(
                slot_id=region_id,
                region_type=canonical_name,
                owner_entity=entity_id,
                candidate_patch_ids=[],
                hidden_type="unknown_hidden",
            )
            memory.hidden_region_slots[region_id] = slot
        slot.stale_frames = 0
        transition_bits = [lifecycle_state]
        if visibility_state:
            transition_bits.append(visibility_state)
        if transition_mode:
            transition_bits.append(transition_mode)
        if transition_phase:
            transition_bits.append(transition_phase)
        if semantic_reasons:
            transition_bits.append("+".join(semantic_reasons))
        if update_source:
            transition_bits.append(update_source)
        slot.last_transition = lifecycle_state
        slot.last_transition_reason = "|".join(transition_bits)
        if lifecycle_state == "newly_revealed":
            # Keep historical hidden slot trace, but do not leave it looking like
            # an active hidden candidate after explicit reveal.
            slot.hidden_type = "revealed_history" if slot.candidate_patch_ids else "revealed"
            reveal_reason = slot.last_transition_reason
            if "revealed" not in reveal_reason.lower():
                reveal_reason = f"{reveal_reason}|revealed"
            slot.last_transition_reason = reveal_reason
        elif lifecycle_state == "newly_occluded":
            self.transition_hidden_slot(slot, "known_hidden", reason=f"graph_lifecycle:{slot.last_transition_reason}")
        self._promote_or_decay_hidden_slot(slot)

    def _upsert_canonical_memory(self, memory: VideoMemory, candidate: CanonicalRegionMemoryEntry) -> None:
        current = memory.canonical_region_memory.get(candidate.record_id)
        if current is None:
            memory.canonical_region_memory[candidate.record_id] = candidate
            return
        if (
            current.canonical_region in self.IDENTITY_SENSITIVE_REGIONS
            and self._is_protected_identity_memory(current)
            and (
                candidate.generated
                or candidate.inferred
                or not candidate.observed_directly
                or self._provenance_reliability(candidate.provenance) < 0.55
                or candidate.evidence_score + 0.12 < current.evidence_score
                or candidate.confidence + 0.12 < current.confidence
            )
        ):
            current.freshness_frames = max(0, candidate.source_frame - current.source_frame)
            if str(current.visibility_state) in {"visible", "partially_visible"} and str(candidate.visibility_state) in {"hidden", "hidden_by_garment", "hidden_by_object", "hidden_by_self"}:
                current.visibility_state = str(candidate.visibility_state)
            if candidate.reveal_lifecycle == "newly_occluded":
                current.reveal_lifecycle = "newly_occluded"
                current.last_transition = "preserve_identity_on_occlusion"
            self._refresh_reuse_policy(current)
            return
        candidate_strength = self._entry_strength(candidate)
        current_strength = self._entry_strength(current)
        current.freshness_frames = max(0, candidate.source_frame - current.source_frame)
        if candidate_strength >= current_strength + 0.02 or (
            candidate.source_frame > current.source_frame and candidate.observed_directly and current.generated
        ) or (
            str(current.visibility_state) in {"hidden", "hidden_by_garment", "hidden_by_object", "hidden_by_self", "unknown_expected_region"}
            and str(candidate.visibility_state) in {"visible", "partially_visible"}
            and candidate.observed_directly
            and candidate.evidence_score >= 0.45
        ):
            transition = str(getattr(candidate, "last_transition", "") or "").strip().lower()
            if transition in {"", "stable"}:
                candidate.last_transition = "refresh"
            memory.canonical_region_memory[candidate.record_id] = candidate
        else:
            current.freshness_frames = max(0, candidate.source_frame - current.source_frame)
            self._refresh_reuse_policy(current)

    def _refresh_canonical_memory_from_descriptor(
        self,
        memory: VideoMemory,
        entity_id: str,
        canonical_region: str,
        source_frame: int,
        evidence_score: float,
        confidence: float,
        observed_directly: bool,
        generated: bool,
    ) -> None:
        canonical = self._canonical_from_region_type(canonical_region)
        if not canonical:
            return
        record_id = make_region_id(entity_id, canonical)
        existing = memory.canonical_region_memory.get(record_id)
        observed_visibility = str(existing.visibility_state if existing is not None else "visible")
        if existing is not None and str(existing.visibility_state) in {"hidden", "hidden_by_garment", "hidden_by_object", "unknown_expected_region"} and observed_directly and evidence_score >= 0.45:
            observed_visibility = "visible"
        visibility = self._resolve_visibility_state(
            existing_visibility=(str(existing.visibility_state) if existing else None),
            observed_visibility=observed_visibility,
            evidence_score=evidence_score,
            observed_directly=observed_directly,
        )
        quality = self._evidence_quality(evidence_score, visibility, observed_directly)
        candidate = CanonicalRegionMemoryEntry(
            record_id=record_id,
            entity_id=entity_id,
            canonical_region=canonical,
            memory_kind=self._memory_kind_for_region(canonical),
            mask_ref=(existing.mask_ref if existing is not None else None),
            region_ref=record_id,
            confidence=confidence,
            visibility_state=visibility,
            provenance=(existing.provenance if existing is not None else "frame_observation"),
            source_frame=source_frame,
            evidence_score=evidence_score,
            evidence_quality=quality,
            observed_directly=observed_directly,
            inferred=not observed_directly,
            generated=generated,
            reliable_for_reuse=False,
            reliable_as_reference=False,
            reference_kind="none",
            suitable_for_reveal=False,
            freshness_frames=0,
            last_observed_frame=(source_frame if observed_directly else (existing.last_observed_frame if existing else None)),
            reveal_lifecycle=(self._visibility_to_reveal_lifecycle(visibility) if observed_directly else (existing.reveal_lifecycle if existing else self._visibility_to_reveal_lifecycle(visibility))),
            last_transition="refresh_from_frame",
        )
        self._refresh_reuse_policy(candidate)
        self._upsert_canonical_memory(memory, candidate)

    def _semantic_region_types(self, person: object) -> list[str]:
        region_types = {"face", "hair", "torso", "garments", "sleeves", "pelvis", "legs"}
        body_parts = getattr(person, "body_parts", [])
        garments = getattr(person, "garments", [])
        for part in body_parts:
            if part.part_type in {"left_arm", "left_upper_arm", "left_lower_arm", "left_hand", "arms"}:
                region_types.add("left_arm")
            if part.part_type in {"right_arm", "right_upper_arm", "right_lower_arm", "right_hand", "arms"}:
                region_types.add("right_arm")
        if garments:
            region_types.add("garments")
        return sorted(region_types)

    def _strategy_hint(self, region_type: str, visibility_phase: str, region_mode: str) -> str:
        if region_type in {"face", "head"} or region_mode == "expression_refine":
            return "identity_patch"
        if region_type in {"garments", "sleeves"} or region_mode.startswith("garment_"):
            return "garment_patch"
        if visibility_phase in {"revealing", "occluding"}:
            return "visibility_transition_patch"
        return "direct_patch"

    def _memory_update_boost(self, region_type: str, context: dict[str, str]) -> float:
        visibility_phase = context.get("visibility_phase", "stable")
        garment_phase = context.get("garment_phase", "worn")
        region_mode = context.get("region_transition_mode", "")
        if visibility_phase == "revealing":
            return 0.5 if region_type in {"torso", "garments", "sleeves"} else 0.42
        if visibility_phase == "occluding":
            return 0.22
        if garment_phase in {"opening", "removed"} and region_type in {"garments", "sleeves", "torso"}:
            return 0.48
        return 0.3

    def _semantic_family(self, region_type: str) -> str:
        if region_type in {"face", "head"}:
            return "face"
        if region_type in {"garments", "sleeves"}:
            return "garment"
        if region_type in {"left_arm", "right_arm", "legs", "pelvis"}:
            return "limb_pose"
        if region_type in {"torso"}:
            return "torso"
        return "generic"

    def _default_coverage_targets(self, region_type: str) -> list[str]:
        mapping = {
            "garments": ["torso"],
            "sleeves": ["left_upper_arm", "right_upper_arm"],
            "torso": ["torso"],
            "pelvis": ["pelvis"],
            "legs": ["legs"],
        }
        return mapping.get(region_type, [region_type])

    def _default_attachment_targets(self, region_type: str) -> list[str]:
        mapping = {
            "garments": ["torso"],
            "sleeves": ["arms", "torso"],
            "torso": ["torso"],
            "pelvis": ["pelvis"],
            "legs": ["pelvis"],
        }
        return mapping.get(region_type, ["torso"])

    def _garment_coverage_compatibility(self, patch: TexturePatchMemory, garment: GarmentSemanticProfile | None) -> float:
        if garment is None or not garment.coverage_targets:
            return 0.0
        overlap = len(set(patch.coverage_targets).intersection(set(garment.coverage_targets)))
        denom = max(1, len(set(garment.coverage_targets)))
        return 0.12 * (overlap / denom)

    def _garment_attachment_compatibility(self, patch: TexturePatchMemory, garment: GarmentSemanticProfile | None) -> float:
        if garment is None or not garment.attachment_targets:
            return 0.0
        overlap = len(set(patch.attachment_targets).intersection(set(garment.attachment_targets)))
        denom = max(1, len(set(garment.attachment_targets)))
        return 0.1 * (overlap / denom)
