from __future__ import annotations

from dataclasses import asdict
import math
import random

from core.semantic_roi import SemanticROIHelper
from core.region_ids import make_region_id, parse_region_id
from core.schema import (
    BBox,
    HiddenRegionSlot,
    MemoryEntry,
    RegionDescriptor,
    SceneGraph,
    TexturePatchMemory,
    VideoMemory,
)
from utils_tensor import crop, mean_color, shape


class MemoryManager:
    _SEMANTIC_REGIONS = ("face", "torso", "sleeves", "garments", "left_arm", "right_arm", "pelvis", "legs")

    def __init__(self) -> None:
        self.roi = SemanticROIHelper()

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
            for garment in person.garments:
                memory.garment_memory[garment.garment_id] = MemoryEntry(entity_id=garment.garment_id, entry_type="garment", embedding=self._encode_visual(f"garment:{garment.garment_id}"), confidence=garment.confidence, last_seen_frames=[scene_graph.frame_index])
        return memory

    def update(self, memory: VideoMemory, scene_graph: SceneGraph) -> VideoMemory:
        return self.update_from_graph(memory, scene_graph)

    def update_from_graph(self, memory: VideoMemory, scene_graph: SceneGraph) -> VideoMemory:
        memory.temporal_history.append(scene_graph)
        observed_entities: set[str] = set()
        visible_regions: set[str] = set()
        for person in scene_graph.persons:
            observed_entities.add(person.person_id)
            identity = memory.identity_memory.get(person.person_id)
            if identity is not None:
                self._refresh_entry(identity, scene_graph.frame_index)
            for part in person.body_parts:
                region_id = make_region_id(person.person_id, part.part_type)
                if part.visibility in ("visible", "partially_visible"):
                    visible_regions.add(region_id)
                    memory.region_descriptors[region_id] = RegionDescriptor(region_id=region_id, entity_id=person.person_id, region_type=part.part_type, bbox=person.bbox, visibility=part.visibility, confidence=part.confidence, last_update_frame=scene_graph.frame_index)
                else:
                    self.apply_visibility_event(memory, {"region_id": region_id, "entity": person.person_id}, {}, visibility="hidden", transition_reason="graph_visibility_hidden")
            for garment in person.garments:
                observed_entities.add(garment.garment_id)
                gid = make_region_id(person.person_id, "garments")
                if garment.visibility in ("visible", "partially_visible"):
                    visible_regions.add(gid)
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
                )
                memory.patch_cache[patch_id] = patch

                descriptor = memory.region_descriptors.get(rid)
                if descriptor:
                    descriptor.last_update_frame = scene_graph.frame_index
                    descriptor.confidence = min(1.0, descriptor.confidence * (1.0 - 0.2 * confidence_boost) + 0.2 * evidence)

                identity = memory.identity_memory.get(person.person_id)
                if identity is not None:
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

    def route_region_retrieval(
        self,
        memory: VideoMemory,
        region_id: str,
        region_type: str,
        entity_id: str,
        query_descriptor: dict[str, float | list[float]] | None = None,
        transition_context: dict[str, str] | None = None,
    ) -> dict[str, object]:
        slot = self.retrieve_hidden_region(memory, region_id)
        phase = (transition_context or {}).get("transition_phase", "single")
        visibility_phase = (transition_context or {}).get("visibility_phase", "stable")
        region_mode = (transition_context or {}).get("region_transition_mode", "")
        hidden_bias = 0.22 if slot and slot.hidden_type == "known_hidden" else -0.05
        recency_weight = 0.2 if phase in {"lower_pelvis", "contact_chair", "garment_opening"} else 0.1
        transition_bias = 0.12 if region_mode.startswith("garment_") or region_mode in {"pose_exposure", "expression_refine"} else 0.0
        hint = self._strategy_hint(region_type, visibility_phase, region_mode)

        patches = [p for p in memory.texture_patches.values() if p.entity_id == entity_id and (p.region_type == region_type or hint == "visibility_transition_patch")]
        if not patches:
            patches = [p for p in memory.texture_patches.values() if p.entity_id == entity_id]
        most_recent = max((p.source_frame for p in patches), default=0)
        scored: list[tuple[float, TexturePatchMemory, dict[str, float]]] = []
        for patch in patches:
            recency = 1.0 / (1.0 + max(0, most_recent - patch.source_frame))
            descriptor_sim = self._descriptor_similarity(query_descriptor, patch.descriptor)
            hidden_slot_contrib = 0.15 if slot and patch.patch_id in slot.candidate_patch_ids else 0.0
            contributions = {
                "base_confidence": patch.confidence,
                "evidence": 0.2 * patch.evidence_score,
                "recency": recency_weight * recency,
                "similarity": 0.25 * descriptor_sim,
                "hidden_slot": hidden_bias + hidden_slot_contrib,
                "transition_bias": transition_bias,
            }
            score = sum(contributions.values())
            scored.append((score, patch, contributions))
        scored.sort(key=lambda item: item[0], reverse=True)
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
            "top_candidate_why": " + ".join(f"{k}:{round(v, 3)}" for k, v in top_contrib.items()) if top_contrib else "none",
            "similarity_contribution": round(top_contrib.get("similarity", 0.0), 4),
            "recency_contribution": round(top_contrib.get("recency", 0.0), 4),
            "hidden_slot_contribution": round(top_contrib.get("hidden_slot", 0.0), 4),
            "transition_bias_contribution": round(top_contrib.get("transition_bias", 0.0), 4),
        }
        return {"candidates": candidates, "strategy_hint": hint, "explanation": explanation}

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

    def _semantic_region_types(self, person: object) -> list[str]:
        region_types = {"face", "torso", "garments", "sleeves", "pelvis", "legs"}
        body_parts = getattr(person, "body_parts", [])
        garments = getattr(person, "garments", [])
        for part in body_parts:
            if part.part_type in {"left_upper_arm", "left_lower_arm", "left_hand"}:
                region_types.add("left_arm")
            if part.part_type in {"right_upper_arm", "right_lower_arm", "right_hand"}:
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
        if region_mode == "pose_exposure":
            return 0.36
        return 0.3
