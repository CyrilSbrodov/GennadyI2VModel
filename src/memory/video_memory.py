from __future__ import annotations

from dataclasses import asdict
import math
import random

from core.schema import (
    BBox,
    HiddenRegionSlot,
    MemoryEntry,
    RegionDescriptor,
    SceneGraph,
    TexturePatchMemory,
    VideoMemory,
)
from utils_tensor import mean_color


class MemoryManager:
    _SEMANTIC_REGIONS = ("face", "hair", "torso", "sleeves", "garments")

    def _encode_visual(self, token: str, dim: int = 8) -> list[float]:
        seed = abs(hash(token)) % (2**32)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]

    def initialize(self, scene_graph: SceneGraph) -> VideoMemory:
        return self.initialize_from_scene(scene_graph)

    def initialize_from_scene(self, scene_graph: SceneGraph) -> VideoMemory:
        memory = VideoMemory(temporal_history=[scene_graph])
        for person in scene_graph.persons:
            memory.identity_memory[person.person_id] = MemoryEntry(
                entity_id=person.person_id,
                entry_type="identity",
                embedding=self._encode_visual(f"identity:{person.person_id}"),
                confidence=person.confidence,
                last_seen_frames=[scene_graph.frame_index],
            )
            self._seed_person_semantic_regions(memory, person.person_id, person.bbox, scene_graph.frame_index)
            for garment in person.garments:
                memory.garment_memory[garment.garment_id] = MemoryEntry(
                    entity_id=garment.garment_id,
                    entry_type="garment",
                    embedding=self._encode_visual(f"garment:{garment.garment_id}"),
                    confidence=garment.confidence,
                    last_seen_frames=[scene_graph.frame_index],
                )
        return memory

    def update(self, memory: VideoMemory, scene_graph: SceneGraph) -> VideoMemory:
        return self.update_from_graph(memory, scene_graph)

    def update_from_graph(self, memory: VideoMemory, scene_graph: SceneGraph) -> VideoMemory:
        memory.temporal_history.append(scene_graph)
        observed_region_ids: set[str] = set()
        observed_entities: set[str] = set()

        for person in scene_graph.persons:
            observed_entities.add(person.person_id)
            identity = memory.identity_memory.get(person.person_id)
            if identity is not None:
                self._refresh_entry(identity, scene_graph.frame_index)
            for part in person.body_parts:
                region_id = f"{person.person_id}:{part.part_type}"
                if part.visibility in ("visible", "partially_visible"):
                    observed_region_ids.add(region_id)
                    memory.region_descriptors[region_id] = RegionDescriptor(
                        region_id=region_id,
                        entity_id=person.person_id,
                        region_type=part.part_type,
                        bbox=BBox(person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h),
                        visibility=part.visibility,
                        confidence=part.confidence,
                        last_update_frame=scene_graph.frame_index,
                    )
                else:
                    self.apply_visibility_event(
                        memory,
                        delta={"region_id": region_id, "entity": person.person_id},
                        masks={},
                        visibility="hidden",
                    )
            for garment in person.garments:
                observed_entities.add(garment.garment_id)
                garment_entry = memory.garment_memory.get(garment.garment_id)
                if garment_entry is not None and garment.visibility in ("visible", "partially_visible"):
                    self._refresh_entry(garment_entry, scene_graph.frame_index)

        for entry in list(memory.identity_memory.values()) + list(memory.garment_memory.values()):
            if entry.entity_id not in observed_entities:
                entry.confidence *= 0.96

        for region_id, descriptor in memory.region_descriptors.items():
            if region_id not in observed_region_ids:
                descriptor.confidence *= 0.95
                descriptor.visibility = "hidden" if descriptor.confidence < 0.3 else descriptor.visibility

        for slot in memory.hidden_region_slots.values():
            slot.stale_frames += 0 if slot.slot_id in observed_region_ids else 1
            slot.confidence = max(0.0, slot.confidence * (0.99 if slot.stale_frames < 3 else 0.95))

        return memory

    def update_from_frame(self, memory: VideoMemory, frame: list, scene_graph: SceneGraph) -> VideoMemory:
        color = mean_color(frame)
        for person in scene_graph.persons:
            patch_id = f"patch::{person.person_id}:latest"
            memory.texture_patches[patch_id] = TexturePatchMemory(
                patch_id=patch_id,
                region_type="person",
                entity_id=person.person_id,
                source_frame=scene_graph.frame_index,
                patch_ref=f"rgb://{color[0]:.3f},{color[1]:.3f},{color[2]:.3f}",
                confidence=min(1.0, 0.6 + person.confidence * 0.3),
            )
        return memory

    def mark_region_revealed(self, memory: VideoMemory, region_id: str, owner_entity: str) -> None:
        self.apply_visibility_event(memory, {"region_id": region_id, "entity": owner_entity}, {}, visibility="revealed")

    def query_hidden_region(self, memory: VideoMemory, region_id: str) -> HiddenRegionSlot | None:
        return memory.hidden_region_slots.get(region_id)

    def retrieve_for_region(self, memory: VideoMemory, region_type: str, owner_entity: str | None = None) -> list[TexturePatchMemory]:
        patches = [p for p in memory.texture_patches.values() if p.region_type == region_type]
        if owner_entity is not None:
            patches = [p for p in patches if p.entity_id == owner_entity]
        return sorted(patches, key=lambda p: p.confidence, reverse=True)

    def apply_visibility_event(self, memory: VideoMemory, delta: dict[str, str], masks: dict[str, object], visibility: str) -> None:
        _ = masks
        region_id = delta.get("region_id", "unknown")
        owner = delta.get("entity", "unknown")
        slot = memory.hidden_region_slots.get(region_id)
        if slot is None:
            memory.hidden_region_slots[region_id] = HiddenRegionSlot(
                slot_id=region_id,
                region_type=region_id.split(":")[-1],
                owner_entity=owner,
                candidate_patch_ids=[],
                confidence=0.55,
            )
            slot = memory.hidden_region_slots[region_id]
        if visibility == "revealed":
            slot.confidence = min(1.0, slot.confidence + 0.15)
            slot.stale_frames = 0
        else:
            slot.confidence = max(0.0, slot.confidence - 0.1)
            slot.stale_frames += 1

    def retrieve(self, memory: VideoMemory, query_embedding: list[float], bank: str = "texture", top_k: int = 3) -> list[dict[str, object]]:
        qn = math.sqrt(sum(v * v for v in query_embedding)) or 1.0
        q = [v / qn for v in query_embedding]

        if bank == "identity":
            entries = list(memory.identity_memory.values())
        elif bank == "garment":
            entries = list(memory.garment_memory.values())
        else:
            entries = [
                MemoryEntry(entity_id=v.patch_id, entry_type="texture", embedding=self._encode_visual(v.patch_id), confidence=v.confidence)
                for v in memory.texture_patches.values()
            ]

        scored: list[tuple[float, MemoryEntry]] = []
        for e in entries:
            en = math.sqrt(sum(v * v for v in e.embedding)) or 1.0
            emb = [v / en for v in e.embedding]
            sim = sum(a * b for a, b in zip(q, emb))
            scored.append((float(sim), e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "entity_id": e.entity_id,
                "similarity": round(sim, 4),
                "confidence": round((sim + e.confidence) / 2.0, 4),
            }
            for sim, e in scored[:top_k]
        ]

    def query_by_region(self, memory: VideoMemory, region_type: str) -> list[RegionDescriptor]:
        return [descriptor for descriptor in memory.region_descriptors.values() if descriptor.region_type == region_type]

    def query_by_entity(self, memory: VideoMemory, entity_id: str) -> dict[str, object]:
        regions = [descriptor for descriptor in memory.region_descriptors.values() if descriptor.entity_id == entity_id]
        patches = [patch for patch in memory.texture_patches.values() if patch.entity_id == entity_id]
        return {
            "identity": memory.identity_memory.get(entity_id),
            "garment": memory.garment_memory.get(entity_id),
            "regions": regions,
            "patches": patches,
        }

    def query_hidden_candidate(self, memory: VideoMemory, region_type: str) -> TexturePatchMemory | None:
        candidates = [p for p in memory.texture_patches.values() if p.region_type == region_type]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.confidence)

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

        return memory

    def _refresh_entry(self, entry: MemoryEntry, frame_index: int) -> None:
        entry.confidence = min(1.0, entry.confidence * 0.9 + 0.1)
        entry.last_seen_frames.append(frame_index)

    def _seed_person_semantic_regions(self, memory: VideoMemory, person_id: str, bbox: BBox, frame_index: int) -> None:
        for offset, region_type in enumerate(self._SEMANTIC_REGIONS):
            region_id = f"{person_id}:{region_type}"
            memory.region_descriptors[region_id] = RegionDescriptor(
                region_id=region_id,
                entity_id=person_id,
                region_type=region_type,
                bbox=BBox(bbox.x, bbox.y + 0.02 * offset, bbox.w, bbox.h),
                visibility="visible" if region_type in ("face", "torso") else "partially_visible",
                confidence=0.8,
                last_update_frame=frame_index,
            )
            patch_id = f"patch::{region_id}"
            memory.texture_patches[patch_id] = TexturePatchMemory(
                patch_id=patch_id,
                region_type=region_type,
                entity_id=person_id,
                source_frame=frame_index,
                patch_ref=f"mem://{patch_id}",
                confidence=0.75,
            )
            if region_type in ("sleeves", "garments"):
                slot_id = f"{person_id}:{region_type}"
                memory.hidden_region_slots[slot_id] = HiddenRegionSlot(
                    slot_id=slot_id,
                    region_type=region_type,
                    owner_entity=person_id,
                    candidate_patch_ids=[patch_id],
                    confidence=0.7,
                )
