from __future__ import annotations

from dataclasses import asdict

from core.schema import (
    BBox,
    HiddenRegionSlot,
    MemoryEntry,
    RegionDescriptor,
    SceneGraph,
    TexturePatchMemory,
    VideoMemory,
)


class MemoryManager:
    _SEMANTIC_REGIONS = ("face", "hair", "torso", "sleeves", "garments")

    def initialize(self, scene_graph: SceneGraph) -> VideoMemory:
        memory = VideoMemory(temporal_history=[scene_graph])
        for person in scene_graph.persons:
            memory.identity_memory[person.person_id] = MemoryEntry(
                entity_id=person.person_id,
                entry_type="identity",
                embedding=[0.1, 0.2, 0.3],
                confidence=person.confidence,
                last_seen_frames=[scene_graph.frame_index],
            )
            self._seed_person_semantic_regions(memory, person.person_id, person.bbox, scene_graph.frame_index)
            for garment in person.garments:
                memory.garment_memory[garment.garment_id] = MemoryEntry(
                    entity_id=garment.garment_id,
                    entry_type="garment",
                    embedding=[0.3, 0.1, 0.4],
                    confidence=garment.confidence,
                    last_seen_frames=[scene_graph.frame_index],
                )
        return memory

    def update(self, memory: VideoMemory, scene_graph: SceneGraph) -> VideoMemory:
        memory.temporal_history.append(scene_graph)
        observed_region_ids: set[str] = set()
        observed_entities: set[str] = set()

        for person in scene_graph.persons:
            observed_entities.add(person.person_id)
            identity = memory.identity_memory.get(person.person_id)
            if identity is not None:
                self._refresh_entry(identity, scene_graph.frame_index)
            for part in person.body_parts:
                if part.visibility in ("visible", "partially_visible"):
                    region_id = f"{person.person_id}:{part.part_type}"
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
