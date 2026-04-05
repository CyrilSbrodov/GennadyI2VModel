from __future__ import annotations

from core.schema import MemoryEntry, SceneGraph, VideoMemory


class MemoryManager:
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
        for entry in list(memory.identity_memory.values()) + list(memory.garment_memory.values()):
            entry.last_seen_frames.append(scene_graph.frame_index)
        return memory
