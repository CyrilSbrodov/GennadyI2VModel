from core.schema import BBox, PersonNode, SceneGraph
from memory.video_memory import MemoryManager


def test_hidden_region_reveal_keeps_texture_similarity() -> None:
    manager = MemoryManager()
    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.5, 0.8), mask_ref=None)
    graph = SceneGraph(frame_index=0, persons=[person], objects=[])
    memory = manager.initialize(graph)

    q = manager._encode_visual("patch::p1:garments")
    before = manager.retrieve(memory, q, bank="texture", top_k=1)[0]

    manager.apply_visibility_event(memory, {"region_id": "p1:garments", "entity": "p1"}, {}, "hidden")
    manager.apply_visibility_event(memory, {"region_id": "p1:garments", "entity": "p1"}, {}, "revealed")

    after = manager.retrieve(memory, q, bank="texture", top_k=1)[0]
    assert before["entity_id"] == after["entity_id"]
