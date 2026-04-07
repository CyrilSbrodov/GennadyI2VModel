from __future__ import annotations

import math

from learned.interfaces import GraphEncoder, GraphEncodingOutput, IdentityAppearanceEncoder
from memory.summaries import AppearanceMemorySummarizer
from core.schema import SceneGraph


class BaselineGraphEncoder(GraphEncoder):
    def encode(self, scene_graph) -> GraphEncodingOutput:
        serialized = _serialize_graph_for_bridge(scene_graph)
        counts = [float(len(scene_graph.persons)), float(len(scene_graph.objects)), float(len(scene_graph.relations)), float(scene_graph.frame_index)]
        n = math.sqrt(sum(v * v for v in counts)) or 1.0
        emb = [v / n for v in counts]
        return GraphEncodingOutput(graph_embedding=emb, serialized_graph=serialized, confidence=0.62)


class BaselineIdentityAppearanceEncoder(IdentityAppearanceEncoder):
    def encode_identity(self, memory_summary: dict[str, object], entity_id: str) -> list[float]:
        identity = memory_summary.get("identity", {}) if isinstance(memory_summary, dict) else {}
        rec = identity.get(entity_id, {}) if isinstance(identity, dict) else {}
        emb = rec.get("embedding", []) if isinstance(rec, dict) else []
        return [float(x) for x in emb] if emb else [0.0] * 8


def summarize_memory(memory) -> dict[str, object]:
    return AppearanceMemorySummarizer().summarize(memory).as_dict()


def _serialize_graph_for_bridge(graph: SceneGraph) -> dict[str, object]:
    return {
        "frame_index": graph.frame_index,
        "persons": [{"person_id": p.person_id, "track_id": p.track_id} for p in graph.persons],
        "objects": [{"object_id": o.object_id, "object_type": o.object_type} for o in graph.objects],
        "relation_count": len(graph.relations),
    }
