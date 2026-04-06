from __future__ import annotations

from core.schema import GraphDelta, SceneGraph


def build_training_targets(graph_sequence: list[SceneGraph]) -> list[GraphDelta]:
    """Build coarse graph deltas between consecutive frames."""
    targets: list[GraphDelta] = []
    for prev_graph, next_graph in zip(graph_sequence, graph_sequence[1:]):
        frame_delta = float(next_graph.frame_index - prev_graph.frame_index)
        targets.append(GraphDelta(pose_deltas={"frame_delta": frame_delta}))
    return targets
