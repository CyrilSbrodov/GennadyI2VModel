from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DynamicsInputs:
    scene_graph_features: list[float]
    action_embedding: list[float]
    planner_context: list[float]
    memory_features: list[float]


class DynamicsModel:
    """Lightweight placeholder model for graph-delta prediction."""

    def forward(self, inputs: DynamicsInputs) -> dict[str, float]:
        merged = inputs.scene_graph_features + inputs.action_embedding + inputs.planner_context + inputs.memory_features
        if not merged:
            return {"delta_scale": 0.0}
        score = sum(merged) / len(merged)
        return {"delta_scale": max(-1.0, min(1.0, score))}
