from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DynamicsInputs:
    serialized_scene_graph: str
    action_tokens: list[str]
    planner_context: list[float]
    memory_features: list[float]


class DynamicsModel:
    """Deterministic lightweight model over serialized graph/action context."""

    def forward(self, inputs: DynamicsInputs) -> dict[str, float]:
        scene_score = (sum(ord(ch) for ch in inputs.serialized_scene_graph) % 1000) / 1000.0
        action_score = len(inputs.action_tokens) / 10.0
        planner_score = sum(inputs.planner_context) / max(1.0, len(inputs.planner_context))
        memory_score = sum(inputs.memory_features) / max(1.0, len(inputs.memory_features))
        raw = 0.35 * scene_score + 0.25 * action_score + 0.2 * planner_score + 0.2 * memory_score
        return {
            "pose": max(0.0, min(1.0, raw + 0.1)),
            "garment": max(0.0, min(1.0, raw)),
            "expression": max(0.0, min(1.0, raw - 0.1)),
            "interaction": max(0.0, min(1.0, raw + 0.05)),
            "visibility": max(0.0, min(1.0, 1.0 - raw)),
        }
