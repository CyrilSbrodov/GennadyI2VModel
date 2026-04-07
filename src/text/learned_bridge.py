from __future__ import annotations

import math

from core.schema import ActionPlan, SceneGraph
from learned.interfaces import TextEncoder, TextEncodingOutput
from text.intent_parser import IntentParser


class BaselineTextEncoderAdapter(TextEncoder):
    """Deterministic parser adapted to learned text-encoder contract."""

    def __init__(self) -> None:
        self.parser = IntentParser()

    def encode(self, text: str, scene_graph: SceneGraph | None = None, action_plan: ActionPlan | None = None) -> TextEncodingOutput:
        plan = action_plan or self.parser.parse(text, scene_graph=scene_graph)
        tokens = [action.type for action in plan.actions]
        target_entities = sorted({a.target_entity for a in plan.actions if a.target_entity})
        target_objects = sorted({a.target_object for a in plan.actions if a.target_object})
        norms = [float((abs(hash(tok)) % 1000) / 1000.0) for tok in tokens] or [0.0]
        nrm = math.sqrt(sum(v * v for v in norms)) or 1.0
        embedding = [v / nrm for v in norms][:16]
        decomposition = [
            {
                "index": idx,
                "action": step.type,
                "start_after": step.start_after,
                "parallel": step.can_run_parallel,
                "duration_sec": step.duration_sec,
            }
            for idx, step in enumerate(plan.actions)
        ]
        avg_conf = 0.0
        if plan.actions:
            confs = [float(step.modifiers.get("parser_confidence", 0.5)) for step in plan.actions]
            avg_conf = sum(confs) / len(confs)

        return TextEncodingOutput(
            action_embedding=embedding,
            structured_action_tokens=tokens,
            target_hints={"entities": target_entities, "objects": target_objects},
            temporal_hints={
                "ordering": plan.temporal_ordering,
                "parallel_groups": plan.parallel_groups,
                "global_style": plan.global_style,
            },
            decomposition_hints=decomposition,
            constraints=plan.constraints[:],
            confidence=avg_conf,
            alignment={
                "source": "baseline_intent_parser",
                "action_count": len(plan.actions),
                "has_constraints": bool(plan.constraints),
            },
            trace=["adapter=baseline_text_encoder", f"actions={tokens}"],
        )
