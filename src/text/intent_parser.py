from __future__ import annotations

import re

from core.schema import ActionPlan, ActionStep


class IntentParser:
    """Rule-assisted parser turning RU text into structured action plan."""

    _patterns = [
        (r"снима[её]т?\s+пальто", "remove_garment", {"target_entity": "coat"}),
        (r"расстегива[её]т?\s+куртк", "open_garment", {"target_entity": "jacket"}),
        (r"садит[сc]я\s+на\s+стул", "sit_down", {"target_object": "chair"}),
        (r"вста[её]т", "stand_up", {}),
        (r"улыба[её]т?с?я", "smile", {"intensity": 0.7}),
        (r"поднима[её]т?\s+рук", "raise_arm", {"target_entity": "arm"}),
        (r"поворачива[её]т?\s+голов", "turn_head", {"target_entity": "head"}),
    ]

    def parse(self, text: str) -> ActionPlan:
        normalized = text.lower()
        actions: list[ActionStep] = []

        for idx, (pattern, action_type, attrs) in enumerate(self._patterns, start=1):
            if re.search(pattern, normalized):
                actions.append(
                    ActionStep(
                        type=action_type,
                        priority=idx,
                        target_entity=attrs.get("target_entity"),
                        target_object=attrs.get("target_object"),
                        intensity=attrs.get("intensity"),
                    )
                )

        if not actions:
            actions.append(ActionStep(type="hold_pose", priority=1))

        actions = sorted(actions, key=lambda a: a.priority)
        return ActionPlan(actions=actions, temporal_ordering=list(range(len(actions))))
