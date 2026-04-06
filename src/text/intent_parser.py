from __future__ import annotations

import re

from core.schema import ActionPlan, ActionStep, SceneGraph


class IntentParser:
    """Rule-assisted parser turning RU text into structured action plan."""

    _patterns = [
        (r"снима[её]т?\s+пальто", "remove_garment", {"target_entity": "coat", "body_part": "hands"}),
        (r"расстегива[её]т?\s+куртк", "open_garment", {"target_entity": "jacket", "body_part": "hands"}),
        (r"садит[сc]я\s+на\s+стул", "sit_down", {"target_object": "chair"}),
        (r"вста[её]т", "stand_up", {}),
        (r"улыба[её]т?с?я", "smile", {"intensity": 0.7, "body_part": "face"}),
        (r"поднима[её]т?\s+рук", "raise_arm", {"target_entity": "arm", "body_part": "arm"}),
        (r"поворачива[её]т?\s+голов", "turn_head", {"target_entity": "head", "body_part": "head"}),
    ]

    _action_vocab = {
        "снимает": "remove_garment",
        "снять": "remove_garment",
        "улыбка": "smile",
        "улыбается": "smile",
        "садится": "sit_down",
        "встает": "stand_up",
        "поднимает": "raise_arm",
        "поворачивает": "turn_head",
    }

    def parse(self, text: str, scene_graph: SceneGraph | None = None) -> ActionPlan:
        normalized = self._normalize_text(text)
        actions: list[ActionStep] = []

        speed = self._extract_speed(normalized)
        style = self._extract_style(normalized)
        intensity_hint = self._extract_intensity(normalized)

        for idx, (pattern, action_type, attrs) in enumerate(self._patterns, start=1):
            if re.search(pattern, normalized):
                step = ActionStep(
                    type=action_type,
                    priority=idx,
                    target_entity=attrs.get("target_entity"),
                    target_object=attrs.get("target_object"),
                    body_part=attrs.get("body_part"),
                    intensity=attrs.get("intensity", intensity_hint),
                    duration_sec=self._estimate_duration(speed),
                    can_run_parallel="одновременно" in normalized,
                    modifiers={"speed": speed, "style": style},
                )
                actions.append(step)

        if not actions:
            actions.append(ActionStep(type="hold_pose", priority=1, duration_sec=1.0))

        actions = sorted(actions, key=lambda a: a.priority)
        for idx, action in enumerate(actions):
            if idx > 0 and any(marker in normalized for marker in ("после", "затем", "потом")):
                action.start_after.append(idx - 1)

        constraints = self._validate_against_scene(actions, scene_graph)
        parallel_groups = [list(range(len(actions)))] if "одновременно" in normalized and len(actions) > 1 else []

        return ActionPlan(
            actions=actions,
            temporal_ordering=list(range(len(actions))),
            constraints=constraints,
            parallel_groups=parallel_groups,
            global_style=style,
        )

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower().replace("ё", "е")
        return normalized

    def _extract_speed(self, text: str) -> str:
        if "быстро" in text:
            return "fast"
        if "медленно" in text:
            return "slow"
        return "normal"

    def _extract_style(self, text: str) -> str:
        if "аккуратно" in text:
            return "careful"
        if "резко" in text:
            return "sharp"
        return "neutral"

    def _extract_intensity(self, text: str) -> float | None:
        if "сильно" in text:
            return 0.9
        if "слегка" in text:
            return 0.4
        return None

    def _estimate_duration(self, speed: str) -> float:
        if speed == "fast":
            return 0.8
        if speed == "slow":
            return 1.8
        return 1.2

    def _validate_against_scene(self, actions: list[ActionStep], scene_graph: SceneGraph | None) -> list[str]:
        if scene_graph is None:
            return []

        object_types = {obj.object_type for obj in scene_graph.objects}
        has_outer_garment = any(
            garment.garment_type in {"coat", "jacket", "hoodie"}
            for person in scene_graph.persons
            for garment in person.garments
        )
        constraints: list[str] = []

        for action in actions:
            if action.type == "sit_down" and "chair" not in object_types:
                constraints.append("requires:chair")
            if action.type == "remove_garment" and not has_outer_garment:
                constraints.append("requires:outer_garment")
        return constraints
