from __future__ import annotations

import re
from dataclasses import dataclass

from core.schema import ActionPlan, ActionStep, SceneGraph


@dataclass(slots=True)
class SemanticIntent:
    text: str
    clauses: list[str]
    modifiers: dict[str, str | float | bool]


class IntentParser:
    _patterns = [
        (r"снима[её]т?\s+пальто", "remove_garment", {"target_entity": "coat", "body_part": "hands"}),
        (r"расстегива[её]т?\s+куртк", "open_garment", {"target_entity": "jacket", "body_part": "hands"}),
        (r"садит[сc]я\s+на\s+стул", "sit_down", {"target_object": "chair"}),
        (r"вста[её]т", "stand_up", {}),
        (r"улыба[её]т?с?я", "smile", {"intensity": 0.7, "body_part": "face"}),
        (r"поднима[её]т?\s+рук", "raise_arm", {"target_entity": "arm", "body_part": "arm"}),
        (r"поворачива[её]т?\s+голов", "turn_head", {"target_entity": "head", "body_part": "head"}),
    ]

    def semantic_encode(self, text: str) -> SemanticIntent:
        normalized = text.lower().replace("ё", "е")
        clauses = [c.strip() for c in re.split(r"[\.,;]", normalized) if c.strip()]
        modifiers: dict[str, str | float | bool] = {
            "speed": self._extract_speed(normalized),
            "style": self._extract_style(normalized),
            "intensity": self._extract_intensity(normalized) or 0.6,
            "simultaneously": "одновременно" in normalized,
            "has_before": "до того" in normalized or "before" in normalized,
            "has_after": "после" in normalized or "after" in normalized or "потом" in normalized,
            "has_while": "пока" in normalized or "while" in normalized,
        }
        return SemanticIntent(text=normalized, clauses=clauses, modifiers=modifiers)

    def structured_decode(self, semantic: SemanticIntent, scene_graph: SceneGraph | None = None) -> ActionPlan:
        actions: list[ActionStep] = []
        intensity_hint = float(semantic.modifiers.get("intensity", 0.6))
        speed = str(semantic.modifiers.get("speed", "normal"))
        style = str(semantic.modifiers.get("style", "neutral"))

        priority = 1
        for clause in semantic.clauses:
            clause_matches = 0
            for pattern, action_type, attrs in self._patterns:
                if re.search(pattern, clause):
                    clause_matches += 1
                    step = ActionStep(
                        type=action_type,
                        priority=priority,
                        target_entity=attrs.get("target_entity"),
                        target_object=attrs.get("target_object"),
                        body_part=attrs.get("body_part"),
                        intensity=float(attrs.get("intensity", intensity_hint)),
                        duration_sec=self._estimate_duration(speed),
                        can_run_parallel=bool(semantic.modifiers.get("simultaneously", False)),
                        constraints=[],
                        modifiers={
                            "speed": speed,
                            "style": style,
                            "schema": {
                                "action_type": action_type,
                                "target": attrs.get("target_entity") or attrs.get("target_object"),
                                "modifiers": {"speed": speed, "style": style},
                                "temporal_dependencies": [],
                                "constraints": [],
                            },
                        },
                    )
                    actions.append(step)
                    priority += 1
            if clause_matches == 0 and clause:
                actions.append(ActionStep(type="hold_pose", priority=priority, duration_sec=1.0, modifiers={"fallback": True}))
                priority += 1

        if not actions:
            actions.append(ActionStep(type="hold_pose", priority=1, duration_sec=1.0))

        actions = sorted(actions, key=lambda a: a.priority)
        for idx, action in enumerate(actions):
            if idx > 0 and bool(semantic.modifiers.get("has_after", False)):
                action.start_after.append(idx - 1)
            if bool(semantic.modifiers.get("has_while", False)) and idx > 0:
                action.can_run_parallel = True

        constraints = self._validate_against_scene(actions, scene_graph)
        parallel_groups = [list(range(len(actions)))] if bool(semantic.modifiers.get("simultaneously", False)) and len(actions) > 1 else []
        return ActionPlan(
            actions=actions,
            temporal_ordering=list(range(len(actions))),
            constraints=constraints,
            parallel_groups=parallel_groups,
            global_style=style,
        )

    def parse(self, text: str, scene_graph: SceneGraph | None = None) -> ActionPlan:
        semantic = self.semantic_encode(text)
        return self.structured_decode(semantic, scene_graph=scene_graph)

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
        body_parts = {part.part_type for person in scene_graph.persons for part in person.body_parts}
        garment_types = {garment.garment_type for person in scene_graph.persons for garment in person.garments}

        has_outer_garment = any(gt in {"coat", "jacket", "hoodie"} for gt in garment_types)
        constraints: list[str] = []

        for action in actions:
            if action.type == "sit_down" and "chair" not in object_types:
                constraints.append("requires:chair")
            if action.type == "remove_garment" and not has_outer_garment:
                constraints.append("requires:outer_garment")
            if action.body_part and action.body_part not in body_parts and action.body_part not in {"face", "arm", "hands"}:
                constraints.append(f"missing:body_part:{action.body_part}")
        return constraints
