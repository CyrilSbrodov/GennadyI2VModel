from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.schema import ActionPlan, ActionStep, SceneGraph


@dataclass(slots=True)
class SemanticIntent:
    text: str
    clauses: list[str]
    modifiers: dict[str, str | float | bool]
    trace: list[str] = field(default_factory=list)


class IntentParser:
    _action_lexicon: dict[str, tuple[str, dict[str, str]]] = {
        "снимает пальто": ("remove_garment", {"target_entity": "coat", "body_part": "hands"}),
        "снять пальто": ("remove_garment", {"target_entity": "coat", "body_part": "hands"}),
        "убирает пальто": ("remove_garment", {"target_entity": "coat", "body_part": "hands"}),
        "садится": ("sit_down", {}),
        "садится на стул": ("sit_down", {"target_object": "chair"}),
        "встает": ("stand_up", {}),
        "улыбается": ("smile", {"body_part": "face"}),
        "улыбнуться": ("smile", {"body_part": "face"}),
        "поднимает руку": ("raise_arm", {"target_entity": "arm", "body_part": "arm"}),
        "поднять руку": ("raise_arm", {"target_entity": "arm", "body_part": "arm"}),
        "поворачивает голову": ("turn_head", {"target_entity": "head", "body_part": "head"}),
        "сделать шаг": ("walk_step", {"target_entity": "legs", "body_part": "legs"}),
        "переносит вес": ("shift_weight", {"target_entity": "torso", "body_part": "torso"}),
    }
    _regex_fallback = [
        (r"снима[её]т?\s+пальто", "remove_garment", {"target_entity": "coat", "body_part": "hands"}),
        (r"садит[сc]я(\s+на\s+стул)?", "sit_down", {"target_object": "chair"}),
        (r"вста[её]т", "stand_up", {}),
        (r"улыба[её]т?с?я", "smile", {"body_part": "face"}),
        (r"поднима[её]т?\s+рук", "raise_arm", {"target_entity": "arm", "body_part": "arm"}),
        (r"поворачива[её]т?\s+голов", "turn_head", {"target_entity": "head", "body_part": "head"}),
    ]

    def semantic_encode(self, text: str) -> SemanticIntent:
        normalized = self._normalize_text(text)
        clauses = [c.strip() for c in re.split(r"[\.,;]", normalized) if c.strip()]
        modifiers: dict[str, str | float | bool] = {
            "speed": self._extract_speed(normalized),
            "style": self._extract_style(normalized),
            "intensity": self._extract_intensity(normalized) or 0.6,
            "simultaneously": any(w in normalized for w in ("одновременно", "вместе")),
            "has_before": any(w in normalized for w in ("до того", "before")),
            "has_after": any(w in normalized for w in ("после", "after", "потом", "затем", "сначала")),
            "has_while": any(w in normalized for w in ("пока", "while")),
        }
        return SemanticIntent(text=normalized, clauses=clauses, modifiers=modifiers, trace=[f"normalized:{normalized}"])

    def structured_decode(self, semantic: SemanticIntent, scene_graph: SceneGraph | None = None) -> ActionPlan:
        actions: list[ActionStep] = []
        intensity_hint = float(semantic.modifiers.get("intensity", 0.6))
        speed = str(semantic.modifiers.get("speed", "normal"))
        style = str(semantic.modifiers.get("style", "neutral"))

        priority = 1
        for clause in semantic.clauses:
            clause_actions = self._extract_actions_from_clause(clause)
            semantic.trace.append(f"clause:{clause}->actions:{[x[0] for x in clause_actions]}")
            if not clause_actions and clause:
                actions.append(ActionStep(type="hold_pose", priority=priority, duration_sec=1.0, modifiers={"fallback": True}))
                priority += 1
                continue

            for action_type, attrs, confidence in clause_actions:
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
                        "parser_confidence": confidence,
                        "parser_trace": semantic.trace[:],
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

        if not actions:
            actions.append(ActionStep(type="hold_pose", priority=1, duration_sec=1.0, modifiers={"parser_confidence": 0.0}))

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

    def _extract_actions_from_clause(self, clause: str) -> list[tuple[str, dict[str, str], float]]:
        found: list[tuple[str, dict[str, str], float]] = []
        for phrase, (action, attrs) in self._action_lexicon.items():
            if phrase in clause:
                merged = dict(attrs)
                if action == "sit_down" and "стул" in clause and "target_object" not in merged:
                    merged["target_object"] = "chair"
                found.append((action, merged, 0.9))

        # lightweight decomposition for conjunctions
        if not found and any(tok in clause for tok in (" и ", " затем ", " потом ")):
            for part in re.split(r"\s+(?:и|затем|потом)\s+", clause):
                found.extend(self._extract_actions_from_clause(part))

        if found:
            return found

        for pattern, action_type, attrs in self._regex_fallback:
            if re.search(pattern, clause):
                found.append((action_type, dict(attrs), 0.65))
        return found

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower().replace("ё", "е")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        synonyms = {
            "присаживается": "садится",
            "улыбнувшись": "улыбается",
            "поднимает руки": "поднимает руку",
            "приподнимает руку": "поднимает руку",
            "разворачивает голову": "поворачивает голову",
        }
        for src, dst in synonyms.items():
            normalized = normalized.replace(src, dst)
        return normalized

    def _extract_speed(self, text: str) -> str:
        if any(x in text for x in ("быстро", "резво")):
            return "fast"
        if any(x in text for x in ("медленно", "плавно")):
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
        if any(x in text for x in ("слегка", "чуть")):
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
            if action.type == "sit_down" and action.target_object == "chair" and "chair" not in object_types:
                constraints.append("requires:chair")
            if action.type == "remove_garment" and not has_outer_garment:
                constraints.append("requires:outer_garment")
            if action.type in {"walk_step", "shift_weight"} and "left_upper_leg" not in body_parts:
                constraints.append("requires:leg_pose")
        return sorted(set(constraints))
