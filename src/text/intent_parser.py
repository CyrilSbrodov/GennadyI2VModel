from __future__ import annotations

import re

from core.schema import ActionPlan, ActionStep, SceneGraph
from text.contracts import (
    ActionCandidate,
    ConstraintHint,
    ParsedClause,
    ParsedIntent,
    ResolvedTarget,
    SemanticIntent,
    TargetReference,
)
from text.grounding import SceneGroundingIndex, build_scene_grounding_index
from text.modifiers import extract_modifiers
from text.temporal import extract_temporal_relations, split_clauses


class IntentParser:
    """Многоступенчатый scene-aware parser для построения ActionPlan через structured intent."""

    _semantic_families: dict[str, list[tuple[str, str]]] = {
        "pose": [
            ("sit", "сад"),
            ("stand", "вста"),
            ("raise_arm", "поднима"),
            ("lower_arm", "опуска"),
            ("turn_head", "поворач"),
            ("bend", "наклон"),
            ("step", "шаг"),
            ("shift_weight", "вес"),
            ("hold_pose", "замр"),
        ],
        "garment": [
            ("remove", "снима"),
            ("open", "расстег"),
            ("loosen", "ослаб"),
            ("adjust", "поправ"),
            ("reveal_inner", "оголя"),
            ("cover", "прикры"),
            ("uncover", "откры"),
        ],
        "expression": [
            ("smile", "улыба"),
            ("frown", "хмур"),
            ("look", "смотр"),
            ("relax_expression", "расслаб"),
        ],
        "interaction": [
            ("touch", "каса"),
            ("hold", "держ"),
            ("lean_on", "опира"),
            ("approach_contact", "подход"),
        ],
        "visibility": [
            ("reveal", "показы"),
            ("hide", "скрыва"),
            ("show", "демонстр"),
            ("uncover", "откры"),
        ],
    }

    _legacy_action_map: dict[tuple[str, str], str] = {
        ("pose", "sit"): "sit_down",
        ("pose", "stand"): "stand_up",
        ("pose", "raise_arm"): "raise_arm",
        ("pose", "lower_arm"): "lower_arm",
        ("pose", "turn_head"): "turn_head",
        ("pose", "bend"): "bend",
        ("pose", "step"): "walk_step",
        ("pose", "shift_weight"): "shift_weight",
        ("pose", "hold_pose"): "hold_pose",
        ("garment", "remove"): "remove_garment",
        ("garment", "open"): "open_garment",
        ("garment", "loosen"): "loosen_garment",
        ("garment", "adjust"): "adjust_garment",
        ("garment", "reveal_inner"): "reveal_inner_layer",
        ("garment", "cover"): "cover",
        ("garment", "uncover"): "uncover",
        ("expression", "smile"): "smile",
        ("expression", "frown"): "frown",
        ("expression", "look"): "look",
        ("expression", "relax_expression"): "relax_expression",
        ("interaction", "touch"): "touch",
        ("interaction", "hold"): "hold",
        ("interaction", "sit_on_support"): "sit_down",
        ("interaction", "lean_on"): "lean_on_object",
        ("interaction", "approach_contact"): "approach_contact",
        ("visibility", "reveal"): "reveal",
        ("visibility", "hide"): "hide",
        ("visibility", "show"): "show",
        ("visibility", "uncover"): "uncover",
    }

    _garment_reference_hints = {
        "куртк": "outerwear",
        "пальт": "outerwear",
        "худи": "upper_garment",
        "кофт": "upper_garment",
        "одежд": "garment_like",
        "верхн": "outerwear",
        "рукав": "sleeve_region",
        "воротник": "neck_region",
    }

    _support_reference_hints = ("стул", "кресл", "табур", "скам")

    def semantic_encode(self, text: str) -> SemanticIntent:
        """Строит structured intent и отдаёт совместимый semantic envelope."""

        parsed_intent = self._parse_to_structured_intent(text, scene_graph=None)
        modifiers = extract_modifiers(parsed_intent.normalized_text).as_dict()
        clauses = [clause.text for clause in parsed_intent.clauses]
        trace = [f"normalized:{parsed_intent.normalized_text}", f"clauses:{len(clauses)}"]
        return SemanticIntent(
            text=parsed_intent.normalized_text,
            clauses=clauses,
            modifiers={k: v for k, v in modifiers.items() if v is not None},
            trace=trace,
            parsed_intent=parsed_intent,
        )

    def structured_decode(self, semantic: SemanticIntent, scene_graph: SceneGraph | None = None) -> ActionPlan:
        """Декодирует structured intent в ActionPlan с temporal и constraints."""

        parsed_intent = semantic.parsed_intent or self._parse_to_structured_intent(semantic.text, scene_graph=None)
        if scene_graph is not None:
            parsed_intent = self._parse_to_structured_intent(parsed_intent.normalized_text, scene_graph=scene_graph)
        return self._build_action_plan(parsed_intent)

    def parse(self, text: str, scene_graph: SceneGraph | None = None) -> ActionPlan:
        """Публичный API: текст -> ActionPlan (совместимо с текущим runtime)."""

        parsed_intent = self._parse_to_structured_intent(text, scene_graph=scene_graph)
        return self._build_action_plan(parsed_intent)

    def parse_to_structured_intent(self, text: str, scene_graph: SceneGraph | None = None) -> ParsedIntent:
        """Публичный API для structured intent без сборки ActionPlan."""

        return self._parse_to_structured_intent(text, scene_graph=scene_graph)

    def _parse_to_structured_intent(self, text: str, scene_graph: SceneGraph | None) -> ParsedIntent:
        """Основной пайплайн: normalize -> clauses -> candidates -> grounding -> temporal."""

        normalized = self._normalize_text(text)
        clause_texts = split_clauses(normalized)
        index = build_scene_grounding_index(scene_graph)
        parsed = ParsedIntent(normalized_text=normalized)

        for i, clause_text in enumerate(clause_texts):
            clause = ParsedClause(index=i, text=clause_text)
            clause.modifiers = extract_modifiers(clause_text)
            clause.action_candidates = self._detect_action_candidates(clause)
            clause.target_references = self._detect_target_references(clause)
            clause.resolved_targets = self._resolve_targets(clause, index)
            clause.constraints = self._extract_constraints(clause, index)
            clause.ambiguities.extend(self._extract_ambiguities(clause))
            parsed.clauses.append(clause)
            parsed.explainability_trace.append(
                {
                    "clause": clause.text,
                    "actions": [c.semantic_action for c in clause.action_candidates],
                    "targets": [r.target_entity_class for r in clause.resolved_targets],
                    "constraints": [c.requirement for c in clause.constraints],
                    "ambiguities": clause.ambiguities[:],
                }
            )

        parsed.temporal_relations = extract_temporal_relations(clause_texts)
        parsed.global_constraints = self._collect_global_constraints(parsed)
        parsed.parser_confidence = self._compute_parser_confidence(parsed)
        return parsed

    def _detect_action_candidates(self, clause: ParsedClause) -> list[ActionCandidate]:
        """Ищет action candidates через semantic families, а не через финальные labels."""

        found: list[ActionCandidate] = []
        for family, markers in self._semantic_families.items():
            for semantic_action, marker in markers:
                if marker in clause.text:
                    found.append(
                        ActionCandidate(
                            clause_index=clause.index,
                            trigger_text=marker,
                            semantic_family=family,
                            semantic_action=semantic_action,
                            confidence=0.82,
                            lexical_reason=f"marker:{marker}",
                        )
                    )

        if not found and any(tok in clause.text for tok in ("и", "потом", "затем")):
            for part in re.split(r"\s+(?:и|потом|затем)\s+", clause.text):
                part_clause = ParsedClause(index=clause.index, text=part)
                found.extend(self._detect_action_candidates(part_clause))

        if not found:
            found.append(
                ActionCandidate(
                    clause_index=clause.index,
                    trigger_text="fallback",
                    semantic_family="pose",
                    semantic_action="hold_pose",
                    confidence=0.3,
                    lexical_reason="fallback_generic",
                )
            )
        return found

    def _detect_target_references(self, clause: ParsedClause) -> list[TargetReference]:
        """Извлекает body/garment/object ссылки как отдельный слой parser."""

        refs: list[TargetReference] = []
        if any(h in clause.text for h in self._support_reference_hints):
            refs.append(TargetReference(clause_index=clause.index, reference_text="support", target_kind="object", target_region="support"))

        for hint, label in self._garment_reference_hints.items():
            if hint in clause.text:
                refs.append(TargetReference(clause_index=clause.index, reference_text=hint, target_kind="garment", target_region=label))

        if "рук" in clause.text:
            refs.append(TargetReference(clause_index=clause.index, reference_text="рука", target_kind="body", target_region="arm"))
        if any(t in clause.text for t in ("голов", "лиц", "улыб")):
            refs.append(TargetReference(clause_index=clause.index, reference_text="face", target_kind="body", target_region="face"))

        if not refs:
            refs.append(TargetReference(clause_index=clause.index, reference_text="self", target_kind="self", target_region=None))
        return refs

    def _resolve_targets(self, clause: ParsedClause, scene_index: SceneGroundingIndex) -> list[ResolvedTarget]:
        """Резолвит target refs в entity_class/id с confidence и причиной."""

        resolved: list[ResolvedTarget] = []
        for ref in clause.target_references:
            if ref.target_kind == "object" and ref.target_region == "support":
                target_id = next(iter(scene_index.object_ids_by_type.get("chair", [])), None)
                resolved.append(
                    ResolvedTarget(
                        clause_index=clause.index,
                        target_entity_class="object",
                        target_entity_id=target_id,
                        target_region="support",
                        target_object="chair",
                        grounding_confidence=0.9 if scene_index.has_support_object else 0.25,
                        resolution_reason="scene_support_lookup",
                        unresolved=not scene_index.has_support_object,
                    )
                )
                continue

            if ref.target_kind == "garment":
                garment_type = "outerwear" if ref.target_region in {"outerwear", "upper_garment"} else "garment_like"
                garment_id = self._pick_garment_id(scene_index)
                resolved.append(
                    ResolvedTarget(
                        clause_index=clause.index,
                        target_entity_class="garment",
                        target_entity_id=garment_id,
                        target_region=ref.target_region,
                        target_object=garment_type,
                        grounding_confidence=0.85 if garment_id else 0.35,
                        resolution_reason="garment_family_grounding",
                        unresolved=garment_id is None,
                    )
                )
                continue

            if ref.target_kind == "body":
                region = ref.target_region or "body"
                available = scene_index.has_face if region == "face" else scene_index.has_usable_arm if region == "arm" else bool(scene_index.body_part_types)
                resolved.append(
                    ResolvedTarget(
                        clause_index=clause.index,
                        target_entity_class="body",
                        target_entity_id=region if available else None,
                        target_region=region,
                        target_object=None,
                        grounding_confidence=0.88 if available else 0.4,
                        resolution_reason="body_region_grounding",
                        unresolved=not available,
                    )
                )
                continue

            resolved.append(
                ResolvedTarget(
                    clause_index=clause.index,
                    target_entity_class=ref.target_kind,
                    target_entity_id=None,
                    target_region=ref.target_region,
                    target_object=None,
                    grounding_confidence=0.5,
                    resolution_reason="generic_self_reference",
                    unresolved=False,
                )
            )

        return resolved

    def _extract_constraints(self, clause: ParsedClause, scene_index: SceneGroundingIndex) -> list[ConstraintHint]:
        """Извлекает required preconditions из actions и grounded targets."""

        constraints: list[ConstraintHint] = []
        actions = {(a.semantic_family, a.semantic_action) for a in clause.action_candidates}

        if ("pose", "sit") in actions or ("interaction", "sit_on_support") in actions:
            constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:support_object", reason="sit_action"))
            if not scene_index.has_support_object:
                constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:chair", reason="support_missing"))
        if any(fam == "garment" and act in {"remove", "open", "loosen"} for fam, act in actions):
            constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:garment_like_entity", reason="garment_action"))
            if not scene_index.has_outer_garment:
                constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:outer_garment", reason="outerwear_missing"))
        if any(fam == "expression" for fam, _ in actions):
            constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:visible_face", reason="expression_action"))
        if any((fam, act) in {("pose", "raise_arm"), ("interaction", "touch"), ("interaction", "hold")} for fam, act in actions):
            constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:usable_arm", reason="arm_action"))

        return constraints

    def _extract_ambiguities(self, clause: ParsedClause) -> list[str]:
        """Фиксирует неоднозначности по действиям/целям/grounding."""

        ambiguities: list[str] = []
        if any(a.confidence < 0.5 for a in clause.action_candidates):
            ambiguities.append("ambiguous_action")
        if any(rt.unresolved for rt in clause.resolved_targets):
            ambiguities.append("weak_grounding")
        if any(rt.target_entity_class == "self" for rt in clause.resolved_targets):
            ambiguities.append("generic_interpretation")
        return ambiguities

    def _collect_global_constraints(self, parsed: ParsedIntent) -> list[ConstraintHint]:
        """Сводит constraints всех клауз в общий набор без дублей."""

        unique: dict[str, ConstraintHint] = {}
        for clause in parsed.clauses:
            for hint in clause.constraints:
                unique.setdefault(hint.requirement, hint)
        return list(unique.values())

    def _compute_parser_confidence(self, parsed: ParsedIntent) -> float:
        """Считает интегральную parser-confidence с учетом ambiguities."""

        scores: list[float] = []
        for clause in parsed.clauses:
            for cand in clause.action_candidates:
                penalty = 0.1 * len(clause.ambiguities)
                scores.append(max(0.05, cand.confidence - penalty))
        return sum(scores) / len(scores) if scores else 0.0

    def _build_action_plan(self, parsed: ParsedIntent) -> ActionPlan:
        """Собирает финальный ActionPlan из structured intent."""

        actions: list[ActionStep] = []
        clause_to_actions: dict[int, list[int]] = {}
        global_speed = "normal"
        global_style = "neutral"

        for clause in parsed.clauses:
            if clause.modifiers.speed != "normal":
                global_speed = clause.modifiers.speed
            if clause.modifiers.carefulness == "careful":
                global_style = "careful"
            elif clause.modifiers.abruptness == "abrupt":
                global_style = "sharp"

            for cand in clause.action_candidates:
                action_type = self._legacy_action_map.get((cand.semantic_family, cand.semantic_action), "hold_pose")
                resolved = clause.resolved_targets[0] if clause.resolved_targets else None
                intensity = clause.modifiers.intensity if clause.modifiers.intensity is not None else 0.6
                duration = self._estimate_duration(clause.modifiers.speed)
                step = ActionStep(
                    type=action_type,
                    priority=len(actions) + 1,
                    target_entity=resolved.target_entity_id if resolved and resolved.target_entity_class != "object" else None,
                    target_object=resolved.target_object if resolved else None,
                    body_part=resolved.target_region if resolved and resolved.target_entity_class == "body" else None,
                    intensity=float(intensity),
                    duration_sec=duration,
                    can_run_parallel=clause.modifiers.simultaneity_hint,
                    constraints=[c.requirement for c in clause.constraints],
                    modifiers={
                        **clause.modifiers.as_dict(),
                        "parser_confidence": round(max(0.0, cand.confidence - 0.1 * len(clause.ambiguities)), 3),
                        "semantic_family": cand.semantic_family,
                        "semantic_action": cand.semantic_action,
                        "ambiguities": clause.ambiguities[:],
                        "trace": {
                            "trigger": cand.trigger_text,
                            "reason": cand.lexical_reason,
                            "resolution": [r.resolution_reason for r in clause.resolved_targets],
                        },
                    },
                )
                action_idx = len(actions)
                clause_to_actions.setdefault(clause.index, []).append(action_idx)
                actions.append(step)

        if not actions:
            actions = [ActionStep(type="hold_pose", priority=1, duration_sec=1.0, modifiers={"parser_confidence": 0.0})]

        parallel_groups: list[list[int]] = []
        for rel in parsed.temporal_relations:
            src_indices = clause_to_actions.get(rel.source_clause, [])
            tgt_indices = clause_to_actions.get(rel.target_clause, [])
            if rel.relation in {"after", "sequence"}:
                for src in src_indices:
                    if tgt_indices:
                        actions[src].start_after.extend(tgt_indices)
            if rel.relation == "parallel":
                group = sorted(set(src_indices + tgt_indices))
                if len(group) > 1:
                    parallel_groups.append(group)
                    for idx in group:
                        actions[idx].can_run_parallel = True

        constraints = sorted({c.requirement for c in parsed.global_constraints})
        for step in actions:
            step.start_after = sorted(set(step.start_after))

        return ActionPlan(
            actions=actions,
            temporal_ordering=list(range(len(actions))),
            constraints=constraints,
            parallel_groups=parallel_groups,
            global_style=global_style if global_style != "neutral" else ("slow" if global_speed == "slow" else "neutral"),
        )

    def _normalize_text(self, text: str) -> str:
        """Нормализует строку и приводит ключевые синонимы к общим формам."""

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

    def _estimate_duration(self, speed: str) -> float:
        """Оценка длительности действия по speed-модификатору."""

        if speed == "fast":
            return 0.8
        if speed == "slow":
            return 1.8
        return 1.2

    def _pick_garment_id(self, scene_index: SceneGroundingIndex) -> str | None:
        """Подбирает id одежды без жёсткой привязки к словарю всех garment labels."""

        for preferred in ("coat", "jacket", "hoodie", "outerwear", "upper_garment"):
            ids = scene_index.garment_ids_by_type.get(preferred, [])
            if ids:
                return ids[0]
        for ids in scene_index.garment_ids_by_type.values():
            if ids:
                return ids[0]
        return None
