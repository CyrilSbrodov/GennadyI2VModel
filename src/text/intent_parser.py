from __future__ import annotations

import re

from core.schema import ActionPlan, ActionStep, RuntimeSemanticTransition, SceneGraph, TransitionPhaseContract, TransitionTargetProfile
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
        "pose_transition": [
            ("seated_pose", "сад"),
            ("upright_pose", "вста"),
            ("arm_elevation", "поднима"),
            ("arm_lowering", "опуска"),
            ("head_rotation", "поворач"),
            ("body_reorientation", "наклон"),
            ("weight_shift", "вес"),
            ("pose_hold", "замр"),
        ],
        "garment_transition": [
            ("outer_layer_removal", "снима"),
            ("outer_layer_opening", "расстег"),
            ("garment_reposition", "ослаб"),
            ("garment_reposition", "поправ"),
            ("inner_layer_reveal", "оголя"),
            ("occlude_region", "прикры"),
            ("reveal_region", "откры"),
        ],
        "expression_transition": [
            ("smile_like", "улыба"),
            ("frown_like", "хмур"),
            ("gaze_shift", "смотр"),
            ("expression_relax", "расслаб"),
        ],
        "interaction_transition": [
            ("touch_contact", "каса"),
            ("touch_contact", "держ"),
            ("lean_support", "опира"),
            ("support_contact", "подход"),
        ],
        "visibility_transition": [
            ("reveal_region", "показы"),
            ("occlude_region", "скрыва"),
            ("reveal_region", "демонстр"),
            ("reveal_region", "откры"),
        ],
    }

    _goal_to_legacy_action_map: dict[str, str] = {
        "seated_pose": "sit_down",
        "upright_pose": "stand_up",
        "arm_elevation": "raise_arm",
        "arm_lowering": "lower_arm",
        "head_rotation": "turn_head",
        "body_reorientation": "bend",
        "weight_shift": "shift_weight",
        "pose_hold": "hold_pose",
        "outer_layer_removal": "remove_garment",
        "outer_layer_opening": "open_garment",
        "garment_reposition": "adjust_garment",
        "inner_layer_reveal": "reveal_inner_layer",
        "smile_like": "smile",
        "frown_like": "frown",
        "gaze_shift": "look",
        "expression_relax": "relax_expression",
        "touch_contact": "touch",
        "lean_support": "lean_on_object",
        "support_contact": "approach_contact",
        "support_release": "stand_up",
        "reveal_region": "reveal",
        "occlude_region": "hide",
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
                    semantic_family="pose_transition",
                    semantic_action="pose_hold",
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

        if ("pose_transition", "seated_pose") in actions:
            constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:support_object", reason="sit_action"))
            if not scene_index.has_support_object:
                constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:chair", reason="support_missing"))
        if any(fam == "garment_transition" and act in {"outer_layer_removal", "outer_layer_opening", "garment_reposition"} for fam, act in actions):
            constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:garment_like_entity", reason="garment_action"))
            if not scene_index.has_outer_garment:
                constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:outer_garment", reason="outerwear_missing"))
        if any(fam == "expression_transition" for fam, _ in actions):
            constraints.append(ConstraintHint(clause_index=clause.index, requirement="requires:visible_face", reason="expression_action"))
        if any((fam, act) in {("pose_transition", "arm_elevation"), ("interaction_transition", "touch_contact")} for fam, act in actions):
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
                resolved = self._select_resolved_target_for_candidate(cand, clause.resolved_targets)
                transition = self._build_runtime_semantic_transition(cand, clause, resolved)
                action_type = self._goal_to_legacy_action_map.get(transition.goal, "hold_pose")
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
                        "semantic_goal": transition.goal,
                        "semantic_transition": self._runtime_semantic_transition_to_dict(transition),
                        "ambiguities": clause.ambiguities[:],
                        "trace": {
                            "trigger": cand.trigger_text,
                            "reason": cand.lexical_reason,
                            "resolution": [r.resolution_reason for r in clause.resolved_targets],
                        },
                    },
                    semantic_transition=transition,
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

    def _select_resolved_target_for_candidate(self, candidate: ActionCandidate, resolved_targets: list[ResolvedTarget]) -> ResolvedTarget | None:
        if not resolved_targets:
            return None
        preferred_by_family = {
            "pose_transition": {"body", "self"},
            "expression_transition": {"body", "self"},
            "garment_transition": {"garment", "self"},
            "interaction_transition": {"object", "self"},
            "visibility_transition": {"body", "garment", "self"},
        }
        preferred = preferred_by_family.get(candidate.semantic_family, {"self"})
        for resolved in resolved_targets:
            if resolved.target_entity_class in preferred:
                return resolved
        return resolved_targets[0]

    def _build_runtime_semantic_transition(
        self,
        candidate: ActionCandidate,
        clause: ParsedClause,
        resolved: ResolvedTarget | None,
    ) -> RuntimeSemanticTransition:
        primary, secondary, context = self._structured_target_regions(candidate.semantic_family, candidate.semantic_action, resolved)
        target_profile = TransitionTargetProfile(
            primary_regions=primary,
            secondary_regions=secondary,
            context_regions=context,
            entity=resolved.target_entity_class if resolved else "self",
            entity_id=resolved.target_entity_id if resolved else None,
            object_role=resolved.target_object if resolved else None,
            support_target="chair" if resolved and resolved.target_region == "support" else None,
        )
        sequencing = "parallel" if clause.modifiers.simultaneity_hint else ("sequential" if clause.modifiers.sequencing_hint else "single")
        global_phase_sequence = self._global_phase_sequence()
        family_subphase_sequence = self._family_subphase_sequence(candidate.semantic_family, candidate.semantic_action)
        phase = TransitionPhaseContract(global_phase="prepare", global_phase_sequence=global_phase_sequence)
        if candidate.semantic_family == "pose_transition":
            phase.pose_subphase = family_subphase_sequence[0] if family_subphase_sequence else "steady"
            phase.pose_subphase_sequence = family_subphase_sequence
        if candidate.semantic_family == "garment_transition":
            phase.garment_subphase = family_subphase_sequence[0] if family_subphase_sequence else "stable"
            phase.garment_subphase_sequence = family_subphase_sequence
        if candidate.semantic_family == "interaction_transition":
            phase.interaction_subphase = family_subphase_sequence[0] if family_subphase_sequence else "free"
            phase.interaction_subphase_sequence = family_subphase_sequence
        if candidate.semantic_family == "expression_transition":
            phase.expression_subphase = family_subphase_sequence[0] if family_subphase_sequence else "neutral"
            phase.expression_subphase_sequence = family_subphase_sequence
        return RuntimeSemanticTransition(
            family=candidate.semantic_family,
            goal=candidate.semantic_action,
            confidence=candidate.confidence,
            lexical_bootstrap_score=self._bootstrap_strength(candidate.lexical_reason),
            lexical_trace=[candidate.lexical_reason],
            target_profile=target_profile,
            phase=phase,
            modifiers={
                "intensity": float(clause.modifiers.intensity if clause.modifiers.intensity is not None else 0.6),
                "speed": clause.modifiers.speed,
                "abruptness": clause.modifiers.abruptness,
                "carefulness": clause.modifiers.carefulness,
                "extent": "full" if (clause.modifiers.degree_hint or "") in {"fully", "полностью"} else "partial",
                "simultaneity": clause.modifiers.simultaneity_hint,
                "sequencing": sequencing,
                "global_phase_sequence": global_phase_sequence,
                "family_subphase_sequence": family_subphase_sequence,
            },
        )

    def _global_phase_sequence(self) -> list[str]:
        return ["prepare", "transition", "contact_or_reveal", "stabilize"]

    def _family_subphase_sequence(self, family: str, goal: str) -> list[str]:
        if family == "pose_transition" and goal == "seated_pose":
            return ["weight_shift", "lowering", "contact_settle", "seated_stabilization"]
        if family == "pose_transition" and goal == "arm_elevation":
            return ["shoulder_lift", "elbow_extension", "arm_lock", "arm_stabilization"]
        if family == "pose_transition" and goal == "head_rotation":
            return ["neck_prepare", "head_rotation", "gaze_lock", "head_stabilization"]
        if family == "garment_transition":
            return ["tensioned", "opening", "partially_detached", "garment_settle"]
        if family == "interaction_transition":
            return ["near_support", "approach_contact", "weight_transfer", "stabilized_contact"]
        if family == "expression_transition":
            return ["subtle_rise", "forming_expression", "stable_expression", "expression_relax"]
        return ["steady"]

    def _runtime_semantic_transition_to_dict(self, transition: RuntimeSemanticTransition) -> dict[str, object]:
        regions = sorted(
            set(
                transition.target_profile.primary_regions
                + transition.target_profile.secondary_regions
                + transition.target_profile.context_regions
            )
        )
        return {
            "family": transition.family,
            "goal": transition.goal,
            "global_phase_sequence": transition.phase.global_phase_sequence,
            "family_subphase_sequence": (
                transition.phase.pose_subphase_sequence
                or transition.phase.garment_subphase_sequence
                or transition.phase.interaction_subphase_sequence
                or transition.phase.expression_subphase_sequence
            ),
            "active_phase": transition.phase.global_phase,
            "targets": {
                "entity": transition.target_profile.entity,
                "entity_id": transition.target_profile.entity_id,
                "regions": regions,
                "primary_regions": transition.target_profile.primary_regions,
                "secondary_regions": transition.target_profile.secondary_regions,
                "context_regions": transition.target_profile.context_regions,
                "object_role": transition.target_profile.object_role,
                "support_target": transition.target_profile.support_target,
            },
            "modifiers": transition.modifiers,
            "confidence": transition.confidence,
            "lexical_bootstrap": transition.lexical_trace,
            "lexical_bootstrap_score": transition.lexical_bootstrap_score,
        }

    def _structured_target_regions(
        self,
        family: str,
        goal: str,
        resolved: ResolvedTarget | None,
    ) -> tuple[list[str], list[str], list[str]]:
        if family == "pose_transition" and goal == "seated_pose":
            return ["legs", "pelvis"], ["torso"], ["support_zone"]
        if family == "pose_transition" and goal == "arm_elevation":
            return ["left_arm", "right_arm"], ["upper_torso", "shoulders"], []
        if family == "pose_transition" and goal == "head_rotation":
            return ["face", "head", "neck"], ["upper_torso"], []
        if family == "garment_transition":
            return ["garments", "sleeves"], ["torso", "inner_garment"], ["support_zone"] if resolved and resolved.target_entity_class == "object" else []
        if family == "expression_transition":
            return ["face", "head", "neck"], ["upper_torso"], []
        if family == "interaction_transition":
            return ["support_zone", "pelvis"], ["legs", "torso"], ["chair"] if resolved and resolved.target_object else []
        return ["torso"], [], []

    def _bootstrap_strength(self, lexical_reason: str) -> float:
        if lexical_reason == "fallback_generic":
            return 1.0
        if lexical_reason.startswith("marker:"):
            return 0.35
        return 0.5

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
