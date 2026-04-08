from __future__ import annotations

from dataclasses import dataclass

from core.schema import SceneGraph
from text.contracts import ParsedIntent
from text.encoder_contracts import GroundedTargetConditioning


_ACTION_FAMILIES = ["pose", "garment", "expression", "interaction", "visibility"]
_SPEED_ORDER = ["slow", "normal", "fast"]
_STYLE_ORDER = ["neutral", "smooth", "abrupt", "careful"]


@dataclass(slots=True)
class ActionEncodingResult:
    """Результат кодирования action-семантики."""

    embedding: list[float]
    structured_action_tokens: list[str]
    family_presence_vector: dict[str, float]
    family_distribution: dict[str, int]


@dataclass(slots=True)
class TargetEncodingResult:
    """Результат кодирования targets и grounding сигналов."""

    embedding: list[float]
    grounded_targets: list[GroundedTargetConditioning]
    scene_alignment_score: float
    unresolved_target_count: int
    weak_grounding_count: int


class ActionSemanticsEncoder:
    """Кодирует семейства действий и разнообразие action semantics."""

    def encode(self, parsed: ParsedIntent) -> ActionEncodingResult:
        action_pairs: list[tuple[str, str]] = []
        for clause in parsed.clauses:
            action_pairs.extend((cand.semantic_family, cand.semantic_action) for cand in clause.action_candidates)

        distribution = {fam: 0 for fam in _ACTION_FAMILIES}
        unique_actions: set[tuple[str, str]] = set()
        for family, action in action_pairs:
            distribution.setdefault(family, 0)
            distribution[family] += 1
            unique_actions.add((family, action))

        total = max(1, len(action_pairs))
        family_presence = {fam: (1.0 if distribution.get(fam, 0) > 0 else 0.0) for fam in _ACTION_FAMILIES}
        embedding = [distribution.get(fam, 0) / total for fam in _ACTION_FAMILIES]
        embedding.extend(
            [
                min(1.0, len(unique_actions) / 8.0),
                min(1.0, len(action_pairs) / 8.0),
                sum(family_presence.values()) / max(1.0, float(len(_ACTION_FAMILIES))),
            ]
        )

        tokens = [action for _, action in sorted(unique_actions)] or ["hold_pose"]
        return ActionEncodingResult(
            embedding=embedding,
            structured_action_tokens=tokens,
            family_presence_vector=family_presence,
            family_distribution=distribution,
        )


class TargetGroundingEncoder:
    """Кодирует target-классы, качество grounding и unresolved сигналы."""

    def encode(self, parsed: ParsedIntent, scene_graph: SceneGraph | None = None) -> TargetEncodingResult:
        grounded_targets: list[GroundedTargetConditioning] = []
        class_counts: dict[str, int] = {"body": 0, "garment": 0, "object": 0, "self": 0}

        unresolved = 0
        weak_grounding = 0
        confidences: list[float] = []

        for clause in parsed.clauses:
            for target in clause.resolved_targets:
                grounded_targets.append(
                    GroundedTargetConditioning(
                        clause_index=target.clause_index,
                        target_entity_class=target.target_entity_class,
                        target_entity_id=target.target_entity_id,
                        target_region=target.target_region,
                        target_object=target.target_object,
                        grounding_confidence=float(target.grounding_confidence),
                        unresolved=bool(target.unresolved),
                        resolution_reason=target.resolution_reason,
                    )
                )
                class_counts.setdefault(target.target_entity_class, 0)
                class_counts[target.target_entity_class] += 1
                confidences.append(float(target.grounding_confidence))
                if target.unresolved:
                    unresolved += 1
                if target.grounding_confidence < 0.6:
                    weak_grounding += 1

        total = max(1, len(grounded_targets))
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        alignment = max(0.0, min(1.0, avg_conf - 0.15 * (unresolved / total)))

        embedding = [
            class_counts.get("body", 0) / total,
            class_counts.get("garment", 0) / total,
            class_counts.get("object", 0) / total,
            class_counts.get("self", 0) / total,
            avg_conf,
            min(1.0, unresolved / total),
            min(1.0, weak_grounding / total),
            1.0 if scene_graph is not None else 0.0,
        ]

        return TargetEncodingResult(
            embedding=embedding,
            grounded_targets=grounded_targets,
            scene_alignment_score=alignment,
            unresolved_target_count=unresolved,
            weak_grounding_count=weak_grounding,
        )


class ModifierEncoder:
    """Кодирует intensity/speed/style/repetition/sequencing сигналы."""

    def encode(self, parsed: ParsedIntent) -> list[float]:
        intensities: list[float] = []
        speed_dist = {name: 0 for name in _SPEED_ORDER}
        style_dist = {name: 0 for name in _STYLE_ORDER}
        repetition = 0
        duration = 0
        sequencing = 0
        simultaneity = 0

        for clause in parsed.clauses:
            mod = clause.modifiers
            if mod.intensity is not None:
                intensities.append(float(mod.intensity))
            speed_dist[mod.speed if mod.speed in speed_dist else "normal"] += 1
            if mod.smoothness == "smooth":
                style_dist["smooth"] += 1
            if mod.abruptness == "abrupt":
                style_dist["abrupt"] += 1
            if mod.carefulness == "careful":
                style_dist["careful"] += 1
            if mod.duration_hint:
                duration += 1
            if mod.repetition_hint:
                repetition += 1
            if mod.sequencing_hint:
                sequencing += 1
            if mod.simultaneity_hint:
                simultaneity += 1

        total = max(1, len(parsed.clauses))
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0.55
        return [
            avg_intensity,
            speed_dist["slow"] / total,
            speed_dist["normal"] / total,
            speed_dist["fast"] / total,
            style_dist["smooth"] / total,
            style_dist["abrupt"] / total,
            style_dist["careful"] / total,
            min(1.0, repetition / total),
            min(1.0, duration / total),
            min(1.0, sequencing / total),
            min(1.0, simultaneity / total),
        ]


class TemporalEncoder:
    """Кодирует temporal relations и anchor-структуру."""

    def encode(self, parsed: ParsedIntent) -> list[float]:
        relations = {"after": 0, "parallel": 0, "sequence": 0, "anchor_first": 0}
        for rel in parsed.temporal_relations:
            relations[rel.relation] = relations.get(rel.relation, 0) + 1

        edges = max(1, len(parsed.temporal_relations))
        clause_count = max(1, len(parsed.clauses))
        return [
            relations.get("after", 0) / edges,
            relations.get("parallel", 0) / edges,
            relations.get("sequence", 0) / edges,
            relations.get("anchor_first", 0) / edges,
            min(1.0, len(parsed.clauses) / 8.0),
            min(1.0, len(parsed.temporal_relations) / 8.0),
            min(1.0, edges / clause_count),
        ]


class ConstraintEncoder:
    """Кодирует preconditions, важные для planner/dynamics/rendering."""

    def encode(self, parsed: ParsedIntent) -> tuple[list[float], list[str]]:
        constraints = sorted({constraint.requirement for constraint in parsed.global_constraints})

        flags = {
            "requires:support_object": 0.0,
            "requires:garment_like_entity": 0.0,
            "requires:visible_face": 0.0,
            "requires:usable_arm": 0.0,
            "requires:object": 0.0,
        }
        for requirement in constraints:
            if requirement in flags:
                flags[requirement] = 1.0
            elif requirement.startswith("requires:"):
                flags["requires:object"] = 1.0

        embedding = [
            flags["requires:support_object"],
            flags["requires:garment_like_entity"],
            flags["requires:visible_face"],
            flags["requires:usable_arm"],
            flags["requires:object"],
            min(1.0, len(constraints) / 8.0),
        ]
        return embedding, constraints


class AmbiguityConfidenceEncoder:
    """Кодирует ambiguity/quality сигнал и confidence агрегаты."""

    def encode(
        self,
        parsed: ParsedIntent,
        *,
        weak_grounding_count: int,
        unresolved_target_count: int,
    ) -> tuple[list[float], float, float]:
        ambiguity_count = sum(len(clause.ambiguities) for clause in parsed.clauses)
        generic_count = sum(
            1
            for clause in parsed.clauses
            for ambiguity in clause.ambiguities
            if ambiguity == "generic_interpretation"
        )

        action_count = max(1, sum(len(clause.action_candidates) for clause in parsed.clauses))
        ambiguity_score = max(
            0.0,
            min(1.0, (ambiguity_count + weak_grounding_count + unresolved_target_count) / (action_count * 2.0)),
        )
        encoder_confidence = max(0.0, min(1.0, 0.7 * parsed.parser_confidence + 0.3 * (1.0 - ambiguity_score)))

        embedding = [
            parsed.parser_confidence,
            min(1.0, ambiguity_count / action_count),
            min(1.0, weak_grounding_count / action_count),
            min(1.0, generic_count / action_count),
            min(1.0, unresolved_target_count / action_count),
            ambiguity_score,
            encoder_confidence,
        ]
        return embedding, ambiguity_score, encoder_confidence
