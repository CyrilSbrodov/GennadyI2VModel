from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModifierBundle:
    """Структурные модификаторы действия, извлеченные из текста."""

    intensity: float | None = None
    speed: str = "normal"
    smoothness: str = "neutral"
    abruptness: str = "neutral"
    carefulness: str = "neutral"
    duration_hint: str | None = None
    repetition_hint: str | None = None
    degree_hint: str | None = None
    simultaneity_hint: bool = False
    sequencing_hint: str | None = None

    def as_dict(self) -> dict[str, str | float | bool | None]:
        """Преобразует bundle в словарь для совместимости с ActionStep.modifiers."""

        return {
            "intensity": self.intensity,
            "speed": self.speed,
            "smoothness": self.smoothness,
            "abruptness": self.abruptness,
            "carefulness": self.carefulness,
            "duration_hint": self.duration_hint,
            "repetition_hint": self.repetition_hint,
            "degree_hint": self.degree_hint,
            "simultaneity_hint": self.simultaneity_hint,
            "sequencing_hint": self.sequencing_hint,
        }


@dataclass(slots=True)
class ActionCandidate:
    """Кандидат действия на уровне семантического семейства, до финального ActionStep."""

    clause_index: int
    trigger_text: str
    semantic_family: str
    semantic_action: str
    confidence: float
    lexical_reason: str


@dataclass(slots=True)
class TargetReference:
    """Ссылка на цель в тексте до scene-grounding."""

    clause_index: int
    reference_text: str
    target_kind: str
    target_region: str | None = None


@dataclass(slots=True)
class ResolvedTarget:
    """Результат разрешения target-reference относительно scene graph."""

    clause_index: int
    target_entity_class: str
    target_entity_id: str | None
    target_region: str | None
    target_object: str | None
    grounding_confidence: float
    resolution_reason: str
    unresolved: bool = False


@dataclass(slots=True)
class TemporalRelation:
    """Явная temporal-связь между действиями."""

    relation: str
    source_clause: int
    target_clause: int
    marker: str


@dataclass(slots=True)
class ConstraintHint:
    """Предпосылка, необходимая для выполнения действия."""

    clause_index: int
    requirement: str
    reason: str


@dataclass(slots=True)
class ParsedClause:
    """Промежуточный слой для отдельной клаузы после сегментации."""

    index: int
    text: str
    action_candidates: list[ActionCandidate] = field(default_factory=list)
    target_references: list[TargetReference] = field(default_factory=list)
    resolved_targets: list[ResolvedTarget] = field(default_factory=list)
    modifiers: ModifierBundle = field(default_factory=ModifierBundle)
    constraints: list[ConstraintHint] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedIntent:
    """Полный structured intent до сборки ActionPlan."""

    normalized_text: str
    clauses: list[ParsedClause] = field(default_factory=list)
    temporal_relations: list[TemporalRelation] = field(default_factory=list)
    global_constraints: list[ConstraintHint] = field(default_factory=list)
    parser_confidence: float = 0.0
    explainability_trace: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class SemanticIntent:
    """Слой совместимости со старым API semantic_encode/structured_decode."""

    text: str
    clauses: list[str]
    modifiers: dict[str, str | float | bool]
    trace: list[str] = field(default_factory=list)
    parsed_intent: ParsedIntent | None = None
