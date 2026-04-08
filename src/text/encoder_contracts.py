from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GroundedTargetConditioning:
    """Структурное представление target после grounding для downstream-модулей."""

    clause_index: int
    target_entity_class: str
    target_entity_id: str | None
    target_region: str | None
    target_object: str | None
    grounding_confidence: float
    unresolved: bool
    resolution_reason: str


@dataclass(slots=True)
class TextEncodingDiagnostics:
    """Диагностика encoder-слоя для explainability и quality-monitoring."""

    action_count: int = 0
    family_distribution: dict[str, int] = field(default_factory=dict)
    grounded_target_count: int = 0
    unresolved_target_count: int = 0
    weak_grounding_count: int = 0
    temporal_relation_count: int = 0
    constraint_count: int = 0
    ambiguity_count: int = 0
    parser_confidence: float = 0.0
    encoder_confidence: float = 0.0
    explainability_summary: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TextEncodingOutput:
    """Единый контракт text conditioning поверх parser-структуры."""

    global_text_embedding: list[float]
    action_embedding: list[float]
    target_embedding: list[float]
    modifier_embedding: list[float]
    temporal_embedding: list[float]
    constraint_embedding: list[float]
    grounding_embedding: list[float]
    structured_action_tokens: list[str]
    grounded_targets: list[GroundedTargetConditioning]
    parser_confidence: float
    encoder_confidence: float
    diagnostics: TextEncodingDiagnostics

    scene_alignment_score: float = 0.0
    ambiguity_score: float = 0.0
    conditioning_hints: dict[str, object] = field(default_factory=dict)
    family_presence_vector: dict[str, float] = field(default_factory=dict)

    # Поля обратной совместимости с текущим learned/runtime стеком.
    target_hints: dict[str, list[str]] = field(default_factory=dict)
    temporal_hints: dict[str, object] = field(default_factory=dict)
    decomposition_hints: list[dict[str, object]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    confidence: float = 0.0
    alignment: dict[str, object] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
