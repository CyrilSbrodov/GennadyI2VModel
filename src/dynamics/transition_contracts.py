from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PlannerTransitionContext:
    """Контекст планировщика, влияющий на фазу и интенсивность перехода."""

    step_index: int = 0
    total_steps: int = 1
    phase: str = "single"
    sequencing_stage: str = "mid"
    target_duration: float = 1.0
    intensity: float = 0.5


@dataclass(slots=True)
class MemoryInfluence:
    """Сигналы памяти как priors/ограничения для переходов."""

    hidden_reveal_evidence: float = 0.0
    garment_memory_strength: float = 0.0
    body_region_memory_strength: float = 0.0
    identity_continuity: float = 0.0
    visibility_safety: float = 0.0


@dataclass(slots=True)
class PoseTransitionIntent:
    target_pose: str = "stable"
    progression: str = "steady"
    weight_shift: float = 0.0


@dataclass(slots=True)
class GarmentTransitionIntent:
    progression_state: str = "worn"
    attachment_delta: float = 0.0
    reveal_bias: float = 0.0


@dataclass(slots=True)
class VisibilityTransitionIntent:
    reveal_regions: list[str] = field(default_factory=list)
    occlude_regions: list[str] = field(default_factory=list)
    stable_regions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InteractionTransitionIntent:
    support_progression: str = "free"
    support_target: str = ""
    contact_bias: float = 0.0


@dataclass(slots=True)
class ExpressionTransitionIntent:
    expression_label: str = "neutral"
    progression: str = "neutral"
    intensity_delta: float = 0.0


@dataclass(slots=True)
class TransitionIntent:
    """Промежуточный контракт намерения перехода состояния."""

    action_families: list[str] = field(default_factory=list)
    target_entity: str = ""
    target_regions: list[str] = field(default_factory=list)
    planner: PlannerTransitionContext = field(default_factory=PlannerTransitionContext)
    memory: MemoryInfluence = field(default_factory=MemoryInfluence)
    pose: PoseTransitionIntent = field(default_factory=PoseTransitionIntent)
    garment: GarmentTransitionIntent = field(default_factory=GarmentTransitionIntent)
    visibility: VisibilityTransitionIntent = field(default_factory=VisibilityTransitionIntent)
    interaction: InteractionTransitionIntent = field(default_factory=InteractionTransitionIntent)
    expression: ExpressionTransitionIntent = field(default_factory=ExpressionTransitionIntent)


@dataclass(slots=True)
class TransitionDiagnostics:
    """Диагностика перехода с объяснимыми метриками по семействам."""

    delta_magnitude: float = 0.0
    family_contribution: dict[str, float] = field(default_factory=dict)
    transition_smoothness_proxy: float = 1.0
    constraint_violations: list[str] = field(default_factory=list)
    visibility_uncertainty: float = 0.0
    garment_uncertainty: float = 0.0
    interaction_uncertainty: float = 0.0
    hidden_transition_uncertainty: float = 0.0
    fallback_usage: list[str] = field(default_factory=list)
    explainability_summary: str = ""
