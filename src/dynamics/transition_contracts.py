from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import TransitionTargetProfile


@dataclass(slots=True)
class PlannerTransitionContext:
    """Контекст планировщика, влияющий на фазу и интенсивность перехода."""

    step_index: int = 0
    total_steps: int = 1
    phase: str = "transition"
    sequencing_stage: str = "transition"
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
    goal: str = "pose_hold"
    progression: str = "stabilize"
    phase_sequence: list[str] = field(default_factory=lambda: ["steady"])
    weight_shift: float = 0.0


@dataclass(slots=True)
class GarmentTransitionIntent:
    goal: str = "garment_static"
    progression_state: str = "worn"
    phase_sequence: list[str] = field(default_factory=lambda: ["tensioned", "opening", "partially_detached", "garment_settle"])
    attachment_delta: float = 0.0
    reveal_bias: float = 0.0


@dataclass(slots=True)
class VisibilityTransitionIntent:
    goal: str = "preserve_identity_region"
    reveal_regions: list[str] = field(default_factory=list)
    occlude_regions: list[str] = field(default_factory=list)
    stable_regions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InteractionTransitionIntent:
    goal: str = "free"
    support_progression: str = "free"
    phase_sequence: list[str] = field(default_factory=lambda: ["near_support", "approach_contact", "weight_transfer", "stabilized_contact"])
    support_target: str = ""
    contact_bias: float = 0.0


@dataclass(slots=True)
class ExpressionTransitionIntent:
    goal: str = "expression_relax"
    expression_label: str = "neutral"
    progression: str = "neutral"
    intensity_delta: float = 0.0


@dataclass(slots=True)
class TransitionIntent:
    """Промежуточный контракт намерения перехода состояния."""

    active_families: list[str] = field(default_factory=list)
    goals: dict[str, str] = field(default_factory=dict)
    target_entity: str = ""
    target_regions: list[str] = field(default_factory=list)
    target_profile: TransitionTargetProfile = field(default_factory=TransitionTargetProfile)
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
