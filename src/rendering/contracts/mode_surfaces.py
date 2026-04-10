from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class UpdatePathContract:
    """Контракт update-пути (learnable-ready surface)."""

    mode: str
    reuse_fraction: float
    synth_fraction: float
    refinement_fraction: float
    target_consistency: float
    training_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RevealPathContract:
    """Контракт reveal-пути (learnable-ready surface)."""

    reveal_type: str
    hidden_mode: str
    memory_usage_ratio: float
    reconstruction_bias: float
    hallucination_budget: float
    training_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InsertionPathContract:
    """Контракт insertion-пути (learnable-ready surface)."""

    entity_type: str
    pose_role: str
    context_conditioning_score: float
    reusable_artifact_expected: bool
    alpha_semantics: str
    uncertainty_semantics: str
    training_tags: list[str] = field(default_factory=list)
