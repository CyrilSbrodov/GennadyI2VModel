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


@dataclass(slots=True)
class LearnedTemporalTransitionContract:
    """Typed learned transition contract used as primary downstream conditioning."""

    predicted_family: str = "pose_transition"
    predicted_phase: str = "transition"
    target_profile: TransitionTargetProfile = field(default_factory=TransitionTargetProfile)
    reveal_score: float = 0.0
    occlusion_score: float = 0.0
    support_contact_score: float = 0.0
    transition_embedding: list[float] = field(default_factory=list)
    confidence: float = 0.0
    teacher_source: str = "weak_manifest_bootstrap"
    is_learned_primary: bool = True

    def to_metadata(self) -> dict[str, object]:
        return {
            "predicted_family": self.predicted_family,
            "predicted_phase": self.predicted_phase,
            "target_profile": {
                "primary_regions": list(self.target_profile.primary_regions),
                "secondary_regions": list(self.target_profile.secondary_regions),
                "context_regions": list(self.target_profile.context_regions),
                "entity": self.target_profile.entity,
                "entity_id": self.target_profile.entity_id,
                "object_role": self.target_profile.object_role,
                "support_target": self.target_profile.support_target,
            },
            "reveal_score": float(self.reveal_score),
            "occlusion_score": float(self.occlusion_score),
            "support_contact_score": float(self.support_contact_score),
            "transition_embedding": list(self.transition_embedding),
            "confidence": float(self.confidence),
            "teacher_source": self.teacher_source,
            "is_learned_primary": bool(self.is_learned_primary),
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, object] | None) -> "LearnedTemporalTransitionContract | None":
        if not isinstance(payload, dict):
            return None
        raw_target = payload.get("target_profile", {})
        target = raw_target if isinstance(raw_target, dict) else {}
        return cls(
            predicted_family=str(payload.get("predicted_family", "pose_transition")),
            predicted_phase=str(payload.get("predicted_phase", "transition")),
            target_profile=TransitionTargetProfile(
                primary_regions=[str(x) for x in target.get("primary_regions", []) if isinstance(x, str)],
                secondary_regions=[str(x) for x in target.get("secondary_regions", []) if isinstance(x, str)],
                context_regions=[str(x) for x in target.get("context_regions", []) if isinstance(x, str)],
                entity=str(target.get("entity", "self")),
                entity_id=str(target.get("entity_id")) if target.get("entity_id") is not None else None,
                object_role=str(target.get("object_role")) if target.get("object_role") is not None else None,
                support_target=str(target.get("support_target")) if target.get("support_target") is not None else None,
            ),
            reveal_score=float(payload.get("reveal_score", 0.0) or 0.0),
            occlusion_score=float(payload.get("occlusion_score", 0.0) or 0.0),
            support_contact_score=float(payload.get("support_contact_score", 0.0) or 0.0),
            transition_embedding=[float(x) for x in payload.get("transition_embedding", []) if isinstance(x, (int, float))],
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            teacher_source=str(payload.get("teacher_source", "weak_manifest_bootstrap")),
            is_learned_primary=bool(payload.get("is_learned_primary", False)),
        )


@dataclass(slots=True)
class LearnedHumanStateContract:
    """Typed learned latent human-state contract with reveal/occlusion memory."""

    predicted_family: str = "pose_transition"
    predicted_phase: str = "transition"
    target_profile: TransitionTargetProfile = field(default_factory=TransitionTargetProfile)
    state_embedding: list[float] = field(default_factory=list)
    region_state_embeddings: dict[str, list[float]] = field(default_factory=dict)
    reveal_memory_embedding: list[float] = field(default_factory=list)
    visibility_state_scores: dict[str, float] = field(default_factory=dict)
    support_contact_state: float = 0.0
    compact_conditioning_embedding: list[float] = field(default_factory=list)
    confidence: float = 0.0
    teacher_source: str = "human_state_bootstrap_manifest"
    is_learned_primary: bool = True

    def to_metadata(self) -> dict[str, object]:
        return {
            "predicted_family": self.predicted_family,
            "predicted_phase": self.predicted_phase,
            "target_profile": {
                "primary_regions": list(self.target_profile.primary_regions),
                "secondary_regions": list(self.target_profile.secondary_regions),
                "context_regions": list(self.target_profile.context_regions),
                "entity": self.target_profile.entity,
                "entity_id": self.target_profile.entity_id,
                "object_role": self.target_profile.object_role,
                "support_target": self.target_profile.support_target,
            },
            "state_embedding": [float(x) for x in self.state_embedding],
            "region_state_embeddings": {str(k): [float(v) for v in vals] for k, vals in self.region_state_embeddings.items()},
            "reveal_memory_embedding": [float(x) for x in self.reveal_memory_embedding],
            "visibility_state_scores": {str(k): float(v) for k, v in self.visibility_state_scores.items()},
            "support_contact_state": float(self.support_contact_state),
            "compact_conditioning_embedding": [float(x) for x in self.compact_conditioning_embedding],
            "confidence": float(self.confidence),
            "teacher_source": self.teacher_source,
            "is_learned_primary": bool(self.is_learned_primary),
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, object] | None) -> "LearnedHumanStateContract | None":
        if not isinstance(payload, dict):
            return None
        target = payload.get("target_profile", {}) if isinstance(payload.get("target_profile", {}), dict) else {}
        raw_region = payload.get("region_state_embeddings", {}) if isinstance(payload.get("region_state_embeddings", {}), dict) else {}
        raw_vis = payload.get("visibility_state_scores", {}) if isinstance(payload.get("visibility_state_scores", {}), dict) else {}
        return cls(
            predicted_family=str(payload.get("predicted_family", "pose_transition")),
            predicted_phase=str(payload.get("predicted_phase", "transition")),
            target_profile=TransitionTargetProfile(
                primary_regions=[str(x) for x in target.get("primary_regions", []) if isinstance(x, str)],
                secondary_regions=[str(x) for x in target.get("secondary_regions", []) if isinstance(x, str)],
                context_regions=[str(x) for x in target.get("context_regions", []) if isinstance(x, str)],
                entity=str(target.get("entity", "self")),
                entity_id=str(target.get("entity_id")) if target.get("entity_id") is not None else None,
                object_role=str(target.get("object_role")) if target.get("object_role") is not None else None,
                support_target=str(target.get("support_target")) if target.get("support_target") is not None else None,
            ),
            state_embedding=[float(x) for x in payload.get("state_embedding", []) if isinstance(x, (int, float))],
            region_state_embeddings={str(k): [float(x) for x in vals if isinstance(x, (int, float))] for k, vals in raw_region.items() if isinstance(vals, list)},
            reveal_memory_embedding=[float(x) for x in payload.get("reveal_memory_embedding", []) if isinstance(x, (int, float))],
            visibility_state_scores={str(k): float(v) for k, v in raw_vis.items() if isinstance(v, (int, float))},
            support_contact_state=float(payload.get("support_contact_state", 0.0) or 0.0),
            compact_conditioning_embedding=[float(x) for x in payload.get("compact_conditioning_embedding", []) if isinstance(x, (int, float))],
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            teacher_source=str(payload.get("teacher_source", "human_state_bootstrap_manifest")),
            is_learned_primary=bool(payload.get("is_learned_primary", False)),
        )
