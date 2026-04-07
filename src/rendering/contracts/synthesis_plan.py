from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import GarmentSemanticProfile


@dataclass(slots=True)
class PatchSynthesisPlan:
    """Контракт локального синтеза перед генерацией patch."""

    region_id: str
    region_type: str
    entity_id: str
    entity_class: str
    selected_family: str
    selected_strategy: str
    retrieval_mode: str
    retrieval_summary: dict[str, object] = field(default_factory=dict)
    hidden_reconstruction_mode: str = "not_hidden"
    transition_mode: str = "stable"
    garment_semantics: GarmentSemanticProfile = field(default_factory=GarmentSemanticProfile)
    risk_profile: dict[str, float] = field(default_factory=dict)
    confidence_prior: float = 0.0
    seam_sensitivity: float = 0.4
    proposal_mode: str = "deterministic"
    refinement_mode: str = "conservative"
    explainability: dict[str, object] = field(default_factory=dict)
