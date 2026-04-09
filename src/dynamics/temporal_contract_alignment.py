from __future__ import annotations

from dynamics.transition_contracts import LearnedTemporalTransitionContract
from core.schema import GraphDelta


def _jaccard(lhs: set[str], rhs: set[str]) -> float:
    if not lhs and not rhs:
        return 1.0
    union = lhs | rhs
    if not union:
        return 1.0
    return float(len(lhs & rhs)) / float(len(union))


def compute_temporal_contract_alignment(
    contract: LearnedTemporalTransitionContract,
    delta: GraphDelta,
    *,
    renderer_transition_mode: str = "",
) -> dict[str, float]:
    target = contract.target_profile
    contract_regions = set(target.primary_regions + target.secondary_regions + target.context_regions)
    delta_regions = set(delta.affected_regions)
    profile_consistency = _jaccard(contract_regions, delta_regions)

    dynamics_alignment = 0.0
    if delta.transition_phase:
        dynamics_alignment = 1.0 if str(delta.transition_phase) == contract.predicted_phase else 0.0

    renderer_alignment = 0.0
    if renderer_transition_mode:
        renderer_alignment = 1.0 if renderer_transition_mode in {"garment_reveal", "visibility_occlusion"} and contract.reveal_score >= 0.5 else 0.5

    reveal_proxy = 1.0 if len(delta.newly_revealed_regions) > 0 else 0.0
    occlusion_proxy = 1.0 if len(delta.newly_occluded_regions) > 0 else 0.0
    support_proxy = min(1.0, abs(float(delta.interaction_deltas.get("support_contact", 0.0))))
    reveal_agreement = 1.0 - abs(float(contract.reveal_score) - reveal_proxy)
    occlusion_agreement = 1.0 - abs(float(contract.occlusion_score) - occlusion_proxy)
    support_agreement = 1.0 - abs(float(contract.support_contact_score) - support_proxy)

    return {
        "learned_contract_to_dynamics_alignment": round(max(0.0, min(1.0, dynamics_alignment)), 6),
        "learned_contract_to_renderer_alignment": round(max(0.0, min(1.0, renderer_alignment)), 6),
        "target_profile_consistency": round(max(0.0, min(1.0, profile_consistency)), 6),
        "reveal_occlusion_support_agreement": round(
            max(0.0, min(1.0, (reveal_agreement + occlusion_agreement + support_agreement) / 3.0)),
            6,
        ),
    }
