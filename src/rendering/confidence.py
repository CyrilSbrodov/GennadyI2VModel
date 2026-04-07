from __future__ import annotations


class PatchConfidenceEstimator:
    """Центральный оценщик confidence и рисков для ROI синтеза."""

    def estimate(self, factors: dict[str, float], strategy: str) -> dict[str, object]:
        strategy_prior = factors.get("strategy_prior", 0.45)
        retrieval_evidence = factors.get("retrieval_evidence", 0.0)
        semantic_compatibility = factors.get("semantic_compatibility", 0.0)
        garment_completeness = factors.get("garment_semantics_completeness", 0.0)
        hidden_prior = factors.get("hidden_prior", 0.0)
        transition_difficulty = factors.get("transition_difficulty", 0.4)
        region_difficulty = factors.get("region_difficulty", 0.4)
        seam_risk = factors.get("seam_risk", 0.3)
        hallucination_risk = factors.get("hallucination_risk", 0.5)
        fallback_penalty = factors.get("fallback_penalty", 0.0)
        missing_evidence_penalty = factors.get("missing_evidence_penalty", 0.0)

        positive = (
            0.33 * strategy_prior
            + 0.22 * retrieval_evidence
            + 0.16 * semantic_compatibility
            + 0.12 * garment_completeness
            + 0.17 * hidden_prior
        )
        negative = (
            0.22 * transition_difficulty
            + 0.15 * region_difficulty
            + 0.18 * seam_risk
            + 0.25 * hallucination_risk
            + 0.1 * fallback_penalty
            + 0.1 * missing_evidence_penalty
        )
        confidence = max(0.05, min(0.99, 0.5 + positive - negative))
        risks = {
            "hallucination_risk": max(0.0, min(1.0, hallucination_risk)),
            "seam_risk": max(0.0, min(1.0, seam_risk)),
            "semantic_mismatch_risk": max(0.0, min(1.0, 1.0 - semantic_compatibility)),
            "identity_drift_risk": max(0.0, min(1.0, 0.65 if strategy == "face_refine" else 0.35 - 0.2 * retrieval_evidence)),
        }
        return {
            "confidence": confidence,
            "decomposition": {
                "positive": round(positive, 4),
                "negative": round(negative, 4),
                "strategy_prior": strategy_prior,
                "retrieval_evidence": retrieval_evidence,
                "semantic_compatibility": semantic_compatibility,
                "garment_completeness": garment_completeness,
                "hidden_prior": hidden_prior,
                "transition_difficulty": transition_difficulty,
                "region_difficulty": region_difficulty,
                "fallback_penalty": fallback_penalty,
                "missing_evidence_penalty": missing_evidence_penalty,
            },
            "risks": risks,
        }
