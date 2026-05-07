from __future__ import annotations


ALLOWED_TARGET_SOURCES = {"runtime_output_patch", "provided_ground_truth_roi", "unknown"}
ALLOWED_TRAINING_TARGET_QUALITIES = {"self_generated_runtime_target", "external_or_observed_target", "unknown"}


def target_supervision_weight(training_target_quality: str) -> float:
    """Deterministic renderer target provenance supervision policy."""
    quality = str(training_target_quality or "unknown").strip().lower()
    if quality == "external_or_observed_target":
        return 1.0
    if quality == "self_generated_runtime_target":
        return 0.35
    return 0.6


def target_supervision_summary(target_source: str, training_target_quality: str) -> dict[str, object]:
    """Return authoritative target provenance conditioning fields for renderer training."""
    quality = str(training_target_quality or "unknown").strip().lower()
    source = str(target_source or "unknown").strip().lower()
    if source not in ALLOWED_TARGET_SOURCES:
        source = "unknown"
    if quality not in ALLOWED_TRAINING_TARGET_QUALITIES:
        quality = "unknown"
    return {
        "target_source": source,
        "training_target_quality": quality,
        "target_is_self_generated": quality == "self_generated_runtime_target",
        "target_is_external_or_observed": quality == "external_or_observed_target",
        "target_supervision_weight": target_supervision_weight(quality),
    }
