from __future__ import annotations

from text.contracts import ModifierBundle


def extract_modifiers(text: str) -> ModifierBundle:
    """Извлекает структурные modifiers без схлопывания в один style-флаг."""

    speed = "normal"
    if any(t in text for t in ("быстро", "резво")):
        speed = "fast"
    elif any(t in text for t in ("медленно", "плавно")):
        speed = "slow"

    smoothness = "smooth" if "плавно" in text else "neutral"
    abruptness = "abrupt" if "резко" in text else "neutral"
    carefulness = "careful" if "аккуратно" in text else "neutral"

    intensity: float | None = None
    degree_hint: str | None = None
    if "сильно" in text or "полностью" in text:
        intensity = 0.9
        degree_hint = "high"
    elif any(t in text for t in ("слегка", "чуть", "чуть-чуть")):
        intensity = 0.4
        degree_hint = "low"

    sequencing_hint: str | None = None
    if "сначала" in text:
        sequencing_hint = "first"
    elif any(t in text for t in ("потом", "затем")):
        sequencing_hint = "then"

    duration_hint: str | None = "extended" if any(t in text for t in ("долго", "некоторое время")) else None
    repetition_hint: str | None = "repeat" if any(t in text for t in ("несколько раз", "повторно")) else None
    simultaneity_hint = any(t in text for t in ("одновременно", "вместе"))

    return ModifierBundle(
        intensity=intensity,
        speed=speed,
        smoothness=smoothness,
        abruptness=abruptness,
        carefulness=carefulness,
        duration_hint=duration_hint,
        repetition_hint=repetition_hint,
        degree_hint=degree_hint,
        simultaneity_hint=simultaneity_hint,
        sequencing_hint=sequencing_hint,
    )
