from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_ACTIONS = (
    "stable_idle",
    "head_turn",
    "expression_smile",
    "torso_shift",
    "arm_raise",
    "garment_reveal_or_adjust",
)


@dataclass(slots=True)
class I2VFramePlanEntry:
    frame_index: int
    action_phase: str
    affected_entities: list[str]
    affected_regions: list[str]
    region_transition_mode: dict[str, str]
    expected_reference_families: dict[str, str]
    use_input_frame_visual_anchors: bool


def plan_i2v_frames(prompt: str, frame_count: int, entity_id: str = "p1") -> list[I2VFramePlanEntry]:
    text = (prompt or "").lower()
    actions = ["stable_idle"]
    if any(k in text for k in ("head", "turn", "look")):
        actions.append("head_turn")
    if any(k in text for k in ("smile", "улыб", "grin")):
        actions.append("expression_smile")
    if any(k in text for k in ("torso", "body", "lean", "shift", "sit", "сад")):
        actions.append("torso_shift")
    if any(k in text for k in ("arm", "hand", "raise", "wave", "рук")):
        actions.append("arm_raise")
    if any(k in text for k in ("garment", "coat", "jacket", "shirt", "adjust", "reveal", "пальт", "одеж")):
        actions.append("garment_reveal_or_adjust")
    unique_actions = []
    for a in actions:
        if a not in unique_actions:
            unique_actions.append(a)
    entries: list[I2VFramePlanEntry] = []
    for i in range(max(1, frame_count)):
        phase = unique_actions[min(len(unique_actions) - 1, (i * len(unique_actions)) // max(1, frame_count))]
        regions = {
            "stable_idle": ["face"],
            "head_turn": ["face", "hair"],
            "expression_smile": ["face"],
            "torso_shift": ["torso"],
            "arm_raise": ["left_arm", "right_arm"],
            "garment_reveal_or_adjust": ["torso", "upper_clothes"],
        }[phase]
        families = {}
        modes = {}
        for r in regions:
            rid = f"{entity_id}:{r}"
            modes[rid] = phase
            families[rid] = "identity" if r in {"face", "hair"} else ("garment" if "clothes" in r else "body_shape")
        entries.append(
            I2VFramePlanEntry(
                frame_index=i,
                action_phase=phase,
                affected_entities=[entity_id],
                affected_regions=[f"{entity_id}:{r}" for r in regions],
                region_transition_mode=modes,
                expected_reference_families=families,
                use_input_frame_visual_anchors=True,
            )
        )
    return entries
