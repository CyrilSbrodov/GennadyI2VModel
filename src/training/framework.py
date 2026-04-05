from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StageConfig:
    name: str
    objective: str


class ContinualTrainingFramework:
    STAGES = [
        StageConfig("stage_1_perception", "image/video -> structured facts"),
        StageConfig("stage_2_representation", "facts -> calibrated scene graph"),
        StageConfig("stage_3_memory", "identity/garment memory learning"),
        StageConfig("stage_4_dynamics", "graph_t + action -> delta"),
        StageConfig("stage_5_patch_renderer", "ROI before/after rendering"),
        StageConfig("stage_6_temporal", "flicker/stability reduction"),
        StageConfig("stage_7_instruction", "text -> structured actions"),
        StageConfig("stage_8_joint_tuning", "careful integration + distillation"),
    ]

    def plan(self) -> list[StageConfig]:
        return self.STAGES.copy()
