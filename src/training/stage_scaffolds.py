from __future__ import annotations

from dataclasses import dataclass, field

from dynamics.learned_bridge import BaselineDynamicsTransitionModel
from rendering.learned_bridge import BaselinePatchSynthesisModel
from rendering.temporal_bridge import BaselineTemporalConsistencyModel
from text.learned_bridge import BaselineTextEncoderAdapter


@dataclass(slots=True)
class StageScaffoldConfig:
    stage_name: str
    model_backend: str = "baseline"
    dataset_path: str = ""
    batch_size: int = 2
    learning_rate: float = 1e-4
    epochs: int = 1
    checkpoint_path: str = "artifacts/checkpoints/stage.ckpt"
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StageScaffoldResult:
    stage_name: str
    backend: str
    checkpoint_path: str
    expected_inputs: list[str]
    expected_outputs: list[str]


class TextEncoderStageRunner:
    def __init__(self, backend: str = "baseline") -> None:
        self.backend = backend
        self.model = BaselineTextEncoderAdapter()

    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        _ = self.model
        return StageScaffoldResult(
            stage_name="text_encoder",
            backend=self.backend,
            checkpoint_path=config.checkpoint_path,
            expected_inputs=["text", "scene_graph(optional)", "action_plan(optional)"],
            expected_outputs=["action_embedding", "structured_action_tokens", "alignment"],
        )


class DynamicsTransitionStageRunner:
    def __init__(self, backend: str = "baseline") -> None:
        self.backend = backend
        self.model = BaselineDynamicsTransitionModel()

    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        _ = self.model
        return StageScaffoldResult(
            stage_name="dynamics_transition",
            backend=self.backend,
            checkpoint_path=config.checkpoint_path,
            expected_inputs=["graph_state", "memory_summary", "text_action_summary", "step_context"],
            expected_outputs=["graph_delta", "confidence", "transition_metadata"],
        )


class PatchSynthesisStageRunner:
    def __init__(self, backend: str = "baseline") -> None:
        self.backend = backend
        self.model = BaselinePatchSynthesisModel()

    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        _ = self.model
        return StageScaffoldResult(
            stage_name="patch_synthesis",
            backend=self.backend,
            checkpoint_path=config.checkpoint_path,
            expected_inputs=["region", "scene_state", "memory_summary", "transition_context", "retrieval_summary"],
            expected_outputs=["rgb_patch", "confidence", "uncertainty_map"],
        )


class TemporalRefinementStageRunner:
    def __init__(self, backend: str = "baseline") -> None:
        self.backend = backend
        self.model = BaselineTemporalConsistencyModel()

    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        _ = self.model
        return StageScaffoldResult(
            stage_name="temporal_refinement",
            backend=self.backend,
            checkpoint_path=config.checkpoint_path,
            expected_inputs=["previous_frame", "current_composed_frame", "changed_regions", "scene_state", "memory_state"],
            expected_outputs=["refined_frame", "region_consistency_scores"],
        )


def build_stage_runner(stage_name: str, backend: str = "baseline"):
    mapping = {
        "text_encoder": TextEncoderStageRunner,
        "dynamics_transition": DynamicsTransitionStageRunner,
        "patch_synthesis": PatchSynthesisStageRunner,
        "temporal_refinement": TemporalRefinementStageRunner,
    }
    if stage_name not in mapping:
        known = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown learned stage {stage_name}. Known: {known}")
    return mapping[stage_name](backend=backend)
