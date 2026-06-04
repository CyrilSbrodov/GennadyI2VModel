from __future__ import annotations

from core.pipeline_contract import ContractValidationError, PipelineStage
from learned.factory import BackendConfig
from evaluation.contracts import (
    build_graph_eval_payload,
    build_hidden_reconstruction_payload,
    build_patch_eval_payload,
    build_temporal_eval_payload,
    build_text_eval_payload,
    graph_transition_eval,
    hidden_region_reconstruction_eval,
    patch_synthesis_eval,
    temporal_consistency_eval,
    text_action_alignment_eval,
)
from training.dynamics_trainer import DynamicsTrainer
from training.stage_scaffolds import StageScaffoldConfig, build_stage_runner
from training.perception_trainer import PerceptionTrainer
from training.renderer_trainer import RendererTrainer
from training.representation_trainer import RepresentationTrainer
from training.temporal_trainer import TemporalTrainer
from training.temporal_transition_trainer import TemporalTransitionTrainer
from training.human_state_transition_trainer import HumanStateTransitionTrainer
from training.types import StageResult, StageTrainer, TrainingConfig

CANONICAL_TRAINING_STAGE_ALIASES = {
    "input": "input",
    "perception": "perception",
    "scene_graph": "scene_graph",
    "memory": "memory",
    "intent": "intent",
    "planning": "planning",
    "dynamics": "dynamics",
    "region_routing": "region_routing",
    "rendering": "rendering",
    "compositing": "compositing",
    "temporal_refinement": "temporal_refinement",
    "output": "output",
    # Explicit, non-ambiguous compatibility aliases.
    "representation": "scene_graph",
    "renderer": "rendering",
    "patch_synthesis": "rendering",
    "text_encoder": "intent",
    "dynamics_transition": "dynamics",
    "stage5_memory": "memory",
}

LEGACY_STAGE_ALIASES: dict[str, str] = {}


def canonical_training_stage_name(stage_name: str) -> str:
    name = str(stage_name or "").strip()
    name = LEGACY_STAGE_ALIASES.get(name, name)
    canonical = CANONICAL_TRAINING_STAGE_ALIASES.get(name)
    if canonical is None:
        known = sorted(set(CANONICAL_TRAINING_STAGE_ALIASES) | set(LEGACY_STAGE_ALIASES))
        raise ContractValidationError(f"Unknown training stage {stage_name!r}. Known stages/aliases: {known}")
    if canonical not in {stage.value for stage in PipelineStage}:
        raise ContractValidationError(f"Training stage {stage_name!r} maps outside canonical pipeline: {canonical!r}")
    return canonical


class MemoryTrainer:
    stage_name = "memory"

    def train(self, config: TrainingConfig) -> StageResult:
        raise NotImplementedError(
            "Memory is a canonical first-class stage, but standalone memory training "
            "is not implemented. Use existing MemoryDataset builders for data "
            "contracts; do not normalize memory to representation."
        )


class ReplayBuffer:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def push(self, event: dict[str, object]) -> None:
        self.events.append(event)


def distillation_hook(stage_name: str, result: StageResult) -> dict[str, float]:
    return {"stage": float(len(stage_name)), "score": float(result.val_metrics.get("score", 0.0))}


def evaluate_stage(result: StageResult) -> dict[str, float]:
    base = {
        "perception_accuracy": result.val_metrics.get("score", 0.0),
        "graph_quality": result.val_metrics.get("score", 0.0),
        "temporal_consistency": result.train_metrics.get("progress", 0.0),
        "identity_preservation": 1.0 - result.train_metrics.get("loss", 1.0),
    }
    contract_payload = result.val_metrics.get("contract_payload")
    stage = canonical_training_stage_name(result.stage_name)
    eval_payloads = []
    if isinstance(contract_payload, dict) and stage == "intent":
        eval_payloads.append(text_action_alignment_eval(build_text_eval_payload(contract_payload)))
    if isinstance(contract_payload, dict) and stage == "dynamics":
        eval_payloads.append(graph_transition_eval(build_graph_eval_payload(contract_payload)))
    if isinstance(contract_payload, dict) and stage == "rendering":
        eval_payloads.append(patch_synthesis_eval(build_patch_eval_payload(contract_payload)))
        eval_payloads.append(hidden_region_reconstruction_eval(build_hidden_reconstruction_payload(contract_payload)))
    if isinstance(contract_payload, dict) and stage == "temporal_refinement":
        eval_payloads.append(temporal_consistency_eval(build_temporal_eval_payload(contract_payload)))
    if not eval_payloads:
        eval_payloads = [
            text_action_alignment_eval({"alignment_score": result.val_metrics.get("score", 0.0)}),
            graph_transition_eval({"delta_match": result.val_metrics.get("score", 0.0)}),
            patch_synthesis_eval({"patch_quality": result.val_metrics.get("score", 0.0), "identity_consistency": base["identity_preservation"]}),
            temporal_consistency_eval({"temporal_drift": 1.0 - base["temporal_consistency"]}),
        ]
    for payload in eval_payloads:
        base.update(payload.metrics)
    return base


def _build_stage_trainers() -> dict[str, StageTrainer]:
    trainers: list[StageTrainer] = [
        PerceptionTrainer(),
        RepresentationTrainer(),
        MemoryTrainer(),
        DynamicsTrainer(),
        RendererTrainer(),
        TemporalTrainer(),
        TemporalTransitionTrainer(),
        HumanStateTransitionTrainer(),
    ]
    return {trainer.stage_name: trainer for trainer in trainers}


def train_learned_stage(stage_name: str, config: TrainingConfig, backend: str = "baseline") -> StageResult:
    canonical = canonical_training_stage_name(stage_name)
    if canonical == "rendering" and backend == "baseline":
        backend = "numpy_local"
    runner_stage = "text_encoder" if canonical == "intent" else ("patch_synthesis" if canonical == "rendering" else ("temporal_refinement" if canonical == "temporal_refinement" else stage_name))
    backend_config = config.learned_backend_config
    if canonical == "rendering" and backend_config is None:
        backend_config = BackendConfig(patch_backend="baseline", dynamics_backend="baseline", temporal_backend="baseline")
    runner = build_stage_runner(runner_stage, backend=backend, backend_config=backend_config)
    scaffold = StageScaffoldConfig(
        stage_name=runner_stage,
        model_backend=backend,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        checkpoint_path=f"{config.checkpoint_dir}/{canonical}.ckpt",
        dataset_path=config.learned_dataset_path,
        backend_config=backend_config,
    )
    result = runner.run(scaffold)
    return StageResult(
        stage_name=canonical,
        train_metrics=result.train_metrics or {"progress": 0.5, "loss": 0.5},
        val_metrics=result.val_metrics or {"score": 0.5},
        checkpoint_path=result.checkpoint_path,
    )


def train_stage(stage_name: str, config: TrainingConfig) -> StageResult:
    canonical = canonical_training_stage_name(stage_name)
    if canonical in {"intent", "rendering", "temporal_refinement"} and stage_name in {"text_encoder", "intent", "patch_synthesis", "renderer", "rendering", "temporal_refinement"}:
        return train_learned_stage(canonical, config)
    if canonical == "dynamics":
        result = DynamicsTrainer().train(config)
        return StageResult(stage_name="dynamics", train_metrics=result.train_metrics, val_metrics=result.val_metrics, checkpoint_path=result.checkpoint_path)
    trainers = _build_stage_trainers()
    trainer_key = "representation" if canonical == "scene_graph" else canonical
    if trainer_key not in trainers:
        known = ", ".join(sorted(set(trainers) | {"scene_graph", "rendering", "intent"}))
        raise ContractValidationError(f"Unknown stage '{stage_name}'. Available stages: {known}")
    result = trainers[trainer_key].train(config)
    return StageResult(stage_name=canonical, train_metrics=result.train_metrics, val_metrics=result.val_metrics, checkpoint_path=result.checkpoint_path)


def train_pipeline(config: TrainingConfig) -> list[StageResult]:
    replay = ReplayBuffer()
    requested = config.stage_order
    normalized = [canonical_training_stage_name(name) for name in requested]
    results: list[StageResult] = []
    for stage_name in normalized:
        result = train_stage(stage_name, config)
        replay.push({"stage": stage_name, "metrics": result.val_metrics})
        _ = distillation_hook(stage_name, result)
        _ = evaluate_stage(result)
        results.append(result)
    return results
