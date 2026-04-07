from __future__ import annotations

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
from training.types import StageResult, StageTrainer, TrainingConfig


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
    stage = result.stage_name
    eval_payloads = []
    if isinstance(contract_payload, dict) and stage == "text_encoder":
        eval_payloads.append(text_action_alignment_eval(build_text_eval_payload(contract_payload)))
    if isinstance(contract_payload, dict) and stage == "dynamics_transition":
        eval_payloads.append(graph_transition_eval(build_graph_eval_payload(contract_payload)))
    if isinstance(contract_payload, dict) and stage == "patch_synthesis":
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
        DynamicsTrainer(),
        RendererTrainer(),
    ]
    return {trainer.stage_name: trainer for trainer in trainers}


def train_learned_stage(stage_name: str, config: TrainingConfig, backend: str = "baseline") -> StageResult:
    runner = build_stage_runner(stage_name, backend=backend, backend_config=config.learned_backend_config)
    scaffold = StageScaffoldConfig(
        stage_name=stage_name,
        model_backend=backend,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        checkpoint_path=f"{config.checkpoint_dir}/{stage_name}.ckpt",
        dataset_path=config.learned_dataset_path,
        backend_config=config.learned_backend_config,
    )
    result = runner.run(scaffold)
    return StageResult(
        stage_name=result.stage_name,
        train_metrics=result.train_metrics or {"progress": 0.5, "loss": 0.5},
        val_metrics=result.val_metrics or {"score": 0.5},
        checkpoint_path=result.checkpoint_path,
    )


def train_stage(stage_name: str, config: TrainingConfig) -> StageResult:
    if stage_name in {"text_encoder", "dynamics_transition", "patch_synthesis", "temporal_refinement"}:
        return train_learned_stage(stage_name, config)
    trainers = _build_stage_trainers()
    if stage_name not in trainers:
        known = ", ".join(sorted(trainers))
        raise ValueError(f"Unknown stage '{stage_name}'. Available stages: {known}")
    return trainers[stage_name].train(config)


def train_pipeline(config: TrainingConfig) -> list[StageResult]:
    stage_map = {
        "stage1_perception": "perception",
        "stage2_representation": "representation",
        "stage3_dynamics": "dynamics",
        "stage4_renderer": "renderer",
        "stage5_memory": "representation",
        "stage6_temporal": "dynamics",
        "stage7_instruction": "perception",
        "stage8_joint_tuning": "renderer",
    }
    replay = ReplayBuffer()
    requested = config.stage_order
    normalized = [stage_map.get(name, name) for name in requested]
    results: list[StageResult] = []
    for stage_name in normalized:
        result = train_stage(stage_name, config)
        replay.push({"stage": stage_name, "metrics": result.val_metrics})
        _ = distillation_hook(stage_name, result)
        _ = evaluate_stage(result)
        results.append(result)
    return results
