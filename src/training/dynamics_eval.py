from __future__ import annotations

import json
from pathlib import Path

from dynamics.graph_delta_predictor import GraphDeltaPredictor
from dynamics.model import DynamicsModel, decode_prediction, featurize_runtime, targets_from_delta
from planning.transition_engine import PlannedState
from training.datasets import DynamicsDataset


def evaluate_dynamics(weights_path: str, dataset_size: int = 8) -> dict[str, float]:
    model = DynamicsModel.load(weights_path)
    dataset = DynamicsDataset.synthetic(dataset_size)
    predictor = GraphDeltaPredictor(strict_mode=True)
    predictor.model = model

    metrics = {
        "pose_mse": 0.0,
        "garment_mse": 0.0,
        "visibility_mse": 0.0,
        "expression_mse": 0.0,
        "interaction_mse": 0.0,
        "region_mse": 0.0,
        "contract_valid_ratio": 0.0,
        "conditioning_sensitivity": 0.0,
        "fallback_free_ratio": 0.0,
    }

    for idx, sample in enumerate(dataset.samples):
        graph = sample["graphs"][0]
        labels = [a.type for a in sample.get("actions", [])] or ["micro_adjust"]
        inputs = featurize_runtime(graph, PlannedState(step_index=idx + 1, labels=labels), {"step_index": float(idx + 1), "total_steps": float(dataset_size + 1), "phase": "mid"}, None)
        pred = model.forward(inputs)
        targets = targets_from_delta(sample["deltas"][0])
        losses = model.compute_losses(pred, targets)
        for key in ("pose", "garment", "visibility", "expression", "interaction", "region"):
            metrics[f"{key}_mse"] += float(losses[f"{key}_loss"])

        decoded = decode_prediction(pred, graph, phase="mid", semantic_reasons=labels)
        contract_ok = bool(decoded.pose_deltas and decoded.visibility_deltas and decoded.region_transition_mode and decoded.state_after)
        metrics["contract_valid_ratio"] += 1.0 if contract_ok else 0.0

        alt_inputs = featurize_runtime(graph, PlannedState(step_index=idx + 1, labels=labels + ["intensity=0.9"]), {"step_index": float(idx + 1), "total_steps": float(dataset_size + 1), "phase": "late"}, None)
        alt_pred = model.forward(alt_inputs)
        metrics["conditioning_sensitivity"] += abs(float(pred.pose[0]) - float(alt_pred.pose[0]))

        delta, _ = predictor.predict(graph, PlannedState(step_index=idx + 1, labels=labels), planner_context={"step_index": float(idx + 1), "total_steps": float(dataset_size + 1), "phase": "mid"})
        metrics["fallback_free_ratio"] += 1.0 if delta.transition_diagnostics.get("runtime_path") == "learned_primary" else 0.0

    n = max(1.0, float(len(dataset.samples)))
    for k in list(metrics):
        metrics[k] = round(metrics[k] / n, 6)
    return metrics


def main() -> None:
    weights = "artifacts/checkpoints/dynamics/dynamics_weights.json"
    out = evaluate_dynamics(weights)
    Path("artifacts/checkpoints/dynamics/eval.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/checkpoints/dynamics/eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
