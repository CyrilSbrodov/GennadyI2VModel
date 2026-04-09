from __future__ import annotations

import json
from pathlib import Path

from dynamics.graph_delta_predictor import GraphDeltaPredictor
from dynamics.model import DynamicsModel, decode_prediction, featurize_runtime, targets_from_delta
from planning.transition_engine import PlannedState
from training.datasets import DynamicsDataset


def evaluate_dynamics(weights_path: str, dataset_size: int = 8, dataset_manifest: str = "") -> dict[str, float | str]:
    model = DynamicsModel.load(weights_path)
    if dataset_manifest:
        payload = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
        is_video_manifest = payload.get("manifest_type") == "video_transition_manifest"
        dataset = DynamicsDataset.from_video_transition_manifest(dataset_manifest, strict=False) if is_video_manifest else DynamicsDataset.from_transition_manifest(dataset_manifest, strict=False)
        if len(dataset.samples) == 0:
            dataset = DynamicsDataset.synthetic(dataset_size)
            dataset_source = "synthetic_dynamics_bootstrap_fallback_manifest_empty"
        else:
            dataset_source = "manifest_video_dynamics_primary_eval" if is_video_manifest else "manifest_dynamics_primary_eval"
    else:
        dataset = DynamicsDataset.synthetic(dataset_size)
        dataset_source = "synthetic_dynamics_bootstrap_eval"
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
        "usable_sample_count": 0.0,
        "invalid_records": 0.0,
        "skipped_records": 0.0,
        "pose_group_coverage": 0.0,
        "garment_group_coverage": 0.0,
        "visibility_group_coverage": 0.0,
        "expression_group_coverage": 0.0,
        "interaction_group_coverage": 0.0,
        "region_group_coverage": 0.0,
    }

    for idx, sample in enumerate(dataset.samples):
        graph = sample["graphs"][0]
        labels = [a.type for a in sample.get("actions", [])] or ["micro_adjust"]
        planner_context = {}
        if isinstance(sample.get("graph_transition_contract"), dict):
            maybe = sample["graph_transition_contract"].get("planner_context")
            if isinstance(maybe, dict):
                planner_context = maybe
        step_index = float(planner_context.get("step_index", idx + 1))
        total_steps = float(planner_context.get("total_steps", len(dataset.samples) + 1))
        phase = str(planner_context.get("phase", "mid"))
        inputs = featurize_runtime(graph, PlannedState(step_index=int(step_index), labels=labels), {"step_index": step_index, "total_steps": total_steps, "phase": phase}, None)
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
        metrics["usable_sample_count"] += 1.0
        tgt = sample["deltas"][0]
        metrics["pose_group_coverage"] += 1.0 if tgt.pose_deltas else 0.0
        metrics["garment_group_coverage"] += 1.0 if tgt.garment_deltas else 0.0
        metrics["visibility_group_coverage"] += 1.0 if tgt.visibility_deltas else 0.0
        metrics["expression_group_coverage"] += 1.0 if tgt.expression_deltas else 0.0
        metrics["interaction_group_coverage"] += 1.0 if tgt.interaction_deltas else 0.0
        metrics["region_group_coverage"] += 1.0 if tgt.region_transition_mode else 0.0

    n = max(1.0, float(len(dataset.samples)))
    for k in list(metrics):
        metrics[k] = round(metrics[k] / n, 6)
    if isinstance(getattr(dataset, "diagnostics", None), dict):
        metrics["invalid_records"] = float(dataset.diagnostics.get("invalid_records", 0))
        metrics["skipped_records"] = float(dataset.diagnostics.get("skipped_records", 0))
    metrics["dataset_source"] = dataset_source
    metrics["summary_score"] = round(max(0.0, 1.0 - 0.45 * metrics["pose_mse"] - 0.35 * metrics["visibility_mse"] - 0.2 * metrics["region_mse"]), 6)
    return metrics


def main() -> None:
    weights = "artifacts/checkpoints/dynamics/dynamics_weights.json"
    out = evaluate_dynamics(weights)
    Path("artifacts/checkpoints/dynamics/eval.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/checkpoints/dynamics/eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
