from __future__ import annotations

import json
from pathlib import Path

from training.datasets import HumanStateTransitionDataset
from training.human_state_transition_trainer import HumanStateTransitionDatasetAdapter, HumanStateTransitionTrainer
from dynamics.human_state_transition import HumanStateTransitionModel


def evaluate_human_state_transition(dataset_manifest: str, dataset_size: int = 8, weights_path: str = "") -> dict[str, float | str]:
    model = HumanStateTransitionModel.load(weights_path) if weights_path else HumanStateTransitionModel()
    if dataset_manifest:
        payload = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
        if payload.get("manifest_type") == "video_transition_manifest":
            dataset = HumanStateTransitionDataset.from_video_transition_manifest(dataset_manifest, strict=False)
            dataset_source = "manifest_video_human_state_transition_primary_eval" if dataset.samples else "synthetic_human_state_transition_bootstrap_fallback_manifest_empty"
        else:
            dataset = HumanStateTransitionDataset(samples=[])
            dataset_source = "synthetic_human_state_transition_bootstrap_eval_non_video_manifest"
    else:
        dataset = HumanStateTransitionDataset(samples=[])
        dataset_source = "synthetic_human_state_transition_bootstrap_eval"
    if not dataset.samples and dataset_size > 0:
        dataset.samples = [
            {
                "human_state_transition_features": [0.01 * (i + j) for j in range(160)],
                "human_state_transition_target": {
                    "family": "pose_transition",
                    "phase": "transition",
                    "region_state_targets": [0.3] * 8,
                    "visibility_targets": [0.7] * 8,
                    "reveal_memory_target": 0.35,
                    "support_contact_target": 0.1,
                },
            }
            for i in range(dataset_size)
        ]
    batches = [HumanStateTransitionDatasetAdapter.sample_to_batch(s) for s in dataset.samples]
    metrics = HumanStateTransitionTrainer.evaluate_model(model, batches, diagnostics=getattr(dataset, "diagnostics", {}))
    metrics["dataset_source"] = dataset_source
    metrics["weights_path"] = weights_path
    return metrics
