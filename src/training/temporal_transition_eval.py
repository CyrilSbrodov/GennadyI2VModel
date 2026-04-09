from __future__ import annotations

import json
from pathlib import Path

from dynamics.temporal_transition_encoder import TemporalTransitionEncoder
from training.datasets import TemporalTransitionDataset
from training.temporal_transition_trainer import TemporalTransitionDatasetAdapter, TemporalTransitionTrainer


def evaluate_temporal_transition(weights_path: str, dataset_size: int = 8, dataset_manifest: str = "") -> dict[str, float | str]:
    model = TemporalTransitionEncoder.load(weights_path)
    if dataset_manifest:
        payload = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
        if payload.get("manifest_type") == "video_transition_manifest":
            dataset = TemporalTransitionDataset.from_video_transition_manifest(dataset_manifest, strict=False)
            if len(dataset.samples) > 0:
                dataset_source = "manifest_video_temporal_transition_primary_eval"
            else:
                dataset = TemporalTransitionDataset.synthetic(dataset_size)
                dataset_source = "synthetic_temporal_transition_bootstrap_fallback_manifest_empty"
        else:
            dataset = TemporalTransitionDataset.synthetic(dataset_size)
            dataset_source = "synthetic_temporal_transition_bootstrap_eval_non_video_manifest"
    else:
        dataset = TemporalTransitionDataset.synthetic(dataset_size)
        dataset_source = "synthetic_temporal_transition_bootstrap_eval"

    batches = [TemporalTransitionDatasetAdapter.sample_to_batch(sample) for sample in dataset.samples]
    metrics = TemporalTransitionTrainer.evaluate_model(model, batches, diagnostics=getattr(dataset, "diagnostics", {}))
    metrics["dataset_source"] = dataset_source
    return metrics
