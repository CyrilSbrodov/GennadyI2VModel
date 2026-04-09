from __future__ import annotations

import numpy as np

from rendering.trainable_patch_renderer import PatchBatch, TemporalLocalPatchModel


def evaluate_temporal_renderer(model: TemporalLocalPatchModel, batches: list[PatchBatch]) -> dict[str, float]:
    if not batches:
        return {
            "roi_reconstruction_mae": 1.0,
            "alpha_mae": 1.0,
            "uncertainty_calibration": 1.0,
            "temporal_consistency_proxy": 0.0,
            "reveal_quality": 0.0,
            "occlusion_boundary_quality": 0.0,
            "preservation_score": 0.0,
            "renderer_summary_score": 0.0,
        }

    eval_rows = [model.eval_step(batch) for batch in batches]
    roi_mae = float(np.mean([row["mae"] for row in eval_rows]))
    alpha_mae = float(np.mean([row["alpha_mae"] for row in eval_rows]))
    uncertainty_cal = float(np.mean([row["uncertainty_calibration_loss"] for row in eval_rows]))
    temporal_cons = float(np.mean([row.get("temporal_consistency_loss", 0.0) for row in eval_rows]))
    reveal_loss = float(np.mean([row.get("reveal_region_focus_loss", 0.0) for row in eval_rows]))
    occlusion_loss = float(np.mean([row.get("occlusion_boundary_loss", 0.0) for row in eval_rows]))
    preservation = float(np.mean([row.get("preservation_loss", 0.0) for row in eval_rows]))

    reveal_quality = float(max(0.0, 1.0 - reveal_loss))
    occlusion_quality = float(max(0.0, 1.0 - occlusion_loss))
    preservation_score = float(max(0.0, 1.0 - preservation))
    temporal_proxy = float(max(0.0, 1.0 - temporal_cons))
    summary = float(
        np.clip(
            1.0 - (roi_mae + 0.55 * alpha_mae + 0.4 * uncertainty_cal + 0.3 * temporal_cons),
            0.0,
            1.0,
        )
    )

    return {
        "roi_reconstruction_mae": roi_mae,
        "alpha_mae": alpha_mae,
        "uncertainty_calibration": uncertainty_cal,
        "temporal_consistency_proxy": temporal_proxy,
        "reveal_quality": reveal_quality,
        "occlusion_boundary_quality": occlusion_quality,
        "preservation_score": preservation_score,
        "renderer_summary_score": summary,
    }
