from __future__ import annotations

from dataclasses import dataclass

from dynamics.model import DynamicsModel, DynamicsTensorBatch, decode_prediction, dynamics_inputs_from_tensor_batch


@dataclass(slots=True)
class DynamicsTrainingSample:
    tensor_batch: DynamicsTensorBatch
    targets: object
    graph_before: object
    action_tokens: list[str]
    source: str


@dataclass(slots=True)
class DynamicsDatasetSurface:
    samples: list[DynamicsTrainingSample]
    source: str
    diagnostics: dict[str, object]


class FamilyAwareDynamicsTrainingModule:
    """Family-aware train/val orchestration over DynamicsTensorBatch first-class surface."""

    def train_epoch(self, model: DynamicsModel, surface: DynamicsDatasetSurface, *, lr: float) -> dict[str, float]:
        if not surface.samples:
            return {"total_loss": 0.0}
        totals: dict[str, float] = {"total_loss": 0.0}
        family_counts: dict[str, float] = {}
        for sample in surface.samples:
            losses = model.train_step(dynamics_inputs_from_tensor_batch(sample.tensor_batch), sample.targets, lr=lr)
            totals["total_loss"] += float(losses.get("total_loss", 0.0))
            family = sample.tensor_batch.family
            family_counts[family] = family_counts.get(family, 0.0) + 1.0
        denom = max(1.0, float(len(surface.samples)))
        out = {"total_loss": round(totals["total_loss"] / denom, 6)}
        for family, count in family_counts.items():
            out[f"{family}_sample_ratio"] = round(count / denom, 6)
        return out

    def validate_epoch(self, model: DynamicsModel, surface: DynamicsDatasetSurface) -> dict[str, float]:
        if not surface.samples:
            return {"score": 0.0}
        metrics: dict[str, float] = {
            "pose_transition_pose_delta_quality": 0.0,
            "pose_transition_visibility_phase_consistency": 0.0,
            "garment_transition_reveal_occlusion_quality": 0.0,
            "garment_transition_visibility_region_correctness": 0.0,
            "interaction_transition_contact_consistency": 0.0,
            "interaction_transition_region_entity_correctness": 0.0,
            "expression_transition_expression_delta_quality": 0.0,
            "expression_transition_face_region_involvement": 0.0,
        }
        fam_counts: dict[str, float] = {}
        for sample in surface.samples:
            family = sample.tensor_batch.family
            pred = model.forward(dynamics_inputs_from_tensor_batch(sample.tensor_batch), family=family)
            losses = model.compute_losses(pred, sample.targets)
            decoded = decode_prediction(pred, sample.graph_before, phase=sample.tensor_batch.phase, semantic_reasons=sample.action_tokens)
            fam_counts[family] = fam_counts.get(family, 0.0) + 1.0
            if family == "pose_transition":
                metrics["pose_transition_pose_delta_quality"] += max(0.0, 1.0 - float(losses.get("pose_loss", 1.0)))
                metrics["pose_transition_visibility_phase_consistency"] += 1.0 if decoded.transition_phase in {"transition", "contact_or_reveal", "stabilize"} else 0.0
            elif family == "garment_transition":
                metrics["garment_transition_reveal_occlusion_quality"] += max(0.0, float(pred.region[0]) + float(pred.region[1]) - 0.25)
                metrics["garment_transition_visibility_region_correctness"] += 1.0 if any(r in {"torso", "garments", "inner_garment"} for r in decoded.affected_regions) else 0.0
            elif family == "interaction_transition":
                metrics["interaction_transition_contact_consistency"] += float(pred.interaction[0])
                metrics["interaction_transition_region_entity_correctness"] += 1.0 if bool(decoded.affected_entities and decoded.affected_regions) else 0.0
            elif family == "expression_transition":
                metrics["expression_transition_expression_delta_quality"] += max(0.0, 1.0 - float(losses.get("expression_loss", 1.0)))
                metrics["expression_transition_face_region_involvement"] += 1.0 if "face" in decoded.affected_regions else 0.0
        for fam, count in fam_counts.items():
            for key in list(metrics.keys()):
                if key.startswith(fam):
                    metrics[key] = round(metrics[key] / max(1.0, count), 6)
        metrics["score"] = round(sum(metrics.values()) / max(1.0, float(len(metrics))), 6)
        return metrics
