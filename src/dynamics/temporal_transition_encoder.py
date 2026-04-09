from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FAMILIES = [
    "pose_transition",
    "garment_transition",
    "expression_transition",
    "interaction_transition",
    "visibility_transition",
]
PHASES = ["prepare", "transition", "contact_or_reveal", "stabilize"]
REGION_KEYS = ["face", "torso", "left_arm", "right_arm", "legs", "garments", "inner_garment", "outer_garment"]


@dataclass(slots=True)
class TemporalTransitionTargets:
    family_index: int
    phase_index: int
    target_profile_regions: list[float]
    reveal_score: float
    occlusion_score: float
    support_contact_score: float


@dataclass(slots=True)
class TemporalTransitionPrediction:
    family_logits: np.ndarray
    phase_logits: np.ndarray
    target_profile_scores: np.ndarray
    reveal_score: float
    occlusion_score: float
    support_contact_score: float
    transition_embedding: np.ndarray


class TemporalTransitionEncoder:
    """Compact trainable structured transition encoder for video manifest pairs/windows."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 72, embed_dim: int = 16, seed: int = 23) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        self.w_in = rng.normal(0.0, 0.08, size=(input_dim, hidden_dim))
        self.b_in = np.zeros(hidden_dim, dtype=np.float64)
        self.w_embed = rng.normal(0.0, 0.08, size=(hidden_dim, embed_dim))
        self.b_embed = np.zeros(embed_dim, dtype=np.float64)
        self.w_family = rng.normal(0.0, 0.07, size=(embed_dim, len(FAMILIES)))
        self.b_family = np.zeros(len(FAMILIES), dtype=np.float64)
        self.w_phase = rng.normal(0.0, 0.07, size=(embed_dim, len(PHASES)))
        self.b_phase = np.zeros(len(PHASES), dtype=np.float64)
        self.w_regions = rng.normal(0.0, 0.07, size=(embed_dim, len(REGION_KEYS)))
        self.b_regions = np.zeros(len(REGION_KEYS), dtype=np.float64)
        self.w_reveal = rng.normal(0.0, 0.07, size=(embed_dim, 1))
        self.b_reveal = np.zeros(1, dtype=np.float64)
        self.w_occlusion = rng.normal(0.0, 0.07, size=(embed_dim, 1))
        self.b_occlusion = np.zeros(1, dtype=np.float64)
        self.w_support = rng.normal(0.0, 0.07, size=(embed_dim, 1))
        self.b_support = np.zeros(1, dtype=np.float64)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - float(np.max(x))
        e = np.exp(z)
        return e / max(1e-8, float(np.sum(e)))

    def _forward_internal(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if x.shape[0] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[0]}")
        h = np.tanh(x @ self.w_in + self.b_in)
        embed = np.tanh(h @ self.w_embed + self.b_embed)
        logits = {
            "family": embed @ self.w_family + self.b_family,
            "phase": embed @ self.w_phase + self.b_phase,
            "regions": embed @ self.w_regions + self.b_regions,
            "reveal": embed @ self.w_reveal + self.b_reveal,
            "occlusion": embed @ self.w_occlusion + self.b_occlusion,
            "support": embed @ self.w_support + self.b_support,
        }
        return h, embed, logits

    def forward(self, feature_vector: list[float] | np.ndarray) -> TemporalTransitionPrediction:
        x = np.asarray(feature_vector, dtype=np.float64)
        _, embed, logits = self._forward_internal(x)
        return TemporalTransitionPrediction(
            family_logits=np.asarray(logits["family"], dtype=np.float64),
            phase_logits=np.asarray(logits["phase"], dtype=np.float64),
            target_profile_scores=self._sigmoid(np.asarray(logits["regions"], dtype=np.float64)),
            reveal_score=float(self._sigmoid(np.asarray(logits["reveal"], dtype=np.float64))[0]),
            occlusion_score=float(self._sigmoid(np.asarray(logits["occlusion"], dtype=np.float64))[0]),
            support_contact_score=float(self._sigmoid(np.asarray(logits["support"], dtype=np.float64))[0]),
            transition_embedding=np.asarray(embed, dtype=np.float64),
        )

    def compute_losses(self, prediction: TemporalTransitionPrediction, targets: TemporalTransitionTargets) -> dict[str, float]:
        family_probs = self._softmax(prediction.family_logits)
        phase_probs = self._softmax(prediction.phase_logits)
        fam_loss = -float(np.log(max(1e-8, family_probs[targets.family_index])))
        phase_loss = -float(np.log(max(1e-8, phase_probs[targets.phase_index])))
        region_tgt = np.asarray(targets.target_profile_regions, dtype=np.float64)
        region_loss = float(np.mean((prediction.target_profile_scores - region_tgt) ** 2))
        reveal_loss = float((prediction.reveal_score - float(targets.reveal_score)) ** 2)
        occ_loss = float((prediction.occlusion_score - float(targets.occlusion_score)) ** 2)
        support_loss = float((prediction.support_contact_score - float(targets.support_contact_score)) ** 2)
        total = fam_loss + phase_loss + region_loss + reveal_loss + occ_loss + support_loss
        return {
            "family_loss": fam_loss,
            "phase_loss": phase_loss,
            "target_profile_loss": region_loss,
            "reveal_loss": reveal_loss,
            "occlusion_loss": occ_loss,
            "support_contact_loss": support_loss,
            "total_loss": total,
        }

    def train_step(self, feature_vector: list[float] | np.ndarray, targets: TemporalTransitionTargets, lr: float = 1e-3) -> dict[str, float]:
        x = np.asarray(feature_vector, dtype=np.float64)
        h, embed, logits = self._forward_internal(x)
        family_probs = self._softmax(logits["family"])
        phase_probs = self._softmax(logits["phase"])
        regions = self._sigmoid(logits["regions"])
        reveal = float(self._sigmoid(logits["reveal"])[0])
        occlusion = float(self._sigmoid(logits["occlusion"])[0])
        support = float(self._sigmoid(logits["support"])[0])

        pred = TemporalTransitionPrediction(
            family_logits=np.asarray(logits["family"]),
            phase_logits=np.asarray(logits["phase"]),
            target_profile_scores=np.asarray(regions),
            reveal_score=reveal,
            occlusion_score=occlusion,
            support_contact_score=support,
            transition_embedding=np.asarray(embed),
        )
        losses = self.compute_losses(pred, targets)

        grad_embed = np.zeros_like(embed)

        grad_family_logits = family_probs.copy()
        grad_family_logits[targets.family_index] -= 1.0
        grad_embed += grad_family_logits @ self.w_family.T
        self.w_family -= lr * np.outer(embed, grad_family_logits)
        self.b_family -= lr * grad_family_logits

        grad_phase_logits = phase_probs.copy()
        grad_phase_logits[targets.phase_index] -= 1.0
        grad_embed += grad_phase_logits @ self.w_phase.T
        self.w_phase -= lr * np.outer(embed, grad_phase_logits)
        self.b_phase -= lr * grad_phase_logits

        tgt_regions = np.asarray(targets.target_profile_regions, dtype=np.float64)
        grad_regions_logits = (2.0 / max(1, len(REGION_KEYS))) * (regions - tgt_regions) * regions * (1.0 - regions)
        grad_embed += grad_regions_logits @ self.w_regions.T
        self.w_regions -= lr * np.outer(embed, grad_regions_logits)
        self.b_regions -= lr * grad_regions_logits

        grad_reveal_logit = np.array([2.0 * (reveal - float(targets.reveal_score)) * reveal * (1.0 - reveal)], dtype=np.float64)
        grad_embed += (grad_reveal_logit @ self.w_reveal.T)
        self.w_reveal -= lr * np.outer(embed, grad_reveal_logit)
        self.b_reveal -= lr * grad_reveal_logit

        grad_occ_logit = np.array([2.0 * (occlusion - float(targets.occlusion_score)) * occlusion * (1.0 - occlusion)], dtype=np.float64)
        grad_embed += (grad_occ_logit @ self.w_occlusion.T)
        self.w_occlusion -= lr * np.outer(embed, grad_occ_logit)
        self.b_occlusion -= lr * grad_occ_logit

        grad_support_logit = np.array([2.0 * (support - float(targets.support_contact_score)) * support * (1.0 - support)], dtype=np.float64)
        grad_embed += (grad_support_logit @ self.w_support.T)
        self.w_support -= lr * np.outer(embed, grad_support_logit)
        self.b_support -= lr * grad_support_logit

        grad_embed = grad_embed * (1.0 - embed * embed)
        grad_h = grad_embed @ self.w_embed.T
        grad_h = grad_h * (1.0 - h * h)
        self.w_embed -= lr * np.outer(h, grad_embed)
        self.b_embed -= lr * grad_embed
        self.w_in -= lr * np.outer(x, grad_h)
        self.b_in -= lr * grad_h
        return losses

    def to_contract(self, prediction: TemporalTransitionPrediction) -> dict[str, object]:
        fam_probs = self._softmax(prediction.family_logits)
        phase_probs = self._softmax(prediction.phase_logits)
        family_idx = int(np.argmax(fam_probs))
        phase_idx = int(np.argmax(phase_probs))
        region_scores = {REGION_KEYS[i]: float(prediction.target_profile_scores[i]) for i in range(len(REGION_KEYS))}
        sorted_regions = sorted(region_scores, key=region_scores.get, reverse=True)
        return {
            "family_logits": prediction.family_logits.tolist(),
            "phase_logits": prediction.phase_logits.tolist(),
            "target_profile_scores": region_scores,
            "reveal_score": float(prediction.reveal_score),
            "occlusion_score": float(prediction.occlusion_score),
            "support_contact_score": float(prediction.support_contact_score),
            "transition_embedding": prediction.transition_embedding.tolist(),
            "predicted_family": FAMILIES[family_idx],
            "predicted_phase": PHASES[phase_idx],
            "target_profile": {
                "primary_regions": sorted_regions[:2],
                "secondary_regions": sorted_regions[2:5],
                "context_regions": [r for r in sorted_regions if r.startswith("outer") or r in {"garments", "legs"}][:3],
            },
        }

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "w_in": self.w_in.tolist(),
            "b_in": self.b_in.tolist(),
            "w_embed": self.w_embed.tolist(),
            "b_embed": self.b_embed.tolist(),
            "w_family": self.w_family.tolist(),
            "b_family": self.b_family.tolist(),
            "w_phase": self.w_phase.tolist(),
            "b_phase": self.b_phase.tolist(),
            "w_regions": self.w_regions.tolist(),
            "b_regions": self.b_regions.tolist(),
            "w_reveal": self.w_reveal.tolist(),
            "b_reveal": self.b_reveal.tolist(),
            "w_occlusion": self.w_occlusion.tolist(),
            "b_occlusion": self.b_occlusion.tolist(),
            "w_support": self.w_support.tolist(),
            "b_support": self.b_support.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TemporalTransitionEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(input_dim=int(payload["input_dim"]), hidden_dim=int(payload["hidden_dim"]), embed_dim=int(payload["embed_dim"]))
        for name in (
            "w_in",
            "b_in",
            "w_embed",
            "b_embed",
            "w_family",
            "b_family",
            "w_phase",
            "b_phase",
            "w_regions",
            "b_regions",
            "w_reveal",
            "b_reveal",
            "w_occlusion",
            "b_occlusion",
            "w_support",
            "b_support",
        ):
            setattr(model, name, np.asarray(payload[name], dtype=np.float64))
        return model
