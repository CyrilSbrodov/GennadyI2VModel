from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.schema import TransitionTargetProfile
from dynamics.transition_contracts import LearnedHumanStateContract

REGION_KEYS = ["face", "torso", "left_arm", "right_arm", "legs", "garments", "inner_garment", "outer_garment"]
PHASES = ["prepare", "transition", "contact_or_reveal", "stabilize"]
FAMILIES = ["pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition"]


@dataclass(slots=True)
class HumanStateTransitionTargets:
    family_index: int
    phase_index: int
    region_state_targets: list[float]
    visibility_targets: list[float]
    reveal_memory_target: float
    support_contact_target: float


@dataclass(slots=True)
class HumanStateTransitionPrediction:
    family_logits: np.ndarray
    phase_logits: np.ndarray
    state_embedding: np.ndarray
    compact_conditioning_embedding: np.ndarray
    region_state_embeddings: np.ndarray
    visibility_state_scores: np.ndarray
    reveal_memory_embedding: np.ndarray
    support_contact_state: float
    confidence: float


class HumanStateTransitionModel:
    """Compact learned bootstrap module for latent human-region transition state."""

    def __init__(self, input_dim: int = 160, hidden_dim: int = 96, state_dim: int = 24, seed: int = 37) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        rng = np.random.default_rng(seed)
        self.w_in = rng.normal(0.0, 0.08, size=(input_dim, hidden_dim))
        self.b_in = np.zeros(hidden_dim, dtype=np.float64)
        self.w_state = rng.normal(0.0, 0.08, size=(hidden_dim, state_dim))
        self.b_state = np.zeros(state_dim, dtype=np.float64)

        self.w_family = rng.normal(0.0, 0.07, size=(state_dim, len(FAMILIES)))
        self.b_family = np.zeros(len(FAMILIES), dtype=np.float64)
        self.w_phase = rng.normal(0.0, 0.07, size=(state_dim, len(PHASES)))
        self.b_phase = np.zeros(len(PHASES), dtype=np.float64)

        self.w_regions = rng.normal(0.0, 0.07, size=(state_dim, len(REGION_KEYS) * 3))
        self.b_regions = np.zeros(len(REGION_KEYS) * 3, dtype=np.float64)
        self.w_visibility = rng.normal(0.0, 0.07, size=(state_dim, len(REGION_KEYS)))
        self.b_visibility = np.zeros(len(REGION_KEYS), dtype=np.float64)
        self.w_reveal = rng.normal(0.0, 0.07, size=(state_dim, 1))
        self.b_reveal = np.zeros(1, dtype=np.float64)
        self.w_support = rng.normal(0.0, 0.07, size=(state_dim, 1))
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
        hidden = np.tanh(x @ self.w_in + self.b_in)
        state = np.tanh(hidden @ self.w_state + self.b_state)
        logits = {
            "family": state @ self.w_family + self.b_family,
            "phase": state @ self.w_phase + self.b_phase,
            "regions": state @ self.w_regions + self.b_regions,
            "visibility": state @ self.w_visibility + self.b_visibility,
            "reveal": state @ self.w_reveal + self.b_reveal,
            "support": state @ self.w_support + self.b_support,
        }
        return hidden, state, logits

    def forward(self, feature_vector: list[float] | np.ndarray) -> HumanStateTransitionPrediction:
        x = np.asarray(feature_vector, dtype=np.float64)
        _, state, logits = self._forward_internal(x)
        region_raw = np.asarray(logits["regions"], dtype=np.float64).reshape(len(REGION_KEYS), 3)
        region_emb = np.tanh(region_raw)
        visibility = self._sigmoid(np.asarray(logits["visibility"], dtype=np.float64))
        reveal = np.tanh(np.asarray(logits["reveal"], dtype=np.float64))
        support = float(self._sigmoid(np.asarray(logits["support"], dtype=np.float64))[0])
        fam_probs = self._softmax(np.asarray(logits["family"], dtype=np.float64))
        phase_probs = self._softmax(np.asarray(logits["phase"], dtype=np.float64))
        confidence = float(np.clip(0.45 * np.max(fam_probs) + 0.35 * np.max(phase_probs) + 0.2 * np.mean(visibility), 0.0, 1.0))
        return HumanStateTransitionPrediction(
            family_logits=np.asarray(logits["family"], dtype=np.float64),
            phase_logits=np.asarray(logits["phase"], dtype=np.float64),
            state_embedding=np.asarray(state, dtype=np.float64),
            compact_conditioning_embedding=np.asarray(np.concatenate([state[:12], visibility[:4], reveal]), dtype=np.float64),
            region_state_embeddings=np.asarray(region_emb, dtype=np.float64),
            visibility_state_scores=np.asarray(visibility, dtype=np.float64),
            reveal_memory_embedding=np.asarray(reveal, dtype=np.float64),
            support_contact_state=support,
            confidence=confidence,
        )

    def compute_losses(
        self,
        prediction: HumanStateTransitionPrediction,
        targets: HumanStateTransitionTargets,
        previous_state_embedding: np.ndarray | None = None,
    ) -> dict[str, float]:
        fam_probs = self._softmax(prediction.family_logits)
        phase_probs = self._softmax(prediction.phase_logits)
        family_loss = -float(np.log(max(1e-8, fam_probs[targets.family_index])))
        phase_loss = -float(np.log(max(1e-8, phase_probs[targets.phase_index])))
        region_tgt = np.asarray(targets.region_state_targets, dtype=np.float64)
        vis_tgt = np.asarray(targets.visibility_targets, dtype=np.float64)
        region_scalar = np.mean(prediction.region_state_embeddings, axis=1)
        region_state_prediction_loss = float(np.mean((region_scalar - region_tgt) ** 2))
        visibility_state_loss = float(np.mean((prediction.visibility_state_scores - vis_tgt) ** 2))
        reveal_memory_alignment_loss = float((float(prediction.reveal_memory_embedding[0]) - float(targets.reveal_memory_target)) ** 2)
        state_transition_consistency_loss = float(0.5 * family_loss + 0.5 * phase_loss)
        temporal_smoothness_loss = 0.0
        if previous_state_embedding is not None and np.asarray(previous_state_embedding).shape == prediction.state_embedding.shape:
            temporal_smoothness_loss = float(np.mean((prediction.state_embedding - np.asarray(previous_state_embedding, dtype=np.float64)) ** 2))
        support_loss = float((prediction.support_contact_state - float(targets.support_contact_target)) ** 2)
        total = (
            state_transition_consistency_loss
            + reveal_memory_alignment_loss
            + visibility_state_loss
            + region_state_prediction_loss
            + temporal_smoothness_loss
            + support_loss
        )
        return {
            "state_transition_consistency_loss": state_transition_consistency_loss,
            "reveal_memory_alignment_loss": reveal_memory_alignment_loss,
            "visibility_state_loss": visibility_state_loss,
            "region_state_prediction_loss": region_state_prediction_loss,
            "temporal_smoothness_loss": temporal_smoothness_loss,
            "support_contact_state_loss": support_loss,
            "total_loss": total,
        }

    def train_step(
        self,
        feature_vector: list[float] | np.ndarray,
        targets: HumanStateTransitionTargets,
        lr: float = 1e-3,
        previous_state_embedding: np.ndarray | None = None,
    ) -> dict[str, float]:
        # Bootstrap-level tiny update: keeps module trainable without giant optimization plumbing.
        x = np.asarray(feature_vector, dtype=np.float64)
        hidden, state, logits = self._forward_internal(x)
        pred = self.forward(x)
        losses = self.compute_losses(pred, targets, previous_state_embedding=previous_state_embedding)

        grad_state = np.zeros_like(state)
        fam_probs = self._softmax(logits["family"])
        fam_grad = fam_probs.copy()
        fam_grad[targets.family_index] -= 1.0
        grad_state += fam_grad @ self.w_family.T
        self.w_family -= lr * np.outer(state, fam_grad)
        self.b_family -= lr * fam_grad

        phase_probs = self._softmax(logits["phase"])
        phase_grad = phase_probs.copy()
        phase_grad[targets.phase_index] -= 1.0
        grad_state += phase_grad @ self.w_phase.T
        self.w_phase -= lr * np.outer(state, phase_grad)
        self.b_phase -= lr * phase_grad

        vis = pred.visibility_state_scores
        vis_tgt = np.asarray(targets.visibility_targets, dtype=np.float64)
        vis_grad = (2.0 / max(1, len(REGION_KEYS))) * (vis - vis_tgt) * vis * (1.0 - vis)
        grad_state += vis_grad @ self.w_visibility.T
        self.w_visibility -= lr * np.outer(state, vis_grad)
        self.b_visibility -= lr * vis_grad

        reveal = float(pred.reveal_memory_embedding[0])
        reveal_target = float(targets.reveal_memory_target)
        reveal_logit_grad = np.array([2.0 * (reveal - reveal_target) * (1.0 - reveal * reveal)], dtype=np.float64)
        grad_state += reveal_logit_grad @ self.w_reveal.T
        self.w_reveal -= lr * np.outer(state, reveal_logit_grad)
        self.b_reveal -= lr * reveal_logit_grad

        support = pred.support_contact_state
        support_tgt = float(targets.support_contact_target)
        support_logit_grad = np.array([2.0 * (support - support_tgt) * support * (1.0 - support)], dtype=np.float64)
        grad_state += support_logit_grad @ self.w_support.T
        self.w_support -= lr * np.outer(state, support_logit_grad)
        self.b_support -= lr * support_logit_grad

        region_tgt = np.asarray(targets.region_state_targets, dtype=np.float64)
        region_scalar = np.mean(pred.region_state_embeddings, axis=1)
        region_delta = ((2.0 / max(1, len(REGION_KEYS))) * (region_scalar - region_tgt))[:, None] / 3.0
        region_act = pred.region_state_embeddings
        region_logit_grad = (region_delta * (1.0 - region_act * region_act)).reshape(-1)
        grad_state += region_logit_grad @ self.w_regions.T
        self.w_regions -= lr * np.outer(state, region_logit_grad)
        self.b_regions -= lr * region_logit_grad

        if previous_state_embedding is not None and np.asarray(previous_state_embedding).shape == state.shape:
            grad_state += (2.0 / max(1, state.shape[0])) * (state - np.asarray(previous_state_embedding, dtype=np.float64))

        grad_state = grad_state * (1.0 - state * state)
        grad_hidden = grad_state @ self.w_state.T
        grad_hidden = grad_hidden * (1.0 - hidden * hidden)
        self.w_state -= lr * np.outer(hidden, grad_state)
        self.b_state -= lr * grad_state
        self.w_in -= lr * np.outer(x, grad_hidden)
        self.b_in -= lr * grad_hidden
        return losses

    def to_typed_contract(self, prediction: HumanStateTransitionPrediction) -> LearnedHumanStateContract:
        fam_idx = int(np.argmax(self._softmax(prediction.family_logits)))
        phase_idx = int(np.argmax(self._softmax(prediction.phase_logits)))
        region_state_embeddings = {
            REGION_KEYS[i]: [float(x) for x in prediction.region_state_embeddings[i].tolist()]
            for i in range(len(REGION_KEYS))
        }
        return LearnedHumanStateContract(
            predicted_family=FAMILIES[fam_idx],
            predicted_phase=PHASES[phase_idx],
            state_embedding=[float(x) for x in prediction.state_embedding.tolist()],
            region_state_embeddings=region_state_embeddings,
            reveal_memory_embedding=[float(x) for x in prediction.reveal_memory_embedding.tolist()],
            visibility_state_scores={REGION_KEYS[i]: float(prediction.visibility_state_scores[i]) for i in range(len(REGION_KEYS))},
            support_contact_state=float(prediction.support_contact_state),
            compact_conditioning_embedding=[float(x) for x in prediction.compact_conditioning_embedding.tolist()],
            target_profile=TransitionTargetProfile(
                primary_regions=[k for k in REGION_KEYS if prediction.visibility_state_scores[REGION_KEYS.index(k)] >= 0.55][:3],
                secondary_regions=[k for k in REGION_KEYS if 0.35 <= prediction.visibility_state_scores[REGION_KEYS.index(k)] < 0.55][:3],
                context_regions=[k for k in REGION_KEYS if prediction.visibility_state_scores[REGION_KEYS.index(k)] < 0.35][:3],
            ),
            confidence=float(prediction.confidence),
            teacher_source="human_state_bootstrap_manifest",
            is_learned_primary=True,
        )


    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "state_dim": self.state_dim,
            "w_in": self.w_in.tolist(),
            "b_in": self.b_in.tolist(),
            "w_state": self.w_state.tolist(),
            "b_state": self.b_state.tolist(),
            "w_family": self.w_family.tolist(),
            "b_family": self.b_family.tolist(),
            "w_phase": self.w_phase.tolist(),
            "b_phase": self.b_phase.tolist(),
            "w_regions": self.w_regions.tolist(),
            "b_regions": self.b_regions.tolist(),
            "w_visibility": self.w_visibility.tolist(),
            "b_visibility": self.b_visibility.tolist(),
            "w_reveal": self.w_reveal.tolist(),
            "b_reveal": self.b_reveal.tolist(),
            "w_support": self.w_support.tolist(),
            "b_support": self.b_support.tolist(),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "HumanStateTransitionModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            input_dim=int(payload.get("input_dim", 160)),
            hidden_dim=int(payload.get("hidden_dim", 96)),
            state_dim=int(payload.get("state_dim", 24)),
        )
        for name in (
            "w_in",
            "b_in",
            "w_state",
            "b_state",
            "w_family",
            "b_family",
            "w_phase",
            "b_phase",
            "w_regions",
            "b_regions",
            "w_visibility",
            "b_visibility",
            "w_reveal",
            "b_reveal",
            "w_support",
            "b_support",
        ):
            if name in payload:
                setattr(model, name, np.asarray(payload[name], dtype=np.float64))
        return model
