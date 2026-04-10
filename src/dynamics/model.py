from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.schema import GraphDelta, RegionRef, SceneGraph, VideoMemory
from planning.transition_engine import PlannedState

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

POSE_KEYS = ["torso_pitch", "head_yaw", "arm_raise", "torso_yaw"]
GARMENT_KEYS = ["attachment_delta", "coverage_delta", "layer_shift"]
VISIBILITY_KEYS = ["face", "torso", "arms", "legs", "garments"]
EXPRESSION_KEYS = ["smile_intensity", "eye_openness", "mouth_open"]
INTERACTION_KEYS = ["support_contact", "hand_contact", "proximity_contact"]
REGION_KEYS = ["reveal_score", "occlude_score", "deform_hint", "stabilize_hint"]
ACTION_VOCAB = [
    "sit_down",
    "stand_up",
    "raise_arm",
    "turn_head",
    "remove_garment",
    "open_garment",
    "smile",
    "touch",
    "support",
    "posture_change",
    "walk",
    "lean",
    "wave",
    "look_away",
]
PHASES = ["prepare", "transition", "contact_or_reveal", "stabilize"]
FAMILIES = ["pose_transition", "garment_transition", "interaction_transition", "expression_transition"]
DYNAMICS_CHECKPOINT_FORMAT = "torch_structured_dynamics_v3"
DYNAMICS_CHECKPOINT_VERSION = 3
VISIBILITY_STATES = ["visible", "partially_visible", "hidden", "unknown"]
SCENE_REGION_KEYS = ["face", "torso", "arms", "legs", "garments", "pelvis", "head", "inner_garment"]

GRAPH_DIM = 27
PLANNER_DIM = 8
ACTION_DIM = 16
MEMORY_DIM = 8
TARGET_DIM = 37
INPUT_DIM = GRAPH_DIM + PLANNER_DIM + ACTION_DIM + MEMORY_DIM + TARGET_DIM

TOKEN_COUNT = 8
TOKEN_INDEX = {"cls": 0, "graph": 1, "planner": 2, "action": 3, "memory": 4, "target": 5, "family": 6, "phase": 7}


class DynamicsModelError(RuntimeError):
    pass


class DynamicsModelContractError(DynamicsModelError):
    pass


@dataclass(slots=True)
class DynamicsInputs:
    graph_features: list[float]
    planner_features: list[float]
    action_features: list[float]
    memory_features: list[float]
    target_features: list[float]

    def as_vector(self) -> np.ndarray:
        vec = np.asarray(
            self.graph_features
            + self.planner_features
            + self.action_features
            + self.memory_features
            + self.target_features,
            dtype=np.float32,
        )
        if vec.ndim != 1:
            raise DynamicsModelContractError("feature vector must be 1D")
        if vec.size != INPUT_DIM:
            raise DynamicsModelContractError(f"expected flattened dynamics input with {INPUT_DIM} values, got {vec.size}")
        return vec


@dataclass(slots=True)
class DynamicsTensorBatch:
    features: np.ndarray
    family: str
    phase: str
    conditioning_summary: dict[str, object]


@dataclass(slots=True)
class DynamicsTargets:
    pose: list[float]
    garment: list[float]
    visibility: list[float]
    expression: list[float]
    interaction: list[float]
    region: list[float]
    family: str = "pose_transition"
    phase: str = "transition"


@dataclass(slots=True)
class DynamicsPrediction:
    pose: np.ndarray
    garment: np.ndarray
    visibility: np.ndarray
    expression: np.ndarray
    interaction: np.ndarray
    region: np.ndarray
    family: str = "pose_transition"
    phase: str = "transition"
    family_logits: np.ndarray | None = None
    phase_logits: np.ndarray | None = None
    transition_confidence: float = 0.0
    aux: dict[str, np.ndarray | float | str] = field(default_factory=dict)


@dataclass(slots=True)
class DynamicsTorchPrediction:
    pose: torch.Tensor
    garment: torch.Tensor
    visibility: torch.Tensor
    expression: torch.Tensor
    interaction: torch.Tensor
    region: torch.Tensor
    family: str = "pose_transition"
    phase: str = "transition"
    family_logits: torch.Tensor | None = None
    phase_logits: torch.Tensor | None = None
    transition_confidence: torch.Tensor | None = None
    aux: dict[str, torch.Tensor | float | str] = field(default_factory=dict)


if nn is not None:

    class _GroupEncoder(nn.Module):
        def __init__(self, in_dim: int, d_model: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class _FamilyAdapter(nn.Module):
        def __init__(self, d_model: int, adapter_dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, adapter_dim)
            self.fc2 = nn.Linear(adapter_dim, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            y = self.norm(x)
            y = F.silu(self.fc1(y))
            y = self.fc2(y)
            return residual + y


    class _Head(nn.Module):
        def __init__(self, d_model: int, out_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


if nn is None:
    class _ModuleBase:
        def __init__(self) -> None:
            pass
else:
    _ModuleBase = nn.Module


class DynamicsModel(_ModuleBase):
    """Structured torch dynamics core with grouped tokenization and family adapters."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        trunk_dim: int = 192,
        seed: int = 17,
        *,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 6,
        ff_dim: int = 768,
        adapter_dim: int = 64,
    ) -> None:
        super().__init__()
        del hidden_dim, trunk_dim
        if torch is None or nn is None:
            raise DynamicsModelError("DynamicsModel requires torch. Install PyTorch to use dynamics core.")

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.adapter_dim = adapter_dim
        self.using_torch = True

        torch.manual_seed(seed)

        self.group_dims = {
            "graph": GRAPH_DIM,
            "planner": PLANNER_DIM,
            "action": ACTION_DIM,
            "memory": MEMORY_DIM,
            "target": TARGET_DIM,
        }
        self.group_offsets = {
            "graph": (0, GRAPH_DIM),
            "planner": (GRAPH_DIM, GRAPH_DIM + PLANNER_DIM),
            "action": (GRAPH_DIM + PLANNER_DIM, GRAPH_DIM + PLANNER_DIM + ACTION_DIM),
            "memory": (
                GRAPH_DIM + PLANNER_DIM + ACTION_DIM,
                GRAPH_DIM + PLANNER_DIM + ACTION_DIM + MEMORY_DIM,
            ),
            "target": (
                GRAPH_DIM + PLANNER_DIM + ACTION_DIM + MEMORY_DIM,
                INPUT_DIM,
            ),
        }

        self.group_encoders = nn.ModuleDict(
            {
                name: _GroupEncoder(in_dim=dim, d_model=d_model)
                for name, dim in self.group_dims.items()
            }
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_embeddings = nn.Embedding(TOKEN_COUNT, d_model)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, TOKEN_COUNT, d_model))
        self.family_embedding = nn.Embedding(len(FAMILIES), d_model)
        self.phase_embedding = nn.Embedding(len(PHASES), d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.family_adapters = nn.ModuleDict(
            {family: _FamilyAdapter(d_model=d_model, adapter_dim=adapter_dim) for family in FAMILIES}
        )

        self.summary_norm = nn.LayerNorm(d_model)
        self.summary_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, 1))

        self.heads = nn.ModuleDict(
            {
                "pose": _Head(d_model=d_model, out_dim=len(POSE_KEYS)),
                "garment": _Head(d_model=d_model, out_dim=len(GARMENT_KEYS)),
                "visibility": _Head(d_model=d_model, out_dim=len(VISIBILITY_KEYS)),
                "expression": _Head(d_model=d_model, out_dim=len(EXPRESSION_KEYS)),
                "interaction": _Head(d_model=d_model, out_dim=len(INTERACTION_KEYS)),
                "region": _Head(d_model=d_model, out_dim=len(REGION_KEYS)),
            }
        )

        self.family_logits_head = _Head(d_model=d_model, out_dim=len(FAMILIES))
        self.phase_logits_head = _Head(d_model=d_model, out_dim=len(PHASES))
        self.transition_confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        self._init_parameters()
        self.optimizer = torch.optim.AdamW(super().parameters(), lr=1e-3, weight_decay=1e-4)

    def _init_parameters(self) -> None:
        if torch is None:
            return
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _family_index(self, family: str) -> int:
        family_name = family if family in FAMILIES else "pose_transition"
        return FAMILIES.index(family_name)

    def _phase_index(self, phase: str) -> int:
        phase_name = phase if phase in PHASES else "transition"
        return PHASES.index(phase_name)

    def _split_groups_tensor(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2:
            raise DynamicsModelContractError(f"expected 2D tensor for dynamics input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise DynamicsModelContractError(
                f"expected dynamics input dim {self.input_dim}, got {x.shape[1]}"
            )
        out: dict[str, torch.Tensor] = {}
        for name, (lo, hi) in self.group_offsets.items():
            out[name] = x[:, lo:hi]
        return out

    def _build_tokens(self, x: torch.Tensor, family_idx: torch.Tensor, phase_idx: torch.Tensor) -> torch.Tensor:
        groups = self._split_groups_tensor(x)
        batch = x.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)

        token_list = [
            cls,
            self.group_encoders["graph"](groups["graph"]).unsqueeze(1),
            self.group_encoders["planner"](groups["planner"]).unsqueeze(1),
            self.group_encoders["action"](groups["action"]).unsqueeze(1),
            self.group_encoders["memory"](groups["memory"]).unsqueeze(1),
            self.group_encoders["target"](groups["target"]).unsqueeze(1),
            self.family_embedding(family_idx).unsqueeze(1),
            self.phase_embedding(phase_idx).unsqueeze(1),
        ]
        tokens = torch.cat(token_list, dim=1)
        type_ids = torch.arange(TOKEN_COUNT, device=x.device).unsqueeze(0).expand(batch, -1)
        tokens = tokens + self.type_embeddings(type_ids) + self.pos_embeddings[:, :TOKEN_COUNT, :]
        return tokens

    def _apply_family_adapter(self, hidden: torch.Tensor, family_idx: torch.Tensor) -> torch.Tensor:
        adapted = hidden.clone()
        for idx, family in enumerate(FAMILIES):
            mask = family_idx == idx
            if torch.any(mask):
                adapted[mask] = self.family_adapters[family](hidden[mask])
        return adapted

    def _summary_vector(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_vec = hidden[:, TOKEN_INDEX["cls"], :]
        mean_vec = hidden.mean(dim=1)
        gate_logits = self.summary_gate(torch.cat([cls_vec, mean_vec], dim=-1))
        gate = torch.sigmoid(gate_logits)
        summary = gate * cls_vec + (1.0 - gate) * mean_vec
        return self.summary_norm(summary), gate.squeeze(-1)

    def _forward_raw(self, x: torch.Tensor, family: str, phase: str) -> dict[str, torch.Tensor]:
        if torch is None:
            raise DynamicsModelError("DynamicsModel requires torch for forward path")
        family_idx_val = self._family_index(family)
        phase_idx_val = self._phase_index(phase)
        family_idx = torch.full((x.shape[0],), family_idx_val, dtype=torch.long, device=x.device)
        phase_idx = torch.full((x.shape[0],), phase_idx_val, dtype=torch.long, device=x.device)

        tokens = self._build_tokens(x, family_idx=family_idx, phase_idx=phase_idx)
        hidden = self.trunk(tokens)
        hidden = self._apply_family_adapter(hidden, family_idx=family_idx)
        summary, gate = self._summary_vector(hidden)

        raw = {name: head(summary) for name, head in self.heads.items()}
        raw["family_logits"] = self.family_logits_head(summary)
        raw["phase_logits"] = self.phase_logits_head(summary)
        raw["transition_confidence"] = self.transition_confidence_head(summary).squeeze(-1)
        raw["summary_gate"] = gate
        raw["hidden_cls"] = hidden[:, TOKEN_INDEX["cls"], :]
        return raw

    def _as_model_input_tensor(self, inputs: DynamicsInputs) -> torch.Tensor:
        vec = inputs.as_vector()
        return torch.from_numpy(vec).to(dtype=torch.float32, device=self.device).reshape(1, -1)

    def _prediction_from_raw_torch(self, raw: dict[str, torch.Tensor], family: str, phase: str) -> DynamicsTorchPrediction:
        pose = torch.tanh(raw["pose"]).squeeze(0)
        garment = torch.tanh(raw["garment"]).squeeze(0)
        visibility = torch.sigmoid(raw["visibility"]).squeeze(0)
        expression = torch.tanh(raw["expression"]).squeeze(0)
        interaction = torch.sigmoid(raw["interaction"]).squeeze(0)
        region = torch.sigmoid(raw["region"]).squeeze(0)
        return DynamicsTorchPrediction(
            pose=pose,
            garment=garment,
            visibility=visibility,
            expression=expression,
            interaction=interaction,
            region=region,
            family=family if family in FAMILIES else "pose_transition",
            phase=phase if phase in PHASES else "transition",
            family_logits=raw["family_logits"].squeeze(0),
            phase_logits=raw["phase_logits"].squeeze(0),
            transition_confidence=torch.sigmoid(raw["transition_confidence"]).reshape(()),
            aux={
                "summary_gate": raw["summary_gate"].reshape(()),
                "hidden_cls_norm": torch.norm(raw["hidden_cls"], p=2).reshape(()),
            },
        )

    def _prediction_from_raw_numpy(self, raw: dict[str, torch.Tensor], family: str, phase: str) -> DynamicsPrediction:
        tpred = self._prediction_from_raw_torch(raw, family=family, phase=phase)
        return DynamicsPrediction(
            pose=tpred.pose.detach().cpu().numpy(),
            garment=tpred.garment.detach().cpu().numpy(),
            visibility=tpred.visibility.detach().cpu().numpy(),
            expression=tpred.expression.detach().cpu().numpy(),
            interaction=tpred.interaction.detach().cpu().numpy(),
            region=tpred.region.detach().cpu().numpy(),
            family=tpred.family,
            phase=tpred.phase,
            family_logits=tpred.family_logits.detach().cpu().numpy() if tpred.family_logits is not None else None,
            phase_logits=tpred.phase_logits.detach().cpu().numpy() if tpred.phase_logits is not None else None,
            transition_confidence=float(tpred.transition_confidence.detach().cpu().item() if tpred.transition_confidence is not None else 0.0),
            aux={
                "summary_gate": float(tpred.aux["summary_gate"].detach().cpu().item()),
                "hidden_cls_norm": float(tpred.aux["hidden_cls_norm"].detach().cpu().item()),
            },
        )

    def forward(self, inputs: DynamicsInputs, family: str = "pose_transition", phase: str = "transition") -> DynamicsTorchPrediction:
        x = self._as_model_input_tensor(inputs)
        raw = self._forward_raw(x, family=family, phase=phase)
        return self._prediction_from_raw_torch(raw, family=family, phase=phase)

    def predict(self, inputs: DynamicsInputs, family: str = "pose_transition", phase: str = "transition") -> DynamicsPrediction:
        prev_mode = self.training
        try:
            self.eval()
            with torch.no_grad():
                x = self._as_model_input_tensor(inputs)
                raw = self._forward_raw(x, family=family, phase=phase)
                return self._prediction_from_raw_numpy(raw, family=family, phase=phase)
        finally:
            self.train(prev_mode)

    def _confidence_target_from_targets(self, targets: DynamicsTargets) -> float:
        pose_mag = float(np.mean(np.abs(np.asarray(targets.pose, dtype=np.float32))))
        garment_mag = float(np.mean(np.abs(np.asarray(targets.garment, dtype=np.float32))))
        expression_mag = float(np.mean(np.abs(np.asarray(targets.expression, dtype=np.float32))))
        interaction_mag = float(np.mean(np.asarray(targets.interaction, dtype=np.float32)))
        visibility_mag = float(np.mean(np.asarray(targets.visibility, dtype=np.float32)))
        region_activity = float(np.mean(np.asarray(targets.region, dtype=np.float32)))
        family_phase_bias = 0.12 if targets.phase in {"contact_or_reveal", "stabilize"} else 0.06
        score = 0.32 * (pose_mag + garment_mag + expression_mag) + 0.26 * (interaction_mag + visibility_mag) + 0.42 * region_activity + family_phase_bias
        return float(np.clip(score, 0.05, 0.95))

    def _validate_target_shape(self, name: str, values: list[float], expected_dim: int) -> None:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.shape[0] != expected_dim:
            raise DynamicsModelContractError(
                f"invalid target shape for {name}: expected {expected_dim}, got {arr.shape[0]}"
            )

    def _validate_targets(self, targets: DynamicsTargets) -> None:
        self._validate_target_shape("pose", targets.pose, len(POSE_KEYS))
        self._validate_target_shape("garment", targets.garment, len(GARMENT_KEYS))
        self._validate_target_shape("visibility", targets.visibility, len(VISIBILITY_KEYS))
        self._validate_target_shape("expression", targets.expression, len(EXPRESSION_KEYS))
        self._validate_target_shape("interaction", targets.interaction, len(INTERACTION_KEYS))
        self._validate_target_shape("region", targets.region, len(REGION_KEYS))
        if targets.family not in FAMILIES:
            raise DynamicsModelContractError(f"unknown target family={targets.family}")
        if targets.phase not in PHASES:
            raise DynamicsModelContractError(f"unknown target phase={targets.phase}")

    def _to_numpy_1d(self, value: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _to_float(self, value: float | torch.Tensor) -> float:
        if torch is not None and isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def compute_losses(self, prediction: DynamicsPrediction | DynamicsTorchPrediction, targets: DynamicsTargets) -> dict[str, float]:
        self._validate_targets(targets)
        losses: dict[str, float] = {}
        family_weights = {
            "pose_transition": {
                "pose": 1.6,
                "garment": 0.6,
                "visibility": 1.0,
                "expression": 0.7,
                "interaction": 0.6,
                "region": 1.0,
            },
            "garment_transition": {
                "pose": 0.8,
                "garment": 1.6,
                "visibility": 1.2,
                "expression": 0.6,
                "interaction": 0.6,
                "region": 1.1,
            },
            "interaction_transition": {
                "pose": 0.7,
                "garment": 0.6,
                "visibility": 0.9,
                "expression": 0.6,
                "interaction": 1.6,
                "region": 1.1,
            },
            "expression_transition": {
                "pose": 0.8,
                "garment": 0.6,
                "visibility": 0.9,
                "expression": 1.6,
                "interaction": 0.7,
                "region": 1.0,
            },
        }
        weights = family_weights[targets.family]

        smooth_l1_heads = {"pose", "garment", "expression"}
        mse_heads = {"visibility", "interaction", "region"}
        total = 0.0

        for head in smooth_l1_heads | mse_heads:
            pred = self._to_numpy_1d(getattr(prediction, head))
            tgt = np.asarray(getattr(targets, head), dtype=np.float32).reshape(-1)
            if pred.shape != tgt.shape:
                raise DynamicsModelContractError(
                    f"shape mismatch for head={head}: {pred.shape} != {tgt.shape}"
                )
            if head in smooth_l1_heads:
                diff = np.abs(pred - tgt)
                base = np.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5).mean()
            else:
                base = float(np.mean((pred - tgt) ** 2))
            weighted = float(base) * float(weights[head])
            losses[f"{head}_loss"] = weighted
            total += weighted

        if prediction.family_logits is not None:
            logits = self._to_numpy_1d(prediction.family_logits)
            stable = logits - np.max(logits)
            probs = np.exp(stable) / max(1e-8, float(np.sum(np.exp(stable))))
            fam_loss = -float(np.log(max(1e-8, float(probs[self._family_index(targets.family)]))))
            losses["family_aux_loss"] = fam_loss * 0.2
            total += losses["family_aux_loss"]

        if prediction.phase_logits is not None:
            logits = self._to_numpy_1d(prediction.phase_logits)
            stable = logits - np.max(logits)
            probs = np.exp(stable) / max(1e-8, float(np.sum(np.exp(stable))))
            phase_loss = -float(np.log(max(1e-8, float(probs[self._phase_index(targets.phase)]))))
            losses["phase_aux_loss"] = phase_loss * 0.15
            total += losses["phase_aux_loss"]

        conf_target = self._confidence_target_from_targets(targets)
        conf_loss = float((self._to_float(prediction.transition_confidence) - conf_target) ** 2) * 0.2
        losses["confidence_loss"] = conf_loss
        total += conf_loss

        losses["total_loss"] = float(total)
        return losses

    def train_step(self, inputs: DynamicsInputs, targets: DynamicsTargets, lr: float = 1e-3) -> dict[str, float]:
        self.train()
        self._validate_targets(targets)
        if self.optimizer is None:
            raise DynamicsModelError("optimizer not initialized")

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        x = self._as_model_input_tensor(inputs)
        raw = self._forward_raw(x, family=targets.family, phase=targets.phase)

        target_map = {
            "pose": torch.tensor(targets.pose, dtype=torch.float32, device=self.device).reshape(1, -1),
            "garment": torch.tensor(targets.garment, dtype=torch.float32, device=self.device).reshape(1, -1),
            "visibility": torch.tensor(targets.visibility, dtype=torch.float32, device=self.device).reshape(1, -1),
            "expression": torch.tensor(targets.expression, dtype=torch.float32, device=self.device).reshape(1, -1),
            "interaction": torch.tensor(targets.interaction, dtype=torch.float32, device=self.device).reshape(1, -1),
            "region": torch.tensor(targets.region, dtype=torch.float32, device=self.device).reshape(1, -1),
        }

        pred_heads = {
            "pose": torch.tanh(raw["pose"]),
            "garment": torch.tanh(raw["garment"]),
            "visibility": torch.sigmoid(raw["visibility"]),
            "expression": torch.tanh(raw["expression"]),
            "interaction": torch.sigmoid(raw["interaction"]),
            "region": torch.sigmoid(raw["region"]),
        }

        family_weights = {
            "pose_transition": {"pose": 1.6, "garment": 0.6, "visibility": 1.0, "expression": 0.7, "interaction": 0.6, "region": 1.0},
            "garment_transition": {"pose": 0.8, "garment": 1.6, "visibility": 1.2, "expression": 0.6, "interaction": 0.6, "region": 1.1},
            "interaction_transition": {"pose": 0.7, "garment": 0.6, "visibility": 0.9, "expression": 0.6, "interaction": 1.6, "region": 1.1},
            "expression_transition": {"pose": 0.8, "garment": 0.6, "visibility": 0.9, "expression": 1.6, "interaction": 0.7, "region": 1.0},
        }
        weights = family_weights[targets.family]

        losses_torch: dict[str, torch.Tensor] = {}
        losses_torch["pose_loss"] = F.smooth_l1_loss(pred_heads["pose"], target_map["pose"]) * weights["pose"]
        losses_torch["garment_loss"] = F.smooth_l1_loss(pred_heads["garment"], target_map["garment"]) * weights["garment"]
        losses_torch["expression_loss"] = F.smooth_l1_loss(pred_heads["expression"], target_map["expression"]) * weights["expression"]

        losses_torch["visibility_loss"] = F.mse_loss(pred_heads["visibility"], target_map["visibility"]) * weights["visibility"]
        losses_torch["interaction_loss"] = F.mse_loss(pred_heads["interaction"], target_map["interaction"]) * weights["interaction"]
        losses_torch["region_loss"] = F.mse_loss(pred_heads["region"], target_map["region"]) * weights["region"]

        family_target = torch.tensor([self._family_index(targets.family)], dtype=torch.long, device=self.device)
        phase_target = torch.tensor([self._phase_index(targets.phase)], dtype=torch.long, device=self.device)
        losses_torch["family_aux_loss"] = F.cross_entropy(raw["family_logits"], family_target) * 0.2
        losses_torch["phase_aux_loss"] = F.cross_entropy(raw["phase_logits"], phase_target) * 0.15
        conf_target = torch.tensor([self._confidence_target_from_targets(targets)], dtype=torch.float32, device=self.device)
        conf_pred = torch.sigmoid(raw["transition_confidence"])
        losses_torch["confidence_loss"] = F.mse_loss(conf_pred, conf_target) * 0.2

        total = sum(losses_torch.values())

        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        out = {name: float(val.detach().cpu().item()) for name, val in losses_torch.items()}
        out["total_loss"] = float(total.detach().cpu().item())
        return out

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": DYNAMICS_CHECKPOINT_VERSION,
            "format": DYNAMICS_CHECKPOINT_FORMAT,
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "adapter_dim": self.adapter_dim,
            "families": list(FAMILIES),
            "phases": list(PHASES),
            "state_dict": super().state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "DynamicsModel":
        model = cls()
        if not Path(path).exists():
            return model

        payload = torch.load(path, map_location="cpu")
        valid, reason = validate_checkpoint_payload(payload)
        if not valid:
            raise DynamicsModelError(f"invalid_checkpoint:{reason}")

        model = cls(
            input_dim=int(payload["input_dim"]),
            d_model=int(payload["d_model"]),
            nhead=int(payload["nhead"]),
            num_layers=int(payload["num_layers"]),
            ff_dim=int(payload["ff_dim"]),
            adapter_dim=int(payload["adapter_dim"]),
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        if model.optimizer is not None and isinstance(payload.get("optimizer_state"), dict) and payload.get("optimizer_state"):
            model.optimizer.load_state_dict(payload["optimizer_state"])
        return model


def validate_checkpoint_payload(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "checkpoint_payload_not_dict"
    if payload.get("format") != DYNAMICS_CHECKPOINT_FORMAT:
        return False, "checkpoint_format_mismatch"
    if int(payload.get("version", -1)) != DYNAMICS_CHECKPOINT_VERSION:
        return False, "checkpoint_version_mismatch"

    required_keys = {
        "input_dim",
        "d_model",
        "nhead",
        "num_layers",
        "ff_dim",
        "adapter_dim",
        "families",
        "phases",
        "state_dict",
    }
    missing = sorted(k for k in required_keys if k not in payload)
    if missing:
        return False, f"missing_keys:{','.join(missing)}"

    if int(payload.get("input_dim", -1)) != INPUT_DIM:
        return False, "input_dim_mismatch"

    families = payload.get("families")
    phases = payload.get("phases")
    if not isinstance(families, list) or [str(f) for f in families] != FAMILIES:
        return False, "family_config_invalid"
    if not isinstance(phases, list) or [str(p) for p in phases] != PHASES:
        return False, "phase_config_invalid"

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        return False, "state_dict_missing_or_empty"
    expected_prefixes = {
        "cls_token",
        "pos_embeddings",
        "group_encoders.graph.net.0.weight",
        "trunk.layers.0.self_attn.in_proj_weight",
        "family_adapters.pose_transition.fc1.weight",
        "heads.pose.net.1.weight",
        "family_logits_head.net.1.weight",
        "phase_logits_head.net.1.weight",
        "transition_confidence_head.1.weight",
    }
    if not expected_prefixes.issubset(set(state_dict.keys())):
        return False, "state_dict_schema_invalid"
    return True, "ok"


def dynamics_inputs_from_tensor_batch(batch: DynamicsTensorBatch) -> DynamicsInputs:
    vec = np.asarray(batch.features, dtype=np.float32).reshape(-1)
    if vec.size < INPUT_DIM:
        vec = np.concatenate([vec, np.zeros((INPUT_DIM - vec.size,), dtype=np.float32)])
    vec = vec[:INPUT_DIM]
    return DynamicsInputs(
        graph_features=vec[:GRAPH_DIM].tolist(),
        planner_features=vec[GRAPH_DIM : GRAPH_DIM + PLANNER_DIM].tolist(),
        action_features=vec[GRAPH_DIM + PLANNER_DIM : GRAPH_DIM + PLANNER_DIM + ACTION_DIM].tolist(),
        memory_features=vec[GRAPH_DIM + PLANNER_DIM + ACTION_DIM : GRAPH_DIM + PLANNER_DIM + ACTION_DIM + MEMORY_DIM].tolist(),
        target_features=vec[GRAPH_DIM + PLANNER_DIM + ACTION_DIM + MEMORY_DIM : INPUT_DIM].tolist(),
    )


def _phase_onehot(phase: str) -> list[float]:
    p = phase if phase in PHASES else "transition"
    return [1.0 if p == v else 0.0 for v in PHASES]


def _visibility_hist(values: list[str]) -> list[float]:
    if not values:
        return [0.0] * len(VISIBILITY_STATES)
    count = max(1.0, float(len(values)))
    return [float(sum(1 for v in values if v == state)) / count for state in VISIBILITY_STATES]


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def featurize_runtime(
    scene_graph: SceneGraph,
    target_state: PlannedState,
    planner_context: dict[str, float] | None,
    memory: VideoMemory | None,
) -> DynamicsInputs:
    context = planner_context or {}
    person = scene_graph.persons[0] if scene_graph.persons else None
    labels = {str(x) for x in target_state.labels if isinstance(x, str)}

    action_features = [1.0 if token in labels else 0.0 for token in ACTION_VOCAB]
    intensity = 0.5
    for item in labels:
        if item.startswith("intensity="):
            try:
                intensity = float(item.split("=", 1)[1])
            except ValueError:
                intensity = 0.5
    action_features.extend([_clamp01(intensity), _clamp01(float(len(labels)) / 10.0)])

    if person is not None:
        pose_angles = person.pose_state.angles
        graph_features = [
            _clamp01(person.bbox.x),
            _clamp01(person.bbox.y),
            _clamp01(person.bbox.w),
            _clamp01(person.bbox.h),
            _clamp01(float(len(scene_graph.persons)) / 4.0),
            _clamp01(float(len(scene_graph.objects)) / 12.0),
            _clamp01(float(len(scene_graph.relations)) / 24.0),
            _clamp01(float(len(person.body_parts)) / 16.0),
            _clamp01(float(len(person.garments)) / 10.0),
            _clamp01(float(person.expression_state.smile_intensity)),
            _clamp01(float(person.expression_state.eye_openness)),
            _clamp01((float(person.orientation.yaw) + 90.0) / 180.0),
            _clamp01((float(person.orientation.pitch) + 45.0) / 90.0),
            _clamp01((float(person.orientation.roll) + 45.0) / 90.0),
            _clamp01((float(pose_angles.get("torso_pitch", 0.0)) + 45.0) / 90.0),
            _clamp01((float(pose_angles.get("head_yaw", 0.0)) + 45.0) / 90.0),
        ]
        body_vis = _visibility_hist([bp.visibility for bp in person.body_parts])
        garment_vis = _visibility_hist([g.visibility for g in person.garments])
        garment_state_counts = [
            float(sum(1 for g in person.garments if g.garment_state == "worn")) / max(1.0, float(len(person.garments))),
            float(sum(1 for g in person.garments if g.garment_state in {"opening", "partially_detached"})) / max(1.0, float(len(person.garments))),
            float(sum(1 for g in person.garments if g.garment_state == "removed")) / max(1.0, float(len(person.garments))),
        ]
        graph_features.extend(body_vis + garment_vis + garment_state_counts)
    else:
        graph_features = [0.0] * GRAPH_DIM

    step_idx = float(context.get("step_index", target_state.step_index))
    total_steps = max(1.0, float(context.get("total_steps", context.get("plan_length", 4.0))))
    progress = _clamp01(step_idx / total_steps)
    duration = float(context.get("target_duration", context.get("duration", 1.0)))
    phase = str(context.get("phase", context.get("sequencing_stage", "transition")))
    planner_features = [
        _clamp01(step_idx / 16.0),
        _clamp01(total_steps / 16.0),
        progress,
        _clamp01(duration / 8.0),
    ] + _phase_onehot(phase)

    family = str(context.get("transition_family", "pose_transition"))
    family_onehot = [1.0 if family == f else 0.0 for f in FAMILIES]

    if memory is not None:
        hidden_slots = list(memory.hidden_region_slots.values())
        hidden_recent = float(
            sum(1 for slot in hidden_slots if hasattr(slot, "stale_frames") and float(getattr(slot, "stale_frames", 99.0)) < 4.0)
        )
        temporal_coverage = float(len(memory.temporal_history)) / 32.0
        memory_features = [
            _clamp01(float(len(memory.identity_memory)) / 16.0),
            _clamp01(float(len(memory.garment_memory)) / 16.0),
            _clamp01(float(len(memory.region_descriptors)) / 32.0),
            _clamp01(float(len(memory.hidden_region_slots)) / 32.0),
            _clamp01(hidden_recent / 16.0),
            _clamp01(temporal_coverage),
            1.0 if memory.last_transition_context else 0.0,
            1.0 if memory.last_transition_context.get("visibility_phase") == "mixed" else 0.0,
        ]
    else:
        memory_features = [0.0] * MEMORY_DIM

    semantic = target_state.semantic_transition
    tp = semantic.target_profile if semantic is not None else None

    target_features = [
        1.0 if any("target_pose=" in token for token in labels) else 0.0,
        1.0 if any("target_visibility=" in token for token in labels) else 0.0,
        1.0 if any("target_expression=" in token for token in labels) else 0.0,
        float(sum(1 for x in labels if x in {"sit_down", "stand_up", "posture_change"})) / 3.0,
        float(sum(1 for x in labels if x in {"remove_garment", "open_garment"})) / 2.0,
        float(sum(1 for x in labels if x in {"smile", "look_away"})) / 2.0,
        1.0 if "chair" in labels or (tp is not None and tp.support_target == "chair") else 0.0,
        1.0 if "support" in labels or (tp is not None and "support_zone" in tp.context_regions) else 0.0,
    ]
    target_features.extend(family_onehot)

    regions = (tp.primary_regions + tp.secondary_regions + tp.context_regions) if tp is not None else []
    # SCENE_REGION_KEYS are feature-space semantic regions from target_profile.
    # REGION_KEYS are decoder head outputs and must stay separate.
    region_presence = [1.0 if r in regions else 0.0 for r in SCENE_REGION_KEYS]
    semantic_conf = float(semantic.confidence) if semantic is not None else 0.0
    lexical_bootstrap = float(semantic.lexical_bootstrap_score) if semantic is not None else 0.0
    target_features.extend(region_presence)
    target_features.extend(
        [
            _clamp01(semantic_conf),
            _clamp01(lexical_bootstrap),
            _clamp01(float(len(labels)) / 10.0),
            _clamp01(float(len(regions)) / 8.0),
        ]
    )

    if len(target_features) < TARGET_DIM:
        target_features.extend([0.0] * (TARGET_DIM - len(target_features)))
    target_features = target_features[:TARGET_DIM]

    return DynamicsInputs(
        graph_features=graph_features[:GRAPH_DIM],
        planner_features=planner_features[:PLANNER_DIM],
        action_features=action_features[:ACTION_DIM],
        memory_features=memory_features[:MEMORY_DIM],
        target_features=target_features[:TARGET_DIM],
    )


def tensorize_dynamics_inputs(inputs: DynamicsInputs, family: str, phase: str) -> DynamicsTensorBatch:
    vec = inputs.as_vector().astype(np.float32)
    fam = family if family in FAMILIES else "pose_transition"
    ph = phase if phase in PHASES else "transition"
    return DynamicsTensorBatch(
        features=vec,
        family=fam,
        phase=ph,
        conditioning_summary={
            "family": fam,
            "phase": ph,
            "input_norm": float(np.linalg.norm(vec)),
            "non_zero": int(np.count_nonzero(vec)),
            "group_norms": {
                "graph": float(np.linalg.norm(vec[:GRAPH_DIM])),
                "planner": float(np.linalg.norm(vec[GRAPH_DIM : GRAPH_DIM + PLANNER_DIM])),
                "action": float(np.linalg.norm(vec[GRAPH_DIM + PLANNER_DIM : GRAPH_DIM + PLANNER_DIM + ACTION_DIM])),
                "memory": float(np.linalg.norm(vec[GRAPH_DIM + PLANNER_DIM + ACTION_DIM : GRAPH_DIM + PLANNER_DIM + ACTION_DIM + MEMORY_DIM])),
                "target": float(np.linalg.norm(vec[GRAPH_DIM + PLANNER_DIM + ACTION_DIM + MEMORY_DIM :])),
            },
        },
    )


def targets_from_delta(delta: GraphDelta, family: str = "pose_transition") -> DynamicsTargets:
    pose = [float(delta.pose_deltas.get(k, 0.0)) / 45.0 for k in POSE_KEYS]
    garment = [
        float(delta.garment_deltas.get("attachment_delta", 0.0)),
        float(delta.garment_deltas.get("coverage_delta", 0.0)),
        float(delta.garment_deltas.get("layer_shift", 0.0)),
    ]
    vis_map = delta.visibility_deltas or delta.predicted_visibility_changes
    visibility = [
        1.0
        if str(vis_map.get(key, "visible")) == "visible"
        else (0.5 if str(vis_map.get(key, "visible")) == "partially_visible" else 0.0)
        for key in VISIBILITY_KEYS
    ]
    expression = [
        float(delta.expression_deltas.get("smile_intensity", 0.0)),
        float(delta.expression_deltas.get("eye_openness", 0.0)),
        1.0 if str(delta.expression_deltas.get("mouth_state", "neutral")) in {"open", "smile"} else 0.0,
    ]
    interaction = [
        _clamp01(float(delta.interaction_deltas.get("support_contact", 0.0))),
        _clamp01(float(delta.interaction_deltas.get("hand_contact", 0.0))),
        _clamp01(float(delta.interaction_deltas.get("proximity_contact", 0.0))),
    ]
    region = [
        1.0 if delta.newly_revealed_regions else 0.0,
        1.0 if delta.newly_occluded_regions else 0.0,
        1.0 if any(v in {"deform", "deform_relation_aware"} for v in delta.region_transition_mode.values()) else 0.0,
        1.0 if delta.transition_phase in {"contact_or_reveal", "stabilize"} else 0.0,
    ]
    return DynamicsTargets(
        pose=pose,
        garment=garment,
        visibility=visibility,
        expression=expression,
        interaction=interaction,
        region=region,
        family=family if family in FAMILIES else "pose_transition",
        phase=delta.transition_phase if delta.transition_phase in PHASES else "transition",
    )


def decode_prediction(
    prediction: DynamicsPrediction,
    scene_graph: SceneGraph,
    phase: str,
    semantic_reasons: list[str],
    planner_context: dict[str, float] | None = None,
) -> GraphDelta:
    context = planner_context or {}
    family = prediction.family if prediction.family in FAMILIES else "pose_transition"
    phase_ctx = str(context.get("phase", prediction.phase if prediction.phase in PHASES else phase))
    phase_name = phase_ctx if phase_ctx in PHASES else (phase if phase in PHASES else "transition")

    person = scene_graph.persons[0] if scene_graph.persons else None
    confidence = float(prediction.transition_confidence)

    pose_scale = 45.0 if family != "expression_transition" else 32.0
    pose = {k: float(prediction.pose[i] * pose_scale) for i, k in enumerate(POSE_KEYS)}

    garment = {
        "attachment_delta": float(prediction.garment[0]),
        "coverage_delta": float(prediction.garment[1]),
        "layer_shift": float(prediction.garment[2]),
    }
    garment_threshold = 0.12 if family == "garment_transition" else 0.18
    if garment["coverage_delta"] < -garment_threshold:
        garment_progression = "removed"
    elif garment["attachment_delta"] < -0.08:
        garment_progression = "opening"
    else:
        garment_progression = "worn"
    garment["garment_progression"] = garment_progression

    visibility_hi = 0.72 - 0.07 * confidence
    visibility_mid = 0.42 - 0.05 * confidence
    visibility_deltas = {
        key: (
            "visible"
            if float(val) >= visibility_hi
            else ("partially_visible" if float(val) >= visibility_mid else "hidden")
        )
        for key, val in zip(VISIBILITY_KEYS, prediction.visibility)
    }

    expression = {
        "smile_intensity": float(prediction.expression[0]),
        "eye_openness": float(prediction.expression[1]),
        "mouth_state": "open" if prediction.expression[2] > (0.3 if family == "expression_transition" else 0.45) else "neutral",
        "expression_progression": "forming_expression" if prediction.expression[0] > 0.06 else "neutral",
    }

    interaction = {
        "support_contact": _clamp01(float(prediction.interaction[0])),
        "hand_contact": _clamp01(float(prediction.interaction[1])),
        "proximity_contact": _clamp01(float(prediction.interaction[2])),
    }

    region_scores = {
        "face": 0.45 * float(prediction.expression[0]) + 0.35 * float(prediction.visibility[0]) + 0.2 * float(prediction.region[2]),
        "head": 0.38 * float(prediction.expression[2]) + 0.34 * float(prediction.visibility[0]) + 0.28 * float(prediction.pose[1] + 1.0) * 0.5,
        "torso": 0.24 * float(prediction.pose[0] + 1.0) * 0.5 + 0.28 * float(prediction.visibility[1]) + 0.24 * float(prediction.region[2]) + 0.24 * float(prediction.region[3]),
        "arms": 0.32 * float(prediction.pose[2] + 1.0) * 0.5 + 0.38 * float(prediction.interaction[1]) + 0.18 * float(prediction.visibility[2]) + 0.12 * float(prediction.region[2]),
        "legs": 0.26 * float(prediction.pose[0] + 1.0) * 0.5 + 0.42 * float(prediction.interaction[2]) + 0.2 * float(prediction.visibility[3]) + 0.12 * float(prediction.region[3]),
        "pelvis": 0.34 * float(prediction.pose[0] + 1.0) * 0.5 + 0.34 * float(prediction.interaction[0]) + 0.16 * float(prediction.visibility[3]) + 0.16 * float(prediction.region[3]),
        "garments": 0.38 * float(max(0.0, -prediction.garment[0])) + 0.28 * float(max(0.0, -prediction.garment[1])) + 0.2 * float(prediction.visibility[4]) + 0.14 * float(prediction.region[0]),
        "inner_garment": 0.3 * float(max(0.0, -prediction.garment[1])) + 0.25 * float(prediction.region[0]) + 0.2 * float(prediction.visibility[1]) + 0.25 * float(prediction.visibility[4]),
    }

    family_bias = {
        "pose_transition": {"torso": 0.16, "legs": 0.14, "arms": 0.08},
        "garment_transition": {"garments": 0.2, "torso": 0.14, "inner_garment": 0.14},
        "interaction_transition": {"arms": 0.2, "torso": 0.14, "pelvis": 0.08},
        "expression_transition": {"face": 0.2, "head": 0.16, "torso": 0.06},
    }[family]
    for reg, add in family_bias.items():
        region_scores[reg] += add
    if phase_name in {"contact_or_reveal", "stabilize"}:
        region_scores["torso"] += 0.05
        region_scores["garments"] += 0.04

    # semantic_reasons act only as weak priors, not primary selection source.
    semantic_prior = {key: 0.0 for key in SCENE_REGION_KEYS}
    for token in semantic_reasons:
        if token in {"raise_arm", "wave", "touch"}:
            semantic_prior["arms"] += 0.04
            semantic_prior["torso"] += 0.02
        elif token in {"sit_down", "stand_up", "walk"}:
            semantic_prior["legs"] += 0.04
            semantic_prior["pelvis"] += 0.03
        elif token in {"smile", "look_away"}:
            semantic_prior["face"] += 0.04
            semantic_prior["head"] += 0.03
        elif token in {"remove_garment", "open_garment"}:
            semantic_prior["garments"] += 0.04
            semantic_prior["inner_garment"] += 0.03
    for reg in SCENE_REGION_KEYS:
        region_scores[reg] += semantic_prior[reg]

    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
    score_threshold = 0.34
    selected = [name for name, score in sorted_regions if score >= score_threshold]
    if not selected:
        selected = [sorted_regions[0][0], sorted_regions[1][0]]
    affected_regions = sorted(set(selected[:4]))

    reveal_t = 0.66 if family == "garment_transition" else 0.72
    occlude_t = 0.66
    deform_t = 0.58
    stabilize_t = 0.7 if phase_name in {"contact_or_reveal", "stabilize"} else 0.78

    region_mode: dict[str, str] = {}
    for region in affected_regions:
        if float(prediction.region[0]) > reveal_t:
            region_mode[region] = "reveal"
        elif float(prediction.region[1]) > occlude_t:
            region_mode[region] = "occlude"
        elif float(prediction.region[2]) > deform_t:
            region_mode[region] = "deform"
        elif float(prediction.region[3]) > stabilize_t:
            region_mode[region] = "stabilize"
        else:
            region_mode[region] = "stable"

    revealed: list[RegionRef] = []
    occluded: list[RegionRef] = []
    if person is not None:
        for region in affected_regions:
            mode = region_mode.get(region)
            if mode == "reveal":
                revealed.append(
                    RegionRef(
                        region_id=f"{person.person_id}:{region}:reveal",
                        bbox=person.bbox,
                        reason="learned_reveal",
                    )
                )
            elif mode == "occlude":
                occluded.append(
                    RegionRef(
                        region_id=f"{person.person_id}:{region}:occlude",
                        bbox=person.bbox,
                        reason="learned_occlude",
                    )
                )

    phase_out = phase_name
    if float(prediction.region[3]) > 0.82 and phase_out not in {"contact_or_reveal", "stabilize"}:
        phase_out = "stabilize"

    transition_diagnostics: dict[str, Any] = {
        "family": family,
        "phase": phase_out,
        "transition_confidence": confidence,
        "summary_gate": prediction.aux.get("summary_gate", 0.0),
        "family_logits": prediction.family_logits.tolist() if prediction.family_logits is not None else [],
        "phase_logits": prediction.phase_logits.tolist() if prediction.phase_logits is not None else [],
        "region_scores": {k: float(v) for k, v in sorted_regions[:5]},
    }

    return GraphDelta(
        pose_deltas=pose,
        garment_deltas=garment,
        expression_deltas=expression,
        interaction_deltas=interaction,
        visibility_deltas=visibility_deltas,
        predicted_visibility_changes=visibility_deltas,
        newly_revealed_regions=revealed,
        newly_occluded_regions=occluded,
        affected_entities=[person.person_id] if person else [],
        affected_regions=affected_regions,
        semantic_reasons=list(semantic_reasons),
        transition_phase=phase_out,
        region_transition_mode=region_mode,
        state_before={
            "pose_phase": person.pose_state.coarse_pose if person else "unknown",
            "expression_phase": person.expression_state.label if person else "unknown",
        },
        state_after={
            "pose_phase": "transition",
            "garment_phase": str(garment_progression),
            "visibility_phase": "mixed" if any(v != "stable" for v in region_mode.values()) else "stable",
            "expression_phase": str(expression["expression_progression"]),
        },
        transition_diagnostics=transition_diagnostics,
    )
