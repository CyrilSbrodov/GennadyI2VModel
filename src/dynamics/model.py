from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.schema import GraphDelta, RegionRef, SceneGraph, VideoMemory
from planning.transition_engine import PlannedState

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

POSE_KEYS = ["torso_pitch", "head_yaw", "arm_raise", "torso_yaw"]
GARMENT_KEYS = ["attachment_delta", "coverage_delta", "layer_shift"]
VISIBILITY_KEYS = ["face", "torso", "arms", "legs", "garments"]
EXPRESSION_KEYS = ["smile_intensity", "eye_openness", "mouth_open"]
INTERACTION_KEYS = ["support_contact", "hand_contact", "proximity_contact"]
REGION_KEYS = ["reveal_score", "occlude_score", "deform_hint", "stabilize_hint"]
ACTION_VOCAB = [
    "sit_down", "stand_up", "raise_arm", "turn_head", "remove_garment", "open_garment", "smile", "touch", "support", "posture_change", "walk", "lean", "wave", "look_away",
]
PHASES = ["prepare", "transition", "contact_or_reveal", "stabilize"]
FAMILIES = ["pose_transition", "garment_transition", "interaction_transition", "expression_transition"]
DYNAMICS_CHECKPOINT_FORMAT = "torch_factorized_dynamics_v2"
DYNAMICS_CHECKPOINT_VERSION = 2
VISIBILITY_STATES = ["visible", "partially_visible", "hidden", "unknown"]


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
        vec = np.asarray(self.graph_features + self.planner_features + self.action_features + self.memory_features + self.target_features, dtype=np.float32)
        if vec.ndim != 1:
            raise DynamicsModelContractError("feature vector must be 1D")
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


if nn is not None:
    class _TorchFamilyDynamics(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, trunk_dim: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, trunk_dim), nn.Tanh())
            self.family_embed = nn.Embedding(len(FAMILIES), trunk_dim)
            self.heads = nn.ModuleDict(
                {
                    "pose_transition": nn.ModuleDict({"pose": nn.Linear(trunk_dim, len(POSE_KEYS)), "visibility": nn.Linear(trunk_dim, len(VISIBILITY_KEYS)), "region": nn.Linear(trunk_dim, len(REGION_KEYS))}),
                    "garment_transition": nn.ModuleDict({"garment": nn.Linear(trunk_dim, len(GARMENT_KEYS)), "visibility": nn.Linear(trunk_dim, len(VISIBILITY_KEYS)), "region": nn.Linear(trunk_dim, len(REGION_KEYS))}),
                    "interaction_transition": nn.ModuleDict({"interaction": nn.Linear(trunk_dim, len(INTERACTION_KEYS)), "region": nn.Linear(trunk_dim, len(REGION_KEYS))}),
                    "expression_transition": nn.ModuleDict({"expression": nn.Linear(trunk_dim, len(EXPRESSION_KEYS)), "pose": nn.Linear(trunk_dim, len(POSE_KEYS)), "region": nn.Linear(trunk_dim, len(REGION_KEYS))}),
                }
            )
            self.shared_heads = nn.ModuleDict(
                {
                    "pose": nn.Linear(trunk_dim, len(POSE_KEYS)),
                    "garment": nn.Linear(trunk_dim, len(GARMENT_KEYS)),
                    "visibility": nn.Linear(trunk_dim, len(VISIBILITY_KEYS)),
                    "expression": nn.Linear(trunk_dim, len(EXPRESSION_KEYS)),
                    "interaction": nn.Linear(trunk_dim, len(INTERACTION_KEYS)),
                    "region": nn.Linear(trunk_dim, len(REGION_KEYS)),
                }
            )

        def forward(self, x: torch.Tensor, family: str) -> dict[str, torch.Tensor]:
            family_idx = torch.tensor([FAMILIES.index(family if family in FAMILIES else "pose_transition")], device=x.device)
            trunk = self.backbone(x)
            trunk = trunk + self.family_embed(family_idx)
            out: dict[str, torch.Tensor] = {}
            for name, head in self.shared_heads.items():
                logits = head(trunk)
                out[name] = torch.sigmoid(logits) if name in {"visibility", "interaction", "region"} else torch.tanh(logits)
            for name, head in self.heads[family].items():
                logits = head(trunk)
                out[name] = torch.sigmoid(logits) if name in {"visibility", "interaction", "region"} else torch.tanh(logits)
            return out


class DynamicsModel:
    """Torch factorized dynamics core with shared backbone + family-aware expert heads."""

    def __init__(self, input_dim: int = 96, hidden_dim: int = 64, trunk_dim: int = 48, seed: int = 17) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.trunk_dim = trunk_dim
        self.device = "cpu"
        self.using_torch = torch is not None and nn is not None
        if self.using_torch:
            torch.manual_seed(seed)
            self.net = _TorchFamilyDynamics(input_dim=input_dim, hidden_dim=hidden_dim, trunk_dim=trunk_dim)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        else:
            self.net = None
            self.optimizer = None

    def _forward_dict(self, x: np.ndarray, family: str) -> dict[str, np.ndarray]:
        if not self.using_torch:
            raise DynamicsModelError("Torch is required for DynamicsModel")
        xt = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            out = self.net(xt, family)
        return {k: v.detach().cpu().numpy().reshape(-1) for k, v in out.items()}

    def forward(self, inputs: DynamicsInputs, family: str = "pose_transition") -> DynamicsPrediction:
        x = inputs.as_vector()
        if x.shape[0] != self.input_dim:
            raise DynamicsModelContractError(f"expected input_dim={self.input_dim}, got {x.shape[0]}")
        act = self._forward_dict(x, family)
        return DynamicsPrediction(
            pose=act["pose"], garment=act["garment"], visibility=act["visibility"], expression=act["expression"], interaction=act["interaction"], region=act["region"], family=family
        )

    def compute_losses(self, prediction: DynamicsPrediction, targets: DynamicsTargets) -> dict[str, float]:
        losses: dict[str, float] = {}
        family_weights = {
            "pose_transition": {"pose": 2.2, "visibility": 1.2, "region": 1.2},
            "garment_transition": {"garment": 2.2, "visibility": 1.4, "region": 1.4},
            "interaction_transition": {"interaction": 2.2, "region": 1.3},
            "expression_transition": {"expression": 2.2, "pose": 0.8, "region": 1.1},
        }
        active = family_weights.get(targets.family, family_weights["pose_transition"])
        total = 0.0
        for head in ("pose", "garment", "visibility", "expression", "interaction", "region"):
            pred = np.asarray(getattr(prediction, head), dtype=np.float32)
            tgt = np.asarray(getattr(targets, head), dtype=np.float32)
            if pred.shape != tgt.shape:
                raise DynamicsModelContractError(f"shape mismatch for head={head}: {pred.shape} != {tgt.shape}")
            base = float(np.mean((pred - tgt) ** 2))
            losses[f"{head}_loss"] = base * float(active.get(head, 0.35))
            total += losses[f"{head}_loss"]
        losses["total_loss"] = float(total)
        return losses

    def train_step(self, inputs: DynamicsInputs, targets: DynamicsTargets, lr: float = 1e-3) -> dict[str, float]:
        if not self.using_torch:
            raise DynamicsModelError("Torch training is unavailable")
        if self.optimizer is None:
            raise DynamicsModelError("optimizer not initialized")
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        x = torch.tensor(inputs.as_vector(), dtype=torch.float32).reshape(1, -1)
        out = self.net(x, targets.family)
        target_map = {
            "pose": torch.tensor(targets.pose, dtype=torch.float32).reshape(1, -1),
            "garment": torch.tensor(targets.garment, dtype=torch.float32).reshape(1, -1),
            "visibility": torch.tensor(targets.visibility, dtype=torch.float32).reshape(1, -1),
            "expression": torch.tensor(targets.expression, dtype=torch.float32).reshape(1, -1),
            "interaction": torch.tensor(targets.interaction, dtype=torch.float32).reshape(1, -1),
            "region": torch.tensor(targets.region, dtype=torch.float32).reshape(1, -1),
        }
        family_weights = {
            "pose_transition": {"pose": 2.2, "visibility": 1.2, "region": 1.2},
            "garment_transition": {"garment": 2.2, "visibility": 1.4, "region": 1.4},
            "interaction_transition": {"interaction": 2.2, "region": 1.3},
            "expression_transition": {"expression": 2.2, "pose": 0.8, "region": 1.1},
        }
        active = family_weights.get(targets.family, family_weights["pose_transition"])
        total = torch.tensor(0.0)
        scalar_losses: dict[str, float] = {}
        for head in ("pose", "garment", "visibility", "expression", "interaction", "region"):
            l = torch.mean((out[head] - target_map[head]) ** 2) * float(active.get(head, 0.35))
            scalar_losses[f"{head}_loss"] = float(l.detach().cpu().item())
            total = total + l
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        scalar_losses["total_loss"] = float(total.detach().cpu().item())
        return scalar_losses

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self.using_torch:
            raise DynamicsModelError("Torch serialization unavailable")
        torch.save({"version": DYNAMICS_CHECKPOINT_VERSION, "format": DYNAMICS_CHECKPOINT_FORMAT, "input_dim": self.input_dim, "hidden_dim": self.hidden_dim, "trunk_dim": self.trunk_dim, "state_dict": self.net.state_dict(), "families": list(FAMILIES)}, path)

    @classmethod
    def load(cls, path: str) -> "DynamicsModel":
        model = cls()
        if not Path(path).exists():
            return model
        if not model.using_torch:
            raise DynamicsModelError("Torch unavailable, cannot load weights")
        payload = torch.load(path, map_location="cpu")
        valid, reason = validate_checkpoint_payload(payload)
        if not valid:
            raise DynamicsModelError(f"invalid_checkpoint:{reason}")
        model = cls(input_dim=int(payload.get("input_dim", 96)), hidden_dim=int(payload.get("hidden_dim", 64)), trunk_dim=int(payload.get("trunk_dim", 48)))
        model.net.load_state_dict(payload["state_dict"], strict=True)
        return model


def validate_checkpoint_payload(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "checkpoint_payload_not_dict"
    if payload.get("format") != DYNAMICS_CHECKPOINT_FORMAT:
        return False, "checkpoint_format_mismatch"
    if int(payload.get("version", -1)) != DYNAMICS_CHECKPOINT_VERSION:
        return False, "checkpoint_version_mismatch"
    for key in ("input_dim", "hidden_dim", "trunk_dim", "state_dict", "families"):
        if key not in payload:
            return False, f"missing_key:{key}"
    if not isinstance(payload.get("state_dict"), dict) or not payload.get("state_dict"):
        return False, "state_dict_missing_or_empty"
    fam = payload.get("families")
    if not isinstance(fam, list) or any(str(x) not in FAMILIES for x in fam):
        return False, "family_config_invalid"
    return True, "ok"


def dynamics_inputs_from_tensor_batch(batch: DynamicsTensorBatch) -> DynamicsInputs:
    vec = np.asarray(batch.features, dtype=np.float32).reshape(-1)
    if vec.size < 96:
        vec = np.concatenate([vec, np.zeros((96-vec.size,), dtype=np.float32)])
    vec = vec[:96]
    return DynamicsInputs(graph_features=vec[:27].tolist(), planner_features=vec[27:35].tolist(), action_features=vec[35:51].tolist(), memory_features=vec[51:59].tolist(), target_features=vec[59:96].tolist())


def _phase_onehot(phase: str) -> list[float]:
    p = phase if phase in PHASES else "transition"
    return [1.0 if p == v else 0.0 for v in PHASES]


def _visibility_hist(values: list[str]) -> list[float]:
    if not values:
        return [0.0] * len(VISIBILITY_STATES)
    hist = [float(sum(1 for v in values if v == state)) / float(len(values)) for state in VISIBILITY_STATES]
    return hist


def featurize_runtime(scene_graph: SceneGraph, target_state: PlannedState, planner_context: dict[str, float] | None, memory: VideoMemory | None) -> DynamicsInputs:
    context = planner_context or {}
    person = scene_graph.persons[0] if scene_graph.persons else None
    labels = {x for x in target_state.labels if isinstance(x, str)}
    action_features = [1.0 if token in labels else 0.0 for token in ACTION_VOCAB]
    intensity = 0.5
    for item in target_state.labels:
        if isinstance(item, str) and item.startswith("intensity="):
            try:
                intensity = float(item.split("=", 1)[1])
            except ValueError:
                intensity = 0.5
    action_features.extend([max(0.0, min(1.0, intensity)), float(len(labels)) / 8.0])

    if person:
        pose_angles = person.pose_state.angles
        graph_features = [person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h, float(len(scene_graph.persons)) / 4.0, float(len(scene_graph.objects)) / 8.0, float(len(scene_graph.relations)) / 16.0, float(len(person.body_parts)) / 16.0, float(len(person.garments)) / 8.0, float(person.expression_state.smile_intensity), float(person.expression_state.eye_openness), float(person.orientation.yaw) / 90.0, float(person.orientation.pitch) / 90.0, float(person.orientation.roll) / 90.0, float(pose_angles.get("torso_pitch", 0.0)) / 45.0, float(pose_angles.get("head_yaw", 0.0)) / 45.0]
        body_vis = _visibility_hist([bp.visibility for bp in person.body_parts])
        garment_vis = _visibility_hist([g.visibility for g in person.garments])
        garment_state_counts = [float(sum(1 for g in person.garments if g.garment_state == "worn")) / max(1, len(person.garments)), float(sum(1 for g in person.garments if g.garment_state in {"opening", "partially_detached"})) / max(1, len(person.garments)), float(sum(1 for g in person.garments if g.garment_state == "removed")) / max(1, len(person.garments))]
        graph_features.extend(body_vis + garment_vis + garment_state_counts)
    else:
        graph_features = [0.0] * 27

    step_idx = float(context.get("step_index", target_state.step_index))
    total_steps = max(1.0, float(context.get("total_steps", context.get("plan_length", 4.0))))
    progress = step_idx / total_steps
    duration = float(context.get("target_duration", context.get("duration", 1.0)))
    phase = str(context.get("phase", context.get("sequencing_stage", "transition")))
    planner_features = [step_idx / 16.0, total_steps / 16.0, progress, duration / 8.0] + _phase_onehot(phase)

    family = str(context.get("transition_family", ""))
    family_onehot = [1.0 if family == f else 0.0 for f in FAMILIES]

    if memory:
        hidden_recent = float(sum(1 for s in memory.hidden_region_slots.values() if hasattr(s, "stale_frames") and float(getattr(s, "stale_frames", 99.0)) < 4.0))
        memory_features = [float(len(memory.identity_memory)) / 8.0, float(len(memory.garment_memory)) / 8.0, float(len(memory.region_descriptors)) / 16.0, float(len(memory.hidden_region_slots)) / 16.0, hidden_recent / 16.0, float(len(memory.temporal_history)) / 16.0, 1.0 if memory.last_transition_context else 0.0, 1.0 if memory.last_transition_context.get("visibility_phase") == "mixed" else 0.0]
    else:
        memory_features = [0.0] * 8

    semantic = target_state.semantic_transition
    target_features = [
        1.0 if any("target_pose=" in str(v) for v in target_state.labels) else 0.0,
        1.0 if any("target_visibility=" in str(v) for v in target_state.labels) else 0.0,
        1.0 if any("target_expression=" in str(v) for v in target_state.labels) else 0.0,
        float(sum(1 for x in labels if x in {"sit_down", "stand_up", "posture_change"})) / 3.0,
        float(sum(1 for x in labels if x in {"remove_garment", "open_garment"})) / 2.0,
        float(sum(1 for x in labels if x in {"smile", "look_away"})) / 2.0,
        1.0 if "chair" in labels or (semantic and semantic.target_profile.support_target == "chair") else 0.0,
        1.0 if "support" in labels or (semantic and "support_zone" in semantic.target_profile.context_regions) else 0.0,
    ] + family_onehot

    merged = graph_features + planner_features + action_features + memory_features + target_features
    if len(merged) > 96:
        merged = merged[:96]
    elif len(merged) < 96:
        merged = merged + [0.0] * (96 - len(merged))
    return DynamicsInputs(graph_features=merged[:27], planner_features=merged[27:35], action_features=merged[35:51], memory_features=merged[51:59], target_features=merged[59:96])


def tensorize_dynamics_inputs(inputs: DynamicsInputs, family: str, phase: str) -> DynamicsTensorBatch:
    vec = inputs.as_vector().astype(np.float32)
    return DynamicsTensorBatch(features=vec, family=family if family in FAMILIES else "pose_transition", phase=phase if phase in PHASES else "transition", conditioning_summary={"family": family, "phase": phase, "input_norm": float(np.linalg.norm(vec)), "non_zero": int(np.count_nonzero(vec))})


def targets_from_delta(delta: GraphDelta, family: str = "pose_transition") -> DynamicsTargets:
    pose = [float(delta.pose_deltas.get(k, 0.0)) / 45.0 for k in POSE_KEYS]
    garment = [float(delta.garment_deltas.get("attachment_delta", 0.0)), float(delta.garment_deltas.get("coverage_delta", 0.0)), float(delta.garment_deltas.get("layer_shift", 0.0))]
    vis_map = delta.visibility_deltas or delta.predicted_visibility_changes
    visibility = [1.0 if str(vis_map.get(key, "visible")) == "visible" else (0.5 if str(vis_map.get(key, "visible")) == "partially_visible" else 0.0) for key in VISIBILITY_KEYS]
    expression = [float(delta.expression_deltas.get("smile_intensity", 0.0)), float(delta.expression_deltas.get("eye_openness", 0.0)), 1.0 if str(delta.expression_deltas.get("mouth_state", "neutral")) in {"open", "smile"} else 0.0]
    interaction = [max(0.0, min(1.0, float(delta.interaction_deltas.get("support_contact", 0.0)))), max(0.0, min(1.0, float(delta.interaction_deltas.get("hand_contact", 0.0)))), max(0.0, min(1.0, float(delta.interaction_deltas.get("proximity_contact", 0.0))))]
    region = [1.0 if delta.newly_revealed_regions else 0.0, 1.0 if delta.newly_occluded_regions else 0.0, 1.0 if any(v in {"deform", "deform_relation_aware"} for v in delta.region_transition_mode.values()) else 0.0, 1.0 if delta.transition_phase in {"contact_or_reveal", "stabilize"} else 0.0]
    return DynamicsTargets(pose=pose, garment=garment, visibility=visibility, expression=expression, interaction=interaction, region=region, family=family)


def decode_prediction(prediction: DynamicsPrediction, scene_graph: SceneGraph, phase: str, semantic_reasons: list[str], planner_context: dict[str, float] | None = None) -> GraphDelta:
    context = planner_context or {}
    person = scene_graph.persons[0] if scene_graph.persons else None
    pose = {k: float(prediction.pose[i] * 45.0) for i, k in enumerate(POSE_KEYS)}
    garment = {"attachment_delta": float(prediction.garment[0]), "coverage_delta": float(prediction.garment[1]), "layer_shift": float(prediction.garment[2]), "garment_progression": "opening" if prediction.garment[0] < -0.1 else ("removed" if prediction.garment[1] < -0.25 else "worn")}
    vis_states = ["hidden", "partially_visible", "visible"]
    visibility_deltas = {key: vis_states[2] if val > 0.7 else (vis_states[1] if val > 0.35 else vis_states[0]) for key, val in zip(VISIBILITY_KEYS, prediction.visibility)}
    expression = {"smile_intensity": float(prediction.expression[0]), "eye_openness": float(prediction.expression[1]), "mouth_state": "open" if prediction.expression[2] > 0.45 else "neutral", "expression_progression": "forming_expression" if prediction.expression[0] > 0.05 else "neutral"}
    interaction = {"support_contact": float(np.clip(prediction.interaction[0], 0.0, 1.0)), "hand_contact": float(np.clip(prediction.interaction[1], 0.0, 1.0)), "proximity_contact": float(np.clip(prediction.interaction[2], 0.0, 1.0))}

    target_region_candidates: list[str] = []
    for token in semantic_reasons:
        if token in {"raise_arm", "wave"}:
            target_region_candidates.extend(["arms", "torso"])
        elif token in {"sit_down", "stand_up", "walk"}:
            target_region_candidates.extend(["legs", "pelvis", "torso"])
        elif token in {"smile", "look_away"}:
            target_region_candidates.extend(["face"])
        elif token in {"remove_garment", "open_garment"}:
            target_region_candidates.extend(["garments", "torso"])
    if not target_region_candidates and person:
        target_region_candidates = [bp.part_type for bp in person.body_parts if bp.visibility != "hidden"][:4]
    affected_regions = sorted(set(target_region_candidates or ["torso"]))

    region_mode: dict[str, str] = {}
    for region in affected_regions:
        if prediction.region[0] > 0.65:
            region_mode[region] = "reveal"
        elif prediction.region[1] > 0.65:
            region_mode[region] = "occlude"
        elif prediction.region[2] > 0.55:
            region_mode[region] = "deform"
        elif prediction.region[3] > 0.7:
            region_mode[region] = "stabilize"
        else:
            region_mode[region] = "stable"

    revealed: list[RegionRef] = []
    occluded: list[RegionRef] = []
    if person:
        for region in affected_regions:
            if region_mode.get(region) == "reveal":
                revealed.append(RegionRef(region_id=f"{person.person_id}:{region}:reveal", bbox=person.bbox, reason="learned_reveal"))
            if region_mode.get(region) == "occlude":
                occluded.append(RegionRef(region_id=f"{person.person_id}:{region}:occlude", bbox=person.bbox, reason="learned_occlude"))

    phase_out = str(context.get("phase", phase))
    if prediction.region[3] > 0.75 and phase_out not in {"contact_or_reveal", "stabilize"}:
        phase_out = "stabilize"
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
        semantic_reasons=semantic_reasons,
        transition_phase=phase_out,
        region_transition_mode=region_mode,
        state_before={"pose_phase": person.pose_state.coarse_pose if person else "unknown", "expression_phase": person.expression_state.label if person else "unknown"},
        state_after={"pose_phase": "transition", "garment_phase": str(garment["garment_progression"]), "visibility_phase": "mixed" if any(v != "stable" for v in region_mode.values()) else "stable", "expression_phase": str(expression["expression_progression"])},
    )
