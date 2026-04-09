from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.schema import GraphDelta, RegionRef, SceneGraph, VideoMemory
from planning.transition_engine import PlannedState

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
        vec = np.asarray(
            self.graph_features + self.planner_features + self.action_features + self.memory_features + self.target_features,
            dtype=np.float64,
        )
        if vec.ndim != 1:
            raise DynamicsModelContractError("feature vector must be 1D")
        return vec


@dataclass(slots=True)
class DynamicsTargets:
    pose: list[float]
    garment: list[float]
    visibility: list[float]
    expression: list[float]
    interaction: list[float]
    region: list[float]


@dataclass(slots=True)
class DynamicsPrediction:
    pose: np.ndarray
    garment: np.ndarray
    visibility: np.ndarray
    expression: np.ndarray
    interaction: np.ndarray
    region: np.ndarray


class DynamicsModel:
    """Compact trainable structured dynamics predictor with shared trunk + per-head decoders."""

    def __init__(self, input_dim: int = 96, hidden_dim: int = 64, trunk_dim: int = 48, seed: int = 17) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.trunk_dim = trunk_dim
        rng = np.random.default_rng(seed)
        self.w_in = rng.normal(0.0, 0.08, size=(input_dim, hidden_dim))
        self.b_in = np.zeros(hidden_dim, dtype=np.float64)
        self.w_trunk = rng.normal(0.0, 0.08, size=(hidden_dim, trunk_dim))
        self.b_trunk = np.zeros(trunk_dim, dtype=np.float64)
        self.heads = {
            "pose": (rng.normal(0.0, 0.07, size=(trunk_dim, len(POSE_KEYS))), np.zeros(len(POSE_KEYS), dtype=np.float64), "tanh"),
            "garment": (rng.normal(0.0, 0.07, size=(trunk_dim, len(GARMENT_KEYS))), np.zeros(len(GARMENT_KEYS), dtype=np.float64), "tanh"),
            "visibility": (rng.normal(0.0, 0.07, size=(trunk_dim, len(VISIBILITY_KEYS))), np.zeros(len(VISIBILITY_KEYS), dtype=np.float64), "sigmoid"),
            "expression": (rng.normal(0.0, 0.07, size=(trunk_dim, len(EXPRESSION_KEYS))), np.zeros(len(EXPRESSION_KEYS), dtype=np.float64), "tanh"),
            "interaction": (rng.normal(0.0, 0.07, size=(trunk_dim, len(INTERACTION_KEYS))), np.zeros(len(INTERACTION_KEYS), dtype=np.float64), "sigmoid"),
            "region": (rng.normal(0.0, 0.07, size=(trunk_dim, len(REGION_KEYS))), np.zeros(len(REGION_KEYS), dtype=np.float64), "sigmoid"),
        }

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _forward_internal(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        if x.shape[0] != self.input_dim:
            raise DynamicsModelContractError(f"expected input_dim={self.input_dim}, got {x.shape[0]}")
        h1_pre = x @ self.w_in + self.b_in
        h1 = np.tanh(h1_pre)
        trunk_pre = h1 @ self.w_trunk + self.b_trunk
        trunk = np.tanh(trunk_pre)
        pre: dict[str, np.ndarray] = {}
        act: dict[str, np.ndarray] = {}
        for name, (w, b, kind) in self.heads.items():
            z = trunk @ w + b
            pre[name] = z
            act[name] = self._sigmoid(z) if kind == "sigmoid" else np.tanh(z)
        return h1, trunk, pre, act

    def forward(self, inputs: DynamicsInputs) -> DynamicsPrediction:
        x = inputs.as_vector()
        _, _, _, act = self._forward_internal(x)
        return DynamicsPrediction(
            pose=act["pose"],
            garment=act["garment"],
            visibility=act["visibility"],
            expression=act["expression"],
            interaction=act["interaction"],
            region=act["region"],
        )

    def compute_losses(self, prediction: DynamicsPrediction, targets: DynamicsTargets) -> dict[str, float]:
        losses: dict[str, float] = {}
        for head in ("pose", "garment", "visibility", "expression", "interaction", "region"):
            pred = np.asarray(getattr(prediction, head), dtype=np.float64)
            tgt = np.asarray(getattr(targets, head), dtype=np.float64)
            if pred.shape != tgt.shape:
                raise DynamicsModelContractError(f"shape mismatch for head={head}: {pred.shape} != {tgt.shape}")
            losses[f"{head}_loss"] = float(np.mean((pred - tgt) ** 2))
        losses["total_loss"] = float(sum(losses.values()))
        return losses

    def train_step(self, inputs: DynamicsInputs, targets: DynamicsTargets, lr: float = 1e-3) -> dict[str, float]:
        x = inputs.as_vector()
        h1, trunk, pre, act = self._forward_internal(x)
        prediction = DynamicsPrediction(
            pose=act["pose"], garment=act["garment"], visibility=act["visibility"], expression=act["expression"], interaction=act["interaction"], region=act["region"]
        )
        losses = self.compute_losses(prediction, targets)

        grad_trunk = np.zeros_like(trunk)
        for name, (w, b, kind) in self.heads.items():
            target = np.asarray(getattr(targets, name), dtype=np.float64)
            pred = act[name]
            grad = (2.0 / max(1, pred.size)) * (pred - target)
            if kind == "sigmoid":
                grad = grad * pred * (1.0 - pred)
            else:
                grad = grad * (1.0 - pred * pred)
            grad_trunk += grad @ w.T
            self.heads[name] = (w - lr * np.outer(trunk, grad), b - lr * grad, kind)

        grad_trunk = grad_trunk * (1.0 - trunk * trunk)
        grad_h1 = grad_trunk @ self.w_trunk.T
        grad_h1 = grad_h1 * (1.0 - h1 * h1)
        self.w_trunk = self.w_trunk - lr * np.outer(h1, grad_trunk)
        self.b_trunk = self.b_trunk - lr * grad_trunk
        self.w_in = self.w_in - lr * np.outer(x, grad_h1)
        self.b_in = self.b_in - lr * grad_h1
        return losses

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "trunk_dim": self.trunk_dim,
            "w_in": self.w_in.tolist(),
            "b_in": self.b_in.tolist(),
            "w_trunk": self.w_trunk.tolist(),
            "b_trunk": self.b_trunk.tolist(),
            "heads": {
                name: {"w": w.tolist(), "b": b.tolist(), "kind": kind}
                for name, (w, b, kind) in self.heads.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "DynamicsModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(input_dim=int(payload["input_dim"]), hidden_dim=int(payload["hidden_dim"]), trunk_dim=int(payload["trunk_dim"]))
        model.w_in = np.asarray(payload["w_in"], dtype=np.float64)
        model.b_in = np.asarray(payload["b_in"], dtype=np.float64)
        model.w_trunk = np.asarray(payload["w_trunk"], dtype=np.float64)
        model.b_trunk = np.asarray(payload["b_trunk"], dtype=np.float64)
        heads = payload.get("heads", {})
        model.heads = {
            name: (
                np.asarray(heads[name]["w"], dtype=np.float64),
                np.asarray(heads[name]["b"], dtype=np.float64),
                str(heads[name]["kind"]),
            )
            for name in heads
        }
        return model


def _phase_onehot(phase: str) -> list[float]:
    p = phase if phase in PHASES else "transition"
    return [1.0 if p == v else 0.0 for v in PHASES]


def _visibility_hist(values: list[str]) -> list[float]:
    if not values:
        return [0.0] * len(VISIBILITY_STATES)
    hist = [float(sum(1 for v in values if v == state)) / float(len(values)) for state in VISIBILITY_STATES]
    return hist


def featurize_runtime(
    scene_graph: SceneGraph,
    target_state: PlannedState,
    planner_context: dict[str, float] | None,
    memory: VideoMemory | None,
) -> DynamicsInputs:
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
        graph_features = [
            person.bbox.x,
            person.bbox.y,
            person.bbox.w,
            person.bbox.h,
            float(len(scene_graph.persons)) / 4.0,
            float(len(scene_graph.objects)) / 8.0,
            float(len(scene_graph.relations)) / 16.0,
            float(len(person.body_parts)) / 16.0,
            float(len(person.garments)) / 8.0,
            float(person.expression_state.smile_intensity),
            float(person.expression_state.eye_openness),
            float(person.orientation.yaw) / 90.0,
            float(person.orientation.pitch) / 90.0,
            float(person.orientation.roll) / 90.0,
            float(pose_angles.get("torso_pitch", 0.0)) / 45.0,
            float(pose_angles.get("head_yaw", 0.0)) / 45.0,
        ]
        body_vis = _visibility_hist([bp.visibility for bp in person.body_parts])
        garment_vis = _visibility_hist([g.visibility for g in person.garments])
        garment_state_counts = [
            float(sum(1 for g in person.garments if g.garment_state == "worn")) / max(1, len(person.garments)),
            float(sum(1 for g in person.garments if g.garment_state in {"opening", "partially_detached"})) / max(1, len(person.garments)),
            float(sum(1 for g in person.garments if g.garment_state == "removed")) / max(1, len(person.garments)),
        ]
        graph_features.extend(body_vis + garment_vis + garment_state_counts)
    else:
        graph_features = [0.0] * 27

    step_idx = float(context.get("step_index", target_state.step_index))
    total_steps = max(1.0, float(context.get("total_steps", context.get("plan_length", 4.0))))
    progress = step_idx / total_steps
    duration = float(context.get("target_duration", context.get("duration", 1.0)))
    phase = str(context.get("phase", context.get("sequencing_stage", "transition")))
    planner_features = [step_idx / 16.0, total_steps / 16.0, progress, duration / 8.0] + _phase_onehot(phase)

    if memory:
        hidden_recent = float(sum(1 for s in memory.hidden_region_slots.values() if hasattr(s, "stale_frames") and float(getattr(s, "stale_frames", 99.0)) < 4.0))
        memory_features = [
            float(len(memory.identity_memory)) / 8.0,
            float(len(memory.garment_memory)) / 8.0,
            float(len(memory.region_descriptors)) / 16.0,
            float(len(memory.hidden_region_slots)) / 16.0,
            hidden_recent / 16.0,
            float(len(memory.temporal_history)) / 16.0,
            1.0 if memory.last_transition_context else 0.0,
            1.0 if memory.last_transition_context.get("visibility_phase") == "mixed" else 0.0,
        ]
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
    ]

    # 27 + 8 + 16 + 8 + 8 = 67 -> pad to model input dim for forward compatibility.
    merged = graph_features + planner_features + action_features + memory_features + target_features
    if len(merged) > 96:
        merged = merged[:96]
    elif len(merged) < 96:
        merged = merged + [0.0] * (96 - len(merged))

    return DynamicsInputs(
        graph_features=merged[:27],
        planner_features=merged[27:35],
        action_features=merged[35:51],
        memory_features=merged[51:59],
        target_features=merged[59:96],
    )


def targets_from_delta(delta: GraphDelta) -> DynamicsTargets:
    pose = [float(delta.pose_deltas.get(k, 0.0)) / 45.0 for k in POSE_KEYS]
    garment = [
        float(delta.garment_deltas.get("attachment_delta", 0.0)),
        float(delta.garment_deltas.get("coverage_delta", 0.0)),
        float(delta.garment_deltas.get("layer_shift", 0.0)),
    ]
    vis_map = delta.visibility_deltas or delta.predicted_visibility_changes
    visibility = [
        1.0 if str(vis_map.get(key, "visible")) == "visible" else (0.5 if str(vis_map.get(key, "visible")) == "partially_visible" else 0.0)
        for key in VISIBILITY_KEYS
    ]
    expression = [
        float(delta.expression_deltas.get("smile_intensity", 0.0)),
        float(delta.expression_deltas.get("eye_openness", 0.0)),
        1.0 if str(delta.expression_deltas.get("mouth_state", "neutral")) in {"open", "smile"} else 0.0,
    ]
    interaction = [
        max(0.0, min(1.0, float(delta.interaction_deltas.get("support_contact", 0.0)))),
        max(0.0, min(1.0, float(delta.interaction_deltas.get("hand_contact", 0.0)))),
        max(0.0, min(1.0, float(delta.interaction_deltas.get("proximity_contact", 0.0)))),
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
    )


def decode_prediction(
    prediction: DynamicsPrediction,
    scene_graph: SceneGraph,
    phase: str,
    semantic_reasons: list[str],
    planner_context: dict[str, float] | None = None,
) -> GraphDelta:
    context = planner_context or {}
    person = scene_graph.persons[0] if scene_graph.persons else None

    pose = {k: float(prediction.pose[i] * 45.0) for i, k in enumerate(POSE_KEYS)}
    garment = {
        "attachment_delta": float(prediction.garment[0]),
        "coverage_delta": float(prediction.garment[1]),
        "layer_shift": float(prediction.garment[2]),
        "garment_progression": "opening" if prediction.garment[0] < -0.1 else ("removed" if prediction.garment[1] < -0.25 else "worn"),
    }

    vis_states = ["hidden", "partially_visible", "visible"]
    visibility_deltas = {
        key: vis_states[2] if val > 0.7 else (vis_states[1] if val > 0.35 else vis_states[0])
        for key, val in zip(VISIBILITY_KEYS, prediction.visibility)
    }

    expression = {
        "smile_intensity": float(prediction.expression[0]),
        "eye_openness": float(prediction.expression[1]),
        "mouth_state": "open" if prediction.expression[2] > 0.45 else "neutral",
        "expression_progression": "forming_expression" if prediction.expression[0] > 0.05 else "neutral",
    }

    interaction = {
        "support_contact": float(np.clip(prediction.interaction[0], 0.0, 1.0)),
        "hand_contact": float(np.clip(prediction.interaction[1], 0.0, 1.0)),
        "proximity_contact": float(np.clip(prediction.interaction[2], 0.0, 1.0)),
    }

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
        state_before={
            "pose_phase": person.pose_state.coarse_pose if person else "unknown",
            "expression_phase": person.expression_state.label if person else "unknown",
        },
        state_after={
            "pose_phase": "transition",
            "garment_phase": str(garment["garment_progression"]),
            "visibility_phase": "mixed" if any(v != "stable" for v in region_mode.values()) else "stable",
            "expression_phase": str(expression["expression_progression"]),
        },
    )
