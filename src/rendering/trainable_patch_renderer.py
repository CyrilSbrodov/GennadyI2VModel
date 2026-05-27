from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.region_ids import parse_region_id
from core.reference_families import (
    ACCESSORY_REFERENCE_REGIONS,
    BODY_SHAPE_REFERENCE_REGIONS,
    CORE_IDENTITY_REGIONS,
    GARMENT_REFERENCE_REGIONS,
    SKIN_REFERENCE_REGIONS,
    reference_kind_for_region,
)
from rendering.patch_conditioning_contract import (
    APPEARANCE_DIM,
    BBOX_DIM,
    DELTA_DIM,
    GLOBAL_COND_DIM,
    GRAPH_DIM,
    MEMORY_DIM,
    MODE_DIM,
    PLANNER_DIM,
    ROLE_DIM,
    SEMANTIC_DIM
)
from core.schema import RegionMemoryBundle, RegionRef
from rendering.patch_tensor_utils import map_to_shape
from rendering.target_provenance_policy import target_supervision_weight
from dynamics.transition_contracts import LearnedHumanStateContract, LearnedTemporalTransitionContract
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from utils_tensor import shape


class RendererInputError(ValueError):
    pass


class RendererInferenceError(RuntimeError):
    pass


IDENTITY_SENSITIVE_REGIONS = {"face", "head", "hair", "mouth", "eyes", "cheek", "neck"}
IDENTITY_BLOCKED_REASONS = (
    "identity_reference_blocked_generated",
    "identity_reference_blocked_inferred",
    "identity_reference_blocked_low_evidence",
)
REFERENCE_FAMILIES = ("skin", "body_shape", "garment", "accessory")
REFERENCE_BLOCKED_REASONS = {
    family: (
        f"{family}_reference_blocked_generated",
        f"{family}_reference_blocked_inferred",
        f"{family}_reference_blocked_low_evidence",
    )
    for family in REFERENCE_FAMILIES
}

ACTION_PHASE_TARGETS: dict[str, set[str]] = {
    "arm_raise": {"left_arm", "right_arm"},
    "expression_smile": {"face", "mouth", "cheek"},
    "head_turn": {"face", "head", "hair"},
    "torso_shift": {"torso", "upper_clothes", "inner_garment"},
    "garment_reveal_or_adjust": {"torso", "upper_clothes", "inner_garment", "outer_garment"},
}


def _i2v_action_targets_region(action_phase: str, region_type: str, region_action_mode: str) -> bool:
    phase = str(action_phase or "").strip().lower()
    mode = str(region_action_mode or "").strip().lower()
    if phase in {"", "stable", "stable_idle", "idle"}:
        return False
    if mode in {"", "stable", "stable_idle", "idle", "unknown", "none"}:
        return False
    if mode != phase:
        return False
    return region_type in ACTION_PHASE_TARGETS.get(phase, set())


def _warp_roi_nearest(roi: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray, amount: float) -> np.ndarray:
    h, w, c = roi.shape
    if h <= 0 or w <= 0 or c != 3 or amount <= 1e-6:
        return np.clip(roi, 0.0, 1.0).astype(np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    sample_x = np.clip(np.rint(xs - flow_x.astype(np.float32) * float(amount) * max(w - 1, 1)), 0, max(w - 1, 0)).astype(np.int32)
    sample_y = np.clip(np.rint(ys - flow_y.astype(np.float32) * float(amount) * max(h - 1, 1)), 0, max(h - 1, 0)).astype(np.int32)
    warped = roi[sample_y, sample_x]
    return np.clip(warped, 0.0, 1.0).astype(np.float32)


def _i2v_action_motion_field(
    h: int,
    w: int,
    region_type: str,
    geometry: dict[str, np.ndarray],
    i2v_action_cond: dict[str, object] | None,
) -> dict[str, np.ndarray | float | bool | str]:
    zeros = np.zeros((h, w), dtype=np.float32)
    cond = i2v_action_cond if isinstance(i2v_action_cond, dict) else {}
    phase = str(cond.get("i2v_action_phase", "stable_idle")).strip().lower()
    mode = str(cond.get("i2v_region_action_mode", "")).strip().lower()
    targeted = _i2v_action_targets_region(phase, region_type, mode)
    intensity = float(np.clip(_safe_float(cond.get("i2v_action_strength"), 0.0), 0.0, 1.0))
    if not targeted:
        return {"active": False, "action_phase": phase, "targeted": False, "flow_x": zeros, "flow_y": zeros, "deformation_mask": zeros, "intensity": intensity}

    lateral = np.clip(geometry.get("lateral", zeros), 0.0, 1.0)
    center = np.clip(geometry.get("center", zeros), 0.0, 1.0)
    upper_band = np.clip(geometry.get("upper_band", zeros), 0.0, 1.0)
    lower_band = np.clip(geometry.get("lower_band", zeros), 0.0, 1.0)
    expr = np.clip(geometry.get("expression_hotspot", zeros), 0.0, 1.0)
    surface = np.clip(geometry.get("surface_band", zeros), 0.0, 1.0)
    split = np.clip(geometry.get("vertical_split", zeros), 0.0, 1.0)
    flow_x = zeros.copy()
    flow_y = zeros.copy()
    deform = zeros.copy()

    if phase == "head_turn":
        flow_x = (0.20 * lateral + 0.10 * center).astype(np.float32)
        deform = (0.20 * lateral + 0.08 * center).astype(np.float32)
    elif phase == "expression_smile":
        smile_hot = np.clip(0.75 * expr + 0.25 * lower_band, 0.0, 1.0)
        flow_y = (-0.22 * smile_hot).astype(np.float32)
        flow_x = (0.08 * lateral * smile_hot).astype(np.float32)
        deform = (0.42 * smile_hot).astype(np.float32)
    elif phase == "arm_raise":
        arm_band = np.clip(0.62 * upper_band + 0.38 * lateral, 0.0, 1.0)
        flow_y = (-0.55 * arm_band).astype(np.float32)
        deform = (0.58 * arm_band).astype(np.float32)
    elif phase == "torso_shift":
        torso_band = np.clip(0.70 * surface + 0.30 * lateral, 0.0, 1.0)
        flow_x = (0.24 * torso_band).astype(np.float32)
        deform = (0.34 * torso_band).astype(np.float32)
    elif phase == "garment_reveal_or_adjust":
        g = np.clip(0.58 * split + 0.42 * surface, 0.0, 1.0)
        flow_y = (-0.30 * g).astype(np.float32)
        flow_x = (0.12 * lateral * g).astype(np.float32)
        deform = (0.52 * g).astype(np.float32)

    return {
        "active": True,
        "action_phase": phase,
        "targeted": True,
        "flow_x": np.clip(flow_x, -1.0, 1.0).astype(np.float32),
        "flow_y": np.clip(flow_y, -1.0, 1.0).astype(np.float32),
        "deformation_mask": np.clip(deform, 0.0, 1.0).astype(np.float32),
        "intensity": intensity,
    }

ROI_FAMILIES = {
    "face_expression": {"face", "head", "mouth", "eyes", "cheek", "neck", "hair"},
    "torso_reveal": {"torso", "inner_garment", "innerwear", "chest", "pelvis"},
    "sleeve_arm_transition": {"left_arm", "right_arm", "arm", "sleeves", "outer_garment", "garments", "legs"},
}

TRANSITION_MODE_ORDER = (
    "garment_surface",
    "garment_reveal",
    "pose_exposure",
    "expression_refine",
    "visibility_occlusion",
    "stable",
)
PROFILE_ROLE_ORDER = ("primary", "secondary", "context")

PIXEL_FEATURE_DIM = 15
CONDITION_FEATURE_DIM = GLOBAL_COND_DIM


@dataclass(slots=True)
class RenderConditioningProfile:
    transition_mode: str
    profile_role: str
    synthesis_family: str
    reveal_like: bool
    edit_strength: float
    preservation_strength: float
    seam_strictness: float
    memory_dependency: float
    uncertainty_bias: float
    alpha_floor: float
    alpha_peak: float
    local_radius_x: float
    local_radius_y: float
    context_support: float




@dataclass(slots=True)
class PatchBatch:
    roi_before: np.ndarray
    roi_after: np.ndarray
    changed_mask: np.ndarray
    alpha_target: np.ndarray
    blend_hint: np.ndarray
    semantic_embed: np.ndarray
    delta_cond: np.ndarray
    planner_cond: np.ndarray
    graph_cond: np.ndarray
    memory_cond: np.ndarray
    appearance_cond: np.ndarray
    bbox_cond: np.ndarray
    reference_rgb: np.ndarray | None = None
    reference_mask: np.ndarray | None = None
    reference_validity: np.ndarray | None = None
    mode_cond: np.ndarray | None = None
    role_cond: np.ndarray | None = None
    preservation_mask: np.ndarray | None = None
    uncertainty_target: np.ndarray | None = None
    seam_prior: np.ndarray | None = None
    i2v_flow_x: np.ndarray | None = None
    i2v_flow_y: np.ndarray | None = None
    i2v_deformation_mask: np.ndarray | None = None
    transition_mode: str = "stable"
    profile_role: str = "primary"
    conditioning_summary: dict[str, object] = field(default_factory=dict)
    previous_roi: np.ndarray | None = None
    predicted_family: str = ""
    predicted_phase: str = ""
    target_profile: dict[str, list[str]] = field(default_factory=dict)
    reveal_score: float = 0.0
    occlusion_score: float = 0.0
    support_contact_score: float = 0.0
    temporal_contract_target: dict[str, object] = field(default_factory=dict)
    graph_delta_target: dict[str, object] = field(default_factory=dict)
    rollout_weight: float = 1.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _clip01(value: np.ndarray | float) -> np.ndarray | float:
    return np.clip(value, 0.0, 1.0)


def _vector_to_size(value: np.ndarray | list[float] | tuple[float, ...] | None, size: int) -> np.ndarray:
    if value is None:
        return np.zeros((size,), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == size:
        return arr.astype(np.float32)
    out = np.zeros((size,), dtype=np.float32)
    copy = min(size, arr.size)
    if copy:
        out[:copy] = arr[:copy]
    return out


def _map_to_shape(value: np.ndarray | None, shape_hw: tuple[int, int], fill: float = 0.0) -> np.ndarray:
    try:
        return map_to_shape(value, shape_hw, fill=fill)
    except RuntimeError as err:
        raise RendererInferenceError(str(err)) from err


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _canonical_transition_mode(raw_mode: str, region_type: str) -> str:
    mode = str(raw_mode or "").strip().lower()
    if mode in {"garment_surface", "open_front", "remove_outer"}:
        return "garment_surface"
    if mode in {"garment_reveal", "visibility_reveal"}:
        return "garment_reveal"
    if mode in {"pose_exposure", "pose_deform", "deform", "deform_relation_aware"}:
        return "pose_exposure"
    if mode in {"expression_refine", "expression_shift", "expression"}:
        return "expression_refine"
    if mode == "visibility_occlusion":
        return "visibility_occlusion"
    if region_type in {"face", "head", "mouth", "eyes", "cheek"}:
        return "expression_refine" if mode else "stable"
    if region_type in {"inner_garment", "outer_garment", "garments", "sleeves"} and mode == "stable":
        return "garment_surface"
    return "stable"


def _default_profile_role(region_type: str, reason: str, delta_regions: list[str]) -> str:
    if region_type in set(delta_regions[:1]):
        return "primary"
    if region_type in set(delta_regions[1:]):
        return "secondary"
    if "context" in reason or "support" in reason:
        return "context"
    return "primary"


def _resolve_profile_role(request: PatchSynthesisRequest, region_type: str) -> str:
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    human_contract = LearnedHumanStateContract.from_metadata(ctx.get("learned_human_state_contract"))
    learned_contract = LearnedTemporalTransitionContract.from_metadata(ctx.get("learned_temporal_contract"))
    if human_contract is not None and human_contract.is_learned_primary:
        primary = set(human_contract.target_profile.primary_regions)
        secondary = set(human_contract.target_profile.secondary_regions)
        context = set(human_contract.target_profile.context_regions)
        if region_type in primary:
            return "primary"
        if region_type in secondary:
            return "secondary"
        if region_type in context:
            return "context"
    if learned_contract is not None and learned_contract.is_learned_primary:
        primary = set(learned_contract.target_profile.primary_regions)
        secondary = set(learned_contract.target_profile.secondary_regions)
        context = set(learned_contract.target_profile.context_regions)
        if region_type in primary:
            return "primary"
        if region_type in secondary:
            return "secondary"
        if region_type in context:
            return "context"
    target_profile = ctx.get("target_profile", {}) if isinstance(ctx.get("target_profile", {}), dict) else {}
    rationale = ctx.get("region_selection_rationale", {}) if isinstance(ctx.get("region_selection_rationale", {}), dict) else {}
    delta = ctx.get("graph_delta")

    primary = {str(x) for x in target_profile.get("primary_regions", [])}
    secondary = {str(x) for x in target_profile.get("secondary_regions", [])}
    context = {str(x) for x in target_profile.get("context_regions", [])}
    if region_type in primary:
        return "primary"
    if region_type in secondary:
        return "secondary"
    if region_type in context:
        return "context"

    rationale_label = str(rationale.get(region_type, "")).strip().lower()
    if rationale_label == "primary_goal_region":
        return "primary"
    if rationale_label == "secondary_influence_region":
        return "secondary"
    if rationale_label == "context_support_region":
        return "context"

    delta_regions = list(getattr(delta, "affected_regions", [])) if delta is not None else []
    return _default_profile_role(region_type, request.region.reason.lower(), delta_regions)


def _build_render_profile(request: PatchSynthesisRequest) -> RenderConditioningProfile:
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    human_contract = LearnedHumanStateContract.from_metadata(ctx.get("learned_human_state_contract"))
    learned_contract = LearnedTemporalTransitionContract.from_metadata(ctx.get("learned_temporal_contract"))
    delta = ctx.get("graph_delta")
    _, region_type = parse_region_id(request.region.region_id)
    raw_mode = ""
    if delta is not None:
        raw_mode = str(getattr(delta, "region_transition_mode", {}).get(region_type, ""))
    if human_contract is not None and human_contract.is_learned_primary:
        vis_scores = list(human_contract.visibility_state_scores.values())
        vis_mean = float(np.mean(vis_scores)) if vis_scores else 0.0
        torso_vis = float(human_contract.visibility_state_scores.get(region_type, human_contract.visibility_state_scores.get("torso", vis_mean)))
        if vis_mean > max(torso_vis, 0.65):
            raw_mode = "visibility_occlusion"
        elif torso_vis >= 0.5:
            raw_mode = "garment_reveal"
        elif human_contract.predicted_family == "expression_transition":
            raw_mode = "expression_refine"
        elif human_contract.predicted_family in {"pose_transition", "interaction_transition"}:
            raw_mode = "pose_exposure"
    elif learned_contract is not None and learned_contract.is_learned_primary:
        if learned_contract.occlusion_score > max(learned_contract.reveal_score, 0.55):
            raw_mode = "visibility_occlusion"
        elif learned_contract.reveal_score >= 0.5:
            raw_mode = "garment_reveal"
        elif learned_contract.predicted_family == "expression_transition":
            raw_mode = "expression_refine"
        elif learned_contract.predicted_family in {"pose_transition", "interaction_transition"}:
            raw_mode = "pose_exposure"
    raw_mode = str(ctx.get("region_transition_mode", raw_mode))
    transition_mode = _canonical_transition_mode(raw_mode, region_type)
    profile_role = _resolve_profile_role(request, region_type)

    base = {
        "stable": RenderConditioningProfile("stable", profile_role, "preserve_context", False, 0.18, 0.92, 0.92, 0.10, 0.08, 0.03, 0.22, 0.18, 0.18, 0.70),
        "garment_surface": RenderConditioningProfile("garment_surface", profile_role, "surface_transition", False, 0.68, 0.62, 0.70, 0.35, 0.22, 0.16, 0.78, 0.30, 0.42, 0.45),
        "garment_reveal": RenderConditioningProfile("garment_reveal", profile_role, "reveal_synthesis", True, 0.90, 0.55, 0.76, 0.78, 0.55, 0.18, 0.88, 0.26, 0.46, 0.50),
        "pose_exposure": RenderConditioningProfile("pose_exposure", profile_role, "pose_conditioned_exposure", True, 0.56, 0.72, 0.82, 0.42, 0.38, 0.12, 0.64, 0.24, 0.36, 0.52),
        "expression_refine": RenderConditioningProfile("expression_refine", profile_role, "expression_micro_edit", False, 0.34, 0.92, 0.96, 0.24, 0.12, 0.08, 0.42, 0.16, 0.18, 0.38),
        "visibility_occlusion": RenderConditioningProfile("visibility_occlusion", profile_role, "occlusion_boundary_control", False, 0.28, 0.84, 0.90, 0.20, 0.33, 0.05, 0.38, 0.28, 0.34, 0.65),
    }[transition_mode]

    identity_bonus = 0.05 if request.identity_embedding and transition_mode == "expression_refine" else 0.0
    if profile_role == "primary":
        base.preservation_strength = max(0.40, base.preservation_strength - 0.10)
        base.alpha_peak = min(0.95, base.alpha_peak + 0.08)
        base.uncertainty_bias = min(1.0, base.uncertainty_bias + 0.02)
    elif profile_role == "secondary":
        base.edit_strength *= 0.66
        base.preservation_strength = min(0.98, base.preservation_strength + 0.12)
        base.alpha_peak *= 0.78
        base.seam_strictness = min(1.0, base.seam_strictness + 0.05)
        base.context_support = min(1.0, base.context_support + 0.08)
    else:
        base.edit_strength *= 0.38
        base.preservation_strength = min(0.995, base.preservation_strength + 0.22)
        base.alpha_peak *= 0.52
        base.alpha_floor *= 0.60
        base.seam_strictness = min(1.0, base.seam_strictness + 0.08)
        base.context_support = min(1.0, base.context_support + 0.18)
        base.uncertainty_bias *= 0.82

    if transition_mode == "expression_refine":
        base.uncertainty_bias = max(0.03, base.uncertainty_bias - identity_bonus)

    bundle_cond = extract_memory_bundle_conditioning(request)
    i2v_action_cond = extract_i2v_action_conditioning(request)
    identity_bias = _identity_preservation_bias(
        region_type,
        float(bundle_cond.get("identity_reference_strength", 0.0)),
        bool(bundle_cond.get("identity_reference_blocked", False)),
    )
    if identity_bias >= 0.9:
        base.preservation_strength = min(0.995, base.preservation_strength + 0.10)
        base.seam_strictness = min(1.0, base.seam_strictness + 0.04)
        base.uncertainty_bias = max(0.02, base.uncertainty_bias - 0.06)
        base.memory_dependency = min(1.0, base.memory_dependency + 0.10)
        if transition_mode != "expression_refine":
            base.edit_strength *= 0.90
    region_action_mode = str(i2v_action_cond.get("i2v_region_action_mode", ""))
    action_phase = str(i2v_action_cond.get("i2v_action_phase", "stable_idle"))
    action_targets_region = _i2v_action_targets_region(action_phase, region_type, region_action_mode)
    if action_targets_region:
        action_boost = 0.06 * _safe_float(i2v_action_cond.get("i2v_action_strength"), 0.0)
        base.edit_strength = min(0.98, base.edit_strength + action_boost)
    if region_type in IDENTITY_SENSITIVE_REGIONS and action_targets_region:
        base.preservation_strength = max(base.preservation_strength, 0.90)
    return base


def _mode_cond(profile: RenderConditioningProfile) -> np.ndarray:
    mode_vec = np.zeros((MODE_DIM,), dtype=np.float32)
    mode_vec[TRANSITION_MODE_ORDER.index(profile.transition_mode)] = 1.0
    mode_vec[6] = float(profile.edit_strength)
    mode_vec[7] = float(profile.seam_strictness)
    return mode_vec


def _role_cond(profile: RenderConditioningProfile) -> np.ndarray:
    role_vec = np.zeros((ROLE_DIM,), dtype=np.float32)
    role_vec[PROFILE_ROLE_ORDER.index(profile.profile_role)] = 1.0
    role_vec[3] = float(profile.edit_strength)
    role_vec[4] = float(profile.preservation_strength)
    role_vec[5] = float(profile.context_support)
    return role_vec


class TrainableLocalPatchModel:
    """Mode-aware ROI renderer with explicit alpha, uncertainty and preservation priors."""

    def __init__(self, pixel_dim: int = PIXEL_FEATURE_DIM, cond_dim: int = CONDITION_FEATURE_DIM, hidden_dim: int = 56, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        self.pixel_dim = pixel_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.W_pixel = (rng.standard_normal((pixel_dim, hidden_dim)) * 0.06).astype(np.float32)
        self.W_cond = (rng.standard_normal((cond_dim, hidden_dim)) * 0.07).astype(np.float32)
        self.b_hidden = np.zeros((hidden_dim,), dtype=np.float32)
        self.W_rgb = (rng.standard_normal((hidden_dim, 3)) * 0.05).astype(np.float32)
        self.b_rgb = np.zeros((3,), dtype=np.float32)
        self.W_alpha = (rng.standard_normal((hidden_dim, 1)) * 0.05).astype(np.float32)
        self.b_alpha = np.zeros((1,), dtype=np.float32)
        self.W_unc = (rng.standard_normal((hidden_dim, 1)) * 0.05).astype(np.float32)
        self.b_unc = np.zeros((1,), dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _conditioning_maps(batch: PatchBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w, _ = batch.roi_before.shape
        preservation = _map_to_shape(batch.preservation_mask, (h, w), fill=0.65 if batch.profile_role != "context" else 0.82)
        uncertainty_target = _map_to_shape(batch.uncertainty_target, (h, w), fill=0.12)
        seam_prior = _map_to_shape(batch.seam_prior, (h, w), fill=0.25)
        return preservation, uncertainty_target, seam_prior

    @staticmethod
    def _to_pixel_features(batch: PatchBatch) -> np.ndarray:
        h, w, _ = batch.roi_before.shape
        preservation, uncertainty_target, seam_prior = TrainableLocalPatchModel._conditioning_maps(batch)
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h, dtype=np.float32), np.linspace(0.0, 1.0, w, dtype=np.float32), indexing="ij")
        return np.concatenate(
            [
                batch.roi_before,
                batch.changed_mask,
                batch.blend_hint,
                preservation,
                batch.alpha_target,
                uncertainty_target,
                seam_prior,
                yy[..., None],
                xx[..., None],
                batch.changed_mask * batch.blend_hint,
                batch.changed_mask * (1.0 - preservation),
                seam_prior * batch.blend_hint,
                np.mean(batch.roi_before, axis=2, keepdims=True),
            ],
            axis=2,
        )

    @staticmethod
    def _global_condition(batch: PatchBatch) -> np.ndarray:
        return np.concatenate(
            [
                _vector_to_size(batch.semantic_embed, SEMANTIC_DIM),
                _vector_to_size(batch.delta_cond, DELTA_DIM),
                _vector_to_size(batch.planner_cond, PLANNER_DIM),
                _vector_to_size(batch.graph_cond, GRAPH_DIM),
                _vector_to_size(batch.memory_cond, MEMORY_DIM),
                _vector_to_size(batch.appearance_cond, APPEARANCE_DIM),
                _vector_to_size(batch.bbox_cond, BBOX_DIM),
                _vector_to_size(batch.mode_cond, MODE_DIM),
                _vector_to_size(batch.role_cond, ROLE_DIM),
            ],
            axis=0,
        ).astype(np.float32)

    def forward(self, batch: PatchBatch) -> dict[str, np.ndarray | float]:
        pix = self._to_pixel_features(batch)
        h, w, _ = pix.shape
        x = pix.reshape(-1, self.pixel_dim)
        cond = self._global_condition(batch)
        cond_proj = cond @ self.W_cond
        hidden_pre = x @ self.W_pixel + cond_proj[None, :] + self.b_hidden[None, :]
        hidden = self._relu(hidden_pre)

        rgb_raw = hidden @ self.W_rgb + self.b_rgb[None, :]
        alpha_raw = hidden @ self.W_alpha + self.b_alpha[None, :]
        unc_raw = hidden @ self.W_unc + self.b_unc[None, :]

        mode_vec = _vector_to_size(batch.mode_cond, MODE_DIM)
        role_vec = _vector_to_size(batch.role_cond, ROLE_DIM)
        mode_edit_scale = float(mode_vec[6]) if mode_vec.size > 6 else 0.5
        edit_strength = float(role_vec[3]) if role_vec.size > 3 else 0.7
        preservation_strength = float(role_vec[4]) if role_vec.size > 4 else 0.6
        context_support = float(role_vec[5]) if role_vec.size > 5 else 0.4

        rgb_before = batch.roi_before.reshape(-1, 3)
        changed = batch.changed_mask.reshape(-1, 1)
        blend = batch.blend_hint.reshape(-1, 1)
        alpha_prior = batch.alpha_target.reshape(-1, 1)
        preservation, uncertainty_target, seam_prior = self._conditioning_maps(batch)
        preservation_flat = preservation.reshape(-1, 1)
        uncertainty_target_flat = uncertainty_target.reshape(-1, 1)
        seam_flat = seam_prior.reshape(-1, 1)

        edit_gate = np.clip(
            changed
            * (0.42 + 0.48 * blend + 0.28 * edit_strength + 0.18 * mode_edit_scale)
            * (1.08 - 0.58 * preservation_flat)
            + seam_flat * context_support * 0.08,
            0.0,
            1.0,
        )
        residual_scale = 0.22 + 0.48 * mode_edit_scale
        residual = np.tanh(rgb_raw) * residual_scale
        rgb = np.clip(rgb_before + residual * edit_gate, 0.0, 1.0)

        alpha_base = _sigmoid(alpha_raw)
        alpha_contract = np.clip(
            alpha_prior * (0.78 + 0.22 * edit_strength)
            + seam_flat * 0.08
            - uncertainty_target_flat * 0.16,
            0.0,
            1.0,
        )
        alpha = np.clip(0.58 * alpha_base + 0.42 * alpha_contract, 0.0, 1.0)

        unc_base = _sigmoid(unc_raw)
        unc_contract = np.clip(
            uncertainty_target_flat
            + preservation_flat * 0.12
            - changed * 0.08
            + (1.0 - edit_gate) * 0.05,
            0.0,
            1.0,
        )
        uncertainty = np.clip(0.55 * unc_base + 0.45 * unc_contract, 0.0, 1.0)

        rgb_img = rgb.reshape(h, w, 3)
        alpha_img = alpha.reshape(h, w)
        uncertainty_img = uncertainty.reshape(h, w)
        edit_gate_img = edit_gate.reshape(h, w)
        drift = float(np.mean(np.abs(rgb_img - batch.roi_before) * preservation))
        ambiguous_edit = float(np.mean(uncertainty_img * alpha_img))
        localized_edit = float(np.mean(alpha_img * edit_gate_img))
        confidence = float(
            np.clip(
                (1.0 - ambiguous_edit)
                * (1.0 - drift * (0.9 + 0.6 * preservation_strength))
                * (0.52 + 0.28 * localized_edit + 0.20 * (1.0 - float(np.mean(uncertainty_target * batch.changed_mask)))),
                0.0,
                1.0,
            )
        )

        return {
            "rgb": rgb_img,
            "alpha": alpha_img,
            "uncertainty": uncertainty_img,
            "confidence": confidence,
            "hidden": hidden,
            "hidden_pre": hidden_pre,
            "pixel_x": x,
            "cond": cond,
            "rgb_raw": rgb_raw,
            "alpha_base": alpha_base.reshape(h, w),
            "unc_base": unc_base.reshape(h, w),
            "edit_gate": edit_gate_img,
            "alpha_contract": alpha_contract.reshape(h, w),
            "uncertainty_contract": unc_contract.reshape(h, w),
            "residual_scale": residual_scale,
        }

    def compute_losses(self, batch: PatchBatch, out: dict[str, np.ndarray | float]) -> dict[str, float]:
        rgb = out["rgb"]
        alpha = out["alpha"]
        unc = out["uncertainty"]
        edit_gate = out["edit_gate"]
        assert isinstance(rgb, np.ndarray) and isinstance(alpha, np.ndarray) and isinstance(unc, np.ndarray) and isinstance(edit_gate, np.ndarray)

        changed = np.clip(batch.changed_mask[..., 0], 0.0, 1.0)
        preservation, uncertainty_target, seam_prior = self._conditioning_maps(batch)
        preservation_map = preservation[..., 0]
        uncertainty_target_map = uncertainty_target[..., 0]
        seam_map = seam_prior[..., 0]
        role_vec = _vector_to_size(batch.role_cond, ROLE_DIM)
        preservation_strength = float(role_vec[4]) if role_vec.size > 4 else 0.6
        context_support = float(role_vec[5]) if role_vec.size > 5 else 0.4

        recon_weights = 0.28 + 1.05 * changed + 0.18 * edit_gate
        recon = float(np.mean(((rgb - batch.roi_after) ** 2) * recon_weights[..., None]))
        alpha_weights = 0.25 + 0.75 * changed + 0.25 * seam_map
        alpha_loss = float(np.mean(((alpha - batch.alpha_target[..., 0]) ** 2) * alpha_weights))
        seam_loss = float(np.mean(np.abs(rgb - batch.roi_before) * seam_map[..., None] * (0.35 + 0.65 * alpha[..., None])))
        appearance = float(np.mean(((rgb - batch.roi_before) ** 2) * preservation_map[..., None] * (0.65 + 0.50 * preservation_strength)))
        region_consistency = float(np.mean((np.mean(rgb, axis=(0, 1)) - np.mean(batch.roi_after, axis=(0, 1))) ** 2))
        uncertainty_loss = float(np.mean(((unc - uncertainty_target_map) ** 2) * (0.40 + 0.60 * changed + 0.20 * seam_map)))
        alpha_contract_target = np.clip(batch.alpha_target[..., 0] * (0.82 + 0.18 * edit_gate) + seam_map * 0.08, 0.0, 1.0)
        alpha_blend_consistency = float(np.mean(np.abs(alpha - alpha_contract_target) * (0.25 + 0.75 * changed)))
        drift_mask = np.clip(preservation_map - changed * 0.35, 0.0, 1.0)
        drift_penalty = float(np.mean(np.abs(rgb - batch.roi_before) * drift_mask[..., None] * (0.30 + 0.70 * context_support)))

        total = (
            recon
            + 0.55 * alpha_loss
            + 0.42 * seam_loss
            + 0.48 * appearance
            + 0.20 * region_consistency
            + 0.40 * uncertainty_loss
            + 0.28 * alpha_blend_consistency
            + 0.22 * drift_penalty
        )
        supervision_weight = target_supervision_weight(
            str((batch.conditioning_summary or {}).get("training_target_quality", "unknown"))
            if isinstance(batch.conditioning_summary, dict)
            else "unknown"
        )
        return {
            "total_loss": total,
            "weighted_total_loss": float(total * supervision_weight),
            "target_supervision_weight": float(supervision_weight),
            "reconstruction_loss": recon,
            "alpha_loss": alpha_loss,
            "seam_loss": seam_loss,
            "appearance_preservation_loss": appearance,
            "region_consistency_loss": region_consistency,
            "uncertainty_calibration_loss": uncertainty_loss,
            "alpha_blend_consistency_loss": alpha_blend_consistency,
            "drift_penalty": drift_penalty,
        }

    def train_step(self, batch: PatchBatch, lr: float = 1e-3) -> dict[str, float]:
        out = self.forward(batch)
        losses = self.compute_losses(batch, out)

        rgb = out["rgb"].reshape(-1, 3)
        alpha = out["alpha"].reshape(-1, 1)
        unc = out["uncertainty"].reshape(-1, 1)
        hidden = out["hidden"]
        hidden_pre = out["hidden_pre"]
        x = out["pixel_x"]
        cond = out["cond"]
        rgb_raw = out["rgb_raw"]
        edit_gate = out["edit_gate"].reshape(-1, 1)
        alpha_base = out["alpha_base"].reshape(-1, 1)
        unc_base = out["unc_base"].reshape(-1, 1)
        assert isinstance(hidden, np.ndarray) and isinstance(hidden_pre, np.ndarray) and isinstance(x, np.ndarray) and isinstance(cond, np.ndarray) and isinstance(rgb_raw, np.ndarray)

        target_rgb = batch.roi_after.reshape(-1, 3)
        rgb_before = batch.roi_before.reshape(-1, 3)
        changed = batch.changed_mask.reshape(-1, 1)
        alpha_t = batch.alpha_target.reshape(-1, 1)
        preservation, uncertainty_target, seam_prior = self._conditioning_maps(batch)
        preservation_flat = preservation.reshape(-1, 1)
        uncertainty_target_flat = uncertainty_target.reshape(-1, 1)
        seam_flat = seam_prior.reshape(-1, 1)
        role_vec = _vector_to_size(batch.role_cond, ROLE_DIM)
        preservation_strength = float(role_vec[4]) if role_vec.size > 4 else 0.6
        context_support = float(role_vec[5]) if role_vec.size > 5 else 0.4
        residual_scale = float(out["residual_scale"])

        recon_weights = (0.28 + 1.05 * changed + 0.18 * edit_gate)
        d_rgb = 2.0 * (rgb - target_rgb) * recon_weights / max(1, rgb.size)
        d_rgb += (2.0 * (rgb - rgb_before) * preservation_flat * (0.65 + 0.50 * preservation_strength) / max(1, rgb.size)) * 0.48
        d_rgb += (np.sign(rgb - rgb_before) * seam_flat * (0.35 + 0.65 * alpha) / max(1, rgb.size)) * 0.42
        drift_mask = np.clip(preservation_flat - changed * 0.35, 0.0, 1.0)
        d_rgb += (np.sign(rgb - rgb_before) * drift_mask * (0.30 + 0.70 * context_support) / max(1, rgb.size)) * 0.22
        d_rgb_raw = d_rgb * edit_gate * (1.0 - np.tanh(rgb_raw) ** 2) * residual_scale

        alpha_contract_target = np.clip(alpha_t * (0.82 + 0.18 * edit_gate) + seam_flat * 0.08, 0.0, 1.0)
        alpha_weights = 0.25 + 0.75 * changed + 0.25 * seam_flat
        d_alpha = (2.0 * (alpha - alpha_t) * alpha_weights / max(1, alpha.size)) * 0.55
        d_alpha += (np.sign(alpha - alpha_contract_target) * (0.25 + 0.75 * changed) / max(1, alpha.size)) * 0.28
        d_alpha_raw = d_alpha * 0.58 * alpha_base * (1.0 - alpha_base)

        unc_weights = 0.40 + 0.60 * changed + 0.20 * seam_flat
        d_unc = (2.0 * (unc - uncertainty_target_flat) * unc_weights / max(1, unc.size)) * 0.40
        d_unc_raw = d_unc * 0.55 * unc_base * (1.0 - unc_base)

        grad_W_rgb = hidden.T @ d_rgb_raw
        grad_b_rgb = d_rgb_raw.sum(axis=0)
        grad_W_alpha = hidden.T @ d_alpha_raw
        grad_b_alpha = d_alpha_raw.sum(axis=0)
        grad_W_unc = hidden.T @ d_unc_raw
        grad_b_unc = d_unc_raw.sum(axis=0)

        d_hidden = d_rgb_raw @ self.W_rgb.T + d_alpha_raw @ self.W_alpha.T + d_unc_raw @ self.W_unc.T
        d_hidden_pre = d_hidden * (hidden_pre > 0.0).astype(np.float32)

        grad_W_pixel = x.T @ d_hidden_pre
        grad_b_hidden = d_hidden_pre.sum(axis=0)
        grad_cond_proj = d_hidden_pre.sum(axis=0)
        grad_W_cond = np.outer(cond, grad_cond_proj)

        self.W_rgb -= lr * grad_W_rgb.astype(np.float32)
        self.b_rgb -= lr * grad_b_rgb.astype(np.float32)
        self.W_alpha -= lr * grad_W_alpha.astype(np.float32)
        self.b_alpha -= lr * grad_b_alpha.astype(np.float32)
        self.W_unc -= lr * grad_W_unc.astype(np.float32)
        self.b_unc -= lr * grad_b_unc.astype(np.float32)
        self.W_pixel -= lr * grad_W_pixel.astype(np.float32)
        self.W_cond -= lr * grad_W_cond.astype(np.float32)
        self.b_hidden -= lr * grad_b_hidden.astype(np.float32)
        return losses

    def eval_step(self, batch: PatchBatch) -> dict[str, float]:
        out = self.forward(batch)
        losses = self.compute_losses(batch, out)
        rgb = out["rgb"]
        alpha = out["alpha"]
        unc = out["uncertainty"]
        assert isinstance(rgb, np.ndarray) and isinstance(alpha, np.ndarray) and isinstance(unc, np.ndarray)
        mae = float(np.mean(np.abs(rgb - batch.roi_after)))
        alpha_mae = float(np.mean(np.abs(alpha - batch.alpha_target[..., 0])))
        uncertainty_mean = float(np.mean(unc))
        losses.update({"mae": mae, "alpha_mae": alpha_mae, "uncertainty_mean": uncertainty_mean})
        return losses

    def infer(self, batch: PatchBatch) -> dict[str, np.ndarray | float]:
        return self.forward(batch)

    def save(self, path: str) -> None:
        payload = {
            "pixel_dim": self.pixel_dim,
            "cond_dim": self.cond_dim,
            "hidden_dim": self.hidden_dim,
            "W_pixel": self.W_pixel.tolist(),
            "W_cond": self.W_cond.tolist(),
            "b_hidden": self.b_hidden.tolist(),
            "W_rgb": self.W_rgb.tolist(),
            "b_rgb": self.b_rgb.tolist(),
            "W_alpha": self.W_alpha.tolist(),
            "b_alpha": self.b_alpha.tolist(),
            "W_unc": self.W_unc.tolist(),
            "b_unc": self.b_unc.tolist(),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TrainableLocalPatchModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(pixel_dim=int(payload["pixel_dim"]), cond_dim=int(payload["cond_dim"]), hidden_dim=int(payload["hidden_dim"]))
        model.W_pixel = np.array(payload["W_pixel"], dtype=np.float32)
        model.W_cond = np.array(payload["W_cond"], dtype=np.float32)
        model.b_hidden = np.array(payload["b_hidden"], dtype=np.float32)
        model.W_rgb = np.array(payload["W_rgb"], dtype=np.float32)
        model.b_rgb = np.array(payload["b_rgb"], dtype=np.float32)
        model.W_alpha = np.array(payload["W_alpha"], dtype=np.float32)
        model.b_alpha = np.array(payload["b_alpha"], dtype=np.float32)
        model.W_unc = np.array(payload["W_unc"], dtype=np.float32)
        model.b_unc = np.array(payload["b_unc"], dtype=np.float32)
        return model


class TemporalLocalPatchModel(TrainableLocalPatchModel):
    """Short-window temporal renderer with explicit temporal conditioning channels."""

    PHASE_ORDER = ("prepare", "transition", "contact_or_reveal", "stabilize")
    FAMILY_ORDER = ("pose_transition", "garment_transition", "expression_transition", "interaction_transition", "visibility_transition")

    def _temporal_feature_vector(self, batch: PatchBatch) -> np.ndarray:
        fam = np.zeros((len(self.FAMILY_ORDER),), dtype=np.float32)
        if batch.predicted_family in self.FAMILY_ORDER:
            fam[self.FAMILY_ORDER.index(batch.predicted_family)] = 1.0
        phase = np.zeros((len(self.PHASE_ORDER),), dtype=np.float32)
        if batch.predicted_phase in self.PHASE_ORDER:
            phase[self.PHASE_ORDER.index(batch.predicted_phase)] = 1.0
        profile = batch.target_profile if isinstance(batch.target_profile, dict) else {}
        primary = float(len(profile.get("primary_regions", [])))
        secondary = float(len(profile.get("secondary_regions", [])))
        context = float(len(profile.get("context_regions", [])))
        return np.concatenate(
            [
                fam,
                phase,
                np.asarray(
                    [
                        float(np.clip(batch.reveal_score, 0.0, 1.0)),
                        float(np.clip(batch.occlusion_score, 0.0, 1.0)),
                        float(np.clip(batch.support_contact_score, 0.0, 1.0)),
                        float(np.clip(primary / 4.0, 0.0, 1.0)),
                        float(np.clip(secondary / 4.0, 0.0, 1.0)),
                        float(np.clip(context / 4.0, 0.0, 1.0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    def forward(self, batch: PatchBatch) -> dict[str, np.ndarray | float]:
        out = super().forward(batch)
        prev = batch.previous_roi if isinstance(batch.previous_roi, np.ndarray) and batch.previous_roi.shape == batch.roi_before.shape else batch.roi_before
        temporal_delta = np.mean(np.abs(batch.roi_before - prev), axis=2, keepdims=True)
        temporal_feat = self._temporal_feature_vector(batch)
        temporal_strength = float(np.clip(np.mean(temporal_feat[-6:]), 0.0, 1.0))
        reveal_boost = float(np.clip(batch.reveal_score - batch.occlusion_score, -1.0, 1.0))
        consistency = np.clip(1.0 - temporal_delta * (0.55 + 0.30 * temporal_strength), 0.0, 1.0)
        edit_gate = np.asarray(out["edit_gate"], dtype=np.float32)
        alpha = np.asarray(out["alpha"], dtype=np.float32)
        rgb = np.asarray(out["rgb"], dtype=np.float32)
        unc = np.asarray(out["uncertainty"], dtype=np.float32)

        temporal_edit_gate = np.clip(edit_gate[..., None] * (0.80 + 0.20 * (1.0 - consistency)) * (1.0 + 0.12 * reveal_boost), 0.0, 1.0)
        residual = np.tanh(rgb - batch.roi_before)
        rgb_temporal = np.clip(batch.roi_before + residual * temporal_edit_gate, 0.0, 1.0)
        alpha_temporal = np.clip(alpha * (0.84 + 0.16 * np.squeeze(temporal_edit_gate, axis=2)), 0.0, 1.0)
        unc_temporal = np.clip(unc * (0.86 + 0.14 * np.squeeze(1.0 - consistency, axis=2)) + np.squeeze(temporal_delta, axis=2) * 0.08, 0.0, 1.0)

        out["rgb"] = rgb_temporal
        out["alpha"] = alpha_temporal
        out["uncertainty"] = unc_temporal
        out["temporal_consistency"] = np.squeeze(consistency, axis=2)
        out["temporal_edit_gate"] = np.squeeze(temporal_edit_gate, axis=2)
        out["temporal_features"] = temporal_feat
        return out

    def compute_losses(self, batch: PatchBatch, out: dict[str, np.ndarray | float]) -> dict[str, float]:
        losses = super().compute_losses(batch, out)
        rgb = np.asarray(out["rgb"], dtype=np.float32)
        prev = batch.previous_roi if isinstance(batch.previous_roi, np.ndarray) and batch.previous_roi.shape == batch.roi_before.shape else batch.roi_before
        changed = np.clip(batch.changed_mask[..., 0], 0.0, 1.0)
        unchanged = np.clip(1.0 - changed, 0.0, 1.0)
        temporal_consistency = float(np.mean(np.abs(rgb - prev) * unchanged[..., None]))
        reveal_focus = float(np.mean(np.abs(rgb - batch.roi_after) * np.clip(changed * float(np.clip(batch.reveal_score, 0.0, 1.0)), 0.0, 1.0)[..., None]))
        occ_edge = np.abs(np.gradient(np.clip(batch.changed_mask[..., 0], 0.0, 1.0), axis=0)) + np.abs(np.gradient(np.clip(batch.changed_mask[..., 0], 0.0, 1.0), axis=1))
        occlusion_boundary = float(np.mean(np.abs(rgb - batch.roi_after) * np.clip(occ_edge * max(batch.occlusion_score, 1e-4), 0.0, 1.0)[..., None]))
        preservation_loss = float(np.mean(np.abs(rgb - batch.roi_before) * unchanged[..., None]))
        rollout_weight = float(np.clip(batch.rollout_weight, 0.1, 3.0))
        rollout_weighted_reconstruction = float(losses["reconstruction_loss"] * rollout_weight)
        total = (
            losses["total_loss"]
            + 0.35 * temporal_consistency
            + 0.25 * reveal_focus
            + 0.25 * occlusion_boundary
            + 0.22 * preservation_loss
            + 0.20 * rollout_weighted_reconstruction
        )
        losses.update(
            {
                "temporal_consistency_loss": temporal_consistency,
                "reveal_region_focus_loss": reveal_focus,
                "occlusion_boundary_loss": occlusion_boundary,
                "preservation_loss": preservation_loss,
                "rollout_weighted_reconstruction_loss": rollout_weighted_reconstruction,
                "total_loss": total,
            }
        )
        return losses


def _to_np_patch(tensor: list) -> np.ndarray:
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RendererInputError("Expected ROI tensor in HxWx3 format")
    return np.clip(arr, 0.0, 1.0)


def _extract_roi(frame: list, region: RegionRef) -> np.ndarray:
    h, w, _ = shape(frame)
    if h <= 0 or w <= 0:
        raise RendererInputError("Current frame is empty")
    x0 = max(0, min(w - 1, int(region.bbox.x * w)))
    y0 = max(0, min(h - 1, int(region.bbox.y * h)))
    x1 = max(x0 + 1, min(w, int((region.bbox.x + region.bbox.w) * w)))
    y1 = max(y0 + 1, min(h, int((region.bbox.y + region.bbox.h) * h)))
    return _to_np_patch([row[x0:x1] for row in frame[y0:y1]])


def summarize_memory_bundle_trace(transition_context: dict[str, object] | None) -> dict[str, object]:
    ctx = transition_context if isinstance(transition_context, dict) else {}
    memory_bundle_raw = ctx.get("region_memory_bundle")
    memory_bundle = memory_bundle_raw if isinstance(memory_bundle_raw, RegionMemoryBundle) else None
    memory_bundle_present = False
    bundle_support_level = "unknown"
    bundle_retrieval_reasons: list[str] = []
    bundle_has_current_reuse = False
    bundle_has_identity_reference = False
    bundle_has_appearance_reference = False
    bundle_has_skin_reference = False
    bundle_has_body_shape_reference = False
    bundle_has_garment_reference = False
    bundle_has_accessory_reference = False
    bundle_has_hidden_slot = False
    bundle_hidden_type = "none"
    bundle_reveal_lifecycle = "unknown"
    serialized_bundle = ctx.get("region_memory_bundle_serialized")
    if isinstance(serialized_bundle, str):
        try:
            parsed_serialized = json.loads(serialized_bundle)
        except json.JSONDecodeError:
            parsed_serialized = {}
        serialized_bundle = parsed_serialized if isinstance(parsed_serialized, dict) else {}
    if memory_bundle is None and isinstance(serialized_bundle, dict):
        bundle_data = serialized_bundle
        if isinstance(bundle_data, dict):
            memory_bundle_present = bool(bundle_data.get("memory_bundle_present", bool(bundle_data)))
            bundle_support_level = str(
                bundle_data.get(
                    "memory_support_level",
                    bundle_data.get("region_memory_support_level", ctx.get("region_memory_support_level", "none" if not bundle_data else "unknown")),
                )
            )
            reasons = bundle_data.get("retrieval_reasons", bundle_data.get("memory_bundle_retrieval_reasons", ctx.get("region_memory_retrieval_reasons", [])))
            bundle_retrieval_reasons = list(reasons) if isinstance(reasons, list) else []
            bundle_has_current_reuse = bool(bundle_data.get("has_current_reuse", bundle_data.get("memory_bundle_has_current_reuse", False)))
            bundle_has_identity_reference = bool(bundle_data.get("has_identity_reference", bundle_data.get("memory_bundle_has_identity_reference", False)))
            bundle_has_appearance_reference = bool(bundle_data.get("has_appearance_reference", bundle_data.get("memory_bundle_has_appearance_reference", False)))
            bundle_has_skin_reference = bool(bundle_data.get("has_skin_reference", bundle_data.get("memory_bundle_has_skin_reference", False)))
            bundle_has_body_shape_reference = bool(bundle_data.get("has_body_shape_reference", bundle_data.get("memory_bundle_has_body_shape_reference", False)))
            bundle_has_garment_reference = bool(bundle_data.get("has_garment_reference", bundle_data.get("memory_bundle_has_garment_reference", False)))
            bundle_has_accessory_reference = bool(bundle_data.get("has_accessory_reference", bundle_data.get("memory_bundle_has_accessory_reference", False)))
            bundle_has_hidden_slot = bool(bundle_data.get("has_hidden_slot", bundle_data.get("memory_bundle_has_hidden_slot", False)))
            hidden_slot = bundle_data.get("hidden_slot")
            bundle_hidden_type = str(bundle_data.get("hidden_type", bundle_data.get("memory_bundle_hidden_type", "none")))
            if isinstance(hidden_slot, dict):
                bundle_hidden_type = str(hidden_slot.get("hidden_type", bundle_hidden_type))
            bundle_reveal_lifecycle = str(bundle_data.get("reveal_lifecycle", bundle_data.get("memory_bundle_reveal_lifecycle", "unknown")))
    else:
        memory_bundle_present = memory_bundle is not None
        bundle_support_level = memory_bundle.memory_support_level if memory_bundle else str(ctx.get("region_memory_support_level", "none"))
        bundle_retrieval_reasons = list(memory_bundle.retrieval_reasons) if memory_bundle else list(ctx.get("region_memory_retrieval_reasons", []))
        bundle_has_current_reuse = bool(memory_bundle.has_current_reuse) if memory_bundle else False
        bundle_has_identity_reference = bool(memory_bundle.has_identity_reference) if memory_bundle else False
        bundle_has_appearance_reference = bool(memory_bundle.has_appearance_reference) if memory_bundle else False
        bundle_has_skin_reference = bool(getattr(memory_bundle, "has_skin_reference", False)) if memory_bundle else False
        bundle_has_body_shape_reference = bool(getattr(memory_bundle, "has_body_shape_reference", False)) if memory_bundle else False
        bundle_has_garment_reference = bool(memory_bundle.has_garment_reference) if memory_bundle else False
        bundle_has_accessory_reference = bool(getattr(memory_bundle, "has_accessory_reference", False)) if memory_bundle else False
        bundle_has_hidden_slot = bool(memory_bundle.has_hidden_slot) if memory_bundle else False
        bundle_hidden_type = memory_bundle.hidden_slot.hidden_type if memory_bundle and memory_bundle.hidden_slot else "none"
        bundle_reveal_lifecycle = str(memory_bundle.reveal_lifecycle) if memory_bundle else "unknown"
    normalized_reasons = {str(reason).strip().lower() for reason in bundle_retrieval_reasons if str(reason).strip()}
    bundle_has_skin_reference = bundle_has_skin_reference or "skin_reference_available" in normalized_reasons
    bundle_has_body_shape_reference = bundle_has_body_shape_reference or "body_shape_reference_available" in normalized_reasons
    bundle_has_garment_reference = bundle_has_garment_reference or "garment_reference_available" in normalized_reasons
    bundle_has_accessory_reference = bundle_has_accessory_reference or "accessory_reference_available" in normalized_reasons
    if isinstance(serialized_bundle, dict):
        serialized_active = serialized_bundle.get("hidden_support_active", serialized_bundle.get("memory_bundle_hidden_support_active"))
    else:
        serialized_active = None
    bundle_hidden_support_active = bool(bundle_has_hidden_slot and bundle_hidden_type not in {"revealed", "revealed_history"})
    if serialized_active is not None:
        bundle_hidden_support_active = bool(serialized_active and bundle_has_hidden_slot and bundle_hidden_type not in {"revealed", "revealed_history"})
    return {
        "memory_bundle_present": memory_bundle_present,
        "memory_support_level": bundle_support_level,
        "memory_bundle_has_current_reuse": bundle_has_current_reuse,
        "memory_bundle_has_identity_reference": bundle_has_identity_reference,
        "memory_bundle_has_appearance_reference": bundle_has_appearance_reference,
        "memory_bundle_has_skin_reference": bundle_has_skin_reference,
        "memory_bundle_has_body_shape_reference": bundle_has_body_shape_reference,
        "memory_bundle_has_garment_reference": bundle_has_garment_reference,
        "memory_bundle_has_accessory_reference": bundle_has_accessory_reference,
        "memory_bundle_has_hidden_slot": bundle_has_hidden_slot,
        "memory_bundle_hidden_type": bundle_hidden_type,
        "memory_bundle_reveal_lifecycle": bundle_reveal_lifecycle,
        "memory_bundle_hidden_support_active": bundle_hidden_support_active,
        "memory_bundle_retrieval_reasons": bundle_retrieval_reasons if memory_bundle_present else list(ctx.get("region_memory_retrieval_reasons", [])),
    }



def _identity_reference_block_state(retrieval_reasons: list[str]) -> tuple[bool, list[str], str]:
    normalized = [str(reason).strip().lower() for reason in retrieval_reasons if str(reason).strip()]
    block_reasons = [reason for reason in IDENTITY_BLOCKED_REASONS if reason in normalized]
    if "identity_reference_blocked_generated" in block_reasons:
        source = "blocked_generated"
    elif "identity_reference_blocked_inferred" in block_reasons:
        source = "blocked_inferred"
    elif "identity_reference_blocked_low_evidence" in block_reasons:
        source = "blocked_low_evidence"
    else:
        source = "none"
    return bool(block_reasons), block_reasons, source


def _identity_reference_strength(
    *,
    has_identity_reference: bool,
    support_level: str,
    retrieval_reasons: list[str],
) -> tuple[float, bool, str, list[str], bool]:
    normalized = [str(reason).strip().lower() for reason in retrieval_reasons if str(reason).strip()]
    blocked, block_reasons, blocked_source = _identity_reference_block_state(normalized)
    support = str(support_level or "none").strip().lower()
    observed_strong = "identity_reference_observed_strong" in normalized
    if blocked:
        return 0.0, True, blocked_source, block_reasons, False
    if has_identity_reference and observed_strong and support != "none":
        return 1.0, False, "observed_strong", [], True
    if has_identity_reference and support == "medium":
        return 0.5, False, "none", [], False
    return 0.0, False, "none", [], False


def _reference_family_strength(
    *,
    family: str,
    has_reference: bool,
    support_level: str,
    retrieval_reasons: list[str],
) -> tuple[float, bool, str, list[str], bool]:
    normalized = [str(reason).strip().lower() for reason in retrieval_reasons if str(reason).strip()]
    blocked_reasons = [reason for reason in REFERENCE_BLOCKED_REASONS.get(family, ()) if reason in normalized]
    if f"{family}_reference_blocked_generated" in blocked_reasons:
        source = "blocked_generated"
    elif f"{family}_reference_blocked_inferred" in blocked_reasons:
        source = "blocked_inferred"
    elif f"{family}_reference_blocked_low_evidence" in blocked_reasons:
        source = "blocked_low_evidence"
    else:
        source = "none"
    if blocked_reasons:
        return 0.0, True, source, blocked_reasons, False
    support = str(support_level or "none").strip().lower()
    observed_strong = f"{family}_reference_observed_strong" in normalized
    available = has_reference or f"{family}_reference_available" in normalized
    if available and observed_strong and support != "none":
        return 1.0, False, "observed_strong", [], True
    if available and support == "medium":
        return 0.5, False, "medium", [], False
    return 0.0, False, "none", [], False


def _reference_family_region_bias(region_type: str, family: str, strength: float, blocked: bool) -> float:
    if blocked:
        return 0.0
    region_sets = {
        "skin": SKIN_REFERENCE_REGIONS,
        "body_shape": BODY_SHAPE_REFERENCE_REGIONS,
        "garment": GARMENT_REFERENCE_REGIONS,
        "accessory": ACCESSORY_REFERENCE_REGIONS,
    }
    if region_type not in region_sets.get(family, set()):
        return 0.0
    return float(np.clip(strength, 0.0, 1.0))


def _identity_preservation_bias(region_type: str, identity_reference_strength: float, identity_reference_blocked: bool) -> float:
    if identity_reference_blocked or region_type not in IDENTITY_SENSITIVE_REGIONS:
        return 0.0
    core_scale = 1.0 if region_type in CORE_IDENTITY_REGIONS else 0.55
    return float(np.clip(identity_reference_strength * core_scale, 0.0, 1.0))



def _payload_untrusted_reason(payload: dict[str, object] | None) -> str:
    if not isinstance(payload, dict):
        return "missing"
    kind = str(payload.get("reference_kind", ""))
    confidence = _safe_float(payload.get("confidence"))
    evidence_score = _safe_float(payload.get("evidence_score"))
    min_conf, min_evidence = (0.65, 0.70) if kind == "identity_reference" else (0.58, 0.58)
    if not bool(payload.get("observed_directly", False)):
        return "not_observed_directly"
    if bool(payload.get("generated", False)):
        return "generated"
    if bool(payload.get("inferred", False)):
        return "inferred"
    descriptor = payload.get("descriptor", {})
    has_descriptor = isinstance(descriptor, dict) and bool(descriptor)
    if not bool(payload.get("patch_id")) and not bool(payload.get("patch_ref")) and not has_descriptor:
        return "missing_patch_cache"
    if confidence < min_conf:
        return "low_confidence"
    if evidence_score < min_evidence:
        return "low_evidence"
    return ""


def summarize_reference_payload_trace(transition_context: dict[str, object] | None) -> dict[str, object]:
    ctx = transition_context if isinstance(transition_context, dict) else {}
    payloads_raw = ctx.get("reference_patch_payloads", [])
    payloads = payloads_raw if isinstance(payloads_raw, list) else []
    payload_dicts = [payload for payload in payloads if isinstance(payload, dict)]
    expected = ctx.get("expected_reference_payload")
    expected_payload = expected if isinstance(expected, dict) else None
    descriptor = expected_payload.get("descriptor", {}) if expected_payload else {}
    descriptor_keys = sorted(str(key) for key in descriptor.keys()) if isinstance(descriptor, dict) else []
    untrusted_reason = _payload_untrusted_reason(expected_payload)
    trusted = bool(expected_payload is not None and untrusted_reason == "")
    trace_reasons = ctx.get("reference_payload_trace_reasons", [])
    return {
        "reference_payload_present": bool(payload_dicts),
        "expected_reference_payload_present": expected_payload is not None,
        "expected_reference_payload_kind": str(expected_payload.get("reference_kind", "")) if expected_payload else "",
        "expected_reference_payload_patch_id_present": bool(expected_payload.get("patch_id")) if expected_payload else False,
        "expected_reference_payload_descriptor_present": bool(descriptor_keys),
        "expected_reference_payload_descriptor_keys": descriptor_keys,
        "expected_reference_payload_confidence": _safe_float(expected_payload.get("confidence")) if expected_payload else 0.0,
        "expected_reference_payload_evidence_score": _safe_float(expected_payload.get("evidence_score")) if expected_payload else 0.0,
        "expected_reference_payload_observed_directly": bool(expected_payload.get("observed_directly", False)) if expected_payload else False,
        "expected_reference_payload_generated": bool(expected_payload.get("generated", False)) if expected_payload else False,
        "expected_reference_payload_inferred": bool(expected_payload.get("inferred", False)) if expected_payload else False,
        "reference_payload_trusted": trusted,
        "reference_payload_untrusted_reason": "" if trusted else untrusted_reason,
        "reference_payload_trace_reasons": list(trace_reasons) if isinstance(trace_reasons, list) else [],
    }


def extract_reference_payload_conditioning(transition_context: dict[str, object] | None) -> dict[str, object]:
    return summarize_reference_payload_trace(transition_context)


def _reference_material_shape(material: dict[str, object] | None) -> list[int]:
    if not isinstance(material, dict):
        return []
    rgb = material.get("rgb_patch")
    try:
        arr = np.asarray(rgb, dtype=np.float32)
    except (TypeError, ValueError):
        return []
    return [int(x) for x in arr.shape] if arr.ndim > 0 else []


def validate_reference_material_for_request(request: PatchSynthesisRequest, material_dict: object) -> dict[str, object]:
    if not isinstance(material_dict, dict):
        return {"valid": False, "reason": "material_missing", "shape": []}
    shape_value = _reference_material_shape(material_dict)
    try:
        entity_id, region_type = parse_region_id(request.region.region_id)
    except Exception:
        entity_id, region_type = "", str(getattr(request.region, "region_id", ""))
    expected_kind = reference_kind_for_region(region_type)
    material_kind = str(material_dict.get("reference_kind", ""))
    material_region = str(material_dict.get("canonical_region", ""))
    identity_subregion_compatible = (
        region_type in {"eyes", "mouth", "cheek"}
        and material_kind == "identity_reference"
        and material_region in CORE_IDENTITY_REGIONS
    )
    if expected_kind != "none" and material_kind != expected_kind and not identity_subregion_compatible:
        return {"valid": False, "reason": "kind_mismatch", "shape": shape_value}
    if material_region != str(region_type) and not identity_subregion_compatible:
        return {"valid": False, "reason": "region_mismatch", "shape": shape_value}
    if str(material_dict.get("entity_id", "")) != str(entity_id):
        return {"valid": False, "reason": "entity_mismatch", "shape": shape_value}
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    expected_payload = ctx.get("expected_reference_payload")
    if isinstance(expected_payload, dict):
        payload_pairs = (
            ("source_patch_id", "patch_id"),
            ("reference_kind", "reference_kind"),
            ("entity_id", "entity_id"),
            ("canonical_region", "canonical_region"),
        )
        for material_key, payload_key in payload_pairs:
            payload_value = expected_payload.get(payload_key)
            if payload_value is not None and str(material_dict.get(material_key, "")) != str(payload_value):
                return {"valid": False, "reason": "payload_material_mismatch", "shape": shape_value}
    source_frame_kind = str(material_dict.get("source_frame_kind", "unknown") or "unknown")
    source_is_input_frame = bool(material_dict.get("source_is_input_frame", False))
    immutable_i2v_anchor = bool(material_dict.get("immutable_i2v_anchor", False))
    if bool(material_dict.get("generated", False)) or bool(material_dict.get("inferred", False)) or source_frame_kind == "generated_runtime_frame":
        return {"valid": False, "reason": "generated_runtime_material_rejected", "shape": shape_value}
    if not source_is_input_frame:
        return {"valid": False, "reason": "non_input_frame_material_rejected", "shape": shape_value}
    if (not immutable_i2v_anchor) or source_frame_kind != "observed_input_frame":
        return {"valid": False, "reason": "missing_i2v_anchor", "shape": shape_value}
    if not bool(material_dict.get("observed_directly", False)):
        return {"valid": False, "reason": "generated_or_unobserved_material_rejected", "shape": shape_value}
    min_confidence, min_evidence = (0.65, 0.70) if material_kind == "identity_reference" else (0.58, 0.58)
    if _safe_float(material_dict.get("confidence")) < min_confidence:
        return {"valid": False, "reason": "low_confidence", "shape": shape_value}
    if _safe_float(material_dict.get("evidence_score")) < min_evidence:
        return {"valid": False, "reason": "low_evidence", "shape": shape_value}
    if not (len(shape_value) == 3 and shape_value[2] == 3 and shape_value[0] > 0 and shape_value[1] > 0):
        return {"valid": False, "reason": "invalid_rgb_shape", "shape": shape_value}
    if not bool(material_dict.get("material_trusted", False)):
        return {"valid": False, "reason": "material_untrusted", "shape": shape_value}
    return {"valid": True, "reason": "", "shape": shape_value}


def summarize_reference_material_trace(transition_context: dict[str, object] | None, request: PatchSynthesisRequest | None = None) -> dict[str, object]:
    ctx = transition_context if isinstance(transition_context, dict) else {}
    material = ctx.get("expected_reference_patch_material")
    material_dict = material if isinstance(material, dict) else None
    payload_trace = summarize_reference_payload_trace(ctx)
    trace_reasons = ctx.get("reference_patch_material_trace_reasons", [])
    if request is not None:
        validation = validate_reference_material_for_request(request, material_dict)
        shape_value = list(validation.get("shape", [])) if isinstance(validation.get("shape", []), list) else []
        validation_valid = bool(validation.get("valid", False))
        material_validated = validation_valid
        payload_trusted = bool(payload_trace.get("reference_payload_trusted", False))
        used = bool(validation_valid and payload_trusted)
        material_trusted = bool(validation_valid and payload_trusted)
        if used:
            missing_reason = ""
        elif not validation_valid:
            missing_reason = str(validation.get("reason", "material_invalid") or "material_invalid")
        elif not payload_trusted:
            payload_reason = str(payload_trace.get("reference_payload_untrusted_reason", "") or "expected_reference_payload_missing")
            missing_reason = "expected_reference_payload_missing" if payload_reason == "missing" else payload_reason
        else:
            missing_reason = "material_missing"
    else:
        shape_value = _reference_material_shape(material_dict)
        arr_valid = len(shape_value) == 3 and shape_value[2] == 3 and shape_value[0] > 0 and shape_value[1] > 0
        material_validated = bool(material_dict and material_dict.get("material_trusted", False) and arr_valid)
        payload_trusted = bool(payload_trace.get("reference_payload_trusted", False))
        material_trusted = bool(material_validated and payload_trusted)
        used = bool(material_trusted)
        if used:
            missing_reason = ""
        elif material_dict is None:
            missing_reason = "material_missing"
        elif not material_validated:
            missing_reason = str(material_dict.get("material_missing_reason", "material_untrusted") or "material_untrusted")
        elif not payload_trusted:
            payload_reason = str(payload_trace.get("reference_payload_untrusted_reason", "") or "expected_reference_payload_missing")
            missing_reason = "expected_reference_payload_missing" if payload_reason == "missing" else payload_reason
        else:
            missing_reason = "material_untrusted"
    source_frame_kind = str(material_dict.get("source_frame_kind", "unknown")) if material_dict else "unknown"
    source_is_input_frame = bool(material_dict.get("source_is_input_frame", False)) if material_dict else False
    immutable_i2v_anchor = bool(material_dict.get("immutable_i2v_anchor", False)) if material_dict else False
    source_frame_index = int(material_dict.get("source_frame_index", 0) or 0) if material_dict else 0
    from_generated = bool(material_dict is not None and (source_frame_kind == "generated_runtime_frame" or (not source_is_input_frame) or bool(material_dict.get("generated", False)) or bool(material_dict.get("inferred", False))))
    from_input = bool(used and source_is_input_frame and immutable_i2v_anchor and source_frame_kind == "observed_input_frame")
    return {
        **payload_trace,
        "reference_patch_material_present": material_dict is not None,
        "reference_patch_material_validated": material_validated,
        "reference_patch_material_trusted": material_trusted,
        "reference_patch_material_used": used,
        "reference_patch_material_source": str(material_dict.get("material_source", "unknown")) if material_dict else "none",
        "reference_patch_material_missing_reason": missing_reason,
        "reference_patch_material_shape": shape_value,
        "reference_patch_material_kind": str(material_dict.get("reference_kind", "")) if material_dict else "",
        "reference_patch_material_confidence": _safe_float(material_dict.get("confidence")) if material_dict else 0.0,
        "reference_patch_material_evidence_score": _safe_float(material_dict.get("evidence_score")) if material_dict else 0.0,
        "reference_patch_material_trace_reasons": list(trace_reasons) if isinstance(trace_reasons, list) else [],
        "i2v_reference_contract_version": "i2v_first_frame_reference_v1",
        "reference_material_from_input_frame": from_input,
        "reference_material_from_generated_frame": from_generated,
        "reference_patch_material_source_frame_kind": source_frame_kind,
        "reference_patch_material_source_frame_index": source_frame_index,
        "reference_patch_material_immutable_i2v_anchor": immutable_i2v_anchor,
        "reference_patch_material_source_is_input_frame": source_is_input_frame,
    }


def extract_reference_material_conditioning(transition_context: dict[str, object] | None, request: PatchSynthesisRequest | None = None) -> dict[str, object]:
    return summarize_reference_material_trace(transition_context, request=request)


def _resize_hw3_nearest(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if arr.ndim != 3 or arr.shape[2] != 3 or h <= 0 or w <= 0:
        return np.zeros((h, w, 3), dtype=np.float32)
    ys = np.linspace(0, arr.shape[0] - 1, h).round().astype(int)
    xs = np.linspace(0, arr.shape[1] - 1, w).round().astype(int)
    return np.clip(arr[ys][:, xs], 0.0, 1.0).astype(np.float32)


def _reference_material_tensors(request: PatchSynthesisRequest, roi_before: np.ndarray, material_cond: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = roi_before.shape
    zeros_rgb = np.zeros_like(roi_before, dtype=np.float32)
    zeros_mask = np.zeros((h, w, 1), dtype=np.float32)
    if not bool(material_cond.get("reference_patch_material_used", False)):
        return zeros_rgb, zeros_mask, zeros_mask.copy()
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    material = ctx.get("expected_reference_patch_material")
    if not isinstance(material, dict):
        return zeros_rgb, zeros_mask, zeros_mask.copy()
    try:
        raw_rgb = np.asarray(material.get("rgb_patch"), dtype=np.float32)
    except (TypeError, ValueError):
        return zeros_rgb, zeros_mask, zeros_mask.copy()
    if raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3 or raw_rgb.shape[0] <= 0 or raw_rgb.shape[1] <= 0:
        return zeros_rgb, zeros_mask, zeros_mask.copy()
    rgb = _resize_hw3_nearest(raw_rgb, (h, w))
    raw_mask = material.get("alpha_or_mask")
    mask = None
    if raw_mask is not None:
        try:
            mask_arr = np.asarray(raw_mask, dtype=np.float32)
            if mask_arr.ndim == 2:
                mask_arr = mask_arr[..., None]
            if mask_arr.ndim == 3:
                if mask_arr.shape[2] != 1:
                    mask_arr = np.mean(mask_arr, axis=2, keepdims=True)
                mask = _resize_hw3_nearest(np.repeat(mask_arr, 3, axis=2), (h, w))[..., :1]
        except (TypeError, ValueError):
            mask = None
    if mask is None:
        mask = np.ones((h, w, 1), dtype=np.float32)
    validity = np.ones((h, w, 1), dtype=np.float32)
    return rgb.astype(np.float32), np.clip(mask, 0.0, 1.0).astype(np.float32), validity


def _descriptor_numeric_signal(descriptor: object) -> float:
    values: list[float] = []
    def collect(value: object) -> None:
        if len(values) >= 32:
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            values.append(float(value))
        elif isinstance(value, (list, tuple)):
            for item in value[:16]:
                collect(item)
        elif isinstance(value, dict):
            for item in list(value.values())[:16]:
                collect(item)
    collect(descriptor)
    if not values:
        return 0.0
    return float(np.clip(abs(float(np.mean(values))), 0.0, 1.0))

def extract_memory_bundle_conditioning_from_context(transition_context: dict[str, object] | None) -> dict[str, object]:
    trace = summarize_memory_bundle_trace(transition_context if isinstance(transition_context, dict) else {})
    support_map = {"none": 0.0, "weak": 0.33, "medium": 0.66, "strong": 1.0}
    support_level = str(trace.get("memory_support_level", "none")).strip().lower()
    retrieval_reasons = [str(x).lower() for x in trace.get("memory_bundle_retrieval_reasons", []) if isinstance(x, str)]
    hidden_type = str(trace.get("memory_bundle_hidden_type", "none")).strip().lower()
    reveal_lifecycle = str(trace.get("memory_bundle_reveal_lifecycle", "unknown")).strip().lower()
    reveal_like_hidden = hidden_type in {"revealed", "revealed_history", "newly_revealed"}
    is_revealed_history = hidden_type == "revealed_history"
    reason_has_newly_revealed = any("newly_revealed" in reason for reason in retrieval_reasons)
    reason_has_low_evidence = any("low_evidence" in reason for reason in retrieval_reasons)
    newly_revealed_signal = reveal_lifecycle == "newly_revealed" or reason_has_newly_revealed
    low_evidence_newly_revealed = (
        newly_revealed_signal and (support_level in {"none", "weak"} or reason_has_low_evidence)
    )
    if not low_evidence_newly_revealed and hidden_type == "newly_revealed":
        low_evidence_newly_revealed = support_level in {"none", "weak"} or reason_has_low_evidence
    reference_payload_cond = extract_reference_payload_conditioning(transition_context)
    reference_material_cond = extract_reference_material_conditioning(transition_context)
    active_hidden_support = bool(trace.get("memory_bundle_hidden_support_active", False) and not reveal_like_hidden)
    has_identity_reference = bool(trace.get("memory_bundle_has_identity_reference", False))
    identity_strength, identity_blocked, identity_source, identity_block_reasons, identity_used = _identity_reference_strength(
        has_identity_reference=has_identity_reference,
        support_level=support_level,
        retrieval_reasons=retrieval_reasons,
    )
    family_conditions: dict[str, dict[str, object]] = {}
    for family in REFERENCE_FAMILIES:
        has_reference = bool(trace.get(f"memory_bundle_has_{family}_reference", False))
        strength, blocked, source, block_reasons, used = _reference_family_strength(
            family=family,
            has_reference=has_reference,
            support_level=support_level,
            retrieval_reasons=retrieval_reasons,
        )
        family_conditions[family] = {
            "has": has_reference,
            "strength": strength,
            "blocked": blocked,
            "source": source,
            "block_reasons": block_reasons,
            "used": used,
        }
    return {
        "memory_bundle_present": bool(trace.get("memory_bundle_present", False)),
        "memory_support_level": support_level,
        "memory_bundle_support_value": float(support_map.get(support_level, 0.0)),
        "has_current_reuse": bool(trace.get("memory_bundle_has_current_reuse", False)),
        "has_identity_reference": has_identity_reference,
        "has_skin_reference": bool(family_conditions["skin"]["has"]),
        "has_body_shape_reference": bool(family_conditions["body_shape"]["has"]),
        "has_garment_reference": bool(family_conditions["garment"]["has"]),
        "has_accessory_reference": bool(family_conditions["accessory"]["has"]),
        "has_appearance_reference": bool(trace.get("memory_bundle_has_appearance_reference", False)),
        "has_hidden_slot": bool(trace.get("memory_bundle_has_hidden_slot", False)),
        "reveal_lifecycle": reveal_lifecycle,
        "retrieval_reasons": retrieval_reasons,
        "identity_reference_strength": identity_strength,
        "identity_reference_used": identity_used,
        "identity_reference_blocked": identity_blocked,
        "identity_reference_source": identity_source,
        "identity_reference_block_reasons": identity_block_reasons,
        "skin_reference_strength": family_conditions["skin"]["strength"],
        "skin_reference_used": family_conditions["skin"]["used"],
        "skin_reference_blocked": family_conditions["skin"]["blocked"],
        "skin_reference_source": family_conditions["skin"]["source"],
        "skin_reference_block_reasons": family_conditions["skin"]["block_reasons"],
        "body_shape_reference_strength": family_conditions["body_shape"]["strength"],
        "body_shape_reference_used": family_conditions["body_shape"]["used"],
        "body_shape_reference_blocked": family_conditions["body_shape"]["blocked"],
        "body_shape_reference_source": family_conditions["body_shape"]["source"],
        "body_shape_reference_block_reasons": family_conditions["body_shape"]["block_reasons"],
        "garment_reference_strength": family_conditions["garment"]["strength"],
        "garment_reference_used": family_conditions["garment"]["used"],
        "garment_reference_blocked": family_conditions["garment"]["blocked"],
        "garment_reference_source": family_conditions["garment"]["source"],
        "garment_reference_block_reasons": family_conditions["garment"]["block_reasons"],
        "accessory_reference_strength": family_conditions["accessory"]["strength"],
        "accessory_reference_used": family_conditions["accessory"]["used"],
        "accessory_reference_blocked": family_conditions["accessory"]["blocked"],
        "accessory_reference_source": family_conditions["accessory"]["source"],
        "accessory_reference_block_reasons": family_conditions["accessory"]["block_reasons"],
        "memory_bundle_has_current_reuse": bool(trace.get("memory_bundle_has_current_reuse", False)),
        "memory_bundle_has_identity_reference": has_identity_reference,
        "memory_bundle_has_appearance_reference": bool(trace.get("memory_bundle_has_appearance_reference", False)),
        "memory_bundle_has_skin_reference": bool(family_conditions["skin"]["has"]),
        "memory_bundle_has_body_shape_reference": bool(family_conditions["body_shape"]["has"]),
        "memory_bundle_has_garment_reference": bool(family_conditions["garment"]["has"]),
        "memory_bundle_has_accessory_reference": bool(family_conditions["accessory"]["has"]),
        "memory_bundle_has_active_hidden_support": active_hidden_support,
        "memory_bundle_is_revealed_history": is_revealed_history,
        "memory_bundle_reveal_lifecycle": reveal_lifecycle,
        "memory_bundle_has_hidden_slot": bool(trace.get("memory_bundle_has_hidden_slot", False)),
        "memory_bundle_hidden_type": hidden_type,
        "memory_bundle_hidden_support_active": active_hidden_support,
        "memory_bundle_retrieval_reasons": retrieval_reasons,
        "memory_bundle_low_evidence_newly_revealed": low_evidence_newly_revealed,
        **reference_payload_cond,
        **reference_material_cond,
    }


def extract_memory_bundle_conditioning(request: PatchSynthesisRequest) -> dict[str, object]:
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    cond = extract_memory_bundle_conditioning_from_context(ctx)
    cond.update(extract_reference_material_conditioning(ctx, request=request))
    return cond


def apply_memory_bundle_conditioning_to_vectors(
    memory_cond: np.ndarray,
    appearance_cond: np.ndarray,
    bundle_cond: dict[str, object],
    *,
    region_id: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    mem_vec = np.asarray(memory_cond, dtype=np.float32).reshape(-1).copy()
    appearance = np.asarray(appearance_cond, dtype=np.float32).reshape(-1).copy()
    support_value = float(bundle_cond.get("memory_bundle_support_value", 0.0))
    has_current_reuse = bool(bundle_cond.get("memory_bundle_has_current_reuse", False))
    has_identity_reference = bool(bundle_cond.get("memory_bundle_has_identity_reference", False))
    has_appearance_reference = bool(bundle_cond.get("memory_bundle_has_appearance_reference", False))
    has_active_hidden = bool(bundle_cond.get("memory_bundle_has_active_hidden_support", False))
    is_revealed_history = bool(bundle_cond.get("memory_bundle_is_revealed_history", False))
    low_evidence_newly_revealed = bool(bundle_cond.get("memory_bundle_low_evidence_newly_revealed", False))
    identity_strength = float(bundle_cond.get("identity_reference_strength", 0.0))
    identity_blocked = bool(bundle_cond.get("identity_reference_blocked", False))
    if mem_vec.size > 6:
        mem_vec[6] = float(np.clip(max(float(mem_vec[6]), support_value), 0.0, 1.0))
    if mem_vec.size > 7:
        mem_vec[7] = float(np.clip(max(float(mem_vec[7]), 1.0 if has_current_reuse else 0.0), 0.0, 1.0))
    if mem_vec.size > 8:
        skin_usable = bool(bundle_cond.get("skin_reference_used", False)) or float(bundle_cond.get("skin_reference_strength", 0.0)) > 0.0
        body_shape_usable = bool(bundle_cond.get("body_shape_reference_used", False)) or float(bundle_cond.get("body_shape_reference_strength", 0.0)) > 0.0
        garment_usable = bool(bundle_cond.get("garment_reference_used", False)) or float(bundle_cond.get("garment_reference_strength", 0.0)) > 0.0
        accessory_usable = bool(bundle_cond.get("accessory_reference_used", False)) or float(bundle_cond.get("accessory_reference_strength", 0.0)) > 0.0
        usable_reference_present = (
            (has_identity_reference and not identity_blocked)
            or has_appearance_reference
            or skin_usable
            or body_shape_usable
            or garment_usable
            or accessory_usable
        )
        mem_vec[8] = float(np.clip(max(float(mem_vec[8]), 0.35 if usable_reference_present else 0.0), 0.0, 1.0))
    if mem_vec.size > 9:
        mem_vec[9] = float(np.clip(max(float(mem_vec[9]), 1.0 if has_active_hidden else 0.0) - (0.35 if is_revealed_history else 0.0), 0.0, 1.0))
    try:
        region_type = parse_region_id(region_id)[1] if isinstance(region_id, str) and region_id.strip() else "unknown"
    except Exception:
        region_type = "unknown"
    if not region_type:
        region_type = "unknown"
    if has_current_reuse and appearance.size >= 6:
        appearance[3:6] = appearance[3:6] * np.float32(0.94)
    identity_bias = _identity_preservation_bias(region_type, identity_strength, identity_blocked)
    if identity_bias > 0.0:
        if mem_vec.size > 2:
            mem_vec[2] = float(np.clip(max(float(mem_vec[2]), 0.35 + 0.55 * identity_bias), 0.0, 1.0))
        if mem_vec.size > 8:
            mem_vec[8] = float(np.clip(max(float(mem_vec[8]), 0.40 + 0.18 * identity_bias), 0.0, 1.0))
        if mem_vec.size > 9:
            mem_vec[9] = float(np.clip(float(mem_vec[9]) + 0.16 * identity_bias, 0.0, 1.0))
        if appearance.size > 7:
            appearance[6] = float(np.clip(appearance[6] + 0.12 * identity_bias, 0.0, 1.0))
            appearance[7] = float(np.clip(max(0.0, appearance[7] - 0.08 * identity_bias), 0.0, 1.0))
    elif identity_blocked and appearance.size > 7:
        appearance[7] = float(np.clip(appearance[7] + 0.04, 0.0, 1.0))
    payload_present = bool(bundle_cond.get("expected_reference_payload_present", False))
    payload_trusted = bool(bundle_cond.get("reference_payload_trusted", False))
    if payload_trusted:
        payload_confidence = float(bundle_cond.get("expected_reference_payload_confidence", 0.0))
        payload_evidence = float(bundle_cond.get("expected_reference_payload_evidence_score", 0.0))
        if mem_vec.size > 8:
            mem_vec[8] = float(np.clip(max(float(mem_vec[8]), 0.45), 0.0, 1.0))
        if appearance.size > 7:
            appearance[6] = float(np.clip(appearance[6] + 0.05 * payload_confidence, 0.0, 1.0))
            appearance[7] = float(np.clip(max(0.0, appearance[7] - 0.03 * payload_evidence), 0.0, 1.0))
    elif payload_present and appearance.size > 7:
        appearance[7] = float(np.clip(appearance[7] + 0.035, 0.0, 1.0))
    if bool(bundle_cond.get("reference_patch_material_used", False)) and mem_vec.size > 8:
        mem_vec[8] = float(np.clip(max(float(mem_vec[8]), 0.92), 0.0, 1.0))
    elif bool(bundle_cond.get("reference_patch_material_present", False)) and not bool(bundle_cond.get("reference_patch_material_used", False)) and appearance.size > 7:
        appearance[7] = float(np.clip(appearance[7] + 0.04, 0.0, 1.0))
    skin_bias = _reference_family_region_bias(
        region_type,
        "skin",
        float(bundle_cond.get("skin_reference_strength", 0.0)),
        bool(bundle_cond.get("skin_reference_blocked", False)),
    )
    body_bias = _reference_family_region_bias(
        region_type,
        "body_shape",
        float(bundle_cond.get("body_shape_reference_strength", 0.0)),
        bool(bundle_cond.get("body_shape_reference_blocked", False)),
    )
    garment_bias = _reference_family_region_bias(
        region_type,
        "garment",
        float(bundle_cond.get("garment_reference_strength", 0.0)),
        bool(bundle_cond.get("garment_reference_blocked", False)),
    )
    accessory_bias = _reference_family_region_bias(
        region_type,
        "accessory",
        float(bundle_cond.get("accessory_reference_strength", 0.0)),
        bool(bundle_cond.get("accessory_reference_blocked", False)),
    )
    if skin_bias > 0.0 and appearance.size > 7:
        appearance[6] = float(np.clip(appearance[6] + 0.08 * skin_bias, 0.0, 1.0))
        appearance[7] = float(np.clip(max(0.0, appearance[7] - 0.035 * skin_bias), 0.0, 1.0))
    if body_bias > 0.0:
        if mem_vec.size > 5:
            mem_vec[5] = float(np.clip(max(float(mem_vec[5]), 0.42 + 0.36 * body_bias), 0.0, 1.0))
        if mem_vec.size > 8:
            mem_vec[8] = float(np.clip(max(float(mem_vec[8]), 0.38 + 0.20 * body_bias), 0.0, 1.0))
        if appearance.size > 7:
            appearance[7] = float(np.clip(max(0.0, appearance[7] - 0.06 * body_bias), 0.0, 1.0))
    if garment_bias > 0.0 and appearance.size > 7:
        appearance[6] = float(np.clip(appearance[6] + 0.1 * garment_bias, 0.0, 1.0))
        appearance[7] = float(np.clip(max(0.0, appearance[7] - 0.045 * garment_bias), 0.0, 1.0))
    if accessory_bias > 0.0 and appearance.size > 7:
        appearance[6] = float(np.clip(appearance[6] + 0.075 * accessory_bias, 0.0, 1.0))
        appearance[7] = float(np.clip(max(0.0, appearance[7] - 0.03 * accessory_bias), 0.0, 1.0))
    if any(bool(bundle_cond.get(f"{family}_reference_blocked", False)) for family in REFERENCE_FAMILIES) and appearance.size > 7:
        appearance[7] = float(np.clip(appearance[7] + 0.025, 0.0, 1.0))
    if low_evidence_newly_revealed and appearance.size > 7:
        appearance[7] = float(np.clip(appearance[7] + 0.08, 0.0, 1.0))
    return mem_vec, appearance


def _semantic_embed(region_id: str, profile: RenderConditioningProfile) -> np.ndarray:
    _, region_type = parse_region_id(region_id)
    if region_type in ROI_FAMILIES["face_expression"]:
        family = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif region_type in ROI_FAMILIES["torso_reveal"]:
        family = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        family = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array(
        [
            family[0],
            family[1],
            family[2],
            float(profile.edit_strength),
            float(profile.memory_dependency),
            float(profile.seam_strictness),
            1.0 if profile.reveal_like else 0.0,
            float(profile.context_support),
        ],
        dtype=np.float32,
    )


def _delta_features(request: PatchSynthesisRequest, profile: RenderConditioningProfile) -> tuple[np.ndarray, np.ndarray]:
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    human_contract = LearnedHumanStateContract.from_metadata(ctx.get("learned_human_state_contract"))
    learned_contract = LearnedTemporalTransitionContract.from_metadata(ctx.get("learned_temporal_contract"))
    delta = ctx.get("graph_delta")
    transition_phase = str(ctx.get("transition_phase", "")).lower()
    step_index = _safe_float(ctx.get("step_index", 0.0))

    delta_vec = np.zeros((DELTA_DIM,), dtype=np.float32)
    if delta is not None:
        numeric_payloads = (delta.pose_deltas, delta.expression_deltas, delta.garment_deltas, delta.interaction_deltas)
        magnitude = sum(abs(float(v)) for payload in numeric_payloads for v in payload.values() if isinstance(v, (int, float)))
        newly_revealed = {r.region_id for r in getattr(delta, "newly_revealed_regions", [])}
        newly_occluded = {r.region_id for r in getattr(delta, "newly_occluded_regions", [])}
        delta_vec = np.array(
            [
                min(1.0, len(delta.pose_deltas) / 3.0),
                min(1.0, len(delta.expression_deltas) / 3.0),
                min(1.0, len(delta.garment_deltas) / 3.0),
                min(1.0, len(delta.semantic_reasons) / 4.0),
                min(1.0, len(delta.affected_regions) / 6.0),
                min(1.0, len(delta.newly_revealed_regions) / 4.0),
                min(1.0, len(delta.newly_occluded_regions) / 4.0),
                1.0 if request.region.region_id in newly_revealed else 0.0,
                1.0 if request.region.region_id in newly_occluded else 0.0,
                min(1.0, magnitude / 2.5),
                1.0 if profile.reveal_like else 0.0,
                1.0 if profile.transition_mode == "visibility_occlusion" else 0.0,
            ],
            dtype=np.float32,
        )

    planner = np.array(
        [
            min(1.0, step_index / 10.0),
            1.0 if transition_phase in {"prepare", "pre", "warmup"} else 0.0,
            1.0 if transition_phase in {"motion", "mid", "execute", "transition"} else 0.0,
            1.0 if transition_phase in {"settle", "post", "cooldown", "stabilize"} else 0.0,
            1.0 if profile.transition_mode == "garment_reveal" else 0.0,
            1.0 if profile.transition_mode == "pose_exposure" else 0.0,
            1.0 if profile.transition_mode == "expression_refine" else 0.0,
            1.0 if profile.transition_mode == "visibility_occlusion" else 0.0,
            1.0 if profile.transition_mode == "stable" else 0.0,
            1.0 if request.scene_state.global_context.source_type == "single_image" else 0.0,
        ],
        dtype=np.float32,
    )
    if learned_contract is not None and learned_contract.is_learned_primary:
        delta_vec[5] = max(delta_vec[5], float(learned_contract.reveal_score))
        delta_vec[6] = max(delta_vec[6], float(learned_contract.occlusion_score))
        delta_vec[10] = max(delta_vec[10], 1.0 if learned_contract.reveal_score >= 0.5 else 0.0)
        delta_vec[11] = max(delta_vec[11], 1.0 if learned_contract.occlusion_score >= 0.5 else 0.0)
        planner[1] = 1.0 if learned_contract.predicted_phase == "prepare" else planner[1]
        planner[2] = 1.0 if learned_contract.predicted_phase in {"transition", "contact_or_reveal"} else planner[2]
        planner[3] = 1.0 if learned_contract.predicted_phase == "stabilize" else planner[3]
    return delta_vec, planner



def extract_region_metadata_conditioning_from_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    metadata = metadata if isinstance(metadata, dict) else {}
    used = bool(metadata) and _safe_float(metadata.get("metadata_completeness_score"), 0.0) > 0.0
    roi_source = str(metadata.get("roi_source", "unknown")) if metadata else "unknown"
    source_node_type = str(metadata.get("source_node_type", "unknown")) if metadata else "unknown"
    bbox_raw = metadata.get("bbox_xywh", {}) if metadata else {}
    if isinstance(bbox_raw, dict):
        bbox_w = _safe_float(bbox_raw.get("w"), 0.0)
        bbox_h = _safe_float(bbox_raw.get("h"), 0.0)
    elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
        bbox_w = _safe_float(bbox_raw[2], 0.0)
        bbox_h = _safe_float(bbox_raw[3], 0.0)
    else:
        bbox_w = 0.0
        bbox_h = 0.0
    area_from_bbox = min(1.0, bbox_w * bbox_h)
    frame_size = metadata.get("mask_frame_size") if metadata else None
    frame_area = 0.0
    if isinstance(frame_size, (tuple, list)) and len(frame_size) == 2:
        frame_area = max(0.0, _safe_float(frame_size[0]) * _safe_float(frame_size[1]))
    if metadata.get("mask_ref") and frame_area > 0.0:
        mask_area_ratio = min(1.0, max(0.0, _safe_float(metadata.get("mask_pixel_count"), 0.0) / frame_area))
    else:
        mask_area_ratio = area_from_bbox if metadata.get("mask_ref") else 0.0
    feature_values = {
        "has_parser_mask": 1.0 if metadata.get("mask_ref") and roi_source == "parser_mask_bbox" else 0.0,
        "mask_area_ratio": mask_area_ratio,
        "source_confidence": min(1.0, _safe_float(metadata.get("source_confidence"), 0.0)),
        "is_identity_sensitive": 1.0 if metadata.get("is_identity_sensitive") else 0.0,
        "is_garment_region": 1.0 if metadata.get("is_garment_region") else 0.0,
        "is_body_region": 1.0 if metadata.get("is_body_region") else 0.0,
        "is_face_region": 1.0 if metadata.get("is_face_region") else 0.0,
        "is_reveal": 1.0 if metadata.get("newly_revealed") or str(metadata.get("reveal_mode", "none")) != "none" else 0.0,
        "is_newly_occluded": 1.0 if metadata.get("newly_occluded") else 0.0,
        "memory_reliable_for_reuse": 1.0 if metadata.get("memory_reliable_for_reuse") else 0.0,
        "memory_suitable_for_reveal": 1.0 if metadata.get("memory_suitable_for_reveal") else 0.0,
        "metadata_completeness_score": min(1.0, _safe_float(metadata.get("metadata_completeness_score"), 0.0)),
        "evidence_strength_score": min(1.0, _safe_float(metadata.get("evidence_strength_score"), 0.0)),
        "roi_source_parser_mask_bbox": 1.0 if roi_source == "parser_mask_bbox" else 0.0,
        "roi_source_body_part_keypoints": 1.0 if roi_source == "body_part_keypoints" else 0.0,
        "roi_source_garment_coverage": 1.0 if roi_source == "garment_coverage" else 0.0,
        "roi_source_person_bbox_fallback": 1.0 if roi_source == "person_bbox_fallback" else 0.0,
        "source_node_body_part": 1.0 if source_node_type == "body_part" else 0.0,
        "source_node_garment": 1.0 if source_node_type == "garment" else 0.0,
        "source_node_face_region": 1.0 if source_node_type == "face_region" else 0.0,
        "source_node_canonical_region": 1.0 if source_node_type == "canonical_region" else 0.0,
        "source_node_fallback": 1.0 if source_node_type == "fallback" else 0.0,
    }
    return {
        "region_metadata_used": used,
        "metadata_completeness_score": feature_values["metadata_completeness_score"],
        "evidence_strength_score": feature_values["evidence_strength_score"],
        "metadata_feature_keys": [k for k, v in feature_values.items() if float(v) != 0.0],
        "mask_ref_present": bool(metadata.get("mask_ref")),
        "roi_source": roi_source,
        "source_node_type": source_node_type,
        "mask_kind": str(metadata.get("mask_kind", "")),
        "feature_values": feature_values,
    }


def extract_region_metadata_conditioning(request: PatchSynthesisRequest) -> dict[str, object]:
    metadata = request.region_metadata if isinstance(getattr(request, "region_metadata", {}), dict) else {}
    return extract_region_metadata_conditioning_from_metadata(metadata)


def extract_i2v_action_conditioning(request: PatchSynthesisRequest) -> dict[str, object]:
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    _, region_type = parse_region_id(request.region.region_id)
    action_phase = str(ctx.get("i2v_action_phase", "") or "stable_idle").strip().lower()
    mode_map = ctx.get("i2v_region_transition_mode", {})
    region_mode = ""
    if isinstance(mode_map, dict):
        region_mode = str(mode_map.get(request.region.region_id) or mode_map.get(region_type) or "").strip().lower()
    if not region_mode:
        region_mode = str(ctx.get("region_transition_mode", "") or "stable").strip().lower()
    active = action_phase not in {"", "stable", "stable_idle", "idle"}
    strength = 0.20 if not active else 0.58
    motion_x = 0.0
    motion_y = 0.0
    expression = 0.0
    pose = 0.0
    garment = 0.0
    if action_phase == "head_turn":
        strength, motion_x, pose = 0.72, 0.35, 0.25
    elif action_phase == "expression_smile":
        strength, expression = 0.68, 0.8
    elif action_phase == "torso_shift":
        strength, motion_x, pose = 0.62, 0.2, 0.7
    elif action_phase == "arm_raise":
        strength, motion_y, pose = 0.82, -0.6, 0.92
    elif action_phase == "garment_reveal_or_adjust":
        strength, garment, pose = 0.7, 0.88, 0.35
    return {
        "i2v_action_phase": action_phase,
        "i2v_action_active": active,
        "i2v_region_action_mode": region_mode,
        "i2v_action_strength": float(np.clip(strength, 0.0, 1.0)),
        "i2v_motion_direction_x": float(np.clip(motion_x, -1.0, 1.0)),
        "i2v_motion_direction_y": float(np.clip(motion_y, -1.0, 1.0)),
        "i2v_expression_bias": float(np.clip(expression, 0.0, 1.0)),
        "i2v_pose_bias": float(np.clip(pose, 0.0, 1.0)),
        "i2v_garment_bias": float(np.clip(garment, 0.0, 1.0)),
    }


def apply_region_metadata_conditioning_to_vectors(
    memory_cond: np.ndarray,
    appearance_cond: np.ndarray,
    region_metadata: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    mem_vec = np.asarray(memory_cond, dtype=np.float32).reshape(-1).copy()
    appearance = np.asarray(appearance_cond, dtype=np.float32).reshape(-1).copy()
    metadata_cond = extract_region_metadata_conditioning_from_metadata(region_metadata)
    metadata_values = metadata_cond["feature_values"] if isinstance(metadata_cond.get("feature_values"), dict) else {}
    if metadata_cond.get("region_metadata_used"):
        if mem_vec.size > 6:
            mem_vec[6] = max(
                float(mem_vec[6]),
                0.20 * _safe_float(metadata_values.get("metadata_completeness_score")) + 0.30 * _safe_float(metadata_values.get("evidence_strength_score")),
            )
        if mem_vec.size > 7:
            mem_vec[7] = max(float(mem_vec[7]), _safe_float(metadata_values.get("has_parser_mask")))
        if mem_vec.size > 8:
            mem_vec[8] = max(float(mem_vec[8]), _safe_float(metadata_values.get("source_confidence")))
        if mem_vec.size > 9:
            mem_vec[9] = min(
                1.0,
                float(mem_vec[9])
                + 0.35 * _safe_float(metadata_values.get("is_reveal"))
                + 0.20 * _safe_float(metadata_values.get("memory_suitable_for_reveal")),
            )
        if appearance.size > 6:
            appearance[6] = min(
                1.0,
                float(appearance[6])
                + 0.05 * _safe_float(metadata_values.get("is_identity_sensitive"))
                + 0.03 * _safe_float(metadata_values.get("is_garment_region")),
            )
        if appearance.size > 7:
            appearance[7] = min(
                1.0,
                float(appearance[7])
                + 0.08 * _safe_float(metadata_values.get("mask_area_ratio"))
                + 0.04 * _safe_float(metadata_values.get("is_newly_occluded")),
            )
    return mem_vec.astype(np.float32), appearance.astype(np.float32)

def _graph_features(request: PatchSynthesisRequest) -> np.ndarray:
    g = request.scene_state
    graph_emb = request.graph_encoding.graph_embedding if request.graph_encoding else []
    persons = float(len(g.persons))
    objects = float(len(g.objects))
    relations = float(len(g.relations))
    fps = float(g.global_context.fps or 0.0)
    frame_w, frame_h = g.global_context.frame_size if g.global_context.frame_size else (0, 0)
    graph_stats = np.asarray(graph_emb[:16], dtype=np.float32) if graph_emb else np.zeros((0,), dtype=np.float32)
    graph_mean = float(np.mean(graph_stats)) if graph_stats.size else 0.0
    graph_std = float(np.std(graph_stats)) if graph_stats.size else 0.0
    return np.array(
        [
            min(1.0, persons / 3.0),
            min(1.0, objects / 4.0),
            min(1.0, relations / 8.0),
            min(1.0, fps / 30.0),
            min(1.0, frame_w / 1920.0),
            min(1.0, frame_h / 1080.0),
            graph_mean,
            graph_std,
        ],
        dtype=np.float32,
    )


def _memory_and_appearance_features(request: PatchSynthesisRequest, roi_before: np.ndarray, profile: RenderConditioningProfile) -> tuple[np.ndarray, np.ndarray]:
    memory_summary = request.memory_summary if isinstance(request.memory_summary, dict) else {}
    channels = request.memory_channels if isinstance(request.memory_channels, dict) else {}
    hidden = channels.get("hidden_regions", {}) if isinstance(channels.get("hidden_regions", {}), dict) else {}
    garments = channels.get("garments", {}) if isinstance(channels.get("garments", {}), dict) else {}
    identity = channels.get("identity", {}) if isinstance(channels.get("identity", {}), dict) else {}
    retrieval = request.retrieval_summary if isinstance(request.retrieval_summary, dict) else {}
    summary = retrieval.get("summary", {}) if isinstance(retrieval.get("summary", {}), dict) else {}
    top_breakdown = retrieval.get("top_score_breakdown", {}) if isinstance(retrieval.get("top_score_breakdown", {}), dict) else {}
    hidden_evidence = (
        _safe_float(top_breakdown.get("same_entity_bonus"))
        + _safe_float(top_breakdown.get("reveal_compatibility"))
        + _safe_float(top_breakdown.get("visibility_lifecycle_compatibility"))
    )
    mem_vec = np.array(
        [
            min(1.0, len(hidden) / 4.0),
            min(1.0, len(garments) / 4.0),
            min(1.0, len(identity) / 3.0),
            min(1.0, len(request.identity_embedding) / 16.0),
            min(1.0, _safe_float(memory_summary.get("hidden_region_count")) / 6.0),
            min(1.0, _safe_float(memory_summary.get("texture_patch_count")) / 10.0),
            min(1.0, _safe_float(retrieval.get("top_score"))),
            1.0 if retrieval.get("profile") == "rich" else 0.0,
            min(1.0, _safe_float(summary.get("candidate_count")) / 6.0),
            min(1.0, hidden_evidence + profile.memory_dependency * 0.25),
        ],
        dtype=np.float32,
    )
    bundle_cond = extract_memory_bundle_conditioning(request)

    mean = np.mean(roi_before, axis=(0, 1))
    std = np.std(roi_before, axis=(0, 1))
    luminance = float(np.mean(mean))
    chroma = float(np.mean(std))
    appearance = np.array(
        [float(mean[0]), float(mean[1]), float(mean[2]), float(std[0]), float(std[1]), float(std[2]), luminance, chroma],
        dtype=np.float32,
    )
    mem_vec, appearance = apply_region_metadata_conditioning_to_vectors(mem_vec, appearance, request.region_metadata)
    mem_vec, appearance = apply_memory_bundle_conditioning_to_vectors(mem_vec, appearance, bundle_cond, region_id=request.region.region_id)
    expected_payload = request.transition_context.get("expected_reference_payload") if isinstance(request.transition_context, dict) else None
    if bool(bundle_cond.get("reference_payload_trusted", False)) and isinstance(expected_payload, dict) and appearance.size > 6:
        descriptor_signal = _descriptor_numeric_signal(expected_payload.get("descriptor", {}))
        if descriptor_signal > 0.0:
            appearance[6] = float(np.clip(appearance[6] + 0.01 * descriptor_signal, 0.0, 1.0))
    material = request.transition_context.get("expected_reference_patch_material") if isinstance(request.transition_context, dict) else None
    if bool(bundle_cond.get("reference_patch_material_used", False)) and isinstance(material, dict):
        try:
            ref = np.asarray(material.get("rgb_patch"), dtype=np.float32)
        except (TypeError, ValueError):
            ref = np.zeros((0, 0, 3), dtype=np.float32)
        if ref.ndim == 3 and ref.shape[2] == 3 and ref.size:
            ref_mean = np.mean(np.clip(ref, 0.0, 1.0), axis=(0, 1))
            ref_std = np.std(np.clip(ref, 0.0, 1.0), axis=(0, 1))
            appearance[:3] = np.clip(0.82 * appearance[:3] + 0.18 * ref_mean.astype(np.float32), 0.0, 1.0)
            appearance[3:6] = np.clip(0.88 * appearance[3:6] + 0.12 * ref_std.astype(np.float32), 0.0, 1.0)
            if mem_vec.size > 8:
                mem_vec[8] = float(np.clip(max(float(mem_vec[8]), 0.92), 0.0, 1.0))
    elif bool(bundle_cond.get("reference_patch_material_present", False)) and appearance.size > 7:
        appearance[7] = float(np.clip(appearance[7] + 0.05, 0.0, 1.0))
    return mem_vec, appearance


def _geometry_priors(h: int, w: int, region_type: str, profile: RenderConditioningProfile) -> dict[str, np.ndarray]:
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h, dtype=np.float32), np.linspace(0.0, 1.0, w, dtype=np.float32), indexing="ij")
    center = np.exp(-(((xx - 0.5) / max(profile.local_radius_x, 1e-3)) ** 2 + ((yy - 0.54) / max(profile.local_radius_y, 1e-3)) ** 2))
    seam_band = np.exp(-((xx - 0.5) ** 2) / 0.028)
    vertical_split = seam_band * (0.42 + 0.58 * yy)
    upper_band = np.clip(1.0 - yy, 0.0, 1.0)
    lower_band = np.clip(yy, 0.0, 1.0)
    edge_distance = np.minimum.reduce([xx, 1.0 - xx, yy, 1.0 - yy])
    edge_band = np.clip(1.0 - edge_distance / 0.25, 0.0, 1.0)

    anchor = 0.5
    if "left" in region_type:
        anchor = 0.35
    elif "right" in region_type:
        anchor = 0.65
    lateral = np.exp(-(((xx - anchor) / 0.20) ** 2 + ((yy - 0.55) / 0.34) ** 2))
    expression_hotspot = np.exp(-(((xx - 0.5) / 0.14) ** 2 + ((yy - 0.68) / 0.12) ** 2))
    eye_hotspot = np.exp(-(((xx - 0.5) / 0.22) ** 2 + ((yy - 0.38) / 0.10) ** 2))
    occlusion_edge = edge_band * (0.55 + 0.45 * (1.0 - center))
    surface_band = np.clip(1.0 - np.abs(xx - 0.5) * 1.7, 0.0, 1.0) * (0.65 + 0.35 * upper_band)

    return {
        "yy": yy,
        "xx": xx,
        "center": center,
        "vertical_split": vertical_split,
        "expression_hotspot": expression_hotspot,
        "eye_hotspot": eye_hotspot,
        "lateral": lateral,
        "surface_band": surface_band,
        "edge_band": edge_band,
        "occlusion_edge": occlusion_edge,
        "upper_band": upper_band,
        "lower_band": lower_band,
    }


def _compose_region_priors(
    profile: RenderConditioningProfile,
    delta_cond: np.ndarray,
    planner_cond: np.ndarray,
    memory_cond: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = geometry["center"]
    vertical_split = geometry["vertical_split"]
    lateral = geometry["lateral"]
    expression_hotspot = geometry["expression_hotspot"]
    eye_hotspot = geometry["eye_hotspot"]
    surface_band = geometry["surface_band"]
    edge_band = geometry["edge_band"]
    occlusion_edge = geometry["occlusion_edge"]
    lower_band = geometry["lower_band"]

    delta_energy = np.clip(0.18 + 0.45 * delta_cond[9] + 0.12 * delta_cond[4] + 0.08 * planner_cond[2], 0.18, 1.0)
    retrieval_evidence = float(memory_cond[6])
    hidden_evidence = float(memory_cond[9])

    if profile.transition_mode == "garment_surface":
        changed = 0.54 * surface_band + 0.24 * center + 0.14 * edge_band
        blend = 0.10 + 0.48 * surface_band + 0.10 * edge_band
    elif profile.transition_mode == "garment_reveal":
        changed = 0.48 * vertical_split + 0.28 * center + 0.18 * lower_band
        blend = 0.08 + 0.42 * vertical_split + 0.18 * center
    elif profile.transition_mode == "pose_exposure":
        changed = 0.44 * lateral + 0.18 * center + 0.16 * edge_band
        blend = 0.06 + 0.32 * lateral + 0.10 * edge_band
    elif profile.transition_mode == "expression_refine":
        changed = 0.66 * expression_hotspot + 0.12 * eye_hotspot + 0.10 * center
        blend = 0.03 + 0.22 * expression_hotspot + 0.05 * eye_hotspot
    elif profile.transition_mode == "visibility_occlusion":
        changed = 0.38 * occlusion_edge + 0.10 * center
        blend = 0.05 + 0.18 * occlusion_edge
    else:
        changed = 0.14 * center + 0.06 * edge_band
        blend = 0.03 + 0.10 * center

    changed = np.clip(changed * profile.edit_strength * delta_energy, 0.0, 1.0)
    preservation = np.clip(
        profile.preservation_strength * 0.72
        + (1.0 - changed) * (0.12 + 0.26 * profile.seam_strictness)
        + edge_band * profile.context_support * 0.08
        - changed * 0.32,
        0.0,
        1.0,
    )

    grad_y, grad_x = np.gradient(changed.astype(np.float32))
    seam_boundary = np.clip(np.abs(grad_x) + np.abs(grad_y), 0.0, 1.0)
    seam_prior = np.clip(edge_band * 0.35 + seam_boundary * (0.42 + 0.40 * profile.seam_strictness) + (1.0 - changed) * 0.10, 0.0, 1.0)

    uncertainty = np.clip(profile.uncertainty_bias + changed * 0.18 + seam_prior * 0.12, 0.0, 1.0)
    if profile.transition_mode == "garment_reveal":
        uncertainty = np.clip(uncertainty + (1.0 - hidden_evidence) * 0.30 + (1.0 - retrieval_evidence) * 0.16, 0.0, 1.0)
    elif profile.transition_mode == "pose_exposure":
        uncertainty = np.clip(uncertainty + (1.0 - retrieval_evidence) * 0.12 + seam_boundary * 0.06, 0.0, 1.0)
    elif profile.transition_mode == "expression_refine":
        uncertainty = np.clip(uncertainty - retrieval_evidence * 0.08 - hidden_evidence * 0.05, 0.0, 1.0)
    elif profile.transition_mode == "stable":
        uncertainty = np.clip(uncertainty - 0.08, 0.0, 1.0)

    blend_hint = np.clip(blend * (0.72 + 0.28 * profile.edit_strength) + seam_boundary * 0.08, 0.0, 1.0)
    alpha_span = max(0.05, profile.alpha_peak - profile.alpha_floor)
    alpha_target = np.clip(
        profile.alpha_floor
        + alpha_span * (0.62 * changed + 0.20 * blend_hint + 0.18 * seam_boundary)
        - uncertainty * 0.12,
        0.0,
        1.0,
    )
    if profile.profile_role == "context":
        alpha_target = np.minimum(alpha_target, 0.32 + 0.32 * changed)
    return (
        changed[..., None].astype(np.float32),
        blend_hint[..., None].astype(np.float32),
        alpha_target[..., None].astype(np.float32),
        uncertainty[..., None].astype(np.float32),
        preservation[..., None].astype(np.float32),
        seam_prior[..., None].astype(np.float32),
    )


def _bootstrap_roi_after(
    roi_before: np.ndarray,
    profile: RenderConditioningProfile,
    appearance_cond: np.ndarray,
    memory_cond: np.ndarray,
    geometry: dict[str, np.ndarray],
    changed_mask: np.ndarray,
    reference_rgb: np.ndarray | None = None,
    reference_validity: np.ndarray | None = None,
    region_type: str = "",
    i2v_action_cond: dict[str, object] | None = None,
) -> np.ndarray:
    changed = changed_mask[..., 0]
    mean_rgb = appearance_cond[:3]
    luminance = float(appearance_cond[6]) if appearance_cond.size > 6 else float(np.mean(mean_rgb))
    retrieval_evidence = float(memory_cond[6]) if memory_cond.size > 6 else 0.0
    hidden_evidence = float(memory_cond[9]) if memory_cond.size > 9 else 0.0

    i2v_action_cond = i2v_action_cond if isinstance(i2v_action_cond, dict) else {}
    action_phase = str(i2v_action_cond.get("i2v_action_phase", "stable_idle"))
    action_strength = _safe_float(i2v_action_cond.get("i2v_action_strength"), 0.2)
    region_action_mode = str(i2v_action_cond.get("i2v_region_action_mode", "")).strip().lower()
    action_targeted = _i2v_action_targets_region(action_phase, region_type, region_action_mode)
    motion = _i2v_action_motion_field(roi_before.shape[0], roi_before.shape[1], region_type, geometry, i2v_action_cond)
    if action_targeted and action_phase == "head_turn":
        tone = np.array([0.025, 0.018, 0.012], dtype=np.float32) * (0.6 + 0.4 * action_strength)
        local = 0.72 * geometry["lateral"] + 0.28 * geometry["center"]
    elif action_targeted and action_phase == "expression_smile":
        tone = np.array([0.04, 0.022, 0.014], dtype=np.float32) * (0.55 + 0.45 * action_strength)
        local = 0.88 * geometry["expression_hotspot"] + 0.12 * geometry["lower_band"]
    elif action_targeted and action_phase == "torso_shift":
        tone = np.array([0.034, 0.026, 0.01], dtype=np.float32) * (0.5 + 0.5 * action_strength)
        local = 0.7 * geometry["surface_band"] + 0.3 * geometry["lateral"]
    elif action_targeted and action_phase == "arm_raise":
        tone = np.array([0.052, 0.03, 0.016], dtype=np.float32) * (0.55 + 0.45 * action_strength)
        local = 0.82 * geometry["lateral"] + 0.18 * geometry["upper_band"]
    elif action_targeted and action_phase == "garment_reveal_or_adjust":
        tone = np.array([0.055, 0.048, -0.02], dtype=np.float32) * (0.5 + 0.5 * action_strength)
        local = 0.75 * geometry["vertical_split"] + 0.25 * geometry["surface_band"]
    elif profile.transition_mode == "garment_surface":
        tone = np.array([0.06, 0.10, -0.02], dtype=np.float32)
        local = geometry["surface_band"]
    elif profile.transition_mode == "garment_reveal":
        reveal_gain = 0.45 + 0.35 * retrieval_evidence + 0.20 * hidden_evidence
        tone = np.array([0.14, 0.05 + 0.03 * luminance, -0.04], dtype=np.float32) * reveal_gain
        local = geometry["vertical_split"] * (0.65 + 0.35 * geometry["lower_band"])
    elif profile.transition_mode == "pose_exposure":
        tone = np.array([0.05, 0.03, 0.01], dtype=np.float32)
        local = geometry["lateral"]
    elif profile.transition_mode == "expression_refine":
        tone = np.array([0.04, 0.02, 0.01], dtype=np.float32)
        local = geometry["expression_hotspot"] + 0.20 * geometry["eye_hotspot"]
    elif profile.transition_mode == "visibility_occlusion":
        tone = np.array([-0.05, -0.05, -0.04], dtype=np.float32)
        local = geometry["occlusion_edge"]
    else:
        tone = np.array([0.01, 0.01, 0.0], dtype=np.float32)
        local = geometry["center"]

    warm_bias = np.asarray(mean_rgb, dtype=np.float32) * 0.10
    after = roi_before.copy()
    action_boost = 0.25 * action_strength if action_targeted else 0.0
    edit_mix = changed * (0.35 + 0.55 * profile.edit_strength + action_boost) * np.clip(local, 0.0, 1.0)
    for c in range(3):
        after[..., c] = np.clip(after[..., c] + edit_mix * (tone[c] + warm_bias[c] * 0.15), 0.0, 1.0)
    if isinstance(reference_rgb, np.ndarray) and reference_rgb.shape == roi_before.shape:
        validity = reference_validity if isinstance(reference_validity, np.ndarray) and reference_validity.shape[:2] == roi_before.shape[:2] else np.zeros((*roi_before.shape[:2], 1), dtype=np.float32)
        support = np.clip(validity[..., 0] * changed * np.clip(local, 0.0, 1.0), 0.0, 1.0)
        material_mix = support * (0.06 + 0.10 * profile.preservation_strength)
        after = np.clip(after * (1.0 - material_mix[..., None]) + reference_rgb * material_mix[..., None], 0.0, 1.0)
    if bool(motion.get("targeted", False)):
        identity_sensitive = region_type in IDENTITY_SENSITIVE_REGIONS
        max_amount = 0.09 if identity_sensitive else 0.26
        warp_amount = float(np.clip((0.02 + 0.20 * float(motion["intensity"])) * (0.7 + 0.3 * profile.edit_strength), 0.0, max_amount))
        warped = _warp_roi_nearest(roi_before, motion["flow_x"], motion["flow_y"], warp_amount)
        deform = np.clip(motion["deformation_mask"] * changed, 0.0, 1.0)
        preserve_gate = 1.0 - np.clip(profile.preservation_strength * (0.82 if identity_sensitive else 0.55), 0.0, 0.95)
        mix = np.clip(deform * preserve_gate, 0.0, 0.35 if identity_sensitive else 0.62)
        after = np.clip(after * (1.0 - mix[..., None]) + warped * mix[..., None], 0.0, 1.0)
    return after.astype(np.float32)


def _build_targets(
    roi_before: np.ndarray,
    profile: RenderConditioningProfile,
    delta_cond: np.ndarray,
    planner_cond: np.ndarray,
    memory_cond: np.ndarray,
    appearance_cond: np.ndarray,
    region_type: str,
    reference_rgb: np.ndarray | None = None,
    reference_validity: np.ndarray | None = None,
    i2v_action_cond: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    h, w, _ = roi_before.shape
    geometry = _geometry_priors(h, w, region_type, profile)
    changed_mask, blend_hint, alpha_target, uncertainty_target, preservation_mask, seam_prior = _compose_region_priors(
        profile,
        delta_cond,
        planner_cond,
        memory_cond,
        geometry,
    )
    if isinstance(i2v_action_cond, dict) and bool(i2v_action_cond.get("i2v_action_active", False)):
        action_phase = str(i2v_action_cond.get("i2v_action_phase", ""))
        region_action_mode = str(i2v_action_cond.get("i2v_region_action_mode", ""))
        gain = _safe_float(i2v_action_cond.get("i2v_action_strength"), 0.0)
        action_targeted = _i2v_action_targets_region(action_phase, region_type, region_action_mode)
        if action_targeted and action_phase == "arm_raise":
            changed_mask = np.clip(changed_mask + 0.28 * geometry["upper_band"][..., None] * gain, 0.0, 1.0)
            blend_hint = np.clip(blend_hint + 0.18 * geometry["lateral"][..., None] * gain, 0.0, 1.0)
        elif action_targeted and action_phase == "expression_smile":
            changed_mask = np.clip(changed_mask + 0.22 * geometry["expression_hotspot"][..., None] * gain, 0.0, 1.0)
            blend_hint = np.clip(blend_hint + 0.15 * geometry["expression_hotspot"][..., None] * gain, 0.0, 1.0)
        elif action_targeted and action_phase == "head_turn":
            changed_mask = np.clip(changed_mask + 0.16 * geometry["lateral"][..., None] * gain, 0.0, 1.0)
        elif action_targeted and action_phase == "torso_shift":
            changed_mask = np.clip(changed_mask + 0.14 * geometry["surface_band"][..., None] * gain, 0.0, 1.0)
        elif action_targeted and action_phase == "garment_reveal_or_adjust":
            changed_mask = np.clip(changed_mask + 0.2 * geometry["vertical_split"][..., None] * gain, 0.0, 1.0)
            blend_hint = np.clip(blend_hint + 0.2 * geometry["surface_band"][..., None] * gain, 0.0, 1.0)
    motion = _i2v_action_motion_field(h, w, region_type, geometry, i2v_action_cond)
    identity_sensitive = region_type in IDENTITY_SENSITIVE_REGIONS
    max_amount = 0.09 if identity_sensitive else 0.26
    warp_amount = float(np.clip((0.02 + 0.20 * float(motion.get("intensity", 0.0))) * (0.7 + 0.3 * profile.edit_strength), 0.0, max_amount)) if bool(motion.get("targeted", False)) else 0.0
    roi_after = _bootstrap_roi_after(
        roi_before,
        profile,
        appearance_cond,
        memory_cond,
        geometry,
        changed_mask,
        reference_rgb,
        reference_validity,
        region_type=region_type,
        i2v_action_cond=i2v_action_cond,
    )
    motion_trace = {
        "i2v_motion_field_active": bool(motion.get("active", False)),
        "i2v_motion_field_targeted": bool(motion.get("targeted", False)),
        "i2v_motion_field_intensity": float(motion.get("intensity", 0.0) or 0.0),
        "i2v_deformation_mask_mean": float(np.mean(motion.get("deformation_mask", np.zeros((h, w), dtype=np.float32)))),
        "i2v_flow_x_mean": float(np.mean(motion.get("flow_x", np.zeros((h, w), dtype=np.float32)))),
        "i2v_flow_y_mean": float(np.mean(motion.get("flow_y", np.zeros((h, w), dtype=np.float32)))),
        "i2v_warp_amount": warp_amount,
    }
    i2v_flow_x = map_to_shape(np.asarray(motion.get("flow_x", np.zeros((h, w), dtype=np.float32)), dtype=np.float32), (h, w), fill=0.0)[..., None]
    i2v_flow_y = map_to_shape(np.asarray(motion.get("flow_y", np.zeros((h, w), dtype=np.float32)), dtype=np.float32), (h, w), fill=0.0)[..., None]
    i2v_deformation_mask = map_to_shape(np.asarray(motion.get("deformation_mask", np.zeros((h, w), dtype=np.float32)), dtype=np.float32), (h, w), fill=0.0)[..., None]
    return roi_after, changed_mask, blend_hint, alpha_target, uncertainty_target, preservation_mask, seam_prior, i2v_flow_x, i2v_flow_y, i2v_deformation_mask, motion_trace


def summarize_patch_batch(batch: PatchBatch) -> dict[str, object]:
    return {
        "transition_mode": batch.transition_mode,
        "profile_role": batch.profile_role,
        "synthesis_family": batch.conditioning_summary.get("synthesis_family", "unknown"),
        "reveal_like": bool(batch.conditioning_summary.get("reveal_like", False)),
        "changed_mask_mean": float(np.mean(batch.changed_mask)),
        "blend_hint_mean": float(np.mean(batch.blend_hint)),
        "alpha_target_mean": float(np.mean(batch.alpha_target)),
        "uncertainty_target_mean": float(np.mean(_map_to_shape(batch.uncertainty_target, batch.roi_before.shape[:2], fill=0.0))),
        "preservation_mean": float(np.mean(_map_to_shape(batch.preservation_mask, batch.roi_before.shape[:2], fill=0.0))),
        "seam_prior_mean": float(np.mean(_map_to_shape(batch.seam_prior, batch.roi_before.shape[:2], fill=0.0))),
        "edit_strength": _safe_float(batch.conditioning_summary.get("edit_strength")),
        "preservation_strength": _safe_float(batch.conditioning_summary.get("preservation_strength")),
        "memory_dependency": _safe_float(batch.conditioning_summary.get("memory_dependency")),
        "memory_bundle_present": bool(batch.conditioning_summary.get("memory_bundle_present", False)),
        "memory_support_level": batch.conditioning_summary.get("memory_support_level", "none"),
        "reference_payload_present": bool(batch.conditioning_summary.get("reference_payload_present", False)),
        "expected_reference_payload_present": bool(batch.conditioning_summary.get("expected_reference_payload_present", False)),
        "expected_reference_payload_kind": batch.conditioning_summary.get("expected_reference_payload_kind", ""),
        "expected_reference_payload_patch_id_present": bool(batch.conditioning_summary.get("expected_reference_payload_patch_id_present", False)),
        "expected_reference_payload_descriptor_present": bool(batch.conditioning_summary.get("expected_reference_payload_descriptor_present", False)),
        "expected_reference_payload_descriptor_keys": list(batch.conditioning_summary.get("expected_reference_payload_descriptor_keys", [])) if isinstance(batch.conditioning_summary.get("expected_reference_payload_descriptor_keys", []), list) else [],
        "reference_payload_trusted": bool(batch.conditioning_summary.get("reference_payload_trusted", False)),
        "reference_payload_untrusted_reason": batch.conditioning_summary.get("reference_payload_untrusted_reason", ""),
        "expected_reference_payload_confidence": _safe_float(batch.conditioning_summary.get("expected_reference_payload_confidence")),
        "expected_reference_payload_evidence_score": _safe_float(batch.conditioning_summary.get("expected_reference_payload_evidence_score")),
        "reference_patch_material_present": bool(batch.conditioning_summary.get("reference_patch_material_present", False)),
        "reference_patch_material_validated": bool(batch.conditioning_summary.get("reference_patch_material_validated", False)),
        "reference_patch_material_trusted": bool(batch.conditioning_summary.get("reference_patch_material_trusted", False)),
        "reference_patch_material_used": bool(batch.conditioning_summary.get("reference_patch_material_used", False)),
        "reference_patch_material_source": batch.conditioning_summary.get("reference_patch_material_source", "none"),
        "reference_patch_material_missing_reason": batch.conditioning_summary.get("reference_patch_material_missing_reason", ""),
        "reference_patch_material_shape": list(batch.conditioning_summary.get("reference_patch_material_shape", [])) if isinstance(batch.conditioning_summary.get("reference_patch_material_shape", []), list) else [],
        "reference_patch_material_kind": batch.conditioning_summary.get("reference_patch_material_kind", ""),
        "reference_patch_material_confidence": _safe_float(batch.conditioning_summary.get("reference_patch_material_confidence")),
        "reference_patch_material_evidence_score": _safe_float(batch.conditioning_summary.get("reference_patch_material_evidence_score")),
        "i2v_reference_contract_version": batch.conditioning_summary.get("i2v_reference_contract_version", "i2v_first_frame_reference_v1"),
        "reference_material_from_input_frame": bool(batch.conditioning_summary.get("reference_material_from_input_frame", False)),
        "reference_material_from_generated_frame": bool(batch.conditioning_summary.get("reference_material_from_generated_frame", False)),
        "reference_patch_material_source_frame_kind": batch.conditioning_summary.get("reference_patch_material_source_frame_kind", "unknown"),
        "reference_patch_material_source_frame_index": int(batch.conditioning_summary.get("reference_patch_material_source_frame_index", 0) or 0),
        "reference_patch_material_immutable_i2v_anchor": bool(batch.conditioning_summary.get("reference_patch_material_immutable_i2v_anchor", False)),
        "reference_patch_material_source_is_input_frame": bool(batch.conditioning_summary.get("reference_patch_material_source_is_input_frame", False)),
        "reference_material_lane_active": bool(batch.conditioning_summary.get("reference_material_lane_active", False)),
        "reference_tensor_input_channels": int(batch.conditioning_summary.get("reference_tensor_input_channels", 0) or 0),
        "local_tensor_input_channels": int(batch.conditioning_summary.get("local_tensor_input_channels", 0) or 0),
        "reference_tensor_input_used": bool(batch.conditioning_summary.get("reference_tensor_input_used", False)),
        "reference_tensor_zero_fallback": bool(batch.conditioning_summary.get("reference_tensor_zero_fallback", True)),

        "material_gate_mean": _safe_float(batch.conditioning_summary.get("material_gate_mean")),
        "material_gate_max": _safe_float(batch.conditioning_summary.get("material_gate_max")),
        "material_gate_cap": _safe_float(batch.conditioning_summary.get("material_gate_cap")),
        "reference_validity_mean": _safe_float(batch.conditioning_summary.get("reference_validity_mean")),
        "reference_mask_mean": _safe_float(batch.conditioning_summary.get("reference_mask_mean")),
        "material_gate_suppressed_by_preservation": _safe_float(batch.conditioning_summary.get("material_gate_suppressed_by_preservation")),
        "identity_reference_used": bool(batch.conditioning_summary.get("identity_reference_used", False)),
        "identity_reference_strength": _safe_float(batch.conditioning_summary.get("identity_reference_strength")),
        "identity_reference_source": batch.conditioning_summary.get("identity_reference_source", "none"),
        "identity_reference_blocked": bool(batch.conditioning_summary.get("identity_reference_blocked", False)),
        "identity_reference_block_reasons": list(batch.conditioning_summary.get("identity_reference_block_reasons", [])) if isinstance(batch.conditioning_summary.get("identity_reference_block_reasons", []), list) else [],
        "identity_preservation_bias": _safe_float(batch.conditioning_summary.get("identity_preservation_bias")),
        "skin_reference_used": bool(batch.conditioning_summary.get("skin_reference_used", False)),
        "skin_reference_strength": _safe_float(batch.conditioning_summary.get("skin_reference_strength")),
        "skin_reference_source": batch.conditioning_summary.get("skin_reference_source", "none"),
        "skin_reference_blocked": bool(batch.conditioning_summary.get("skin_reference_blocked", False)),
        "skin_reference_block_reasons": list(batch.conditioning_summary.get("skin_reference_block_reasons", [])) if isinstance(batch.conditioning_summary.get("skin_reference_block_reasons", []), list) else [],
        "body_shape_reference_used": bool(batch.conditioning_summary.get("body_shape_reference_used", False)),
        "body_shape_reference_strength": _safe_float(batch.conditioning_summary.get("body_shape_reference_strength")),
        "body_shape_reference_source": batch.conditioning_summary.get("body_shape_reference_source", "none"),
        "body_shape_reference_blocked": bool(batch.conditioning_summary.get("body_shape_reference_blocked", False)),
        "body_shape_reference_block_reasons": list(batch.conditioning_summary.get("body_shape_reference_block_reasons", [])) if isinstance(batch.conditioning_summary.get("body_shape_reference_block_reasons", []), list) else [],
        "garment_reference_used": bool(batch.conditioning_summary.get("garment_reference_used", False)),
        "garment_reference_strength": _safe_float(batch.conditioning_summary.get("garment_reference_strength")),
        "garment_reference_source": batch.conditioning_summary.get("garment_reference_source", "none"),
        "garment_reference_blocked": bool(batch.conditioning_summary.get("garment_reference_blocked", False)),
        "garment_reference_block_reasons": list(batch.conditioning_summary.get("garment_reference_block_reasons", [])) if isinstance(batch.conditioning_summary.get("garment_reference_block_reasons", []), list) else [],
        "accessory_reference_used": bool(batch.conditioning_summary.get("accessory_reference_used", False)),
        "accessory_reference_strength": _safe_float(batch.conditioning_summary.get("accessory_reference_strength")),
        "accessory_reference_source": batch.conditioning_summary.get("accessory_reference_source", "none"),
        "accessory_reference_blocked": bool(batch.conditioning_summary.get("accessory_reference_blocked", False)),
        "accessory_reference_block_reasons": list(batch.conditioning_summary.get("accessory_reference_block_reasons", [])) if isinstance(batch.conditioning_summary.get("accessory_reference_block_reasons", []), list) else [],
        "region_metadata_used": bool(batch.conditioning_summary.get("region_metadata_used", False)),
        "metadata_completeness_score": _safe_float(batch.conditioning_summary.get("metadata_completeness_score")),
        "evidence_strength_score": _safe_float(batch.conditioning_summary.get("evidence_strength_score")),
        "metadata_feature_keys": list(batch.conditioning_summary.get("metadata_feature_keys", [])) if isinstance(batch.conditioning_summary.get("metadata_feature_keys", []), list) else [],
        "mask_ref_present": bool(batch.conditioning_summary.get("mask_ref_present", False)),
        "roi_source": batch.conditioning_summary.get("roi_source", "unknown"),
        "source_node_type": batch.conditioning_summary.get("source_node_type", "unknown"),
        "mask_kind": batch.conditioning_summary.get("mask_kind", ""),
        "i2v_action_phase": batch.conditioning_summary.get("i2v_action_phase", "stable_idle"),
        "i2v_action_active": bool(batch.conditioning_summary.get("i2v_action_active", False)),
        "i2v_region_action_mode": batch.conditioning_summary.get("i2v_region_action_mode", ""),
        "i2v_action_strength": _safe_float(batch.conditioning_summary.get("i2v_action_strength")),
        "i2v_motion_direction_x": _safe_float(batch.conditioning_summary.get("i2v_motion_direction_x")),
        "i2v_motion_direction_y": _safe_float(batch.conditioning_summary.get("i2v_motion_direction_y")),
        "i2v_expression_bias": _safe_float(batch.conditioning_summary.get("i2v_expression_bias")),
        "i2v_pose_bias": _safe_float(batch.conditioning_summary.get("i2v_pose_bias")),
        "i2v_garment_bias": _safe_float(batch.conditioning_summary.get("i2v_garment_bias")),
        "i2v_motion_field_active": bool(batch.conditioning_summary.get("i2v_motion_field_active", False)),
        "i2v_motion_field_targeted": bool(batch.conditioning_summary.get("i2v_motion_field_targeted", False)),
        "i2v_motion_field_intensity": _safe_float(batch.conditioning_summary.get("i2v_motion_field_intensity")),
        "i2v_deformation_mask_mean": _safe_float(batch.conditioning_summary.get("i2v_deformation_mask_mean")),
        "i2v_flow_x_mean": _safe_float(batch.conditioning_summary.get("i2v_flow_x_mean")),
        "i2v_flow_y_mean": _safe_float(batch.conditioning_summary.get("i2v_flow_y_mean")),
        "i2v_warp_amount": _safe_float(batch.conditioning_summary.get("i2v_warp_amount")),
        "i2v_motion_input_channels": int(batch.conditioning_summary.get("i2v_motion_input_channels", 0) or 0),
        "i2v_motion_tensor_used": bool(batch.conditioning_summary.get("i2v_motion_tensor_used", False)),
        "i2v_motion_tensor_zero_fallback": bool(batch.conditioning_summary.get("i2v_motion_tensor_zero_fallback", True)),
    }


def summarize_request_contract(request: PatchSynthesisRequest) -> dict[str, object]:
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    delta = ctx.get("graph_delta")
    _, region_type = parse_region_id(request.region.region_id)
    raw_mode = ""
    if delta is not None:
        raw_mode = str(getattr(delta, "region_transition_mode", {}).get(region_type, ""))
    transition_mode = _canonical_transition_mode(str(ctx.get("region_transition_mode", raw_mode)), region_type)
    metadata_cond = extract_region_metadata_conditioning(request)
    return {
        "requested_transition_mode": transition_mode,
        "requested_profile_role": _resolve_profile_role(request, region_type),
        "region_type": region_type,
        "reason": request.region.reason,
        "has_graph_delta": delta is not None,
        "region_metadata_used": metadata_cond["region_metadata_used"],
        "metadata_completeness_score": metadata_cond["metadata_completeness_score"],
        "evidence_strength_score": metadata_cond["evidence_strength_score"],
        "metadata_feature_keys": metadata_cond["metadata_feature_keys"],
        "mask_ref_present": metadata_cond["mask_ref_present"],
        "roi_source": metadata_cond["roi_source"],
        "source_node_type": metadata_cond["source_node_type"],
    }


def build_patch_batch(request: PatchSynthesisRequest, roi_before: np.ndarray) -> PatchBatch:
    _, region_type = parse_region_id(request.region.region_id)
    profile = _build_render_profile(request)
    semantic = _semantic_embed(request.region.region_id, profile)
    delta_cond, planner_cond = _delta_features(request, profile)
    graph_cond = _graph_features(request)
    bundle_cond = extract_memory_bundle_conditioning(request)
    i2v_action_cond = extract_i2v_action_conditioning(request)
    memory_cond, appearance_cond = _memory_and_appearance_features(request, roi_before, profile)
    material_cond = extract_reference_material_conditioning(request.transition_context if isinstance(request.transition_context, dict) else {}, request=request)
    reference_rgb, reference_mask, reference_validity = _reference_material_tensors(request, roi_before, material_cond)
    bbox_cond = np.array(
        [
            float(request.region.bbox.x),
            float(request.region.bbox.y),
            float(request.region.bbox.w),
            float(request.region.bbox.h),
            min(1.0, float(request.region.bbox.w * request.region.bbox.h)),
            min(1.0, float(request.region.bbox.w / max(request.region.bbox.h, 1e-5)) / 3.0),
        ],
        dtype=np.float32,
    )
    (
        roi_after,
        changed_mask,
        blend_hint,
        alpha_target,
        uncertainty_target,
        preservation_mask,
        seam_prior,
        i2v_flow_x,
        i2v_flow_y,
        i2v_deformation_mask,
        motion_trace,
    ) = _build_targets(
        roi_before,
        profile,
        delta_cond,
        planner_cond,
        memory_cond,
        appearance_cond,
        region_type,
        reference_rgb,
        reference_validity,
        i2v_action_cond=i2v_action_cond,
    )
    if bundle_cond["memory_bundle_low_evidence_newly_revealed"]:
        uncertainty_target = np.clip(uncertainty_target + 0.08 * changed_mask, 0.0, 1.0).astype(np.float32)

    metadata_cond = extract_region_metadata_conditioning(request)
    identity_preservation_bias = _identity_preservation_bias(
        region_type,
        float(bundle_cond.get("identity_reference_strength", 0.0)),
        bool(bundle_cond.get("identity_reference_blocked", False)),
    )
    i2v_motion_tensor_used = bool(
        np.any(np.abs(i2v_flow_x) > 1e-8)
        or np.any(np.abs(i2v_flow_y) > 1e-8)
        or np.any(i2v_deformation_mask > 1e-8)
    )
    conditioning_summary = {
        "transition_mode": profile.transition_mode,
        "profile_role": profile.profile_role,
        "synthesis_family": profile.synthesis_family,
        "reveal_like": profile.reveal_like,
        "edit_strength": profile.edit_strength,
        "preservation_strength": profile.preservation_strength,
        "seam_strictness": profile.seam_strictness,
        "memory_dependency": profile.memory_dependency,
        "alpha_floor": profile.alpha_floor,
        "alpha_peak": profile.alpha_peak,
        "memory_bundle_present": bundle_cond["memory_bundle_present"],
        "memory_support_level": bundle_cond["memory_support_level"],
        "memory_bundle_support_value": bundle_cond["memory_bundle_support_value"],
        "memory_bundle_has_current_reuse": bundle_cond["memory_bundle_has_current_reuse"],
        "memory_bundle_has_identity_reference": bundle_cond["memory_bundle_has_identity_reference"],
        "memory_bundle_has_skin_reference": bundle_cond["memory_bundle_has_skin_reference"],
        "memory_bundle_has_body_shape_reference": bundle_cond["memory_bundle_has_body_shape_reference"],
        "memory_bundle_has_garment_reference": bundle_cond["memory_bundle_has_garment_reference"],
        "memory_bundle_has_accessory_reference": bundle_cond["memory_bundle_has_accessory_reference"],
        "memory_bundle_has_active_hidden_support": bundle_cond["memory_bundle_has_active_hidden_support"],
        "memory_bundle_is_revealed_history": bundle_cond["memory_bundle_is_revealed_history"],
        "memory_bundle_reveal_lifecycle": bundle_cond["memory_bundle_reveal_lifecycle"],
        "memory_bundle_low_evidence_newly_revealed": bundle_cond["memory_bundle_low_evidence_newly_revealed"],
        "retrieval_reasons": bundle_cond["retrieval_reasons"],
        "reference_payload_present": bundle_cond["reference_payload_present"],
        "expected_reference_payload_present": bundle_cond["expected_reference_payload_present"],
        "expected_reference_payload_kind": bundle_cond["expected_reference_payload_kind"],
        "expected_reference_payload_patch_id_present": bundle_cond["expected_reference_payload_patch_id_present"],
        "expected_reference_payload_descriptor_present": bundle_cond["expected_reference_payload_descriptor_present"],
        "expected_reference_payload_descriptor_keys": bundle_cond["expected_reference_payload_descriptor_keys"],
        "reference_payload_trusted": bundle_cond["reference_payload_trusted"],
        "reference_payload_untrusted_reason": bundle_cond["reference_payload_untrusted_reason"],
        "reference_payload_trace_reasons": bundle_cond.get("reference_payload_trace_reasons", []),
        "expected_reference_payload_confidence": bundle_cond["expected_reference_payload_confidence"],
        "expected_reference_payload_evidence_score": bundle_cond["expected_reference_payload_evidence_score"],
        "reference_patch_material_present": material_cond["reference_patch_material_present"],
        "reference_patch_material_validated": material_cond["reference_patch_material_validated"],
        "reference_patch_material_trusted": material_cond["reference_patch_material_trusted"],
        "reference_patch_material_used": material_cond["reference_patch_material_used"],
        "reference_patch_material_source": material_cond["reference_patch_material_source"],
        "reference_patch_material_missing_reason": material_cond["reference_patch_material_missing_reason"],
        "reference_patch_material_shape": material_cond["reference_patch_material_shape"],
        "reference_patch_material_kind": material_cond["reference_patch_material_kind"],
        "reference_patch_material_confidence": material_cond["reference_patch_material_confidence"],
        "reference_patch_material_evidence_score": material_cond["reference_patch_material_evidence_score"],
        "reference_patch_material_trace_reasons": material_cond.get("reference_patch_material_trace_reasons", []),
        "reference_material_lane_active": bool(material_cond["reference_patch_material_used"]),
        "reference_memory_lane_semantics": "mem_vec[8]=visual_reference_lane; strong boost requires trusted material tensor",
        "reference_tensor_input_channels": 5,
        "reference_tensor_input_used": bool(material_cond["reference_patch_material_used"] and np.any(reference_validity > 0.0)),
        "reference_tensor_zero_fallback": bool(not (material_cond["reference_patch_material_used"] and np.any(reference_validity > 0.0))),
        "i2v_motion_input_channels": 3,
        "i2v_motion_tensor_used": i2v_motion_tensor_used,
        "i2v_motion_tensor_zero_fallback": bool(not i2v_motion_tensor_used),
        "i2v_flow_x_mean": float(np.mean(i2v_flow_x)),
        "i2v_flow_y_mean": float(np.mean(i2v_flow_y)),
        "i2v_deformation_mask_mean": float(np.mean(i2v_deformation_mask)),
        "identity_sensitive_region": region_type in IDENTITY_SENSITIVE_REGIONS,
        "core_identity_region": region_type in CORE_IDENTITY_REGIONS,
        "identity_reference_used": bundle_cond["identity_reference_used"],
        "identity_reference_strength": bundle_cond["identity_reference_strength"],
        "identity_reference_source": bundle_cond["identity_reference_source"],
        "identity_reference_blocked": bundle_cond["identity_reference_blocked"],
        "identity_reference_block_reasons": bundle_cond["identity_reference_block_reasons"],
        "identity_preservation_bias": identity_preservation_bias,
        "skin_reference_used": bundle_cond["skin_reference_used"],
        "skin_reference_strength": bundle_cond["skin_reference_strength"],
        "skin_reference_source": bundle_cond["skin_reference_source"],
        "skin_reference_blocked": bundle_cond["skin_reference_blocked"],
        "skin_reference_block_reasons": bundle_cond["skin_reference_block_reasons"],
        "body_shape_reference_used": bundle_cond["body_shape_reference_used"],
        "body_shape_reference_strength": bundle_cond["body_shape_reference_strength"],
        "body_shape_reference_source": bundle_cond["body_shape_reference_source"],
        "body_shape_reference_blocked": bundle_cond["body_shape_reference_blocked"],
        "body_shape_reference_block_reasons": bundle_cond["body_shape_reference_block_reasons"],
        "garment_reference_used": bundle_cond["garment_reference_used"],
        "garment_reference_strength": bundle_cond["garment_reference_strength"],
        "garment_reference_source": bundle_cond["garment_reference_source"],
        "garment_reference_blocked": bundle_cond["garment_reference_blocked"],
        "garment_reference_block_reasons": bundle_cond["garment_reference_block_reasons"],
        "accessory_reference_used": bundle_cond["accessory_reference_used"],
        "accessory_reference_strength": bundle_cond["accessory_reference_strength"],
        "accessory_reference_source": bundle_cond["accessory_reference_source"],
        "accessory_reference_blocked": bundle_cond["accessory_reference_blocked"],
        "accessory_reference_block_reasons": bundle_cond["accessory_reference_block_reasons"],
        "region_metadata_used": metadata_cond["region_metadata_used"],
        "metadata_completeness_score": metadata_cond["metadata_completeness_score"],
        "evidence_strength_score": metadata_cond["evidence_strength_score"],
        "metadata_feature_keys": metadata_cond["metadata_feature_keys"],
        "mask_ref_present": metadata_cond["mask_ref_present"],
        "roi_source": metadata_cond["roi_source"],
        "source_node_type": metadata_cond["source_node_type"],
        "mask_kind": metadata_cond["mask_kind"],
        **i2v_action_cond,
        **motion_trace,
        "i2v_region_id": request.region.region_id,
    }
    if isinstance(request.transition_context, dict):
        existing_contract = request.transition_context.get("renderer_batch_contract")
        contract = dict(existing_contract) if isinstance(existing_contract, dict) else {}
        contract["i2v_flow_x"] = i2v_flow_x.tolist()
        contract["i2v_flow_y"] = i2v_flow_y.tolist()
        contract["i2v_deformation_mask"] = i2v_deformation_mask.tolist()
        request.transition_context["renderer_batch_contract"] = contract
    return PatchBatch(
        roi_before=roi_before,
        roi_after=roi_after,
        changed_mask=changed_mask,
        alpha_target=alpha_target,
        blend_hint=blend_hint,
        semantic_embed=semantic,
        delta_cond=delta_cond,
        planner_cond=planner_cond,
        graph_cond=graph_cond,
        memory_cond=memory_cond,
        appearance_cond=appearance_cond,
        bbox_cond=bbox_cond,
        reference_rgb=reference_rgb,
        reference_mask=reference_mask,
        reference_validity=reference_validity,
        mode_cond=_mode_cond(profile),
        role_cond=_role_cond(profile),
        preservation_mask=preservation_mask,
        uncertainty_target=uncertainty_target,
        seam_prior=seam_prior,
        i2v_flow_x=i2v_flow_x,
        i2v_flow_y=i2v_flow_y,
        i2v_deformation_mask=i2v_deformation_mask,
        transition_mode=profile.transition_mode,
        profile_role=profile.profile_role,
        conditioning_summary=conditioning_summary,
    )


def _strategy_name(batch: PatchBatch | None, renderer_path: str) -> str:
    if batch is None:
        return f"LEARNED_{renderer_path.upper()}"
    return f"LEARNED_{batch.transition_mode.upper()}_{batch.profile_role.upper()}"


def _synthesis_mode(batch: PatchBatch | None, renderer_path: str) -> str:
    if batch is None:
        return "learned_primary" if renderer_path == "learned_primary" else "legacy_fallback"
    mode = batch.transition_mode
    if mode == "garment_reveal":
        return "learned_reveal_synthesis"
    if mode == "pose_exposure":
        return "learned_pose_exposure"
    if mode == "expression_refine":
        return "learned_expression_micro_edit"
    if mode == "visibility_occlusion":
        return "learned_visibility_occlusion"
    if mode == "garment_surface":
        return "learned_surface_transition"
    return "learned_context_preserve"


def output_from_prediction(
    request: PatchSynthesisRequest,
    pred: dict[str, np.ndarray | float],
    renderer_path: str,
    diagnostics: dict[str, object],
    batch: PatchBatch | None = None,
) -> PatchSynthesisOutput:
    rgb = pred["rgb"]
    alpha = pred["alpha"]
    uncertainty = pred["uncertainty"]
    conf = float(pred["confidence"])
    assert isinstance(rgb, np.ndarray) and isinstance(alpha, np.ndarray) and isinstance(uncertainty, np.ndarray)
    h, w, _ = rgb.shape
    conditioning_summary = summarize_patch_batch(batch) if batch is not None else {}
    if batch is not None:
        preservation = _map_to_shape(batch.preservation_mask, (h, w), fill=0.0)[..., 0]
        seam_prior = _map_to_shape(batch.seam_prior, (h, w), fill=0.0)[..., 0]
        uncertainty_target = _map_to_shape(batch.uncertainty_target, (h, w), fill=0.0)[..., 0]
        alpha = np.clip(alpha * (0.48 + 0.34 * batch.blend_hint[..., 0] + 0.18 * (1.0 - preservation)) + seam_prior * 0.04, 0.0, 1.0)
        uncertainty = np.clip(0.62 * uncertainty + 0.38 * uncertainty_target, 0.0, 1.0)
        preserved_drift = float(np.mean(np.abs(rgb - batch.roi_before) * _map_to_shape(batch.preservation_mask, (h, w), fill=0.0)))
        conf = float(np.clip(conf * (1.0 - preserved_drift * 1.35) * (1.0 - 0.18 * float(np.mean(uncertainty))), 0.0, 1.0))

    memory_bundle_trace = summarize_memory_bundle_trace(request.transition_context)


    i2v_motion_diag = {
        "i2v_motion_input_channels": int(pred.get("i2v_motion_input_channels", conditioning_summary.get("i2v_motion_input_channels", 0)) or 0),
        "i2v_motion_tensor_used": bool(pred.get("i2v_motion_tensor_used", conditioning_summary.get("i2v_motion_tensor_used", False))),
        "i2v_motion_tensor_zero_fallback": bool(pred.get("i2v_motion_tensor_zero_fallback", conditioning_summary.get("i2v_motion_tensor_zero_fallback", True))),
        "i2v_flow_x_mean": _safe_float(pred.get("i2v_flow_x_mean", conditioning_summary.get("i2v_flow_x_mean"))),
        "i2v_flow_y_mean": _safe_float(pred.get("i2v_flow_y_mean", conditioning_summary.get("i2v_flow_y_mean"))),
        "i2v_deformation_mask_mean": _safe_float(pred.get("i2v_deformation_mask_mean", conditioning_summary.get("i2v_deformation_mask_mean"))),
        "i2v_warp_amount": _safe_float(pred.get("i2v_warp_amount", conditioning_summary.get("i2v_warp_amount"))),
        "i2v_motion_region_reconstruction_loss": _safe_float(pred.get("i2v_motion_region_reconstruction_loss", conditioning_summary.get("i2v_motion_region_reconstruction_loss"))),
        "i2v_motion_preservation_penalty": _safe_float(pred.get("i2v_motion_preservation_penalty", conditioning_summary.get("i2v_motion_preservation_penalty"))),
    }
    material_diag = {
        "material_gate_mean": _safe_float(pred.get("material_gate_mean")),
        "material_gate_max": _safe_float(pred.get("material_gate_max")),
        "material_gate_cap": _safe_float(pred.get("material_gate_cap")),
        "reference_validity_mean": _safe_float(pred.get("reference_validity_mean")),
        "reference_mask_mean": _safe_float(pred.get("reference_mask_mean")),
        "material_consistency_loss": _safe_float(pred.get("material_consistency_loss")),
        "material_gate_regularization": _safe_float(pred.get("material_gate_regularization")),
        "material_gate_preservation_penalty": _safe_float(pred.get("material_gate_preservation_penalty")),
        "material_gate_invalidity_penalty": _safe_float(pred.get("material_gate_invalidity_penalty")),
        "material_gate_area_penalty": _safe_float(pred.get("material_gate_area_penalty")),
        "material_gate_suppressed_by_preservation": _safe_float(pred.get("material_gate_suppressed_by_preservation")),
    }
    exec_trace = {
        "renderer_path": renderer_path,
        "selected_render_strategy": _strategy_name(batch, renderer_path),
        "synthesis_mode": _synthesis_mode(batch, renderer_path),
        "transition_mode": batch.transition_mode if batch is not None else diagnostics.get("requested_transition_mode", "unknown"),
        "profile_role": batch.profile_role if batch is not None else diagnostics.get("requested_profile_role", "unknown"),
        "reveal_like": bool(batch.conditioning_summary.get("reveal_like", False)) if batch is not None else False,
        "diagnostics": diagnostics,
        "conditioning_summary": conditioning_summary,
        "region_metadata_used": bool(conditioning_summary.get("region_metadata_used", False)),
        "region_metadata_completeness_score": _safe_float(conditioning_summary.get("metadata_completeness_score")),
        "region_metadata_evidence_strength_score": _safe_float(conditioning_summary.get("evidence_strength_score")),
        "region_metadata_roi_source": conditioning_summary.get("roi_source", "unknown"),
        "region_metadata_source_node_type": conditioning_summary.get("source_node_type", "unknown"),
        "region_metadata_mask_kind": conditioning_summary.get("mask_kind", ""),
        "region_metadata_mask_ref_present": bool(conditioning_summary.get("mask_ref_present", False)),
        "region_metadata_feature_keys": list(conditioning_summary.get("metadata_feature_keys", [])) if isinstance(conditioning_summary.get("metadata_feature_keys", []), list) else [],
        "identity_reference_used": bool(conditioning_summary.get("identity_reference_used", False)),
        "identity_reference_strength": _safe_float(conditioning_summary.get("identity_reference_strength")),
        "identity_reference_source": conditioning_summary.get("identity_reference_source", "none"),
        "identity_reference_blocked": bool(conditioning_summary.get("identity_reference_blocked", False)),
        "identity_reference_block_reasons": list(conditioning_summary.get("identity_reference_block_reasons", [])) if isinstance(conditioning_summary.get("identity_reference_block_reasons", []), list) else [],
        "identity_preservation_bias": _safe_float(conditioning_summary.get("identity_preservation_bias")),
        "skin_reference_used": bool(conditioning_summary.get("skin_reference_used", False)),
        "skin_reference_strength": _safe_float(conditioning_summary.get("skin_reference_strength")),
        "skin_reference_source": conditioning_summary.get("skin_reference_source", "none"),
        "skin_reference_blocked": bool(conditioning_summary.get("skin_reference_blocked", False)),
        "skin_reference_block_reasons": list(conditioning_summary.get("skin_reference_block_reasons", [])) if isinstance(conditioning_summary.get("skin_reference_block_reasons", []), list) else [],
        "body_shape_reference_used": bool(conditioning_summary.get("body_shape_reference_used", False)),
        "body_shape_reference_strength": _safe_float(conditioning_summary.get("body_shape_reference_strength")),
        "body_shape_reference_source": conditioning_summary.get("body_shape_reference_source", "none"),
        "body_shape_reference_blocked": bool(conditioning_summary.get("body_shape_reference_blocked", False)),
        "body_shape_reference_block_reasons": list(conditioning_summary.get("body_shape_reference_block_reasons", [])) if isinstance(conditioning_summary.get("body_shape_reference_block_reasons", []), list) else [],
        "garment_reference_used": bool(conditioning_summary.get("garment_reference_used", False)),
        "garment_reference_strength": _safe_float(conditioning_summary.get("garment_reference_strength")),
        "garment_reference_source": conditioning_summary.get("garment_reference_source", "none"),
        "garment_reference_blocked": bool(conditioning_summary.get("garment_reference_blocked", False)),
        "garment_reference_block_reasons": list(conditioning_summary.get("garment_reference_block_reasons", [])) if isinstance(conditioning_summary.get("garment_reference_block_reasons", []), list) else [],
        "accessory_reference_used": bool(conditioning_summary.get("accessory_reference_used", False)),
        "accessory_reference_strength": _safe_float(conditioning_summary.get("accessory_reference_strength")),
        "accessory_reference_source": conditioning_summary.get("accessory_reference_source", "none"),
        "accessory_reference_blocked": bool(conditioning_summary.get("accessory_reference_blocked", False)),
        "accessory_reference_block_reasons": list(conditioning_summary.get("accessory_reference_block_reasons", [])) if isinstance(conditioning_summary.get("accessory_reference_block_reasons", []), list) else [],
        "reference_payload_present": bool(conditioning_summary.get("reference_payload_present", False)),
        "expected_reference_payload_present": bool(conditioning_summary.get("expected_reference_payload_present", False)),
        "expected_reference_payload_kind": conditioning_summary.get("expected_reference_payload_kind", ""),
        "expected_reference_payload_patch_id_present": bool(conditioning_summary.get("expected_reference_payload_patch_id_present", False)),
        "i2v_reference_contract_version": conditioning_summary.get("i2v_reference_contract_version", "i2v_first_frame_reference_v1"),
        "reference_material_from_input_frame": bool(conditioning_summary.get("reference_material_from_input_frame", False)),
        "reference_material_from_generated_frame": bool(conditioning_summary.get("reference_material_from_generated_frame", False)),
        "reference_patch_material_source_frame_kind": conditioning_summary.get("reference_patch_material_source_frame_kind", "unknown"),
        "reference_patch_material_source_frame_index": int(conditioning_summary.get("reference_patch_material_source_frame_index", 0) or 0),
        "reference_patch_material_immutable_i2v_anchor": bool(conditioning_summary.get("reference_patch_material_immutable_i2v_anchor", False)),
        "reference_patch_material_source_is_input_frame": bool(conditioning_summary.get("reference_patch_material_source_is_input_frame", False)),
        "expected_reference_payload_descriptor_present": bool(conditioning_summary.get("expected_reference_payload_descriptor_present", False)),
        "expected_reference_payload_descriptor_keys": list(conditioning_summary.get("expected_reference_payload_descriptor_keys", [])) if isinstance(conditioning_summary.get("expected_reference_payload_descriptor_keys", []), list) else [],
        "reference_payload_trusted": bool(conditioning_summary.get("reference_payload_trusted", False)),
        "reference_payload_untrusted_reason": conditioning_summary.get("reference_payload_untrusted_reason", ""),
        "expected_reference_payload_confidence": _safe_float(conditioning_summary.get("expected_reference_payload_confidence")),
        "expected_reference_payload_evidence_score": _safe_float(conditioning_summary.get("expected_reference_payload_evidence_score")),
        "reference_patch_material_present": bool(conditioning_summary.get("reference_patch_material_present", False)),
        "reference_patch_material_validated": bool(conditioning_summary.get("reference_patch_material_validated", False)),
        "reference_patch_material_trusted": bool(conditioning_summary.get("reference_patch_material_trusted", False)),
        "reference_patch_material_used": bool(conditioning_summary.get("reference_patch_material_used", False)),
        "reference_patch_material_source": conditioning_summary.get("reference_patch_material_source", "none"),
        "reference_patch_material_missing_reason": conditioning_summary.get("reference_patch_material_missing_reason", ""),
        "reference_patch_material_shape": list(conditioning_summary.get("reference_patch_material_shape", [])) if isinstance(conditioning_summary.get("reference_patch_material_shape", []), list) else [],
        "reference_patch_material_kind": conditioning_summary.get("reference_patch_material_kind", ""),
        "reference_patch_material_confidence": _safe_float(conditioning_summary.get("reference_patch_material_confidence")),
        "reference_patch_material_evidence_score": _safe_float(conditioning_summary.get("reference_patch_material_evidence_score")),
        "reference_material_lane_active": bool(conditioning_summary.get("reference_material_lane_active", False)),
        "reference_tensor_input_channels": int(conditioning_summary.get("reference_tensor_input_channels", 0) or 0),
        "local_tensor_input_channels": int(conditioning_summary.get("local_tensor_input_channels", 0) or 0),
        "reference_tensor_input_used": bool(conditioning_summary.get("reference_tensor_input_used", False)),
        "reference_tensor_zero_fallback": bool(conditioning_summary.get("reference_tensor_zero_fallback", True)),
        "memory_bundle_present": bool(conditioning_summary.get("memory_bundle_present", memory_bundle_trace.get("memory_bundle_present", False))),
        "memory_support_level": conditioning_summary.get("memory_support_level", memory_bundle_trace.get("memory_support_level", "none")),
        "i2v_action_phase": conditioning_summary.get("i2v_action_phase", "stable_idle"),
        "i2v_action_active": bool(conditioning_summary.get("i2v_action_active", False)),
        "i2v_region_action_mode": conditioning_summary.get("i2v_region_action_mode", ""),
        "i2v_action_strength": _safe_float(conditioning_summary.get("i2v_action_strength")),
        "i2v_motion_direction_x": _safe_float(conditioning_summary.get("i2v_motion_direction_x")),
        "i2v_motion_direction_y": _safe_float(conditioning_summary.get("i2v_motion_direction_y")),
        "i2v_expression_bias": _safe_float(conditioning_summary.get("i2v_expression_bias")),
        "i2v_pose_bias": _safe_float(conditioning_summary.get("i2v_pose_bias")),
        "i2v_garment_bias": _safe_float(conditioning_summary.get("i2v_garment_bias")),
        "i2v_motion_field_active": bool(conditioning_summary.get("i2v_motion_field_active", False)),
        "i2v_motion_field_targeted": bool(conditioning_summary.get("i2v_motion_field_targeted", False)),
        "i2v_motion_field_intensity": _safe_float(conditioning_summary.get("i2v_motion_field_intensity")),
        "i2v_deformation_mask_mean": _safe_float(conditioning_summary.get("i2v_deformation_mask_mean")),
        "i2v_flow_x_mean": _safe_float(conditioning_summary.get("i2v_flow_x_mean")),
        "i2v_flow_y_mean": _safe_float(conditioning_summary.get("i2v_flow_y_mean")),
        "i2v_warp_amount": _safe_float(conditioning_summary.get("i2v_warp_amount")),
        **i2v_motion_diag,
        "alpha_semantics": "blend_probability_for_true_changed_region_with_seam_support",
        "uncertainty_semantics": "local_synthesis_ambiguity_map_for_compositor_and_confidence",
        "confidence_semantics": "patch_reliability_summary_after_preservation_and_ambiguity_checks",
        "output_provenance": "trainable_local_patch_model",
        "torch_backend_used": bool(diagnostics.get("torch_backend_used", False)),
        "model_family": str(diagnostics.get("model_family", "")),
        "fallback_used": bool(diagnostics.get("fallback_used", False)),
        "fallback_reason": str(diagnostics.get("fallback_reason", "")),
        "fallback_message": str(diagnostics.get("fallback_message", "")),
        "checkpoint_requested": bool(diagnostics.get("checkpoint_requested", False)),
        "checkpoint_loaded": bool(diagnostics.get("checkpoint_loaded", False)),
        "checkpoint_path": str(diagnostics.get("checkpoint_path", "")),
        "checkpoint_backend": str(diagnostics.get("checkpoint_backend", "")),
        "checkpoint_fallback_used": bool(diagnostics.get("checkpoint_fallback_used", False)),
        "checkpoint_fallback_backend": str(diagnostics.get("checkpoint_fallback_backend", "")),
        "checkpoint_load_error": str(diagnostics.get("checkpoint_load_error", "")),
        "checkpoint_contract_version": str(diagnostics.get("checkpoint_contract_version", "")),
        "checkpoint_model_family": str(diagnostics.get("checkpoint_model_family", "")),
        "checkpoint_runtime_loadable": bool(diagnostics.get("checkpoint_runtime_loadable", False)),
        "checkpoint_global_cond_dim": int(diagnostics.get("checkpoint_global_cond_dim", 0) or 0),
        **memory_bundle_trace,
        **material_diag,
    }
    return PatchSynthesisOutput(
        region=request.region,
        rgb_patch=rgb.tolist(),
        alpha_mask=alpha.tolist(),
        height=h,
        width=w,
        channels=3,
        confidence=conf,
        z_index=1,
        uncertainty_map=uncertainty.tolist(),
        debug_trace=[
            f"renderer_path={renderer_path}",
            f"region={request.region.region_id}",
            f"mode={exec_trace['transition_mode']}",
            f"role={exec_trace['profile_role']}",
            f"alpha_mean={float(np.mean(alpha)):.4f}",
            f"unc_mean={float(np.mean(uncertainty)):.4f}",
            f"mat_gate_mean={float(material_diag.get('material_gate_mean', 0.0) or 0.0):.4f}",
        ],
        execution_trace=exec_trace,
        metadata={
            "renderer_path": renderer_path,
            "diagnostics": diagnostics,
            "transition_mode": exec_trace["transition_mode"],
            "profile_role": exec_trace["profile_role"],
            "conditioning_summary": conditioning_summary,
            "identity_reference_used": bool(conditioning_summary.get("identity_reference_used", False)),
            "identity_reference_strength": _safe_float(conditioning_summary.get("identity_reference_strength")),
            "identity_reference_source": conditioning_summary.get("identity_reference_source", "none"),
            "identity_reference_blocked": bool(conditioning_summary.get("identity_reference_blocked", False)),
            "identity_reference_block_reasons": list(conditioning_summary.get("identity_reference_block_reasons", [])) if isinstance(conditioning_summary.get("identity_reference_block_reasons", []), list) else [],
            "identity_preservation_bias": _safe_float(conditioning_summary.get("identity_preservation_bias")),
            "skin_reference_used": bool(conditioning_summary.get("skin_reference_used", False)),
            "skin_reference_strength": _safe_float(conditioning_summary.get("skin_reference_strength")),
            "skin_reference_blocked": bool(conditioning_summary.get("skin_reference_blocked", False)),
            "body_shape_reference_used": bool(conditioning_summary.get("body_shape_reference_used", False)),
            "body_shape_reference_strength": _safe_float(conditioning_summary.get("body_shape_reference_strength")),
            "body_shape_reference_blocked": bool(conditioning_summary.get("body_shape_reference_blocked", False)),
            "garment_reference_used": bool(conditioning_summary.get("garment_reference_used", False)),
            "garment_reference_strength": _safe_float(conditioning_summary.get("garment_reference_strength")),
            "garment_reference_blocked": bool(conditioning_summary.get("garment_reference_blocked", False)),
            "accessory_reference_used": bool(conditioning_summary.get("accessory_reference_used", False)),
            "accessory_reference_strength": _safe_float(conditioning_summary.get("accessory_reference_strength")),
            "accessory_reference_blocked": bool(conditioning_summary.get("accessory_reference_blocked", False)),
            "memory_bundle_present": bool(conditioning_summary.get("memory_bundle_present", memory_bundle_trace.get("memory_bundle_present", False))),
            "memory_support_level": conditioning_summary.get("memory_support_level", memory_bundle_trace.get("memory_support_level", "none")),
            "i2v_action_phase": conditioning_summary.get("i2v_action_phase", "stable_idle"),
            "i2v_action_active": bool(conditioning_summary.get("i2v_action_active", False)),
            "i2v_region_action_mode": conditioning_summary.get("i2v_region_action_mode", ""),
            "i2v_action_strength": _safe_float(conditioning_summary.get("i2v_action_strength")),
            "i2v_motion_direction_x": _safe_float(conditioning_summary.get("i2v_motion_direction_x")),
            "i2v_motion_direction_y": _safe_float(conditioning_summary.get("i2v_motion_direction_y")),
            "i2v_expression_bias": _safe_float(conditioning_summary.get("i2v_expression_bias")),
            "i2v_pose_bias": _safe_float(conditioning_summary.get("i2v_pose_bias")),
            "i2v_garment_bias": _safe_float(conditioning_summary.get("i2v_garment_bias")),
            "reference_payload_present": bool(conditioning_summary.get("reference_payload_present", False)),
            "expected_reference_payload_present": bool(conditioning_summary.get("expected_reference_payload_present", False)),
            "expected_reference_payload_kind": conditioning_summary.get("expected_reference_payload_kind", ""),
            "expected_reference_payload_patch_id_present": bool(conditioning_summary.get("expected_reference_payload_patch_id_present", False)),
            "expected_reference_payload_descriptor_present": bool(conditioning_summary.get("expected_reference_payload_descriptor_present", False)),
            "reference_payload_trusted": bool(conditioning_summary.get("reference_payload_trusted", False)),
            "reference_payload_untrusted_reason": conditioning_summary.get("reference_payload_untrusted_reason", ""),
            "expected_reference_payload_confidence": _safe_float(conditioning_summary.get("expected_reference_payload_confidence")),
            "expected_reference_payload_evidence_score": _safe_float(conditioning_summary.get("expected_reference_payload_evidence_score")),
            "reference_patch_material_present": bool(conditioning_summary.get("reference_patch_material_present", False)),
            "reference_patch_material_validated": bool(conditioning_summary.get("reference_patch_material_validated", False)),
            "reference_patch_material_trusted": bool(conditioning_summary.get("reference_patch_material_trusted", False)),
            "reference_patch_material_used": bool(conditioning_summary.get("reference_patch_material_used", False)),
            "reference_patch_material_source": conditioning_summary.get("reference_patch_material_source", "none"),
            "reference_patch_material_missing_reason": conditioning_summary.get("reference_patch_material_missing_reason", ""),
            "reference_patch_material_shape": list(conditioning_summary.get("reference_patch_material_shape", [])) if isinstance(conditioning_summary.get("reference_patch_material_shape", []), list) else [],
            "reference_patch_material_kind": conditioning_summary.get("reference_patch_material_kind", ""),
            "reference_patch_material_confidence": _safe_float(conditioning_summary.get("reference_patch_material_confidence")),
            "reference_patch_material_evidence_score": _safe_float(conditioning_summary.get("reference_patch_material_evidence_score")),
            "reference_material_lane_active": bool(conditioning_summary.get("reference_material_lane_active", False)),
            "reference_tensor_input_channels": int(conditioning_summary.get("reference_tensor_input_channels", 0) or 0),
            "local_tensor_input_channels": int(conditioning_summary.get("local_tensor_input_channels", 0) or 0),
            "reference_tensor_input_used": bool(conditioning_summary.get("reference_tensor_input_used", False)),
            "reference_tensor_zero_fallback": bool(conditioning_summary.get("reference_tensor_zero_fallback", True)),
            "blend_hint_mean": float(np.mean(batch.blend_hint)) if batch is not None else 0.0,
            "changed_mask_mean": float(np.mean(batch.changed_mask)) if batch is not None else 0.0,
            "alpha_mean": float(np.mean(alpha)),
            "uncertainty_mean": float(np.mean(uncertainty)),
            **material_diag,
        },
    )
