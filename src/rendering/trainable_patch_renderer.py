from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.region_ids import parse_region_id
from core.schema import RegionRef
from dynamics.transition_contracts import LearnedTemporalTransitionContract
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from utils_tensor import shape


class RendererInputError(ValueError):
    pass


class RendererInferenceError(RuntimeError):
    pass


ROI_FAMILIES = {
    "face_expression": {"face", "head", "mouth", "eyes", "cheek", "neck"},
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

SEMANTIC_DIM = 8
DELTA_DIM = 12
PLANNER_DIM = 10
GRAPH_DIM = 8
MEMORY_DIM = 10
APPEARANCE_DIM = 8
BBOX_DIM = 6
MODE_DIM = 8
ROLE_DIM = 6
PIXEL_FEATURE_DIM = 15
CONDITION_FEATURE_DIM = (
    SEMANTIC_DIM
    + DELTA_DIM
    + PLANNER_DIM
    + GRAPH_DIM
    + MEMORY_DIM
    + APPEARANCE_DIM
    + BBOX_DIM
    + MODE_DIM
    + ROLE_DIM
)


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
    mode_cond: np.ndarray | None = None
    role_cond: np.ndarray | None = None
    preservation_mask: np.ndarray | None = None
    uncertainty_target: np.ndarray | None = None
    seam_prior: np.ndarray | None = None
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
    return float(value) if isinstance(value, (int, float)) else default


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
    h, w = shape_hw
    if value is None:
        return np.full((h, w, 1), fill, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[:2] != (h, w):
        raise RendererInferenceError(f"Unexpected map shape {arr.shape}, expected {(h, w)}")
    if arr.shape[2] != 1:
        arr = np.mean(arr, axis=2, keepdims=True)
    return _clip01(arr).astype(np.float32)


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
    learned_contract = LearnedTemporalTransitionContract.from_metadata(ctx.get("learned_temporal_contract"))
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
    learned_contract = LearnedTemporalTransitionContract.from_metadata(ctx.get("learned_temporal_contract"))
    delta = ctx.get("graph_delta")
    _, region_type = parse_region_id(request.region.region_id)
    raw_mode = ""
    if delta is not None:
        raw_mode = str(getattr(delta, "region_transition_mode", {}).get(region_type, ""))
    if learned_contract is not None and learned_contract.is_learned_primary:
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
        return {
            "total_loss": total,
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
    mean = np.mean(roi_before, axis=(0, 1))
    std = np.std(roi_before, axis=(0, 1))
    luminance = float(np.mean(mean))
    chroma = float(np.mean(std))
    appearance = np.array(
        [float(mean[0]), float(mean[1]), float(mean[2]), float(std[0]), float(std[1]), float(std[2]), luminance, chroma],
        dtype=np.float32,
    )
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
) -> np.ndarray:
    changed = changed_mask[..., 0]
    mean_rgb = appearance_cond[:3]
    luminance = float(appearance_cond[6]) if appearance_cond.size > 6 else float(np.mean(mean_rgb))
    retrieval_evidence = float(memory_cond[6]) if memory_cond.size > 6 else 0.0
    hidden_evidence = float(memory_cond[9]) if memory_cond.size > 9 else 0.0

    if profile.transition_mode == "garment_surface":
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
    edit_mix = changed * (0.40 + 0.60 * profile.edit_strength) * np.clip(local, 0.0, 1.0)
    for c in range(3):
        after[..., c] = np.clip(after[..., c] + edit_mix * (tone[c] + warm_bias[c] * 0.15), 0.0, 1.0)
    return after.astype(np.float32)


def _build_targets(
    roi_before: np.ndarray,
    profile: RenderConditioningProfile,
    delta_cond: np.ndarray,
    planner_cond: np.ndarray,
    memory_cond: np.ndarray,
    appearance_cond: np.ndarray,
    region_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = roi_before.shape
    geometry = _geometry_priors(h, w, region_type, profile)
    changed_mask, blend_hint, alpha_target, uncertainty_target, preservation_mask, seam_prior = _compose_region_priors(
        profile,
        delta_cond,
        planner_cond,
        memory_cond,
        geometry,
    )
    roi_after = _bootstrap_roi_after(roi_before, profile, appearance_cond, memory_cond, geometry, changed_mask)
    return roi_after, changed_mask, blend_hint, alpha_target, uncertainty_target, preservation_mask, seam_prior


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
    }


def summarize_request_contract(request: PatchSynthesisRequest) -> dict[str, object]:
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    delta = ctx.get("graph_delta")
    _, region_type = parse_region_id(request.region.region_id)
    raw_mode = ""
    if delta is not None:
        raw_mode = str(getattr(delta, "region_transition_mode", {}).get(region_type, ""))
    transition_mode = _canonical_transition_mode(str(ctx.get("region_transition_mode", raw_mode)), region_type)
    return {
        "requested_transition_mode": transition_mode,
        "requested_profile_role": _resolve_profile_role(request, region_type),
        "region_type": region_type,
        "reason": request.region.reason,
        "has_graph_delta": delta is not None,
    }


def build_patch_batch(request: PatchSynthesisRequest, roi_before: np.ndarray) -> PatchBatch:
    _, region_type = parse_region_id(request.region.region_id)
    profile = _build_render_profile(request)
    semantic = _semantic_embed(request.region.region_id, profile)
    delta_cond, planner_cond = _delta_features(request, profile)
    graph_cond = _graph_features(request)
    memory_cond, appearance_cond = _memory_and_appearance_features(request, roi_before, profile)
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
    roi_after, changed_mask, blend_hint, alpha_target, uncertainty_target, preservation_mask, seam_prior = _build_targets(
        roi_before,
        profile,
        delta_cond,
        planner_cond,
        memory_cond,
        appearance_cond,
        region_type,
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
    }
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
        mode_cond=_mode_cond(profile),
        role_cond=_role_cond(profile),
        preservation_mask=preservation_mask,
        uncertainty_target=uncertainty_target,
        seam_prior=seam_prior,
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

    exec_trace = {
        "renderer_path": renderer_path,
        "selected_render_strategy": _strategy_name(batch, renderer_path),
        "synthesis_mode": _synthesis_mode(batch, renderer_path),
        "transition_mode": batch.transition_mode if batch is not None else diagnostics.get("requested_transition_mode", "unknown"),
        "profile_role": batch.profile_role if batch is not None else diagnostics.get("requested_profile_role", "unknown"),
        "reveal_like": bool(batch.conditioning_summary.get("reveal_like", False)) if batch is not None else False,
        "diagnostics": diagnostics,
        "conditioning_summary": conditioning_summary,
        "alpha_semantics": "blend_probability_for_true_changed_region_with_seam_support",
        "uncertainty_semantics": "local_synthesis_ambiguity_map_for_compositor_and_confidence",
        "confidence_semantics": "patch_reliability_summary_after_preservation_and_ambiguity_checks",
        "output_provenance": "trainable_local_patch_model",
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
        ],
        execution_trace=exec_trace,
        metadata={
            "renderer_path": renderer_path,
            "diagnostics": diagnostics,
            "transition_mode": exec_trace["transition_mode"],
            "profile_role": exec_trace["profile_role"],
            "conditioning_summary": conditioning_summary,
            "blend_hint_mean": float(np.mean(batch.blend_hint)) if batch is not None else 0.0,
            "changed_mask_mean": float(np.mean(batch.changed_mask)) if batch is not None else 0.0,
            "alpha_mean": float(np.mean(alpha)),
            "uncertainty_mean": float(np.mean(uncertainty)),
        },
    )
