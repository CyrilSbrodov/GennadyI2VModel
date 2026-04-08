from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.region_ids import parse_region_id
from core.schema import RegionRef
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from utils_tensor import shape


class RendererInputError(ValueError):
    pass


class RendererInferenceError(RuntimeError):
    pass


ROI_FAMILIES = {
    "face_expression": {"face", "head", "mouth", "eyes", "cheek"},
    "torso_reveal": {"torso", "inner_garment", "innerwear", "chest", "pelvis"},
    "sleeve_arm_transition": {"left_arm", "right_arm", "arm", "sleeves", "outer_garment", "garments"},
}


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


class TrainableLocalPatchModel:
    """Conditioned local synthesis model: pixel encoder + structured conditioning fusion + RGB/alpha/uncertainty heads."""

    def __init__(self, pixel_dim: int = 9, cond_dim: int = 48, hidden_dim: int = 48, seed: int = 11) -> None:
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
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _to_pixel_features(batch: PatchBatch) -> np.ndarray:
        h, w, _ = batch.roi_before.shape
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h, dtype=np.float32), np.linspace(0.0, 1.0, w, dtype=np.float32), indexing="ij")
        return np.concatenate(
            [
                batch.roi_before,
                batch.changed_mask,
                batch.blend_hint,
                yy[..., None],
                xx[..., None],
                (batch.changed_mask * batch.blend_hint),
                np.mean(batch.roi_before, axis=2, keepdims=True),
            ],
            axis=2,
        )

    @staticmethod
    def _global_condition(batch: PatchBatch) -> np.ndarray:
        return np.concatenate(
            [
                batch.semantic_embed,
                batch.delta_cond,
                batch.planner_cond,
                batch.graph_cond,
                batch.memory_cond,
                batch.appearance_cond,
                batch.bbox_cond,
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

        residual = np.tanh(rgb_raw) * 0.65
        rgb = np.clip(batch.roi_before.reshape(-1, 3) + residual, 0.0, 1.0)
        alpha = self._sigmoid(alpha_raw)
        uncertainty = self._sigmoid(unc_raw)
        confidence = float(np.clip((1.0 - np.mean(uncertainty)) * (0.6 + 0.4 * np.mean(alpha)), 0.0, 1.0))

        return {
            "rgb": rgb.reshape(h, w, 3),
            "alpha": alpha.reshape(h, w),
            "uncertainty": uncertainty.reshape(h, w),
            "confidence": confidence,
            "hidden": hidden,
            "hidden_pre": hidden_pre,
            "pixel_x": x,
            "cond": cond,
            "rgb_raw": rgb_raw,
        }

    def compute_losses(self, batch: PatchBatch, out: dict[str, np.ndarray | float]) -> dict[str, float]:
        rgb = out["rgb"]
        alpha = out["alpha"]
        unc = out["uncertainty"]
        assert isinstance(rgb, np.ndarray) and isinstance(alpha, np.ndarray) and isinstance(unc, np.ndarray)

        changed = np.clip(batch.changed_mask[..., 0], 0.0, 1.0)
        inv = 1.0 - changed
        recon = float(np.mean(((rgb - batch.roi_after) ** 2) * (0.4 + 0.9 * changed[..., None])))
        alpha_loss = float(np.mean((alpha - batch.alpha_target[..., 0]) ** 2))
        seam_loss = float(np.mean(np.abs((rgb - batch.roi_before) * (1.0 - batch.blend_hint)[..., None])))
        appearance = float(np.mean(((rgb - batch.roi_before) ** 2) * inv[..., None]))
        region_consistency = float(np.mean((np.mean(rgb, axis=(0, 1)) - np.mean(batch.roi_after, axis=(0, 1))) ** 2))
        err_map = np.mean(np.abs(rgb - batch.roi_after), axis=2)
        uncertainty_target = np.clip(err_map + 0.25 * (1.0 - changed), 0.0, 1.0)
        uncertainty_loss = float(np.mean((unc - uncertainty_target) ** 2))
        alpha_blend_consistency = float(np.mean(np.abs((alpha - batch.blend_hint[..., 0]) * changed)))

        total = recon + 0.45 * alpha_loss + 0.32 * seam_loss + 0.2 * appearance + 0.2 * region_consistency + 0.35 * uncertainty_loss + 0.2 * alpha_blend_consistency
        return {
            "total_loss": total,
            "reconstruction_loss": recon,
            "alpha_loss": alpha_loss,
            "seam_loss": seam_loss,
            "appearance_preservation_loss": appearance,
            "region_consistency_loss": region_consistency,
            "uncertainty_calibration_loss": uncertainty_loss,
            "alpha_blend_consistency_loss": alpha_blend_consistency,
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
        assert isinstance(hidden, np.ndarray) and isinstance(hidden_pre, np.ndarray) and isinstance(x, np.ndarray) and isinstance(cond, np.ndarray) and isinstance(rgb_raw, np.ndarray)

        target_rgb = batch.roi_after.reshape(-1, 3)
        changed = batch.changed_mask.reshape(-1, 1)
        alpha_t = batch.alpha_target.reshape(-1, 1)
        blend_t = batch.blend_hint.reshape(-1, 1)

        w_recon = (0.4 + 0.9 * changed)
        d_rgb = 2.0 * (rgb - target_rgb) * w_recon / max(1, rgb.size)
        d_residual = d_rgb
        d_rgb_raw = d_residual * (1.0 - np.tanh(rgb_raw) ** 2) * 0.65

        d_alpha = (2.0 * (alpha - alpha_t) / max(1, alpha.size)) * 0.45
        d_alpha += (np.sign((alpha - blend_t)) * changed / max(1, alpha.size)) * 0.2
        d_alpha_raw = d_alpha * alpha * (1.0 - alpha)

        err_map = np.mean(np.abs(rgb.reshape(batch.roi_after.shape) - batch.roi_after), axis=2, keepdims=True).reshape(-1, 1)
        unc_t = np.clip(err_map + 0.25 * (1.0 - changed), 0.0, 1.0)
        d_unc = (2.0 * (unc - unc_t) / max(1, unc.size)) * 0.35
        d_unc_raw = d_unc * unc * (1.0 - unc)

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


def _semantic_embed(region_id: str) -> np.ndarray:
    _, region_type = parse_region_id(region_id)
    if region_type in ROI_FAMILIES["face_expression"]:
        return np.array([1.0, 0.0, 0.0, 0.9, 0.1, 0.2], dtype=np.float32)
    if region_type in ROI_FAMILIES["torso_reveal"]:
        return np.array([0.0, 1.0, 0.0, 0.2, 0.85, 0.4], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0, 0.15, 0.45, 0.9], dtype=np.float32)


def _delta_features(request: PatchSynthesisRequest) -> tuple[np.ndarray, np.ndarray]:
    delta = request.transition_context.get("graph_delta") if isinstance(request.transition_context, dict) else None
    transition_phase = str(request.transition_context.get("transition_phase", "") if isinstance(request.transition_context, dict) else "")
    step_index = float(request.transition_context.get("step_index", 0.0) if isinstance(request.transition_context, dict) else 0.0)
    modes = [0.0, 0.0, 0.0, 0.0]
    delta_vec = np.zeros((9,), dtype=np.float32)
    if delta is not None:
        _, rtype = parse_region_id(request.region.region_id)
        mode = str(delta.region_transition_mode.get(rtype, ""))
        modes = [
            1.0 if mode in {"garment_reveal", "open_front", "remove_outer"} else 0.0,
            1.0 if mode in {"pose_exposure", "pose_deform"} else 0.0,
            1.0 if mode in {"expression_shift", "expression"} else 0.0,
            1.0 if mode in {"stable", ""} else 0.0,
        ]
        delta_vec = np.array(
            [
                float(len(delta.pose_deltas)),
                float(len(delta.expression_deltas)),
                float(len(delta.garment_deltas)),
                float(len(delta.semantic_reasons)),
                float(len(delta.affected_regions)),
                float(len(delta.newly_revealed_regions)),
                float(len(delta.newly_occluded_regions)),
                1.0 if request.region in delta.newly_revealed_regions else 0.0,
                1.0 if request.region in delta.newly_occluded_regions else 0.0,
            ],
            dtype=np.float32,
        )
    planner = np.array(
        [
            float(min(1.0, step_index / 10.0)),
            1.0 if transition_phase in {"prepare", "pre", "warmup"} else 0.0,
            1.0 if transition_phase in {"motion", "mid", "execute"} else 0.0,
            1.0 if transition_phase in {"settle", "post", "cooldown"} else 0.0,
            *modes,
        ],
        dtype=np.float32,
    )
    return delta_vec, planner


def _graph_features(request: PatchSynthesisRequest) -> np.ndarray:
    g = request.scene_state
    graph_emb = request.graph_encoding.graph_embedding if request.graph_encoding else []
    persons = float(len(g.persons))
    objects = float(len(g.objects))
    relations = float(len(g.relations))
    fps = float(g.global_context.fps or 0.0)
    frame_w, frame_h = g.global_context.frame_size if g.global_context.frame_size else (0, 0)
    return np.array(
        [persons, objects, relations, min(1.0, fps / 30.0), min(1.0, frame_w / 1920.0), min(1.0, frame_h / 1080.0), float(np.mean(graph_emb[:16])) if graph_emb else 0.0],
        dtype=np.float32,
    )


def _memory_and_appearance_features(request: PatchSynthesisRequest, roi_before: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    memory_summary = request.memory_summary if isinstance(request.memory_summary, dict) else {}
    channels = request.memory_channels if isinstance(request.memory_channels, dict) else {}
    hidden = channels.get("hidden_regions", {}) if isinstance(channels.get("hidden_regions", {}), dict) else {}
    garments = channels.get("garments", {}) if isinstance(channels.get("garments", {}), dict) else {}
    identity = channels.get("identity", {}) if isinstance(channels.get("identity", {}), dict) else {}
    retrieval = request.retrieval_summary if isinstance(request.retrieval_summary, dict) else {}
    mem_vec = np.array(
        [
            float(len(hidden)),
            float(len(garments)),
            float(len(identity)),
            float(len(request.identity_embedding)),
            float(memory_summary.get("hidden_region_count", 0.0) if isinstance(memory_summary.get("hidden_region_count", 0.0), (int, float)) else 0.0),
            float(memory_summary.get("texture_patch_count", 0.0) if isinstance(memory_summary.get("texture_patch_count", 0.0), (int, float)) else 0.0),
            float(retrieval.get("top_score", 0.0) if isinstance(retrieval.get("top_score", 0.0), (int, float)) else 0.0),
            1.0 if retrieval.get("profile") == "rich" else 0.0,
        ],
        dtype=np.float32,
    )
    mean = np.mean(roi_before, axis=(0, 1))
    std = np.std(roi_before, axis=(0, 1))
    appearance = np.array([
        float(mean[0]), float(mean[1]), float(mean[2]), float(std[0]), float(std[1]), float(std[2])
    ], dtype=np.float32)
    return mem_vec, appearance


def _build_targets(roi_before: np.ndarray, planner_cond: np.ndarray, delta_cond: np.ndarray, semantic_embed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = roi_before.shape
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h, dtype=np.float32), np.linspace(0.0, 1.0, w, dtype=np.float32), indexing="ij")
    center = np.exp(-((xx - 0.5) ** 2 + (yy - 0.55) ** 2) / 0.12)
    orientation = np.clip(1.0 - np.abs(xx - 0.5) * (0.6 + 0.4 * planner_cond[2]), 0.0, 1.0)
    reveal_strength = np.clip(0.25 + 0.45 * planner_cond[4] + 0.2 * delta_cond[5], 0.05, 0.8)
    motion_strength = np.clip(0.2 + 0.35 * planner_cond[5] + 0.2 * delta_cond[0], 0.05, 0.8)
    face_strength = np.clip(0.18 + 0.45 * semantic_embed[3], 0.05, 0.8)
    changed = np.clip(center * (0.35 + 0.25 * orientation + reveal_strength * 0.2 + motion_strength * 0.15 + face_strength * 0.25), 0.0, 1.0)
    blend_hint = np.clip(0.2 + 0.7 * changed, 0.0, 1.0)

    after = roi_before.copy()
    warmth = np.array([0.11 * face_strength, 0.07 * reveal_strength, -0.04 * motion_strength], dtype=np.float32)
    cool = np.array([-0.03 * motion_strength, 0.02 * motion_strength, 0.08 * motion_strength], dtype=np.float32)
    for c in range(3):
        after[..., c] = np.clip(after[..., c] + changed * warmth[c] + (1.0 - np.abs(xx - 0.5)) * changed * cool[c] * (1 if c == 2 else 0.5), 0.0, 1.0)
    return after.astype(np.float32), changed[..., None].astype(np.float32), blend_hint[..., None].astype(np.float32)


def build_patch_batch(request: PatchSynthesisRequest, roi_before: np.ndarray) -> PatchBatch:
    semantic = _semantic_embed(request.region.region_id)
    delta_cond, planner_cond = _delta_features(request)
    graph_cond = _graph_features(request)
    memory_cond, appearance_cond = _memory_and_appearance_features(request, roi_before)
    roi_after, changed_mask, blend_hint = _build_targets(roi_before, planner_cond, delta_cond, semantic)
    alpha_target = np.clip(0.15 + 0.8 * changed_mask + 0.05 * blend_hint, 0.0, 1.0)
    bbox_cond = np.array([request.region.bbox.x, request.region.bbox.y, request.region.bbox.w, request.region.bbox.h], dtype=np.float32)

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
    )


def output_from_prediction(request: PatchSynthesisRequest, pred: dict[str, np.ndarray | float], renderer_path: str, diagnostics: dict[str, object], batch: PatchBatch | None = None) -> PatchSynthesisOutput:
    rgb = pred["rgb"]
    alpha = pred["alpha"]
    uncertainty = pred["uncertainty"]
    conf = float(pred["confidence"])
    assert isinstance(rgb, np.ndarray) and isinstance(alpha, np.ndarray) and isinstance(uncertainty, np.ndarray)
    h, w, _ = rgb.shape
    if batch is not None:
        alpha = np.clip(alpha * (0.6 + 0.4 * batch.blend_hint[..., 0]), 0.0, 1.0)
        uncertainty = np.clip(0.7 * uncertainty + 0.3 * (1.0 - batch.changed_mask[..., 0]), 0.0, 1.0)
    exec_trace = {
        "renderer_path": renderer_path,
        "selected_render_strategy": f"LEARNED_{renderer_path.upper()}",
        "synthesis_mode": "learned_primary" if renderer_path == "learned_primary" else "legacy_fallback",
        "diagnostics": diagnostics,
        "alpha_semantics": "seam_aware_blend_probability",
        "uncertainty_semantics": "pixel_uncertainty_for_compositor_confidence_weighting",
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
        debug_trace=[f"renderer_path={renderer_path}", f"region={request.region.region_id}", f"alpha_mean={float(np.mean(alpha)):.4f}", f"unc_mean={float(np.mean(uncertainty)):.4f}"],
        execution_trace=exec_trace,
        metadata={
            "renderer_path": renderer_path,
            "diagnostics": diagnostics,
            "blend_hint_mean": float(np.mean(batch.blend_hint)) if batch is not None else 0.0,
            "changed_mask_mean": float(np.mean(batch.changed_mask)) if batch is not None else 0.0,
        },
    )
