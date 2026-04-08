from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.schema import BBox, RegionRef
from learned.interfaces import TemporalRefinementOutput, TemporalRefinementRequest
from utils_tensor import shape


class TemporalInputError(ValueError):
    pass


class TemporalInferenceError(RuntimeError):
    pass


@dataclass(slots=True)
class TemporalBatch:
    previous_frame: np.ndarray
    current_frame: np.ndarray
    target_frame: np.ndarray
    changed_mask: np.ndarray
    alpha_hint: np.ndarray
    confidence_hint: np.ndarray
    transition_cond: np.ndarray
    memory_cond: np.ndarray
    history_cond: np.ndarray


class TrainableTemporalConsistencyModel:
    """Region-aware temporal refinement model with explicit history/transition conditioning."""

    def __init__(self, pixel_dim: int = 16, cond_dim: int = 16, hidden_dim: int = 56, seed: int = 23) -> None:
        rng = np.random.default_rng(seed)
        self.pixel_dim = pixel_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.W_pixel = (rng.standard_normal((pixel_dim, hidden_dim)) * 0.05).astype(np.float32)
        self.W_cond = (rng.standard_normal((cond_dim, hidden_dim)) * 0.05).astype(np.float32)
        self.b_hidden = np.zeros((hidden_dim,), dtype=np.float32)
        self.W_residual = (rng.standard_normal((hidden_dim, 3)) * 0.05).astype(np.float32)
        self.b_residual = np.zeros((3,), dtype=np.float32)
        self.W_gate = (rng.standard_normal((hidden_dim, 1)) * 0.04).astype(np.float32)
        self.b_gate = np.zeros((1,), dtype=np.float32)
        self.W_conf = (rng.standard_normal((hidden_dim, 1)) * 0.04).astype(np.float32)
        self.b_conf = np.zeros((1,), dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _pixel_features(batch: TemporalBatch) -> np.ndarray:
        h, w, _ = batch.current_frame.shape
        delta = batch.current_frame - batch.previous_frame
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h, dtype=np.float32), np.linspace(0.0, 1.0, w, dtype=np.float32), indexing="ij")
        return np.concatenate(
            [
                batch.previous_frame,
                batch.current_frame,
                delta,
                batch.changed_mask,
                batch.alpha_hint,
                batch.confidence_hint,
                yy[..., None],
                xx[..., None],
                batch.changed_mask * batch.alpha_hint,
                batch.changed_mask * batch.confidence_hint,
            ],
            axis=2,
        )

    @staticmethod
    def _global_condition(batch: TemporalBatch) -> np.ndarray:
        return np.concatenate([batch.transition_cond, batch.memory_cond, batch.history_cond], axis=0).astype(np.float32)

    def forward(self, batch: TemporalBatch) -> dict[str, np.ndarray | float]:
        pix = self._pixel_features(batch)
        h, w, _ = pix.shape
        x = pix.reshape(-1, self.pixel_dim)
        cond = self._global_condition(batch)
        hidden_pre = x @ self.W_pixel + (cond @ self.W_cond)[None, :] + self.b_hidden[None, :]
        hidden = self._relu(hidden_pre)

        residual_raw = hidden @ self.W_residual + self.b_residual[None, :]
        gate_raw = hidden @ self.W_gate + self.b_gate[None, :]
        confidence_raw = hidden @ self.W_conf + self.b_conf[None, :]

        residual = np.tanh(residual_raw) * 0.4
        gate = self._sigmoid(gate_raw)
        temporal_conf = self._sigmoid(confidence_raw)

        prev = batch.previous_frame.reshape(-1, 3)
        cur = batch.current_frame.reshape(-1, 3)
        changed = batch.changed_mask.reshape(-1, 1)
        alpha = batch.alpha_hint.reshape(-1, 1)
        conf = batch.confidence_hint.reshape(-1, 1)

        stable_weight = gate * (1.0 - changed) * (0.2 + 0.8 * conf) * (0.15 + 0.85 * (1.0 - alpha))
        refined = np.clip(cur + residual + stable_weight * (prev - cur), 0.0, 1.0)
        confidence = float(np.clip(np.mean(temporal_conf) * (0.5 + 0.5 * np.mean(conf)), 0.0, 1.0))

        return {
            "refined": refined.reshape(h, w, 3),
            "gate": gate.reshape(h, w),
            "temporal_conf": temporal_conf.reshape(h, w),
            "stable_weight": stable_weight.reshape(h, w),
            "hidden": hidden,
            "hidden_pre": hidden_pre,
            "x": x,
            "cond": cond,
            "residual_raw": residual_raw,
            "gate_raw": gate_raw,
            "confidence_raw": confidence_raw,
            "confidence": confidence,
        }

    def compute_losses(self, batch: TemporalBatch, out: dict[str, np.ndarray | float]) -> dict[str, float]:
        refined = out["refined"]
        gate = out["gate"]
        temporal_conf = out["temporal_conf"]
        stable_weight = out["stable_weight"]
        assert isinstance(refined, np.ndarray) and isinstance(gate, np.ndarray) and isinstance(temporal_conf, np.ndarray) and isinstance(stable_weight, np.ndarray)

        changed = np.clip(batch.changed_mask[..., 0], 0.0, 1.0)
        inv = 1.0 - changed
        recon = float(np.mean(((refined - batch.target_frame) ** 2) * (0.45 + 0.95 * changed[..., None])))

        pred_delta = refined - batch.previous_frame
        target_delta = batch.target_frame - batch.previous_frame
        flicker = float(np.mean(np.abs(pred_delta - target_delta) * (0.2 + 0.8 * changed[..., None])))

        region_stability = float(np.mean(((refined - batch.current_frame) ** 2) * inv[..., None]))

        seam_band = np.abs(changed - np.pad(changed[1:, :], ((0, 1), (0, 0)), mode="edge"))
        seam_band += np.abs(changed - np.pad(changed[:, 1:], ((0, 0), (0, 1)), mode="edge"))
        seam_band = np.clip(seam_band, 0.0, 1.0)
        seam_temporal = float(np.mean(np.abs(refined - batch.previous_frame) * seam_band[..., None]))

        confidence_target = np.clip(1.0 - np.mean(np.abs(refined - batch.target_frame), axis=2), 0.0, 1.0)
        conf_calibration = float(np.mean((temporal_conf - confidence_target) ** 2))

        gate_reg = float(np.mean(np.abs(gate - (inv * (0.3 + 0.7 * batch.confidence_hint[..., 0])))))
        stable_consistency = float(np.mean(np.abs(stable_weight - (inv * (0.15 + 0.5 * batch.confidence_hint[..., 0])))))

        total = recon + 0.35 * flicker + 0.25 * region_stability + 0.18 * seam_temporal + 0.22 * conf_calibration + 0.12 * gate_reg + 0.1 * stable_consistency
        return {
            "total_loss": total,
            "reconstruction_loss": recon,
            "flicker_loss": flicker,
            "region_stability_loss": region_stability,
            "seam_temporal_loss": seam_temporal,
            "confidence_calibration_loss": conf_calibration,
            "gate_regularization_loss": gate_reg,
            "stable_weight_consistency_loss": stable_consistency,
        }

    def train_step(self, batch: TemporalBatch, lr: float = 1e-3) -> dict[str, float]:
        out = self.forward(batch)
        losses = self.compute_losses(batch, out)

        refined = out["refined"].reshape(-1, 3)
        gate = out["gate"].reshape(-1, 1)
        temporal_conf = out["temporal_conf"].reshape(-1, 1)
        hidden = out["hidden"]
        hidden_pre = out["hidden_pre"]
        x = out["x"]
        cond = out["cond"]
        residual_raw = out["residual_raw"]
        gate_raw = out["gate_raw"]
        confidence_raw = out["confidence_raw"]
        assert isinstance(hidden, np.ndarray) and isinstance(hidden_pre, np.ndarray) and isinstance(x, np.ndarray) and isinstance(cond, np.ndarray)
        assert isinstance(residual_raw, np.ndarray) and isinstance(gate_raw, np.ndarray) and isinstance(confidence_raw, np.ndarray)

        target = batch.target_frame.reshape(-1, 3)
        changed = batch.changed_mask.reshape(-1, 1)
        inv = 1.0 - changed

        w_recon = (0.45 + 0.95 * changed)
        d_refined = 2.0 * (refined - target) * w_recon / max(1, refined.size)
        d_residual = d_refined
        d_residual_raw = d_residual * (1.0 - np.tanh(residual_raw) ** 2) * 0.4

        gate_t = inv * (0.3 + 0.7 * batch.confidence_hint.reshape(-1, 1))
        d_gate = np.sign(gate - gate_t) * 0.12 / max(1, gate.size)
        d_gate_raw = d_gate * gate * (1.0 - gate)

        conf_target = np.clip(1.0 - np.mean(np.abs(refined - target), axis=1, keepdims=True), 0.0, 1.0)
        d_conf = 2.0 * (temporal_conf - conf_target) * 0.22 / max(1, temporal_conf.size)
        d_conf_raw = d_conf * temporal_conf * (1.0 - temporal_conf)

        grad_W_res = hidden.T @ d_residual_raw
        grad_b_res = d_residual_raw.sum(axis=0)
        grad_W_gate = hidden.T @ d_gate_raw
        grad_b_gate = d_gate_raw.sum(axis=0)
        grad_W_conf = hidden.T @ d_conf_raw
        grad_b_conf = d_conf_raw.sum(axis=0)

        d_hidden = d_residual_raw @ self.W_residual.T + d_gate_raw @ self.W_gate.T + d_conf_raw @ self.W_conf.T
        d_hidden_pre = d_hidden * (hidden_pre > 0.0).astype(np.float32)

        grad_W_pixel = x.T @ d_hidden_pre
        grad_b_hidden = d_hidden_pre.sum(axis=0)
        grad_cond_proj = d_hidden_pre.sum(axis=0)
        grad_W_cond = np.outer(cond, grad_cond_proj)

        self.W_residual -= lr * grad_W_res.astype(np.float32)
        self.b_residual -= lr * grad_b_res.astype(np.float32)
        self.W_gate -= lr * grad_W_gate.astype(np.float32)
        self.b_gate -= lr * grad_b_gate.astype(np.float32)
        self.W_conf -= lr * grad_W_conf.astype(np.float32)
        self.b_conf -= lr * grad_b_conf.astype(np.float32)
        self.W_pixel -= lr * grad_W_pixel.astype(np.float32)
        self.W_cond -= lr * grad_W_cond.astype(np.float32)
        self.b_hidden -= lr * grad_b_hidden.astype(np.float32)
        return losses

    def eval_step(self, batch: TemporalBatch) -> dict[str, float]:
        out = self.forward(batch)
        losses = self.compute_losses(batch, out)
        refined = out["refined"]
        assert isinstance(refined, np.ndarray)
        recon_mae = float(np.mean(np.abs(refined - batch.target_frame)))
        flicker_delta_mae = float(np.mean(np.abs((refined - batch.previous_frame) - (batch.target_frame - batch.previous_frame))))
        region_consistency = float(np.mean(np.abs((refined - batch.current_frame) * (1.0 - batch.changed_mask))))
        losses.update(
            {
                "reconstruction_mae": recon_mae,
                "flicker_delta_mae": flicker_delta_mae,
                "region_consistency_mae": region_consistency,
            }
        )
        return losses

    def infer(self, batch: TemporalBatch) -> dict[str, np.ndarray | float]:
        return self.forward(batch)

    def save(self, path: str) -> None:
        payload = {
            "pixel_dim": self.pixel_dim,
            "cond_dim": self.cond_dim,
            "hidden_dim": self.hidden_dim,
            "W_pixel": self.W_pixel.tolist(),
            "W_cond": self.W_cond.tolist(),
            "b_hidden": self.b_hidden.tolist(),
            "W_residual": self.W_residual.tolist(),
            "b_residual": self.b_residual.tolist(),
            "W_gate": self.W_gate.tolist(),
            "b_gate": self.b_gate.tolist(),
            "W_conf": self.W_conf.tolist(),
            "b_conf": self.b_conf.tolist(),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TrainableTemporalConsistencyModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(pixel_dim=int(payload["pixel_dim"]), cond_dim=int(payload["cond_dim"]), hidden_dim=int(payload["hidden_dim"]))
        model.W_pixel = np.asarray(payload["W_pixel"], dtype=np.float32)
        model.W_cond = np.asarray(payload["W_cond"], dtype=np.float32)
        model.b_hidden = np.asarray(payload["b_hidden"], dtype=np.float32)
        model.W_residual = np.asarray(payload["W_residual"], dtype=np.float32)
        model.b_residual = np.asarray(payload["b_residual"], dtype=np.float32)
        model.W_gate = np.asarray(payload["W_gate"], dtype=np.float32)
        model.b_gate = np.asarray(payload["b_gate"], dtype=np.float32)
        model.W_conf = np.asarray(payload["W_conf"], dtype=np.float32)
        model.b_conf = np.asarray(payload["b_conf"], dtype=np.float32)
        return model


def _np_frame(frame: list, field: str) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise TemporalInputError(f"{field} must be HxWx3")
    return np.clip(arr, 0.0, 1.0)


def _bbox_pixels(bbox: BBox, h: int, w: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(w - 1, int(bbox.x * w)))
    y0 = max(0, min(h - 1, int(bbox.y * h)))
    x1 = max(x0 + 1, min(w, int((bbox.x + bbox.w) * w)))
    y1 = max(y0 + 1, min(h, int((bbox.y + bbox.h) * h)))
    return x0, y0, x1, y1


def _roi_mask(h: int, w: int, regions: list[RegionRef]) -> np.ndarray:
    mask = np.zeros((h, w, 1), dtype=np.float32)
    for r in regions:
        x0, y0, x1, y1 = _bbox_pixels(r.bbox, h, w)
        mask[y0:y1, x0:x1, 0] = 1.0
    return mask


def build_temporal_batch(request: TemporalRefinementRequest, target_frame: list | None = None, history_frame: list | None = None) -> TemporalBatch:
    prev = _np_frame(request.previous_frame, "previous_frame")
    cur = _np_frame(request.current_composed_frame, "current_composed_frame")
    if prev.shape != cur.shape:
        raise TemporalInputError("previous_frame and current_composed_frame shape mismatch")
    tgt = _np_frame(target_frame if target_frame is not None else request.current_composed_frame, "target_frame")
    if tgt.shape != cur.shape:
        raise TemporalInputError("target_frame shape mismatch")

    h, w, _ = cur.shape
    changed_mask = _roi_mask(h, w, request.changed_regions)

    channels = request.memory_channels if isinstance(request.memory_channels, dict) else {}
    patch_alpha = channels.get("patch_alpha", {}) if isinstance(channels.get("patch_alpha", {}), dict) else {}
    patch_conf = channels.get("patch_confidence", {}) if isinstance(channels.get("patch_confidence", {}), dict) else {}
    hidden = channels.get("hidden_regions", {}) if isinstance(channels.get("hidden_regions", {}), dict) else {}
    body = channels.get("body_regions", {}) if isinstance(channels.get("body_regions", {}), dict) else {}

    alpha_hint = np.clip(np.full((h, w, 1), float(patch_alpha.get("mean_alpha", 0.35)), dtype=np.float32) + 0.35 * changed_mask, 0.0, 1.0)
    conf_hint = np.clip(np.full((h, w, 1), float(patch_conf.get("mean_confidence", 0.65)), dtype=np.float32), 0.0, 1.0)

    drift = float(hidden.get("drift", 0.0))
    roi_count = float(body.get("roi_count", len(request.changed_regions)))
    transition_cond = np.array(
        [
            min(1.0, len(request.changed_regions) / 8.0),
            min(1.0, roi_count / 8.0),
            min(1.0, drift * 5.0),
            1.0 if drift > 0.08 else 0.0,
            1.0 if drift <= 0.08 else 0.0,
            float(np.mean(changed_mask)),
        ],
        dtype=np.float32,
    )
    memory_cond = np.array(
        [
            float(channels.get("identity", {}) != {}),
            float(channels.get("body_regions", {}) != {}),
            float(channels.get("hidden_regions", {}) != {}),
            float(channels.get("patch_alpha", {}) != {}),
            float(channels.get("patch_confidence", {}) != {}),
            float(patch_alpha.get("edge_alpha", patch_alpha.get("mean_alpha", 0.3))),
        ],
        dtype=np.float32,
    )
    hist_strength = 0.0
    if history_frame is not None:
        hist = _np_frame(history_frame, "history_frame")
        if hist.shape != cur.shape:
            raise TemporalInputError("history_frame shape mismatch")
        hist_strength = float(np.mean(np.abs(hist - prev)))
    history_cond = np.array([hist_strength, float(np.mean(np.abs(cur - prev))), float(np.mean(conf_hint)), float(np.mean(alpha_hint))], dtype=np.float32)

    return TemporalBatch(
        previous_frame=prev,
        current_frame=cur,
        target_frame=tgt,
        changed_mask=changed_mask,
        alpha_hint=alpha_hint,
        confidence_hint=conf_hint,
        transition_cond=transition_cond,
        memory_cond=memory_cond,
        history_cond=history_cond,
    )


def output_from_temporal_prediction(request: TemporalRefinementRequest, pred: dict[str, np.ndarray | float], temporal_path: str, metadata: dict[str, object]) -> TemporalRefinementOutput:
    refined = pred["refined"]
    gate = pred["gate"]
    stable_weight = pred["stable_weight"]
    temporal_conf = pred["temporal_conf"]
    if not isinstance(refined, np.ndarray) or not isinstance(gate, np.ndarray) or not isinstance(stable_weight, np.ndarray) or not isinstance(temporal_conf, np.ndarray):
        raise TemporalInferenceError("Temporal model prediction is malformed")

    region_scores: dict[str, float] = {}
    h, w, _ = refined.shape
    prev = np.asarray(request.previous_frame, dtype=np.float32)
    cur = np.asarray(request.current_composed_frame, dtype=np.float32)
    for region in request.changed_regions:
        x0, y0, x1, y1 = _bbox_pixels(region.bbox, h, w)
        if x1 <= x0 or y1 <= y0:
            continue
        local_refined = refined[y0:y1, x0:x1]
        local_prev = prev[y0:y1, x0:x1]
        local_cur = cur[y0:y1, x0:x1]
        drift_reduction = float(np.mean(np.abs(local_cur - local_prev)) - np.mean(np.abs(local_refined - local_prev)))
        conf = float(np.mean(temporal_conf[y0:y1, x0:x1]))
        score = float(np.clip(0.5 + 0.6 * drift_reduction + 0.3 * conf, 0.0, 1.0))
        region_scores[region.region_id] = score

    return TemporalRefinementOutput(
        refined_frame=np.clip(refined, 0.0, 1.0).tolist(),
        region_consistency_scores=region_scores,
        metadata={
            "backend": "trainable_temporal_consistency",
            "temporal_path": temporal_path,
            "learned_ready_usage": metadata,
            "temporal_confidence": float(pred.get("confidence", 0.0)),
            "gate_mean": float(np.mean(gate)),
            "stable_weight_mean": float(np.mean(stable_weight)),
            "changed_region_count": len(request.changed_regions),
            "history_used": bool(metadata.get("history_used", False)),
        },
    )


def extract_history_frame(memory_state: object) -> list | None:
    history = getattr(memory_state, "last_frames", None)
    if isinstance(history, list) and history:
        candidate = history[-1]
        if isinstance(candidate, list):
            h, w, c = shape(candidate)
            if h > 0 and w > 0 and c == 3:
                return candidate
    return None
