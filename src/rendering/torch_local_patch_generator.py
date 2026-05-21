from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rendering.patch_conditioning_contract import GLOBAL_COND_DIM, MODE_DIM, ROLE_DIM

BASE_LOCAL_INPUT_CHANNELS = 9
REFERENCE_TENSOR_INPUT_CHANNELS = 5
LOCAL_INPUT_CHANNELS = BASE_LOCAL_INPUT_CHANNELS + REFERENCE_TENSOR_INPUT_CHANNELS
from rendering.patch_tensor_utils import map_to_shape

from rendering.target_provenance_policy import target_supervision_weight

if TYPE_CHECKING:
    from rendering.trainable_patch_renderer import PatchBatch

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class _UnavailableTorchModule:
    def __init__(self, *args, **kwargs) -> None:
        raise TorchBackendUnavailableError("TorchLocalPatchGenerator requires torch, but torch is unavailable")


@dataclass(slots=True)
class _TorchBatch:
    roi_before: "torch.Tensor"
    roi_after: "torch.Tensor"
    local_maps: "torch.Tensor"
    global_cond: "torch.Tensor"


class TorchLocalPatchGeneratorNet(nn.Module if nn is not None else _UnavailableTorchModule):
    def __init__(self, global_dim: int, hidden: int = 48, input_channels: int = LOCAL_INPUT_CHANNELS) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoder = nn.Sequential(nn.Conv2d(self.input_channels, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.cond = nn.Sequential(nn.Linear(global_dim, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden * 2))
        self.residual = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, 3, 1))
        self.alpha = nn.Sequential(nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden // 2, 1, 1))
        self.uncertainty = nn.Sequential(nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden // 2, 1, 1))

    def forward(self, local_maps: "torch.Tensor", global_cond: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        feat = self.encoder(local_maps)
        gamma, beta = torch.chunk(self.cond(global_cond), 2, dim=1)
        feat = feat * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        return {"rgb_residual": torch.tanh(self.residual(feat)), "alpha_logits": self.alpha(feat), "uncertainty_logits": self.uncertainty(feat)}


class TorchBackendUnavailableError(RuntimeError):
    pass


class TorchLocalPatchGenerator:
    def __init__(self, device: str | None = None, seed: int = 11) -> None:
        if torch is None:
            raise TorchBackendUnavailableError("TorchLocalPatchGenerator requires torch, but torch is unavailable")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.net = TorchLocalPatchGeneratorNet(global_dim=GLOBAL_COND_DIM, input_channels=LOCAL_INPUT_CHANNELS).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-4)


    @staticmethod
    def _reference_inputs(batch: "PatchBatch", shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        h, w = shape_hw
        ref_rgb = getattr(batch, "reference_rgb", None)
        if isinstance(ref_rgb, np.ndarray) and ref_rgb.shape == (h, w, 3):
            rgb = np.clip(ref_rgb.astype(np.float32), 0.0, 1.0)
        else:
            rgb = np.zeros((h, w, 3), dtype=np.float32)
        ref_mask = getattr(batch, "reference_mask", None)
        mask = map_to_shape(ref_mask if isinstance(ref_mask, np.ndarray) else None, (h, w), 0.0)
        ref_validity = getattr(batch, "reference_validity", None)
        validity = map_to_shape(ref_validity if isinstance(ref_validity, np.ndarray) else None, (h, w), 0.0)
        used = bool(np.any(validity > 0.0))
        return rgb, mask, validity, used

    @staticmethod
    def _global_cond(batch: "PatchBatch") -> np.ndarray:
        out = np.concatenate([
            batch.semantic_embed,
            batch.delta_cond,
            batch.planner_cond,
            batch.graph_cond,
            batch.memory_cond,
            batch.appearance_cond,
            batch.bbox_cond,
            batch.mode_cond if batch.mode_cond is not None else np.zeros((MODE_DIM,), dtype=np.float32),
            batch.role_cond if batch.role_cond is not None else np.zeros((ROLE_DIM,), dtype=np.float32),
        ]).astype(np.float32)
        if out.shape[0] != GLOBAL_COND_DIM:
            raise RuntimeError(f"Global conditioning dim mismatch: expected {GLOBAL_COND_DIM}, got {out.shape[0]}")
        return out

    def _to_torch_batch(self, batch: "PatchBatch") -> _TorchBatch:
        h, w, _ = batch.roi_before.shape
        maps = [batch.changed_mask, batch.blend_hint, batch.alpha_target, map_to_shape(batch.preservation_mask, (h, w), 0.0), map_to_shape(batch.uncertainty_target, (h, w), 0.0), map_to_shape(batch.seam_prior, (h, w), 0.0)]
        reference_rgb, reference_mask, reference_validity, reference_used = self._reference_inputs(batch, (h, w))
        local = np.concatenate([batch.roi_before.astype(np.float32), *maps, reference_rgb, reference_mask, reference_validity], axis=2)
        if isinstance(batch.conditioning_summary, dict):
            batch.conditioning_summary["reference_tensor_input_channels"] = REFERENCE_TENSOR_INPUT_CHANNELS
            batch.conditioning_summary["reference_tensor_input_used"] = bool(reference_used)
            batch.conditioning_summary["reference_tensor_zero_fallback"] = bool(not reference_used)
            batch.conditioning_summary["local_tensor_input_channels"] = int(local.shape[2])
        return _TorchBatch(
            roi_before=torch.from_numpy(batch.roi_before.transpose(2, 0, 1)[None]).float().to(self.device),
            roi_after=torch.from_numpy(batch.roi_after.transpose(2, 0, 1)[None]).float().to(self.device),
            local_maps=torch.from_numpy(local.transpose(2, 0, 1)[None]).float().to(self.device),
            global_cond=torch.from_numpy(self._global_cond(batch)[None]).float().to(self.device),
        )

    def _predict_tensors(self, batch: "PatchBatch") -> dict[str, "torch.Tensor"]:
        tb = self._to_torch_batch(batch)
        out = self.net(tb.local_maps, tb.global_cond)
        changed, blend, alpha_target = tb.local_maps[:, 3:4], tb.local_maps[:, 4:5], tb.local_maps[:, 5:6]
        preservation, uncertainty_target, seam = tb.local_maps[:, 6:7], tb.local_maps[:, 7:8], tb.local_maps[:, 8:9]
        edit_gate = torch.clamp(0.6 * changed + 0.4 * blend, 0.0, 1.0) * (1.0 - 0.85 * preservation)
        rgb_base = torch.clamp(tb.roi_before + 0.25 * out["rgb_residual"] * edit_gate, 0.0, 1.0)
        reference_rgb = tb.local_maps[:, 9:12]
        reference_mask = tb.local_maps[:, 12:13]
        reference_validity = tb.local_maps[:, 13:14]
        reference_gate = torch.clamp(reference_mask * reference_validity * edit_gate * 0.10, 0.0, 0.18)
        rgb = torch.clamp(rgb_base * (1.0 - reference_gate) + reference_rgb * reference_gate, 0.0, 1.0)
        alpha = torch.sigmoid(out["alpha_logits"]) * (0.55 * blend + 0.45 * alpha_target)
        uncertainty = torch.clamp(torch.sigmoid(out["uncertainty_logits"]) * 0.6 + uncertainty_target * 0.4 + seam * 0.05, 0.0, 1.0)
        return {"rgb": rgb, "alpha": alpha, "uncertainty": uncertainty, "tb": tb, "reference_tensor_input_channels": REFERENCE_TENSOR_INPUT_CHANNELS, "reference_tensor_input_used": bool(float(torch.sum(reference_validity).detach().cpu()) > 0.0), "reference_tensor_zero_fallback": bool(float(torch.sum(reference_validity).detach().cpu()) <= 0.0)}

    def infer(self, batch: "PatchBatch") -> dict[str, np.ndarray | float]:
        self.net.eval()
        with torch.no_grad():
            pred = self._predict_tensors(batch)
        rgb = pred["rgb"][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        alpha = pred["alpha"][0, 0].cpu().numpy().astype(np.float32)
        uncertainty = pred["uncertainty"][0, 0].cpu().numpy().astype(np.float32)
        drift = float(np.mean(np.abs(rgb - batch.roi_before) * map_to_shape(batch.preservation_mask, alpha.shape, 0.0)))
        conf = float(np.clip(1.0 - float(np.mean(uncertainty)) - drift, 0.0, 1.0))
        return {"rgb": rgb, "alpha": alpha, "uncertainty": uncertainty, "confidence": conf, "residual_mean": float(np.mean(np.abs(rgb - batch.roi_before))), "alpha_mean": float(np.mean(alpha)), "uncertainty_mean": float(np.mean(uncertainty)), "preservation_drift": drift, "memory_cond_norm": float(np.linalg.norm(batch.memory_cond)), "appearance_cond_norm": float(np.linalg.norm(batch.appearance_cond)), "reference_tensor_input_channels": REFERENCE_TENSOR_INPUT_CHANNELS, "reference_tensor_input_used": bool(pred.get("reference_tensor_input_used", False)), "reference_tensor_zero_fallback": bool(pred.get("reference_tensor_zero_fallback", True))}

    def train_step(self, batch: "PatchBatch", lr: float = 1e-4) -> dict[str, float]:
        for g in self.optim.param_groups:
            g["lr"] = lr
        self.net.train()
        pred = self._predict_tensors(batch)
        tb = pred["tb"]
        changed, preservation, seam = tb.local_maps[:, 3:4], tb.local_maps[:, 6:7], tb.local_maps[:, 8:9]
        uncertainty_target, alpha_target = tb.local_maps[:, 7:8], tb.local_maps[:, 5:6]
        reconstruction_loss = ((pred["rgb"] - tb.roi_after) ** 2 * (0.2 + changed)).mean()
        alpha_loss = ((pred["alpha"] - alpha_target) ** 2).mean()
        uncertainty_loss = ((pred["uncertainty"] - uncertainty_target) ** 2).mean()
        preservation_loss = (torch.abs(pred["rgb"] - tb.roi_before) * preservation).mean()
        seam_loss = (torch.abs(pred["rgb"] - tb.roi_after) * seam).mean()
        drift_penalty = torch.abs(pred["rgb"] - tb.roi_before).mean()
        total = reconstruction_loss + 0.3 * alpha_loss + 0.25 * uncertainty_loss + 0.4 * preservation_loss + 0.15 * seam_loss + 0.1 * drift_penalty
        self.optim.zero_grad(); total.backward(); self.optim.step()
        supervision_weight = target_supervision_weight(
            str((batch.conditioning_summary or {}).get("training_target_quality", "unknown"))
            if isinstance(batch.conditioning_summary, dict)
            else "unknown"
        )
        total_value = float(total.item())
        return {"total_loss": total_value, "weighted_total_loss": float(total_value * supervision_weight), "target_supervision_weight": float(supervision_weight), "reconstruction_loss": float(reconstruction_loss.item()), "alpha_loss": float(alpha_loss.item()), "uncertainty_calibration_loss": float(uncertainty_loss.item()), "appearance_preservation_loss": float(preservation_loss.item()), "seam_loss": float(seam_loss.item()), "drift_penalty": float(drift_penalty.item())}

    def eval_step(self, batch: "PatchBatch") -> dict[str, float]:
        self.net.eval()
        with torch.no_grad():
            pred = self._predict_tensors(batch)
            tb = pred["tb"]
            total = ((pred["rgb"] - tb.roi_after) ** 2).mean() + ((pred["alpha"] - tb.local_maps[:, 5:6]) ** 2).mean()
        return {"total_loss": float(total.item()), "alpha_mae": float(torch.abs(pred["alpha"] - tb.local_maps[:, 5:6]).mean().item()), "uncertainty_mean": float(pred["uncertainty"].mean().item())}

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.net.state_dict(), "local_input_channels": LOCAL_INPUT_CHANNELS, "reference_tensor_input_channels": REFERENCE_TENSOR_INPUT_CHANNELS}, str(path))

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "TorchLocalPatchGenerator":
        inst = cls(device=device)
        data = torch.load(str(path), map_location=inst.device)
        state = data.get("state_dict", data) if isinstance(data, dict) else data
        if not isinstance(state, dict):
            raise ValueError("torch local patch checkpoint missing state_dict")
        upgraded = False
        current = inst.net.state_dict()
        load_state = dict(state)
        first_key = "encoder.0.weight"
        if first_key in load_state and first_key in current and tuple(load_state[first_key].shape) != tuple(current[first_key].shape):
            old_weight = load_state[first_key]
            new_weight = current[first_key].clone()
            if (
                len(old_weight.shape) == 4
                and len(new_weight.shape) == 4
                and int(old_weight.shape[0]) == int(new_weight.shape[0])
                and int(old_weight.shape[2]) == int(new_weight.shape[2])
                and int(old_weight.shape[3]) == int(new_weight.shape[3])
                and int(old_weight.shape[1]) == BASE_LOCAL_INPUT_CHANNELS
                and int(new_weight.shape[1]) == LOCAL_INPUT_CHANNELS
            ):
                new_weight.zero_()
                new_weight[:, :BASE_LOCAL_INPUT_CHANNELS, :, :] = old_weight
                load_state[first_key] = new_weight
                upgraded = True
            else:
                raise ValueError(
                    "TorchLocalPatchGenerator checkpoint first conv input channel mismatch: "
                    f"checkpoint_shape={tuple(old_weight.shape)}, runtime_shape={tuple(new_weight.shape)}"
                )
        incompatible = [
            (key, tuple(value.shape), tuple(current[key].shape))
            for key, value in load_state.items()
            if key in current and tuple(value.shape) != tuple(current[key].shape)
        ]
        if incompatible:
            key, checkpoint_shape, runtime_shape = incompatible[0]
            raise ValueError(f"TorchLocalPatchGenerator checkpoint tensor shape mismatch for {key}: checkpoint={checkpoint_shape}, runtime={runtime_shape}")
        missing, unexpected = inst.net.load_state_dict(load_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "TorchLocalPatchGenerator checkpoint key mismatch after reference-channel upgrade: "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
        inst.checkpoint_compatibility = {
            "checkpoint_reference_channel_upgrade": bool(upgraded),
            "checkpoint_local_input_channels": int(data.get("local_input_channels", BASE_LOCAL_INPUT_CHANNELS)) if isinstance(data, dict) else BASE_LOCAL_INPUT_CHANNELS,
            "runtime_local_input_channels": LOCAL_INPUT_CHANNELS,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        inst.net.eval()
        return inst
