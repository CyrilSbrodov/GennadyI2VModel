from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rendering.patch_conditioning_contract import GLOBAL_COND_DIM, MODE_DIM, ROLE_DIM

BASE_LOCAL_INPUT_CHANNELS = 9
REFERENCE_TENSOR_INPUT_CHANNELS = 5
I2V_MOTION_INPUT_CHANNELS = 3
LOCAL_INPUT_CHANNELS = BASE_LOCAL_INPUT_CHANNELS + REFERENCE_TENSOR_INPUT_CHANNELS + I2V_MOTION_INPUT_CHANNELS
MATERIAL_GATE_MAX = 0.35
MATERIAL_GATE_INIT_BIAS = -3.0
from rendering.patch_tensor_utils import map_to_shape

from rendering.target_provenance_policy import target_supervision_weight

if TYPE_CHECKING:
    from rendering.trainable_patch_renderer import PatchBatch

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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


class _ConvBlock(nn.Module if nn is not None else _UnavailableTorchModule):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.SiLU())

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class TorchLocalPatchGeneratorNet(nn.Module if nn is not None else _UnavailableTorchModule):
    def __init__(self, global_dim: int, hidden: int = 32, input_channels: int = LOCAL_INPUT_CHANNELS) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.global_dim = int(global_dim)
        self.stem = _ConvBlock(self.input_channels, hidden)
        self.down1 = _ConvBlock(hidden, hidden * 2)
        self.down2 = _ConvBlock(hidden * 2, hidden * 4)
        self.pool = nn.AvgPool2d(2, 2)
        self.film = nn.Sequential(nn.Linear(global_dim, hidden * 8), nn.SiLU(), nn.Linear(hidden * 8, hidden * 8))
        self.up1 = _ConvBlock(hidden * 4 + hidden * 2, hidden * 2)
        self.up2 = _ConvBlock(hidden * 2 + hidden, hidden)
        self.rgb_residual = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, 3, 1))
        self.alpha = nn.Sequential(nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden // 2, 1, 1))
        self.uncertainty = nn.Sequential(nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden // 2, 1, 1))
        self.material_gate = nn.Sequential(nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden // 2, 1, 1))
        nn.init.constant_(self.material_gate[-1].bias, MATERIAL_GATE_INIT_BIAS)

    def forward(self, local_maps: "torch.Tensor", global_cond: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        s0 = self.stem(local_maps)
        d1 = self.down1(self.pool(s0))
        d2 = self.down2(self.pool(d1))
        gamma, beta = torch.chunk(self.film(global_cond), 2, dim=1)
        d2 = d2 * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        u1 = F.interpolate(d2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, d1], dim=1))
        u2 = F.interpolate(u1, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, s0], dim=1))
        return {
            "rgb_residual": torch.tanh(self.rgb_residual(u2)),
            "alpha_logits": self.alpha(u2),
            "uncertainty_logits": self.uncertainty(u2),
            "material_gate_logits": self.material_gate(u2),
        }


class TorchBackendUnavailableError(RuntimeError):
    pass


class TorchLocalPatchGenerator:
    @staticmethod
    def _i2v_tensor_used_np(flow_x: np.ndarray, flow_y: np.ndarray, deformation_mask: np.ndarray, eps: float = 1e-8) -> bool:
        return bool(np.any(np.abs(flow_x) > eps) or np.any(np.abs(flow_y) > eps) or np.any(deformation_mask > eps))

    @staticmethod
    def _i2v_tensor_used_torch(flow_x: "torch.Tensor", flow_y: "torch.Tensor", deformation_mask: "torch.Tensor", eps: float = 1e-8) -> bool:
        return bool(bool((flow_x.abs() > eps).any().detach().cpu().item()) or bool((flow_y.abs() > eps).any().detach().cpu().item()) or bool((deformation_mask > eps).any().detach().cpu().item()))

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
        mask = map_to_shape(getattr(batch, "reference_mask", None), (h, w), 0.0)
        validity = map_to_shape(getattr(batch, "reference_validity", None), (h, w), 0.0)
        used = bool(np.any(validity > 0.0))
        return rgb, mask, validity, used

    @staticmethod
    def _global_cond(batch: "PatchBatch") -> np.ndarray:
        mode_cond = batch.mode_cond if batch.mode_cond is not None else np.zeros((MODE_DIM,), dtype=np.float32)
        role_cond = batch.role_cond if batch.role_cond is not None else np.zeros((ROLE_DIM,), dtype=np.float32)
        out = np.concatenate(
            [
                batch.semantic_embed,
                batch.delta_cond,
                batch.planner_cond,
                batch.graph_cond,
                batch.memory_cond,
                batch.appearance_cond,
                batch.bbox_cond,
                mode_cond,
                role_cond,
            ]
        ).astype(np.float32)
        if out.shape[0] != GLOBAL_COND_DIM:
            raise RuntimeError(f"Global conditioning dim mismatch: expected {GLOBAL_COND_DIM}, got {out.shape[0]}")
        return out

    @staticmethod
    def _i2v_motion_inputs(batch: "PatchBatch", shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        h, w = shape_hw
        fx = map_to_shape(getattr(batch, "i2v_flow_x", None), (h, w), 0.0)
        fy = map_to_shape(getattr(batch, "i2v_flow_y", None), (h, w), 0.0)
        dm = map_to_shape(getattr(batch, "i2v_deformation_mask", None), (h, w), 0.0)
        used = TorchLocalPatchGenerator._i2v_tensor_used_np(fx, fy, dm)
        return fx, fy, dm, used

    def _to_torch_batch(self, batch: "PatchBatch") -> _TorchBatch:
        h, w, _ = batch.roi_before.shape
        if h < 4 or w < 4:
            raise ValueError(f"TorchLocalPatchGenerator requires ROI >= 4x4, got {h}x{w}")
        maps = [
            batch.changed_mask,
            batch.blend_hint,
            batch.alpha_target,
            map_to_shape(batch.preservation_mask, (h, w), 0.0),
            map_to_shape(batch.uncertainty_target, (h, w), 0.0),
            map_to_shape(batch.seam_prior, (h, w), 0.0),
        ]
        reference_rgb, reference_mask, reference_validity, reference_used = self._reference_inputs(batch, (h, w))
        i2v_flow_x, i2v_flow_y, i2v_deformation_mask, i2v_used = self._i2v_motion_inputs(batch, (h, w))
        local = np.concatenate([batch.roi_before.astype(np.float32), *maps, reference_rgb, reference_mask, reference_validity, i2v_flow_x, i2v_flow_y, i2v_deformation_mask], axis=2)
        if isinstance(batch.conditioning_summary, dict):
            batch.conditioning_summary.update(
                {
                    "reference_tensor_input_channels": REFERENCE_TENSOR_INPUT_CHANNELS,
                    "reference_tensor_input_used": bool(reference_used),
                    "reference_tensor_zero_fallback": bool(not reference_used),
                    "local_tensor_input_channels": int(local.shape[2]),
                    "reference_validity_mean": float(np.mean(reference_validity)),
                    "reference_mask_mean": float(np.mean(reference_mask)),
                    "i2v_motion_input_channels": I2V_MOTION_INPUT_CHANNELS,
                    "i2v_motion_tensor_used": bool(i2v_used),
                    "i2v_motion_tensor_zero_fallback": bool(not i2v_used),
                    "i2v_flow_x_mean": float(np.mean(i2v_flow_x)),
                    "i2v_flow_y_mean": float(np.mean(i2v_flow_y)),
                    "i2v_deformation_mask_mean": float(np.mean(i2v_deformation_mask)),
                }
            )
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
        reference_rgb, reference_mask, reference_validity = tb.local_maps[:, 9:12], tb.local_maps[:, 12:13], tb.local_maps[:, 13:14]
        i2v_flow_x, i2v_flow_y, i2v_deformation_mask = tb.local_maps[:, 14:15], tb.local_maps[:, 15:16], tb.local_maps[:, 16:17]
        edit_strength = torch.clamp(0.6 * changed + 0.4 * blend, 0.0, 1.0)
        edit_gate = torch.clamp(edit_strength * (1.0 - 0.95 * preservation), 0.0, 1.0)
        rgb_base = torch.clamp(tb.roi_before + 0.35 * out["rgb_residual"] * edit_gate, 0.0, 1.0)
        learned_gate = torch.sigmoid(out["material_gate_logits"])
        material_gate_candidate = learned_gate * reference_mask * reference_validity * edit_strength
        material_gate_raw = material_gate_candidate * (1.0 - preservation)
        material_gate = torch.clamp(material_gate_raw, 0.0, MATERIAL_GATE_MAX)
        rgb = torch.clamp(rgb_base * (1.0 - material_gate) + reference_rgb * material_gate, 0.0, 1.0)
        alpha = torch.sigmoid(out["alpha_logits"]) * (0.55 * blend + 0.45 * alpha_target) * (1.0 - 0.7 * preservation)
        ref_sensitive = torch.clamp(changed * reference_mask, 0.0, 1.0)
        ref_missing = ref_sensitive * (1.0 - reference_validity)
        uncertainty = torch.clamp(torch.sigmoid(out["uncertainty_logits"]) * 0.5 + uncertainty_target * 0.35 + seam * 0.12 + ref_missing * 0.18 + changed * 0.08, 0.0, 1.0)
        used = bool(float(torch.sum(reference_validity).detach().cpu()) > 0.0)
        suppressed = torch.clamp(material_gate_raw - material_gate, 0.0, 1.0) + material_gate_raw * preservation
        return {
            "rgb": rgb,
            "alpha": alpha,
            "uncertainty": uncertainty,
            "tb": tb,
            "reference_tensor_input_channels": REFERENCE_TENSOR_INPUT_CHANNELS,
            "reference_tensor_input_used": used,
            "reference_tensor_zero_fallback": not used,
            "material_gate": material_gate,
            "reference_validity": reference_validity,
            "reference_mask": reference_mask,
            "edit_gate": edit_gate,
            "material_gate_max": float(material_gate.max().item()),
            "material_gate_cap": float(MATERIAL_GATE_MAX),
            "material_gate_suppressed_by_preservation": float((material_gate_candidate * preservation).mean().item()),
            "reference_validity_mean": float(reference_validity.mean().item()),
            "reference_mask_mean": float(reference_mask.mean().item()),
            "material_gate_mean": float(material_gate.mean().item()),
            "suppressed_gate_mean": float(suppressed.mean().item()),
            "i2v_motion_input_channels": I2V_MOTION_INPUT_CHANNELS,
            "i2v_motion_tensor_used": self._i2v_tensor_used_torch(i2v_flow_x, i2v_flow_y, i2v_deformation_mask),
            "i2v_motion_tensor_zero_fallback": bool(not self._i2v_tensor_used_torch(i2v_flow_x, i2v_flow_y, i2v_deformation_mask)),
            "i2v_flow_x_mean": float(i2v_flow_x.mean().item()),
            "i2v_flow_y_mean": float(i2v_flow_y.mean().item()),
            "i2v_deformation_mask_mean": float(i2v_deformation_mask.mean().item()),
        }

    def infer(self, batch: "PatchBatch") -> dict[str, np.ndarray | float]:
        self.net.eval()
        with torch.no_grad():
            pred = self._predict_tensors(batch)
        rgb = pred["rgb"][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        alpha = pred["alpha"][0, 0].cpu().numpy().astype(np.float32)
        uncertainty = pred["uncertainty"][0, 0].cpu().numpy().astype(np.float32)
        drift = float(np.mean(np.abs(rgb - batch.roi_before) * map_to_shape(batch.preservation_mask, alpha.shape, 0.0)))
        return {"rgb": rgb, "alpha": alpha, "uncertainty": uncertainty, "confidence": float(np.clip(1.0 - float(np.mean(uncertainty)) - drift, 0.0, 1.0)), "residual_mean": float(np.mean(np.abs(rgb - batch.roi_before))), "alpha_mean": float(np.mean(alpha)), "uncertainty_mean": float(np.mean(uncertainty)), "preservation_drift": drift, "memory_cond_norm": float(np.linalg.norm(batch.memory_cond)), "appearance_cond_norm": float(np.linalg.norm(batch.appearance_cond)), "reference_tensor_input_channels": REFERENCE_TENSOR_INPUT_CHANNELS, "reference_tensor_input_used": bool(pred.get("reference_tensor_input_used", False)), "reference_tensor_zero_fallback": bool(pred.get("reference_tensor_zero_fallback", True)), "material_gate_mean": float(pred.get("material_gate_mean", 0.0)), "material_gate_max": float(pred.get("material_gate_max", 0.0)), "material_gate_cap": float(pred.get("material_gate_cap", MATERIAL_GATE_MAX)), "material_gate_suppressed_by_preservation": float(pred.get("material_gate_suppressed_by_preservation", 0.0)), "reference_validity_mean": float(pred.get("reference_validity_mean", 0.0)), "reference_mask_mean": float(pred.get("reference_mask_mean", 0.0)), "i2v_motion_input_channels": float(pred.get("i2v_motion_input_channels", I2V_MOTION_INPUT_CHANNELS)), "i2v_motion_tensor_used": bool(pred.get("i2v_motion_tensor_used", False)), "i2v_motion_tensor_zero_fallback": bool(pred.get("i2v_motion_tensor_zero_fallback", True)), "i2v_flow_x_mean": float(pred.get("i2v_flow_x_mean", 0.0)), "i2v_flow_y_mean": float(pred.get("i2v_flow_y_mean", 0.0)), "i2v_deformation_mask_mean": float(pred.get("i2v_deformation_mask_mean", 0.0)), "i2v_motion_region_reconstruction_loss": float(pred.get("i2v_motion_region_reconstruction_loss", 0.0)), "i2v_motion_preservation_penalty": float(pred.get("i2v_motion_preservation_penalty", 0.0))}

    def train_step(self, batch: "PatchBatch", lr: float = 1e-4) -> dict[str, float]:
        for g in self.optim.param_groups:
            g["lr"] = lr
        self.net.train()
        pred = self._predict_tensors(batch)
        tb = pred["tb"]
        changed = tb.local_maps[:, 3:4]
        preservation = tb.local_maps[:, 6:7]
        i2v_deformation_mask = tb.local_maps[:, 16:17]
        seam = tb.local_maps[:, 8:9]
        blend = tb.local_maps[:, 4:5]
        uncertainty_target, alpha_target = tb.local_maps[:, 7:8], tb.local_maps[:, 5:6]
        ref_mask, ref_validity, ref_rgb = pred["reference_mask"], pred["reference_validity"], tb.local_maps[:, 9:12]
        reconstruction_loss = ((pred["rgb"] - tb.roi_after) ** 2 * (0.2 + changed)).mean()
        alpha_loss = ((pred["alpha"] - alpha_target) ** 2).mean()
        uncertainty_loss = ((pred["uncertainty"] - uncertainty_target) ** 2).mean()
        preservation_loss = (torch.abs(pred["rgb"] - tb.roi_before) * preservation).mean()
        seam_loss = (torch.abs(pred["rgb"] - tb.roi_after) * seam).mean()
        drift_penalty = torch.abs(pred["rgb"] - tb.roi_before).mean()
        mat_region = ref_validity * ref_mask * torch.clamp(0.65 * changed + 0.35 * blend, 0.0, 1.0) * (1.0 - preservation)
        eps = 1e-6
        mat_denom = torch.clamp(mat_region.sum(), min=eps)
        material_consistency_loss = ((torch.abs(pred["rgb"] - ref_rgb) * mat_region).sum() / (mat_denom * 3.0))
        material_gate_preservation_penalty = (pred["material_gate"] * preservation).mean()
        material_gate_invalidity_penalty = ((1.0 - ref_validity) * pred["material_gate"]).mean()
        valid_scalar = torch.clamp(ref_validity.mean(), 0.0, 1.0)
        material_gate_area_penalty = torch.clamp(pred["material_gate"].mean() - 0.22, min=0.0) * valid_scalar
        material_gate_regularization = material_gate_preservation_penalty + material_gate_invalidity_penalty + material_gate_area_penalty
        i2v_motion_region_reconstruction_loss = (((pred["rgb"] - tb.roi_after) ** 2) * i2v_deformation_mask).mean()
        i2v_motion_preservation_penalty = (torch.abs(pred["rgb"] - tb.roi_before) * (1.0 - i2v_deformation_mask) * preservation).mean()
        total = (
            reconstruction_loss
            + 0.3 * alpha_loss
            + 0.25 * uncertainty_loss
            + 0.4 * preservation_loss
            + 0.15 * seam_loss
            + 0.1 * drift_penalty
            + 0.2 * material_consistency_loss
            + 0.1 * material_gate_regularization
            + 0.15 * i2v_motion_region_reconstruction_loss
            + 0.1 * i2v_motion_preservation_penalty
        )
        self.optim.zero_grad()
        total.backward()
        self.optim.step()
        supervision_weight = target_supervision_weight(str((batch.conditioning_summary or {}).get("training_target_quality", "unknown")) if isinstance(batch.conditioning_summary, dict) else "unknown")
        return {
            "total_loss": float(total.item()),
            "weighted_total_loss": float(total.item() * supervision_weight),
            "target_supervision_weight": float(supervision_weight),
            "reconstruction_loss": float(reconstruction_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "uncertainty_calibration_loss": float(uncertainty_loss.item()),
            "appearance_preservation_loss": float(preservation_loss.item()),
            "seam_loss": float(seam_loss.item()),
            "drift_penalty": float(drift_penalty.item()),
            "material_consistency_loss": float(material_consistency_loss.item()),
            "material_gate_regularization": float(material_gate_regularization.item()),
            "material_gate_preservation_penalty": float(material_gate_preservation_penalty.item()),
            "material_gate_invalidity_penalty": float(material_gate_invalidity_penalty.item()),
            "material_gate_area_penalty": float(material_gate_area_penalty.item()),
            "material_gate_mean": float(pred.get("material_gate_mean", 0.0)),
            "material_gate_max": float(pred.get("material_gate_max", 0.0)),
            "material_gate_cap": float(pred.get("material_gate_cap", MATERIAL_GATE_MAX)),
            "material_gate_suppressed_by_preservation": float(pred.get("material_gate_suppressed_by_preservation", 0.0)),
            "reference_validity_mean": float(pred.get("reference_validity_mean", 0.0)),
            "reference_tensor_input_used": float(pred.get("reference_tensor_input_used", False)),
            "reference_tensor_zero_fallback": float(pred.get("reference_tensor_zero_fallback", True)),
            "i2v_motion_region_reconstruction_loss": float(i2v_motion_region_reconstruction_loss.item()),
            "i2v_motion_preservation_penalty": float(i2v_motion_preservation_penalty.item()),
            "i2v_motion_input_channels": float(I2V_MOTION_INPUT_CHANNELS),
            "i2v_motion_tensor_used": float(pred.get("i2v_motion_tensor_used", False)),
            "i2v_motion_tensor_zero_fallback": float(pred.get("i2v_motion_tensor_zero_fallback", True)),
        }

    def eval_step(self, batch: "PatchBatch") -> dict[str, float]:
        self.net.eval()
        with torch.no_grad():
            pred = self._predict_tensors(batch)
            tb = pred["tb"]
            mat_region = pred["reference_validity"] * pred["reference_mask"] * (1.0 - tb.local_maps[:, 6:7])
            mat_denom = torch.clamp(mat_region.sum(), min=1e-6)
            mat_consistency = (torch.abs(pred["rgb"] - tb.local_maps[:, 9:12]) * mat_region).sum() / (mat_denom * 3.0)
            i2v_deformation_mask = tb.local_maps[:, 16:17]
            i2v_motion_region_reconstruction_loss = (((pred["rgb"] - tb.roi_after) ** 2) * i2v_deformation_mask).mean()
            i2v_motion_preservation_penalty = (torch.abs(pred["rgb"] - tb.roi_before) * (1.0 - i2v_deformation_mask) * tb.local_maps[:, 6:7]).mean()
            total = ((pred["rgb"] - tb.roi_after) ** 2).mean() + ((pred["alpha"] - tb.local_maps[:, 5:6]) ** 2).mean() + 0.2 * mat_consistency + 0.15 * i2v_motion_region_reconstruction_loss + 0.1 * i2v_motion_preservation_penalty
        return {
            "total_loss": float(total.item()),
            "alpha_mae": float(torch.abs(pred["alpha"] - tb.local_maps[:, 5:6]).mean().item()),
            "uncertainty_mean": float(pred["uncertainty"].mean().item()),
            "material_consistency_loss": float(mat_consistency.item()),
            "material_gate_mean": float(pred.get("material_gate_mean", 0.0)),
            "material_gate_max": float(pred.get("material_gate_max", 0.0)),
            "material_gate_cap": float(pred.get("material_gate_cap", MATERIAL_GATE_MAX)),
            "reference_validity_mean": float(pred.get("reference_validity_mean", 0.0)),
            "reference_mask_mean": float(pred.get("reference_mask_mean", 0.0)),
            "material_gate_suppressed_by_preservation": float(pred.get("material_gate_suppressed_by_preservation", 0.0)),
            "i2v_motion_region_reconstruction_loss": float(i2v_motion_region_reconstruction_loss.item()),
            "i2v_motion_preservation_penalty": float(i2v_motion_preservation_penalty.item()),
            "i2v_motion_tensor_used": float(pred.get("i2v_motion_tensor_used", False)),
            "i2v_motion_tensor_zero_fallback": float(pred.get("i2v_motion_tensor_zero_fallback", True)),
            "i2v_deformation_mask_mean": float(pred.get("i2v_deformation_mask_mean", 0.0)),
        }

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.net.state_dict(), "local_input_channels": LOCAL_INPUT_CHANNELS, "reference_tensor_input_channels": REFERENCE_TENSOR_INPUT_CHANNELS, "i2v_motion_input_channels": I2V_MOTION_INPUT_CHANNELS, "patch_batch_contract_version": "patch_batch_v3_i2v_motion_guidance"}, str(path))

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "TorchLocalPatchGenerator":
        inst = cls(device=device)
        data = torch.load(str(path), map_location=inst.device)
        state = data.get("state_dict", data) if isinstance(data, dict) else data
        if not isinstance(state, dict):
            raise ValueError("torch local patch checkpoint missing state_dict")
        checkpoint_reference_channel_upgrade = False
        checkpoint_i2v_motion_channel_upgrade = False
        checkpoint_motion_channel_upgrade_from_9 = False
        current = inst.net.state_dict()
        load_state = dict(state)
        if "encoder.0.weight" in load_state:
            raise ValueError("legacy tiny encoder checkpoint is not compatible with reference-guided UNet; retrain renderer or provide v2 checkpoint")
        if "stem.net.0.weight" in load_state and tuple(load_state["stem.net.0.weight"].shape) != tuple(current["stem.net.0.weight"].shape):
            old_weight = load_state["stem.net.0.weight"]
            new_weight = current["stem.net.0.weight"].clone()
            if (
                int(old_weight.shape[1]) in {BASE_LOCAL_INPUT_CHANNELS + REFERENCE_TENSOR_INPUT_CHANNELS, BASE_LOCAL_INPUT_CHANNELS}
                and int(new_weight.shape[1]) == LOCAL_INPUT_CHANNELS
                and old_weight.shape[0] == new_weight.shape[0]
                and old_weight.shape[2:] == new_weight.shape[2:]
            ):
                new_weight.zero_()
                old_channels = int(old_weight.shape[1])
                new_weight[:, :old_channels, :, :] = old_weight
                load_state["stem.net.0.weight"] = new_weight
                checkpoint_reference_channel_upgrade = old_channels == BASE_LOCAL_INPUT_CHANNELS
                checkpoint_i2v_motion_channel_upgrade = old_channels in {BASE_LOCAL_INPUT_CHANNELS + REFERENCE_TENSOR_INPUT_CHANNELS, BASE_LOCAL_INPUT_CHANNELS}
                checkpoint_motion_channel_upgrade_from_9 = old_channels == BASE_LOCAL_INPUT_CHANNELS
            else:
                raise ValueError(f"TorchLocalPatchGenerator checkpoint first conv input channel mismatch: checkpoint_shape={tuple(old_weight.shape)}, runtime_shape={tuple(new_weight.shape)}")
        incompatible = [(k, tuple(v.shape), tuple(current[k].shape)) for k, v in load_state.items() if k in current and tuple(v.shape) != tuple(current[k].shape)]
        if incompatible:
            k, cs, rs = incompatible[0]
            raise ValueError(f"TorchLocalPatchGenerator checkpoint tensor shape mismatch for {k}: checkpoint={cs}, runtime={rs}")
        missing, unexpected = inst.net.load_state_dict(load_state, strict=False)
        if missing or unexpected:
            raise ValueError(f"TorchLocalPatchGenerator checkpoint key mismatch after reference-channel upgrade: missing={list(missing)}, unexpected={list(unexpected)}")
        inst.checkpoint_compatibility = {
            "checkpoint_reference_channel_upgrade": bool(checkpoint_reference_channel_upgrade),
            "checkpoint_i2v_motion_channel_upgrade": bool(checkpoint_i2v_motion_channel_upgrade),
            "checkpoint_motion_channel_upgrade_from_9": bool(checkpoint_motion_channel_upgrade_from_9),
            "checkpoint_local_input_channels": int(data.get("local_input_channels", BASE_LOCAL_INPUT_CHANNELS))
            if isinstance(data, dict)
            else BASE_LOCAL_INPUT_CHANNELS,
            "runtime_local_input_channels": LOCAL_INPUT_CHANNELS,
            "patch_batch_contract_version": str(data.get("patch_batch_contract_version", "")) if isinstance(data, dict) else "",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        inst.net.eval()
        return inst
