from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - runtime optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_ModuleBase = nn.Module if nn is not None else object


TorchPathName = Literal["existing_update", "reveal", "insertion"]
CheckpointStatus = Literal[
    "torch_unavailable",
    "bootstrap_only",
    "checkpoint_directory_missing",
    "checkpoint_file_missing",
    "checkpoint_invalid",
    "checkpoint_loaded",
]


@dataclass(slots=True)
class RendererTensorBatch:
    roi: object
    changed_mask: object
    alpha_hint: object
    conditioning_map: object
    context_vector: object
    memory_hint: object
    region_geometry: object
    reveal_signal: object
    insertion_signal: object
    conditioning_summary: dict[str, object]
    path_type: TorchPathName


@dataclass(slots=True)
class TorchBackendResult:
    rgb: list[list[list[float]]]
    alpha: list[list[float]]
    uncertainty: list[list[float]]
    confidence: float
    backend_trace: dict[str, object]


@dataclass(slots=True)
class TorchBackendCheckpoint:
    version: int
    model_name: str
    path_name: TorchPathName
    state_dict: dict[str, object]


class _SharedRendererBackbone(_ModuleBase):  # type: ignore[misc]
    def __init__(self, in_ch: int = 15, hidden_ch: int = 28) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _ModeHead(_ModuleBase):  # type: ignore[misc]
    def __init__(self, hidden_ch: int = 28) -> None:
        super().__init__()
        self.rgb = nn.Conv2d(hidden_ch, 3, kernel_size=1)
        self.alpha = nn.Conv2d(hidden_ch, 1, kernel_size=1)
        self.uncertainty = nn.Conv2d(hidden_ch, 1, kernel_size=1)
        self.confidence = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_ch, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, feat: torch.Tensor, roi: torch.Tensor, alpha_hint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_residual = torch.tanh(self.rgb(feat)) * 0.25
        rgb = torch.clamp(roi + rgb_residual, 0.0, 1.0)
        alpha = torch.clamp(0.6 * torch.sigmoid(self.alpha(feat)) + 0.4 * alpha_hint, 0.0, 1.0)
        unc = torch.clamp(torch.sigmoid(self.uncertainty(feat)), 0.0, 1.0)
        conf = self.confidence(feat).flatten(start_dim=1)
        return rgb, alpha, unc, conf


class _PathConditioning(_ModuleBase):  # type: ignore[misc]
    def __init__(self, *, context_dim: int, hidden_ch: int = 28) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(context_dim, hidden_ch)
        self.to_beta = nn.Linear(context_dim, hidden_ch)
        self.path_gate = nn.Sequential(nn.Linear(context_dim, hidden_ch), nn.Sigmoid())

    def forward(self, feat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(context).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(context).unsqueeze(-1).unsqueeze(-1)
        gate = self.path_gate(context).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + torch.tanh(gamma) * 0.35) + beta * 0.15 + gate * feat * 0.2


class _TorchPathModule(_ModuleBase):  # type: ignore[misc]
    def __init__(self, *, context_dim: int) -> None:
        super().__init__()
        self.backbone = _SharedRendererBackbone(in_ch=15)
        self.conditioning = _PathConditioning(context_dim=context_dim)
        self.head = _ModeHead()

    def forward(self, inputs: torch.Tensor, roi: torch.Tensor, alpha_hint: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(inputs)
        conditioned = self.conditioning(feat, context)
        return self.head(conditioned, roi, alpha_hint)


class TorchExistingRegionUpdater:
    path_name: TorchPathName = "existing_update"

    def __init__(self, device: str = "cpu") -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise RuntimeError("torch is unavailable")
        self.device = torch.device(device)
        self.model = _TorchPathModule(context_dim=16).to(self.device)

    def forward(self, batch: RendererTensorBatch) -> TorchBackendResult:
        return _run_model(self.model, self.device, batch, module="torch_existing_region_updater")


class TorchRevealRegionSynthesizer:
    path_name: TorchPathName = "reveal"

    def __init__(self, device: str = "cpu") -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise RuntimeError("torch is unavailable")
        self.device = torch.device(device)
        self.model = _TorchPathModule(context_dim=16).to(self.device)

    def forward(self, batch: RendererTensorBatch) -> TorchBackendResult:
        return _run_model(self.model, self.device, batch, module="torch_reveal_region_synthesizer")


class TorchNewEntityInserter:
    path_name: TorchPathName = "insertion"

    def __init__(self, device: str = "cpu") -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise RuntimeError("torch is unavailable")
        self.device = torch.device(device)
        self.model = _TorchPathModule(context_dim=16).to(self.device)

    def forward(self, batch: RendererTensorBatch) -> TorchBackendResult:
        return _run_model(self.model, self.device, batch, module="torch_new_entity_inserter")


def _run_model(model: _TorchPathModule, device: torch.device, batch: RendererTensorBatch, *, module: str) -> TorchBackendResult:
    roi = torch.tensor(batch.roi, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    changed = torch.tensor(batch.changed_mask, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    alpha_hint = torch.tensor(batch.alpha_hint, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    memory_hint = torch.tensor(batch.memory_hint, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    conditioning_map = torch.tensor(batch.conditioning_map, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    reveal_signal = torch.tensor(batch.reveal_signal, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    insertion_signal = torch.tensor(batch.insertion_signal, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    h, w = int(roi.shape[2]), int(roi.shape[3])
    ctx = torch.tensor(batch.context_vector, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, h, device=device), torch.linspace(0.0, 1.0, w, device=device), indexing="ij")
    ctx_bias = torch.sigmoid(ctx.mean()).expand(1, 1, h, w)
    geom = torch.tensor(batch.region_geometry, dtype=torch.float32, device=device).view(1, 4, 1, 1).expand(1, 4, h, w)
    inp = torch.cat(
        [roi, changed, alpha_hint, memory_hint, conditioning_map[:, :1, ...], reveal_signal, insertion_signal, ctx_bias, geom[:, :3, ...], yy[None, None, ...], xx[None, None, ...]],
        dim=1,
    )
    model.eval()
    with torch.no_grad():
        rgb, alpha, unc, conf = model(inp, roi, alpha_hint, ctx.unsqueeze(0))
    return TorchBackendResult(
        rgb=rgb[0].detach().cpu().permute(1, 2, 0).tolist(),
        alpha=alpha[0, 0].detach().cpu().tolist(),
        uncertainty=unc[0, 0].detach().cpu().tolist(),
        confidence=float(conf[0, 0].detach().cpu().item()),
        backend_trace={
            "module": module,
            "backend_type": "torch_learned",
            "path_name": batch.path_type,
            "device": str(device),
            "tensor_contract": {
                "roi": _shape3(batch.roi),
                "changed_mask": _shape3(batch.changed_mask),
                "alpha_hint": _shape3(batch.alpha_hint),
                "memory_hint": _shape3(batch.memory_hint),
                "conditioning_map": _shape3(batch.conditioning_map),
                "reveal_signal": _shape3(batch.reveal_signal),
                "insertion_signal": _shape3(batch.insertion_signal),
                "context_vector": len(list(batch.context_vector)),
            },
            "conditioning_summary": dict(batch.conditioning_summary),
        },
    )


class TorchRendererBackendBundle:
    def __init__(self, *, device: str = "cpu", allow_random_init_for_dev: bool = False) -> None:
        self.available = torch is not None and nn is not None
        self.device = device
        self.allow_random_init_for_dev = allow_random_init_for_dev
        self.existing: TorchExistingRegionUpdater | None = None
        self.reveal: TorchRevealRegionSynthesizer | None = None
        self.insertion: TorchNewEntityInserter | None = None
        self.checkpoint_status: dict[TorchPathName, CheckpointStatus] = {
            "existing_update": "torch_unavailable" if not self.available else "bootstrap_only",
            "reveal": "torch_unavailable" if not self.available else "bootstrap_only",
            "insertion": "torch_unavailable" if not self.available else "bootstrap_only",
        }
        self.checkpoint_details: dict[TorchPathName, dict[str, object]] = {
            "existing_update": {},
            "reveal": {},
            "insertion": {},
        }
        if self.available:
            self.existing = TorchExistingRegionUpdater(device=device)
            self.reveal = TorchRevealRegionSynthesizer(device=device)
            self.insertion = TorchNewEntityInserter(device=device)

    def backend_runtime_status(self, path_name: TorchPathName) -> dict[str, object]:
        status = self.checkpoint_status.get(path_name, "torch_unavailable")
        checkpoint_loaded = status == "checkpoint_loaded"
        usable = checkpoint_loaded or (self.available and self.allow_random_init_for_dev)
        return {
            "backend_requested_mode": "torch_learned",
            "path_name": path_name,
            "torch_available": self.available,
            "checkpoint_status": status,
            "usable_for_inference": usable,
            "random_init_allowed": self.allow_random_init_for_dev,
            "runtime_status": "usable" if usable else "fallback_required",
            "details": dict(self.checkpoint_details.get(path_name, {})),
        }

    def save_checkpoint(self, checkpoint_dir: str) -> dict[str, str]:
        if not self.available or torch is None:
            return {}
        out: dict[str, str] = {}
        root = Path(checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        mapping = {
            "existing": ("torch_existing_region_updater", self.existing),
            "reveal": ("torch_reveal_region_synthesizer", self.reveal),
            "insertion": ("torch_new_entity_inserter", self.insertion),
        }
        for key, (name, module) in mapping.items():
            if module is None:
                continue
            payload = {
                "version": 1,
                "model_name": name,
                "path_name": module.path_name,
                "state_dict": module.model.state_dict(),
            }
            path = root / f"{key}.pt"
            torch.save(payload, path)
            out[key] = str(path)
        return out

    def load_checkpoint(self, checkpoint_dir: str) -> dict[str, object]:
        trace: dict[str, object] = {"loaded": [], "missing": [], "invalid": []}
        if not self.available or torch is None:
            trace["reason"] = "torch_unavailable"
            return trace
        root = Path(checkpoint_dir)
        if not root.exists():
            for path_name in self.checkpoint_status:
                self.checkpoint_status[path_name] = "checkpoint_directory_missing"
                self.checkpoint_details[path_name] = {"checkpoint_dir": checkpoint_dir}
            trace["reason"] = "checkpoint_directory_missing"
            return trace
        mapping = {
            "existing": ("existing_update", self.existing),
            "reveal": ("reveal", self.reveal),
            "insertion": ("insertion", self.insertion),
        }
        for key, (path_name, module) in mapping.items():
            path = root / f"{key}.pt"
            if module is None:
                continue
            if not path.exists():
                trace["missing"].append(key)
                self.checkpoint_status[path_name] = "checkpoint_file_missing"
                self.checkpoint_details[path_name] = {"checkpoint_file": str(path)}
                continue
            try:
                payload = torch.load(path, map_location=self.device)
                if not isinstance(payload, dict):
                    raise ValueError("checkpoint payload must be dict")
                state_dict = payload.get("state_dict", {})
                if not isinstance(state_dict, dict) or not state_dict:
                    raise ValueError("state_dict missing or empty")
                module.model.load_state_dict(state_dict, strict=True)
                trace["loaded"].append(key)
                self.checkpoint_status[path_name] = "checkpoint_loaded"
                self.checkpoint_details[path_name] = {"checkpoint_file": str(path), "model_name": payload.get("model_name", "")}
            except Exception as exc:
                trace["invalid"].append({"path": key, "error": str(exc)})
                self.checkpoint_status[path_name] = "checkpoint_invalid"
                self.checkpoint_details[path_name] = {"checkpoint_file": str(path), "error": str(exc)}
        return trace


def build_renderer_tensor_batch(
    *,
    roi: list[list[list[float]]],
    path_type: TorchPathName,
    transition_mode: str,
    hidden_mode: str,
    retrieval_top_score: float,
    memory_hint_strength: float,
    lifecycle: str = "already_existing",
    region_role: str = "primary",
    region_type: str = "generic",
    entity_type: str = "generic_entity",
    insertion_type: str = "none",
    reveal_type: str = "none",
    transition_strength: float = 0.0,
    retrieval_evidence: float = 0.0,
    reveal_memory_strength: float = 0.0,
    insertion_context_strength: float = 0.0,
    appearance_conditioning_strength: float = 0.0,
    scene_context_strength: float = 0.0,
    pose_role: str = "neutral",
    bbox_summary: tuple[float, float, float, float] | None = None,
) -> RendererTensorBatch:
    roi_np = _to_float_grid(roi)
    h, w, _ = _shape3(roi_np)
    path_base = {"existing_update": 0.36, "reveal": 0.64, "insertion": 0.72}[path_type]
    changed = [[[path_base + 0.1 * float(max(0.0, min(1.0, transition_strength)))] for _ in range(w)] for _ in range(h)]
    alpha_hint = [[[0.64 if hidden_mode == "known_hidden" else (0.52 if hidden_mode == "unknown_hidden" else 0.46)] for _ in range(w)] for _ in range(h)]
    mem_strength = max(0.0, min(1.0, memory_hint_strength))
    memory_hint = [[[mem_strength] for _ in range(w)] for _ in range(h)]
    mode_code = {
        "stable": 0.1,
        "garment_surface": 0.3,
        "garment_reveal": 0.6,
        "pose_exposure": 0.45,
        "expression_refine": 0.35,
        "visibility_occlusion": 0.75,
    }.get(transition_mode, 0.2)
    path_code = {"existing_update": 0.2, "reveal": 0.6, "insertion": 0.9}[path_type]
    hidden_code = {"not_hidden": 0.1, "known_hidden": 0.8, "unknown_hidden": 0.55}.get(hidden_mode, 0.2)
    lifecycle_code = {
        "already_existing": 0.2,
        "newly_inserted": 0.95,
        "previously_hidden_now_revealed": 0.78,
        "still_hidden": 0.65,
        "interaction_boundary": 0.5,
        "stable_context": 0.35,
    }.get(lifecycle, 0.3)
    region_role_code = {"primary": 0.82, "context": 0.52, "background": 0.22}.get(region_role, 0.3)
    region_type_code = {"face": 0.72, "torso": 0.64, "left_arm": 0.58, "right_arm": 0.58, "legs": 0.54}.get(region_type, 0.4)
    entity_type_code = {"person": 0.9, "garment": 0.7, "object": 0.55, "generic_entity": 0.4}.get(entity_type, 0.4)
    insertion_code = {"none": 0.0, "new_entity": 0.9, "artifact": 0.65}.get(insertion_type, 0.2)
    reveal_code = {"none": 0.0, "garment_change_reveal": 0.9, "pose_exposure_reveal": 0.75, "occlusion_reveal": 0.7, "generic_reveal": 0.5}.get(reveal_type, 0.2)
    pose_role_code = {"neutral": 0.4, "standing": 0.66, "active": 0.78, "contextual": 0.58}.get(pose_role, 0.45)
    context_vector = [
        mode_code,
        lifecycle_code,
        path_code,
        hidden_code,
        region_role_code,
        region_type_code,
        entity_type_code,
        insertion_code,
        reveal_code,
        float(max(0.0, min(1.0, retrieval_top_score))),
        float(max(0.0, min(1.0, retrieval_evidence))),
        float(max(0.0, min(1.0, reveal_memory_strength))),
        float(max(0.0, min(1.0, insertion_context_strength))),
        float(max(0.0, min(1.0, appearance_conditioning_strength))),
        float(max(0.0, min(1.0, scene_context_strength))),
        pose_role_code,
    ]
    reveal_signal = [[[float(max(0.0, min(1.0, reveal_memory_strength)))] for _ in range(w)] for _ in range(h)]
    insertion_signal = [[[float(max(0.0, min(1.0, insertion_context_strength)))] for _ in range(w)] for _ in range(h)]
    conditioning_map = [[[context_vector[9] * 0.4 + context_vector[11] * 0.6] for _ in range(w)] for _ in range(h)]
    bbox = bbox_summary or (0.5, 0.5, 1.0, 1.0)
    return RendererTensorBatch(
        roi=roi_np,
        changed_mask=changed,
        alpha_hint=alpha_hint,
        conditioning_map=conditioning_map,
        context_vector=context_vector,
        memory_hint=memory_hint,
        region_geometry=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        reveal_signal=reveal_signal,
        insertion_signal=insertion_signal,
        conditioning_summary={
            "transition_mode": transition_mode,
            "lifecycle": lifecycle,
            "hidden_mode": hidden_mode,
            "path_type": path_type,
            "region_role": region_role,
            "region_type": region_type,
            "entity_type": entity_type,
            "insertion_type": insertion_type,
            "reveal_type": reveal_type,
        },
        path_type=path_type,
    )


def serialize_tensor_batch(batch: RendererTensorBatch) -> dict[str, object]:
    return {
        "path_type": batch.path_type,
        "roi_shape": _shape3(batch.roi),
        "changed_mask_shape": _shape3(batch.changed_mask),
        "alpha_hint_shape": _shape3(batch.alpha_hint),
        "memory_hint_shape": _shape3(batch.memory_hint),
        "conditioning_map_shape": _shape3(batch.conditioning_map),
        "reveal_signal_shape": _shape3(batch.reveal_signal),
        "insertion_signal_shape": _shape3(batch.insertion_signal),
        "context_vector": list(batch.context_vector),
        "conditioning_summary": dict(batch.conditioning_summary),
    }


def _to_float_grid(value: object) -> list[list[list[float]]]:
    if np is not None:
        arr = np.asarray(value, dtype=np.float32)
        return arr.tolist()
    return value if isinstance(value, list) else list(value)  # type: ignore[arg-type]


def _shape3(value: object) -> list[int]:
    if np is not None and hasattr(value, "shape"):
        s = list(value.shape)  # type: ignore[union-attr]
        return [int(s[0]), int(s[1]), int(s[2])]
    h = len(value)  # type: ignore[arg-type]
    w = len(value[0]) if h else 0  # type: ignore[index]
    c = len(value[0][0]) if h and w else 0  # type: ignore[index]
    return [h, w, c]
