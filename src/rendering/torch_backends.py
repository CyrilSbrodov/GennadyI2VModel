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


@dataclass(slots=True)
class RendererTensorBatch:
    roi: object
    changed_mask: object
    alpha_hint: object
    context_vector: object
    memory_hint: object
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
    def __init__(self, in_ch: int = 9, hidden_ch: int = 24) -> None:
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
    def __init__(self, hidden_ch: int = 24) -> None:
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


class _TorchPathModule(_ModuleBase):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _SharedRendererBackbone(in_ch=9)
        self.head = _ModeHead()

    def forward(self, inputs: torch.Tensor, roi: torch.Tensor, alpha_hint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(inputs)
        return self.head(feat, roi, alpha_hint)


class TorchExistingRegionUpdater:
    path_name: TorchPathName = "existing_update"

    def __init__(self, device: str = "cpu") -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise RuntimeError("torch is unavailable")
        self.device = torch.device(device)
        self.model = _TorchPathModule().to(self.device)

    def forward(self, batch: RendererTensorBatch) -> TorchBackendResult:
        return _run_model(self.model, self.device, batch, module="torch_existing_region_updater")


class TorchRevealRegionSynthesizer:
    path_name: TorchPathName = "reveal"

    def __init__(self, device: str = "cpu") -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise RuntimeError("torch is unavailable")
        self.device = torch.device(device)
        self.model = _TorchPathModule().to(self.device)

    def forward(self, batch: RendererTensorBatch) -> TorchBackendResult:
        return _run_model(self.model, self.device, batch, module="torch_reveal_region_synthesizer")


class TorchNewEntityInserter:
    path_name: TorchPathName = "insertion"

    def __init__(self, device: str = "cpu") -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise RuntimeError("torch is unavailable")
        self.device = torch.device(device)
        self.model = _TorchPathModule().to(self.device)

    def forward(self, batch: RendererTensorBatch) -> TorchBackendResult:
        return _run_model(self.model, self.device, batch, module="torch_new_entity_inserter")


def _run_model(model: _TorchPathModule, device: torch.device, batch: RendererTensorBatch, *, module: str) -> TorchBackendResult:
    roi = torch.tensor(batch.roi, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    changed = torch.tensor(batch.changed_mask, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    alpha_hint = torch.tensor(batch.alpha_hint, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    memory_hint = torch.tensor(batch.memory_hint, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    h, w = int(roi.shape[2]), int(roi.shape[3])
    ctx = torch.tensor(batch.context_vector, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, h, device=device), torch.linspace(0.0, 1.0, w, device=device), indexing="ij")
    ctx_bias = torch.sigmoid(ctx.mean()).expand(1, 1, h, w)
    inp = torch.cat([roi, changed, alpha_hint, memory_hint, yy[None, None, ...], xx[None, None, ...], ctx_bias], dim=1)
    model.eval()
    with torch.no_grad():
        rgb, alpha, unc, conf = model(inp, roi, alpha_hint)
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
                "context_vector": len(list(batch.context_vector)),
            },
        },
    )


class TorchRendererBackendBundle:
    def __init__(self, *, device: str = "cpu") -> None:
        self.available = torch is not None and nn is not None
        self.device = device
        self.existing: TorchExistingRegionUpdater | None = None
        self.reveal: TorchRevealRegionSynthesizer | None = None
        self.insertion: TorchNewEntityInserter | None = None
        if self.available:
            self.existing = TorchExistingRegionUpdater(device=device)
            self.reveal = TorchRevealRegionSynthesizer(device=device)
            self.insertion = TorchNewEntityInserter(device=device)

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
        trace: dict[str, object] = {"loaded": [], "missing": []}
        if not self.available or torch is None:
            trace["reason"] = "torch_unavailable"
            return trace
        mapping = {
            "existing": self.existing,
            "reveal": self.reveal,
            "insertion": self.insertion,
        }
        for key, module in mapping.items():
            path = Path(checkpoint_dir) / f"{key}.pt"
            if module is None:
                continue
            if not path.exists():
                trace["missing"].append(key)
                continue
            payload = torch.load(path, map_location=self.device)
            state_dict = payload.get("state_dict", {}) if isinstance(payload, dict) else {}
            module.model.load_state_dict(state_dict, strict=False)
            trace["loaded"].append(key)
        return trace


def build_renderer_tensor_batch(
    *,
    roi: list[list[list[float]]],
    path_type: TorchPathName,
    transition_mode: str,
    hidden_mode: str,
    retrieval_top_score: float,
    memory_hint_strength: float,
) -> RendererTensorBatch:
    roi_np = _to_float_grid(roi)
    h, w, _ = _shape3(roi_np)
    changed = [[[0.45 if path_type == "existing_update" else 0.68] for _ in range(w)] for _ in range(h)]
    alpha_hint = [[[0.62 if hidden_mode == "known_hidden" else 0.48] for _ in range(w)] for _ in range(h)]
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
    context_vector = [mode_code, path_code, hidden_code, float(max(0.0, min(1.0, retrieval_top_score)))]
    return RendererTensorBatch(
        roi=roi_np,
        changed_mask=changed,
        alpha_hint=alpha_hint,
        context_vector=context_vector,
        memory_hint=memory_hint,
        path_type=path_type,
    )


def serialize_tensor_batch(batch: RendererTensorBatch) -> dict[str, object]:
    return {
        "path_type": batch.path_type,
        "roi_shape": _shape3(batch.roi),
        "changed_mask_shape": _shape3(batch.changed_mask),
        "alpha_hint_shape": _shape3(batch.alpha_hint),
        "memory_hint_shape": _shape3(batch.memory_hint),
        "context_vector": list(batch.context_vector),
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
