from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.schema import RegionRef
from learned.interfaces import PatchSynthesisModel, PatchSynthesisOutput, PatchSynthesisRequest
from rendering.roi_renderer_contract import ROIRenderValidationError, validate_roi_render_output, wrap_roi_render_output
from rendering.learned_bridge import LegacyDeterministicPatchSynthesisModel

CONTRACT_VERSION = "lightweight_roi_renderer_v1"
TRAINING_MANIFEST_VERSION = "renderer_observed_pair_manifest_v3"
ROI_RENDERER_CONTRACT_VERSION = "roi_renderer_contract_v1"
SAFE_OUTPUT_AUTHORITIES = {"generated", "generated_from_memory", "weak"}
UNSAFE_ROUTE_MARKERS = {"blocked", "diagnostic", "private", "optional", "unknown", "deferred", "identity_risk"}

@dataclass(slots=True)
class LightweightROIModelConfig:
    input_channels: int = 16
    base_channels: int = 16
    output_channels: int = 4
    conditioning_channels: int = 12

class TinyROIRenderer(nn.Module):
    """Small residual encoder/decoder; no attention, transformer, diffusion, or external weights."""
    def __init__(self, config: LightweightROIModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or LightweightROIModelConfig()
        c = self.config.base_channels
        self.net = nn.Sequential(
            nn.Conv2d(self.config.input_channels, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 2, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, self.config.output_channels, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = self.net(x)
        if y.shape[-2:] != x.shape[-2:]:
            y = torch.nn.functional.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"rgb": y[:, :3], "alpha": y[:, 3:4]}

@dataclass(slots=True)
class ROIFeatureResult:
    tensor: torch.Tensor
    metadata: dict[str, object]

class ROIFeatureBuilder:
    fields = ("render_mode","route_decision_type","canonical_region_id","memory_family","memory_authority","output_authority","source_delta_type","source_reveal_decision_type","action_type","phase_id")
    flag_fields = ("identity_locked", "weak_memory_candidate")
    def __init__(self, conditioning_channels: int = 12) -> None:
        self.conditioning_channels = conditioning_channels
    def build(self, roi_rgb: Any, roi_request: Mapping[str, Any], *, reference_roi: Any | None = None, alpha: Any | None = None) -> ROIFeatureResult:
        self.validate_request(roi_request)
        rgb = _to_chw(roi_rgb)
        h, w = rgb.shape[-2:]
        planes = [rgb]
        # Reference ROI support remains metadata-compatible for future expansion; v1 keeps
        # the default tensor tiny by using source RGB + alpha + 12 conditioning planes.
        a = _to_mask(alpha, h, w) if alpha is not None else torch.ones(1, h, w)
        planes.append(a)
        vals = []
        for f in self.fields:
            vals.append(_stable_hash(str(roi_request.get(f, "none"))))
        vals.append(1.0 if roi_request.get("identity_locked") else 0.0)
        vals.append(1.0 if roi_request.get("weak_memory_candidate") else 0.0)
        cond = torch.tensor(vals[: self.conditioning_channels], dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
        planes.append(cond)
        x = torch.cat(planes, dim=0)
        meta = {f: roi_request.get(f) for f in self.fields}
        meta.update({f: bool(roi_request.get(f, False)) for f in self.flag_fields})
        return ROIFeatureResult(x, meta)
    def validate_request(self, r: Mapping[str, Any]) -> None:
        if str(r.get("output_authority", "generated")) not in SAFE_OUTPUT_AUTHORITIES:
            raise ROIRenderValidationError("unsafe_output_authority", "learned ROI renderer output authority must remain generated/weak")
        if bool(r.get("private_or_optional_region", False)) or bool(r.get("blocked", False)) or bool(r.get("diagnostic_only", False)):
            raise ROIRenderValidationError("unsafe_route", "private/blocked/diagnostic route cannot build learned ROI features")
        route = str(r.get("route_decision_type", "")).lower()
        reasons = " ".join(str(x).lower() for x in r.get("route_validation_reasons", []) or [])
        if any(m in route or m in reasons for m in UNSAFE_ROUTE_MARKERS) or "weak_identity" in route:
            raise ROIRenderValidationError("unsafe_route", "unsafe route cannot build learned ROI features")
        identity_region = str(r.get("canonical_region_id", "")) in {"face", "head", "hair", "scalp"} or bool(r.get("identity_locked", False))
        if route == "route_reveal_weak_memory" and identity_region:
            raise ROIRenderValidationError("weak_identity_reveal_forbidden", "Weak identity reveal cannot be rendered by learned ROI backend")
        flags = r.get("forbidden_operation_flags", {}) or {}
        if any(bool(v) for v in dict(flags).values()):
            raise ROIRenderValidationError("forbidden_operation", "forbidden operation flag must be false")

def _stable_hash(text: str) -> float:
    h = 2166136261
    for b in text.encode("utf8"):
        h = (h ^ b) * 16777619 & 0xFFFFFFFF
    return (h % 997) / 996.0

def _to_chw(value: Any) -> torch.Tensor:
    arr = np.asarray(value, dtype=np.float32)
    if arr.max(initial=0) > 1.0: arr = arr / 255.0
    if arr.ndim == 2: arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] != 3: raise ValueError("ROI RGB must have 3 channels")
    return torch.from_numpy(arr).permute(2,0,1).contiguous().float().clamp(0,1)

def _to_mask(value: Any, h: int, w: int) -> torch.Tensor:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3: arr = arr[...,0]
    if arr.max(initial=0) > 1.0: arr = arr / 255.0
    t = torch.from_numpy(arr).view(1, *arr.shape).float().clamp(0,1)
    if t.shape[-2:] != (h,w): t = torch.nn.functional.interpolate(t[None], size=(h,w), mode="nearest")[0]
    return t

def _load_array(v: Any) -> Any:
    if isinstance(v, (list, tuple)): return v
    p = Path(str(v))
    if p.suffix == ".npy": return np.load(p)
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

def roi_request_to_dict(obj: Any) -> dict[str, Any]:
    """Normalize Sprint-8 ROI request forms while preserving original nested data."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        if "requests" in obj and obj["requests"]:
            return roi_request_to_dict(obj["requests"][0])
        return _flatten_roi_request_dict(dict(obj))
    if hasattr(obj, "to_training_metadata"):
        d = dict(obj.to_training_metadata())
        d.update({
            "region_id": getattr(obj, "region_id", ""),
            "entity_id": getattr(obj, "entity_id", ""),
            "route_validation_reasons": getattr(obj, "route_validation_reasons", ()),
        })
        return _flatten_roi_request_dict(d)
    if is_dataclass(obj):
        return _flatten_roi_request_dict(asdict(obj))
    return {}


def _flatten_roi_request_dict(raw: dict[str, Any]) -> dict[str, Any]:
    d = dict(raw)
    memory = d.get("memory_usage") if isinstance(d.get("memory_usage"), Mapping) else {}
    identity = d.get("identity_policy") if isinstance(d.get("identity_policy"), Mapping) else {}
    safety = d.get("safety_policy") if isinstance(d.get("safety_policy"), Mapping) else {}
    trace = d.get("trace") if isinstance(d.get("trace"), Mapping) else {}
    route = d.get("source_route_decision") if isinstance(d.get("source_route_decision"), Mapping) else {}

    def fill(key: str, value: Any) -> None:
        if key not in d or d.get(key) in (None, ""):
            d[key] = value

    fill("memory_family", memory.get("memory_family", route.get("memory_family", "none")))
    fill("memory_authority", memory.get("memory_authority", route.get("memory_authority", "none")))
    fill("source_memory_authority", memory.get("source_memory_authority", memory.get("memory_authority", "none")))
    fill("output_authority", memory.get("output_reference_authority", route.get("output_reference_authority", d.get("output_authority", "generated"))))
    fill("identity_locked", bool(identity.get("identity_locked", route.get("identity_locked", False))))
    fill("allowed_to_modify_identity", bool(identity.get("allowed_to_modify_identity", route.get("allowed_to_modify_identity", False))))
    fill("renderer_must_preserve_identity", bool(identity.get("renderer_must_preserve_identity", route.get("renderer_must_preserve_identity", False))))
    fill("weak_memory_candidate", bool(memory.get("weak_memory_candidate", route.get("weak_memory_candidate", False))))
    fill("private_or_optional_region", bool(route.get("private_or_optional_region", d.get("private_or_optional_region", False))))
    fill("blocked", bool(route.get("blocked", d.get("blocked", False))))
    fill("diagnostic_only", bool(route.get("diagnostic_only", d.get("diagnostic_only", False))))
    fill("forbidden_operation_flags", trace.get("forbidden_operation_flags", d.get("forbidden_operation_flags", {})))
    if safety:
        d["safety_policy"] = dict(safety)
    return d

@dataclass(slots=True)
class RendererObservedPairDataset(Dataset):
    manifest_path: str
    roi_size: int = 16
    strict: bool = False
    max_samples: int | None = None
    samples: list[dict[str, Any]] = field(init=False, default_factory=list)
    skipped_reasons: dict[str, int] = field(init=False, default_factory=dict)
    def __post_init__(self) -> None:
        payload = json.loads(Path(self.manifest_path).read_text())
        records = payload.get("samples", payload.get("records", []))
        for rec in records[: self.max_samples]:
            reason = self._skip_reason(rec)
            if reason:
                self.skipped_reasons[reason] = self.skipped_reasons.get(reason,0)+1
                continue
            self.samples.append(rec)
        if self.strict and not self.samples: raise ValueError("no usable observed ROI samples")
    def _skip_reason(self, r: Mapping[str, Any]) -> str:
        required = ["roi_before", "roi_after", "canonical_region_id", "route_decision_type", "render_mode"]
        if any(k not in r for k in required):
            if self.strict: raise ValueError("missing_required_fields")
            return "missing_required_fields"
        if str(r.get("target_provenance", r.get("roi_after_provenance", "observed"))) != "observed": return "target_not_observed"
        if str(r.get("target_role", "observed_target")) in {"generated_runtime_output", "bootstrap_self_generated"}: return "generated_target_forbidden"
        if bool(r.get("private_or_optional_region", False)): return "private_blocked"
        if bool(r.get("diagnostic_only", False)): return "diagnostic_blocked"
        if bool(r.get("blocked", False)): return "blocked_route"
        if "unknown" in str(r.get("route_decision_type", "")).lower() or bool(r.get("unknown_or_deferred", False)): return "unknown_blocked"
        if bool(r.get("identity_risk", False)): return "identity_risk_blocked"
        if str(r.get("output_authority", "generated")) not in SAFE_OUTPUT_AUTHORITIES: return "unsafe_output_authority"
        return ""
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.samples[idx]
        builder = ROIFeatureBuilder()
        feat = builder.build(_resize(_load_array(r["roi_before"]), self.roi_size), r, alpha=r.get("alpha"))
        target = _to_chw(_resize(_load_array(r["roi_after"]), self.roi_size))
        alpha_t = _to_mask(r.get("alpha_target", np.ones((self.roi_size,self.roi_size))), self.roi_size, self.roi_size)
        return {"x": feat.tensor, "rgb": target, "alpha": alpha_t, "meta": feat.metadata}

def _resize(arr: Any, size: int) -> np.ndarray:
    t = _to_chw(arr)[None]
    return torch.nn.functional.interpolate(t, size=(size,size), mode="bilinear", align_corners=False)[0].permute(1,2,0).numpy()

def save_lightweight_roi_checkpoint(path: str, model: TinyROIRenderer, metrics: Mapping[str, Any], *, manifest_version: str = TRAINING_MANIFEST_VERSION) -> str:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(model.config), "contract_version": CONTRACT_VERSION, "training_manifest_contract_version": manifest_version, "roi_renderer_contract_version": ROI_RENDERER_CONTRACT_VERSION, "region_vocab": {"hash":"fnv1a_mod_997"}, "render_mode_vocab": "stable_hash_planes", "safety_metadata": {"no_raw_images": True, "output_authorities": sorted(SAFE_OUTPUT_AUTHORITIES)}, "training_metrics_summary": dict(metrics)}, p)
    return str(p)

def load_lightweight_roi_checkpoint(path: str, map_location: str = "cpu") -> tuple[TinyROIRenderer, dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = TinyROIRenderer(LightweightROIModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model, ckpt

def train_lightweight_roi_renderer(manifest: str, out: str, epochs: int=1, batch_size: int=1, lr: float=1e-3, device: str="cpu", roi_size: int=16, seed: int=0, max_samples: int|None=None, strict: bool=False, resume: str="") -> dict[str, Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    ds = RendererObservedPairDataset(manifest, roi_size, strict, max_samples)
    if len(ds) == 0 and strict: raise ValueError("no usable observed ROI samples")
    model = load_lightweight_roi_checkpoint(resume)[0] if resume else TinyROIRenderer()
    model.to(dev); opt = torch.optim.Adam(model.parameters(), lr=lr)
    metrics: dict[str, Any] = {"sample_count": len(ds), "skipped_count": sum(ds.skipped_reasons.values()), "skipped_reasons": ds.skipped_reasons, "identity_locked_count": sum(1 for s in ds.samples if s.get("identity_locked")), "weak_memory_count": sum(1 for s in ds.samples if s.get("weak_memory_candidate")), "private_blocked_count": ds.skipped_reasons.get("private_blocked",0), "unknown_blocked_count": ds.skipped_reasons.get("unknown_blocked",0)}
    if len(ds) == 0:
        metrics.update({"train_loss": None, "rgb_l1": None, "alpha_l1": None, "checkpoint_path": ""}); return metrics
    for _ in range(epochs):
        losses=[]; rgbs=[]; alphas=[]
        for b in DataLoader(ds, batch_size=batch_size, shuffle=False):
            x=b["x"].to(dev); rgb=b["rgb"].to(dev); alpha=b["alpha"].to(dev)
            pred=model(x); rgb_l1=torch.nn.functional.l1_loss(pred["rgb"], rgb); alpha_l1=torch.nn.functional.l1_loss(pred["alpha"], alpha); loss=rgb_l1+0.25*alpha_l1
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu())); rgbs.append(float(rgb_l1.detach().cpu())); alphas.append(float(alpha_l1.detach().cpu()))
    metrics.update({"train_loss": float(np.mean(losses)), "rgb_l1": float(np.mean(rgbs)), "alpha_l1": float(np.mean(alphas))})
    metrics["checkpoint_path"] = save_lightweight_roi_checkpoint(out, model.cpu(), metrics)
    return metrics

class LightweightROIRendererBackend(PatchSynthesisModel):
    def __init__(self, checkpoint_path: str = "", enabled: bool = True, fallback: PatchSynthesisModel | None = None, device: str = "cpu") -> None:
        self.checkpoint_path=checkpoint_path; self.enabled=enabled; self.fallback=fallback or LegacyDeterministicPatchSynthesisModel(); self.device=torch.device(device); self.builder=ROIFeatureBuilder(); self.model=None; self.checkpoint_loaded=False; self.load_error=""
        if checkpoint_path:
            try:
                self.model,_=load_lightweight_roi_checkpoint(checkpoint_path, str(self.device)); self.model.to(self.device); self.checkpoint_loaded=True
            except Exception as e:
                self.load_error=str(e)
    def synthesize_patch(self, request: PatchSynthesisRequest) -> PatchSynthesisOutput:
        roi_contract = roi_request_to_dict(request.transition_context.get("roi_renderer_contract"))
        if not roi_contract: raise ROIRenderValidationError("missing_roi_renderer_contract", "Lightweight ROI backend requires Sprint 8 roi_renderer_contract")
        # Validate before any learned or fallback rendering so unsafe Sprint-8 routes never render.
        self.builder.validate_request(roi_contract)
        if not self.enabled or not self.model:
            out = self.fallback.synthesize_patch(request)
            wrapped = self._wrap_roi_output(roi_contract, out.height, out.width, patch_ref="lightweight_roi_fallback_patch")
            out.execution_trace = dict(out.execution_trace)
            out.execution_trace.update(self._trace(roi_contract, True, "disabled_or_checkpoint_missing"))
            out.execution_trace["roi_renderer_output"] = wrapped.as_dict()
            out.metadata = dict(out.metadata)
            out.metadata.update({"output_authority": wrapped.output_authority, "roi_renderer_output": wrapped.as_dict()})
            return out
        feat = self.builder.build(_extract_region(request.current_frame, request.region), roi_contract)
        with torch.no_grad(): pred = self.model(feat.tensor[None].to(self.device))
        rgb = pred["rgb"][0].cpu().permute(1,2,0).numpy().clip(0,1).tolist(); alpha = pred["alpha"][0,0].cpu().numpy().clip(0,1).tolist()
        h=len(rgb); w=len(rgb[0]) if h else 0
        authority = str(roi_contract.get("output_authority", "weak" if roi_contract.get("weak_memory_candidate") else "generated"))
        if authority not in SAFE_OUTPUT_AUTHORITIES: authority="generated"
        wrapped = self._wrap_roi_output(roi_contract, h, w, patch_ref="lightweight_roi_runtime_patch")
        return PatchSynthesisOutput(request.region, rgb, alpha, h, w, 3, 0.6, execution_trace=self._trace(roi_contract, False, ""), metadata={"output_authority": authority, "roi_renderer_output": wrapped.as_dict()})
    def _wrap_roi_output(self, roi_contract: Mapping[str, Any], height: int, width: int, *, patch_ref: str) -> Any:
        wrapped = wrap_roi_render_output(request=_DictReq(roi_contract), patch_ref=patch_ref, patch_shape=(height, width, 3))
        return validate_roi_render_output(wrapped, request=_DictReq(roi_contract))

    def _trace(self, c: Mapping[str, Any], fallback: bool, reason: str) -> dict[str, Any]:
        authority = c.get("output_authority", "generated")
        if authority not in SAFE_OUTPUT_AUTHORITIES:
            authority = "generated"
        return {"lightweight_roi_renderer_used": not fallback, "checkpoint_loaded": self.checkpoint_loaded, "checkpoint_path": self.checkpoint_path, "renderer_contract_validated": True, "render_mode": c.get("render_mode"), "route_decision_type": c.get("route_decision_type"), "output_authority": authority, "identity_locked": bool(c.get("identity_locked")), "weak_memory_candidate": bool(c.get("weak_memory_candidate")), "fallback_used": fallback, "fallback_reason": reason, "forbidden_operation_flags": {"observed_evidence_created": False, "identity_memory_created": False, "memory_write_performed": False, "scene_graph_mutated": False, "private_region_rendered": False, "hidden_anatomy_generated": False, "clothing_removed": False}}

class _DictReq:
    def __init__(self, d: Mapping[str, Any]) -> None:
        self.region_id=str(d.get("region_id","")); self.canonical_region_id=str(d.get("canonical_region_id","")); self.render_mode=str(d.get("render_mode","preserve")); self.route_decision_type=str(d.get("route_decision_type","route_preserve_visible")); self.memory_usage=type("M",(),{"weak_memory_candidate":bool(d.get("weak_memory_candidate",False)),"memory_authority":str(d.get("memory_authority","none"))})(); self.source_route_decision={}; self.reference_usage=None; self.identity_policy=None; self.safety_policy=None; self.trace=None

def _extract_region(frame: Any, region: RegionRef) -> Any:
    arr = np.asarray(frame, dtype=np.float32)
    h, w = arr.shape[:2]
    bx, by, bw, bh = float(region.bbox.x), float(region.bbox.y), float(region.bbox.w), float(region.bbox.h)
    normalized = all(0.0 <= v <= 1.0 for v in (bx, by, bw, bh))
    if normalized:
        x = int(round(bx * w)); y = int(round(by * h)); ww = int(round(bw * w)); hh = int(round(bh * h))
    else:
        x = int(round(bx)); y = int(round(by)); ww = int(round(bw)); hh = int(round(bh))
    x = min(max(0, x), max(0, w - 1)); y = min(max(0, y), max(0, h - 1))
    ww = max(1, ww); hh = max(1, hh)
    x2 = min(w, max(x + 1, x + ww)); y2 = min(h, max(y + 1, y + hh))
    return arr[y:y2, x:x2]

def main(argv: Sequence[str] | None = None) -> None:
    p=argparse.ArgumentParser(description="Train lightweight Sprint-9 ROI renderer")
    p.add_argument("train_roi_renderer", nargs="?"); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=1); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--device", default="cpu"); p.add_argument("--image-size", "--roi-size", dest="roi_size", type=int, default=16); p.add_argument("--seed", type=int, default=0); p.add_argument("--max-samples", type=int); p.add_argument("--strict", action="store_true"); p.add_argument("--resume", default="")
    a=p.parse_args(argv); print(json.dumps(train_lightweight_roi_renderer(a.manifest,a.out,a.epochs,a.batch_size,a.lr,a.device,a.roi_size,a.seed,a.max_samples,a.strict,a.resume), indent=2))
if __name__ == "__main__": main()
