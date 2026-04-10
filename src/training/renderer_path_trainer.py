from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from rendering.torch_backends import RendererTensorBatch, TorchPathName, TorchRendererBackendBundle, build_renderer_tensor_batch

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


TrainPathMode = Literal["existing_update", "reveal", "insertion", "mixed"]


@dataclass(slots=True)
class RendererPathTrainingSample:
    tensor_batch: RendererTensorBatch
    target_rgb: np.ndarray
    target_alpha: np.ndarray
    target_uncertainty_proxy: np.ndarray
    path_type: TorchPathName
    lifecycle: str
    hidden_mode: str
    summary: dict[str, object]


@dataclass(slots=True)
class RendererPathTrainingBatch:
    samples: list[RendererPathTrainingSample]


@dataclass(slots=True)
class RendererPathDatasetSurface:
    samples: list[RendererPathTrainingSample]
    source: str
    diagnostics: dict[str, object]


class RendererPathBootstrapDatasetBuilder:
    """Bootstrap renderer training surface aligned with scene/path semantics."""

    def __init__(self, *, patch_size: tuple[int, int] = (16, 16), seed: int = 0) -> None:
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)

    def build(self, *, per_path: int = 8) -> RendererPathDatasetSurface:
        samples: list[RendererPathTrainingSample] = []
        for path in ("existing_update", "reveal", "insertion"):
            for idx in range(per_path):
                samples.append(self._make_sample(path, idx))
        self.rng.shuffle(samples)
        return RendererPathDatasetSurface(
            samples=samples,
            source="bootstrap_scene_semantic_renderer_surface",
            diagnostics={"per_path": per_path, "total": len(samples), "bootstrap": True},
        )

    def _make_sample(self, path_type: TorchPathName, idx: int) -> RendererPathTrainingSample:
        h, w = self.patch_size
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        base = np.stack(
            [
                0.18 + 0.28 * xx,
                0.22 + 0.25 * yy,
                0.2 + 0.2 * (0.5 * xx + 0.5 * yy),
            ],
            axis=-1,
        ).astype(np.float32)
        noise = self.rng.normal(0.0, 0.02, size=(h, w, 3)).astype(np.float32)
        roi = np.clip(base + noise, 0.0, 1.0)

        lifecycle = "already_existing"
        hidden_mode = "not_hidden"
        transition_mode = "stable"
        reveal_type = "none"
        insertion_type = "none"
        transition_strength = 0.2
        reveal_memory_strength = 0.2
        insertion_context_strength = 0.2
        retrieval_score = float(np.clip(0.45 + 0.03 * idx, 0.0, 1.0))

        if path_type == "existing_update":
            transition_mode = "expression_refine"
            transition_strength = 0.25
        elif path_type == "reveal":
            lifecycle = "previously_hidden_now_revealed"
            hidden_mode = "known_hidden" if idx % 2 == 0 else "unknown_hidden"
            transition_mode = "garment_reveal"
            reveal_type = "garment_change_reveal"
            transition_strength = 0.55
            reveal_memory_strength = float(np.clip(0.72 - 0.05 * (idx % 3), 0.1, 1.0))
        else:
            lifecycle = "newly_inserted"
            transition_mode = "pose_exposure"
            insertion_type = "new_entity"
            transition_strength = 0.62
            insertion_context_strength = float(np.clip(0.7 + 0.03 * (idx % 4), 0.1, 1.0))

        batch = build_renderer_tensor_batch(
            roi=roi.tolist(),
            path_type=path_type,
            transition_mode=transition_mode,
            hidden_mode=hidden_mode,
            retrieval_top_score=retrieval_score,
            memory_hint_strength=retrieval_score,
            lifecycle=lifecycle,
            region_role="primary",
            region_type="torso" if path_type != "existing_update" else "face",
            entity_type="person",
            insertion_type=insertion_type,
            reveal_type=reveal_type,
            transition_strength=transition_strength,
            retrieval_evidence=retrieval_score,
            reveal_memory_strength=reveal_memory_strength,
            insertion_context_strength=insertion_context_strength,
            appearance_conditioning_strength=0.5,
            scene_context_strength=0.5,
            pose_role="active" if path_type != "existing_update" else "neutral",
            bbox_summary=(0.5, 0.5, 0.8, 0.8),
        )

        changed = np.asarray(batch.changed_mask, dtype=np.float32)
        reveal_signal = np.asarray(batch.reveal_signal, dtype=np.float32)
        insertion_signal = np.asarray(batch.insertion_signal, dtype=np.float32)

        target_rgb = roi.copy()
        if path_type == "existing_update":
            refine = np.stack([0.02 * xx, 0.01 * yy, 0.015 * (xx - yy)], axis=-1).astype(np.float32)
            target_rgb = np.clip(roi + refine * changed, 0.0, 1.0)
        elif path_type == "reveal":
            reveal_texture = np.stack([0.55 + 0.25 * yy, 0.35 + 0.3 * xx, 0.3 + 0.2 * (1.0 - yy)], axis=-1).astype(np.float32)
            blend = np.clip(0.35 + 0.45 * reveal_signal, 0.0, 1.0)
            target_rgb = np.clip(roi * (1.0 - blend) + reveal_texture * blend, 0.0, 1.0)
        else:
            cx, cy = 0.5, 0.5
            occ = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / 0.05).astype(np.float32)[..., None]
            entity_rgb = np.stack([0.72 + 0.15 * xx, 0.2 + 0.2 * yy, 0.2 + 0.1 * xx], axis=-1).astype(np.float32)
            blend = np.clip(0.4 * occ + 0.4 * insertion_signal, 0.0, 1.0)
            target_rgb = np.clip(roi * (1.0 - blend) + entity_rgb * blend, 0.0, 1.0)

        if path_type == "existing_update":
            target_alpha = np.clip(0.25 + 0.45 * changed, 0.0, 1.0)
            target_uncertainty_proxy = np.clip(0.2 + 0.4 * (1.0 - changed), 0.0, 1.0)
        elif path_type == "reveal":
            hidden_weight = 0.8 if hidden_mode == "unknown_hidden" else 0.55
            target_alpha = np.clip(0.35 + 0.55 * reveal_signal, 0.0, 1.0)
            target_uncertainty_proxy = np.clip(hidden_weight * (1.0 - reveal_signal) + 0.15, 0.0, 1.0)
        else:
            cx, cy = 0.5, 0.5
            occupancy = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / 0.06).astype(np.float32)[..., None]
            target_alpha = np.clip(0.2 + 0.7 * occupancy, 0.0, 1.0)
            target_uncertainty_proxy = np.clip(0.25 + 0.35 * (1.0 - occupancy) + 0.2 * insertion_signal, 0.0, 1.0)

        return RendererPathTrainingSample(
            tensor_batch=batch,
            target_rgb=target_rgb.astype(np.float32),
            target_alpha=target_alpha.astype(np.float32),
            target_uncertainty_proxy=target_uncertainty_proxy.astype(np.float32),
            path_type=path_type,
            lifecycle=lifecycle,
            hidden_mode=hidden_mode,
            summary={"bootstrap": True, "index": idx, "path_type": path_type},
        )


class RendererPathBatchBuilder:
    def build_batches(
        self,
        surface: RendererPathDatasetSurface,
        *,
        mode: TrainPathMode,
        batch_size: int,
        balanced_mixed: bool = True,
    ) -> list[RendererPathTrainingBatch]:
        if mode == "mixed":
            selected = list(surface.samples)
            if balanced_mixed:
                pools: dict[str, list[RendererPathTrainingSample]] = {
                    "existing_update": [s for s in surface.samples if s.path_type == "existing_update"],
                    "reveal": [s for s in surface.samples if s.path_type == "reveal"],
                    "insertion": [s for s in surface.samples if s.path_type == "insertion"],
                }
                min_count = min(len(v) for v in pools.values()) if pools else 0
                selected = []
                for i in range(min_count):
                    selected.extend([pools["existing_update"][i], pools["reveal"][i], pools["insertion"][i]])
        else:
            selected = [s for s in surface.samples if s.path_type == mode]

        out: list[RendererPathTrainingBatch] = []
        for i in range(0, len(selected), max(1, batch_size)):
            chunk = selected[i : i + max(1, batch_size)]
            if chunk:
                out.append(RendererPathTrainingBatch(samples=chunk))
        return out


class RendererPathTrainer:
    def __init__(self, *, bundle: TorchRendererBackendBundle | None = None, device: str = "cpu", learning_rate: float = 2e-3) -> None:
        self.bundle = bundle or TorchRendererBackendBundle(device=device)
        self.device = torch.device(device) if torch is not None else None
        self.learning_rate = learning_rate
        self.batch_builder = RendererPathBatchBuilder()

    @staticmethod
    def _as_t(x: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

    def _model_for(self, path: TorchPathName):
        return self.bundle.get_path_module(path)

    def _forward(self, sample: RendererPathTrainingSample):
        from rendering.torch_backends import tensor_batch_to_torch_inputs

        if torch is None or F is None or self.device is None:
            raise RuntimeError("torch unavailable")
        model = self._model_for(sample.path_type)
        if model is None:
            raise RuntimeError(f"missing model for path {sample.path_type}")
        inp, roi, alpha_hint, context = tensor_batch_to_torch_inputs(sample.tensor_batch, device=self.device)
        rgb, alpha, unc, conf = model(inp, roi, alpha_hint, context)
        return rgb, alpha, unc, conf

    def _compute_path_loss(self, sample: RendererPathTrainingSample, outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        if torch is None or F is None or self.device is None:
            raise RuntimeError("torch unavailable")
        rgb, alpha, unc, conf = outputs
        target_rgb = self._as_t(sample.target_rgb, self.device)
        target_alpha = self._as_t(sample.target_alpha, self.device)
        target_unc = self._as_t(sample.target_uncertainty_proxy, self.device)
        changed = self._as_t(np.asarray(sample.tensor_batch.changed_mask, dtype=np.float32), self.device)
        reveal_signal = self._as_t(np.asarray(sample.tensor_batch.reveal_signal, dtype=np.float32), self.device)
        insertion_signal = self._as_t(np.asarray(sample.tensor_batch.insertion_signal, dtype=np.float32), self.device)

        recon = F.l1_loss(rgb, target_rgb)
        alpha_loss = F.l1_loss(alpha, target_alpha)
        unc_loss = F.l1_loss(unc, target_unc)

        if sample.path_type == "existing_update":
            preservation = torch.mean(torch.abs(rgb - self._as_t(np.asarray(sample.tensor_batch.roi, dtype=np.float32), self.device)) * (1.0 - changed))
            refinement_weight = 1.0 + 1.5 * float(sample.tensor_batch.context_vector[13])
            total = recon + 0.7 * alpha_loss + 0.45 * preservation + 0.2 * refinement_weight * unc_loss
            parts = {
                "reconstruction_loss": float(recon.item()),
                "alpha_loss": float(alpha_loss.item()),
                "preservation_bias_loss": float(preservation.item()),
                "refinement_weighted_uncertainty": float((refinement_weight * unc_loss).item()),
            }
        elif sample.path_type == "reveal":
            hidden_weight = 1.3 if sample.hidden_mode == "unknown_hidden" else 1.0
            reveal_unc = torch.mean(torch.abs(unc - target_unc) * (0.65 + reveal_signal))
            total = recon + 0.8 * alpha_loss + hidden_weight * 0.5 * reveal_unc
            parts = {
                "reconstruction_loss": float(recon.item()),
                "alpha_loss": float(alpha_loss.item()),
                "reveal_uncertainty_proxy_loss": float(reveal_unc.item()),
                "hidden_mode_weight": float(hidden_weight),
            }
        else:
            occupancy = torch.clamp(insertion_signal * 0.5 + changed * 0.5, 0.0, 1.0)
            silhouette = torch.mean(torch.abs(alpha - target_alpha) * (0.35 + occupancy))
            unc_reg = torch.mean(torch.abs(unc - target_unc) * (0.4 + occupancy))
            total = recon + 0.85 * alpha_loss + 0.55 * silhouette + 0.3 * unc_reg
            parts = {
                "reconstruction_loss": float(recon.item()),
                "alpha_loss": float(alpha_loss.item()),
                "insertion_silhouette_loss": float(silhouette.item()),
                "uncertainty_regularization_proxy": float(unc_reg.item()),
            }
        parts["confidence_mean"] = float(conf.mean().item())
        parts["total_loss"] = float(total.item())
        return total, parts

    def train_epoch(self, batches: Iterable[RendererPathTrainingBatch], *, mode: TrainPathMode) -> dict[str, float]:
        if torch is None or self.device is None:
            raise RuntimeError("torch unavailable")
        meters: dict[str, list[float]] = {}
        path_counts = {"existing_update": 0.0, "reveal": 0.0, "insertion": 0.0}
        opts: dict[str, torch.optim.Optimizer] = {}

        for path in ("existing_update", "reveal", "insertion"):
            if mode not in {"mixed", path}:
                continue
            model = self._model_for(path)  # type: ignore[arg-type]
            if model is None:
                continue
            model.train()
            opts[path] = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for batch in batches:
            for sample in batch.samples:
                opt = opts.get(sample.path_type)
                if opt is None:
                    continue
                opt.zero_grad(set_to_none=True)
                outputs = self._forward(sample)
                total, parts = self._compute_path_loss(sample, outputs)
                total.backward()
                opt.step()
                path_counts[sample.path_type] += 1.0
                for k, v in parts.items():
                    meters.setdefault(f"{sample.path_type}.{k}", []).append(v)

        summary = {k: float(np.mean(v)) for k, v in meters.items()}
        summary.update({f"trained_count_{k}": v for k, v in path_counts.items()})
        return summary

    def validate_epoch(self, batches: Iterable[RendererPathTrainingBatch]) -> dict[str, float]:
        if torch is None or self.device is None:
            raise RuntimeError("torch unavailable")
        path_stats: dict[str, dict[str, list[float]]] = {
            "existing_update": {},
            "reveal": {},
            "insertion": {},
        }

        for path in ("existing_update", "reveal", "insertion"):
            model = self._model_for(path)  # type: ignore[arg-type]
            if model is not None:
                model.eval()

        with torch.no_grad():
            for batch in batches:
                for sample in batch.samples:
                    rgb, alpha, unc, _ = self._forward(sample)
                    target_rgb = self._as_t(sample.target_rgb, self.device)
                    target_alpha = self._as_t(sample.target_alpha, self.device)
                    target_unc = self._as_t(sample.target_uncertainty_proxy, self.device)
                    ps = path_stats[sample.path_type]
                    ps.setdefault("rgb_mae", []).append(float(torch.mean(torch.abs(rgb - target_rgb)).item()))
                    ps.setdefault("alpha_mae", []).append(float(torch.mean(torch.abs(alpha - target_alpha)).item()))
                    ps.setdefault("uncertainty_proxy_mae", []).append(float(torch.mean(torch.abs(unc - target_unc)).item()))
                    ps.setdefault("confidence", []).append(float(torch.mean(1.0 - unc).item()))

        metrics: dict[str, float] = {}
        path_scores: list[float] = []
        for path, stats in path_stats.items():
            if not stats:
                metrics[f"{path}.sample_count"] = 0.0
                continue
            rgb_mae = float(np.mean(stats["rgb_mae"]))
            alpha_mae = float(np.mean(stats["alpha_mae"]))
            unc_mae = float(np.mean(stats["uncertainty_proxy_mae"]))
            confidence = float(np.mean(stats["confidence"]))
            score = max(0.0, 1.0 - (rgb_mae + 0.7 * alpha_mae + 0.5 * unc_mae))
            path_scores.append(score)
            metrics.update(
                {
                    f"{path}.rgb_mae": rgb_mae,
                    f"{path}.alpha_mae": alpha_mae,
                    f"{path}.uncertainty_proxy_mae": unc_mae,
                    f"{path}.confidence_summary": confidence,
                    f"{path}.path_score": score,
                    f"{path}.sample_count": float(len(stats["rgb_mae"])),
                }
            )
        metrics["score"] = float(np.mean(path_scores)) if path_scores else 0.0
        return metrics

    def train(
        self,
        *,
        train_surface: RendererPathDatasetSurface,
        val_surface: RendererPathDatasetSurface,
        mode: TrainPathMode = "mixed",
        epochs: int = 2,
        batch_size: int = 4,
        checkpoint_dir: str,
    ) -> dict[str, object]:
        train_batches = self.batch_builder.build_batches(train_surface, mode=mode, batch_size=batch_size, balanced_mixed=True)
        val_mode: TrainPathMode = "mixed" if mode == "mixed" else mode
        val_batches = self.batch_builder.build_batches(val_surface, mode=val_mode, batch_size=batch_size, balanced_mixed=False)
        history: list[dict[str, object]] = []
        best_score = -1.0
        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        best_dir = out_dir / "best"

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_batches, mode=mode)
            val_metrics = self.validate_epoch(val_batches)
            history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            if val_metrics.get("score", 0.0) > best_score:
                best_score = float(val_metrics["score"])
                best_dir.mkdir(parents=True, exist_ok=True)
                self.bundle.save_checkpoint(str(best_dir))

        latest_dir = out_dir / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        self.bundle.save_checkpoint(str(latest_dir))

        return {
            "history": history,
            "best_score": best_score,
            "best_checkpoint_dir": str(best_dir),
            "latest_checkpoint_dir": str(latest_dir),
            "trained_paths": ["existing_update", "reveal", "insertion"] if mode == "mixed" else [mode],
            "dataset_sources": {"train": train_surface.source, "val": val_surface.source},
            "loss_policy": {
                "existing_update": ["reconstruction", "alpha", "preservation_bias", "refinement_weighted_uncertainty_proxy"],
                "reveal": ["reconstruction", "alpha", "reveal_uncertainty_proxy", "hidden_mode_weighting"],
                "insertion": ["reconstruction", "alpha", "silhouette_occupancy_consistency", "uncertainty_regularization_proxy"],
            },
            "usable_checkpoint_policy": "torch_renderer_backend_bundle_compatible",
        }
