from __future__ import annotations

import json
from collections import Counter
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


@dataclass(slots=True)
class RendererManifestDatasetConfig:
    strict: bool = False
    patch_size: tuple[int, int] = (16, 16)


class RendererPathManifestDatasetBuilder:
    """Build renderer-path training samples from real manifest-backed entries."""

    _SUPPORTED_PATHS: tuple[TorchPathName, ...] = ("existing_update", "reveal", "insertion")
    _REQUIRED_IDENTITY_FIELDS = (
        "sample_id",
        "video_id",
        "frame_before_path",
        "frame_after_path",
        "frame_before_index",
        "frame_after_index",
    )
    _REQUIRED_SEMANTIC_FIELDS = ("path_type", "transition_family", "transition_mode")
    _REQUIRED_ENTITY_FIELDS = ("entity_id", "entity_type", "region_id", "region_type", "lifecycle", "hidden_mode", "region_role")
    _REQUIRED_GEOMETRY_FIELDS = ("roi_bbox_before", "roi_bbox_after")

    def __init__(self, *, config: RendererManifestDatasetConfig | None = None) -> None:
        self.config = config or RendererManifestDatasetConfig()

    def build_surface(self, manifest_path: str) -> RendererPathDatasetSurface:
        payload, records = self._read_manifest(manifest_path)
        diagnostics: dict[str, object] = {
            "source": "renderer_manifest_backed",
            "manifest_path": manifest_path,
            "manifest_type": payload.get("manifest_type", "renderer_path_manifest"),
            "manifest_version": payload.get("manifest_version", 1),
            "total_records": len(records),
            "usable_samples": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
            "samples_per_path": {"existing_update": 0, "reveal": 0, "insertion": 0},
            "supervision_mode_counts": {"explicit": 0, "derived": 0},
            "missing_optional_fields": {},
            "hidden_mode_counts": {},
            "reveal_type_counts": {},
            "insertion_type_counts": {},
            "transition_mode_counts": {},
            "transition_family_counts": {},
        }
        missing_optional = Counter()
        hidden_modes = Counter()
        reveal_types = Counter()
        insertion_types = Counter()
        transition_modes = Counter()
        transition_families = Counter()
        diagnostics_notes: list[dict[str, object]] = []
        samples: list[RendererPathTrainingSample] = []

        for idx, rec in enumerate(records):
            try:
                sample, build_info = self._build_sample_from_record(rec=rec, manifest_path=manifest_path)
                samples.append(sample)
                diagnostics["usable_samples"] = int(diagnostics["usable_samples"]) + 1
                per_path = diagnostics["samples_per_path"]
                assert isinstance(per_path, dict)
                per_path[sample.path_type] = int(per_path.get(sample.path_type, 0)) + 1
                mode_counts = diagnostics["supervision_mode_counts"]
                assert isinstance(mode_counts, dict)
                mode_counts[build_info["supervision_mode"]] = int(mode_counts.get(build_info["supervision_mode"], 0)) + 1
                for field_name in build_info.get("missing_optional_fields", []):
                    missing_optional[str(field_name)] += 1
                hidden_modes[str(build_info["hidden_mode"])] += 1
                reveal_types[str(build_info["reveal_type"])] += 1
                insertion_types[str(build_info["insertion_type"])] += 1
                transition_modes[str(build_info["transition_mode"])] += 1
                transition_families[str(build_info["transition_family"])] += 1
                if len(diagnostics_notes) < 20:
                    diagnostics_notes.append(
                        {
                            "sample_id": build_info["sample_id"],
                            "path_type": build_info["path_type"],
                            "supervision_mode": build_info["supervision_mode"],
                            "notes": list(build_info.get("notes", [])),
                        }
                    )
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                bad_examples = diagnostics["invalid_examples"]
                assert isinstance(bad_examples, list)
                if len(bad_examples) < 16:
                    bad_examples.append({"index": idx, "error": str(exc)})
                if self.config.strict:
                    raise ValueError(f"renderer manifest record {idx} invalid: {exc}") from exc

        diagnostics["missing_optional_fields"] = dict(sorted(missing_optional.items()))
        diagnostics["hidden_mode_counts"] = dict(sorted(hidden_modes.items()))
        diagnostics["reveal_type_counts"] = dict(sorted(reveal_types.items()))
        diagnostics["insertion_type_counts"] = dict(sorted(insertion_types.items()))
        diagnostics["transition_mode_counts"] = dict(sorted(transition_modes.items()))
        diagnostics["transition_family_counts"] = dict(sorted(transition_families.items()))
        diagnostics["notes_preview"] = diagnostics_notes

        return RendererPathDatasetSurface(samples=samples, source="manifest_renderer_path_training_surface", diagnostics=diagnostics)

    def _read_manifest(self, manifest_path: str) -> tuple[dict[str, object], list[dict[str, object]]]:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError("renderer manifest must contain list field 'records'")
        typed_records: list[dict[str, object]] = []
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise ValueError(f"renderer manifest record {idx} must be an object")
            typed_records.append(rec)
        return payload, typed_records

    def _validate_required_fields(self, rec: dict[str, object]) -> None:
        missing = [field for field in self._REQUIRED_IDENTITY_FIELDS if rec.get(field) in {None, ""}]
        missing.extend([field for field in self._REQUIRED_SEMANTIC_FIELDS if rec.get(field) in {None, ""}])
        missing.extend([field for field in self._REQUIRED_ENTITY_FIELDS if rec.get(field) in {None, ""}])
        missing.extend([field for field in self._REQUIRED_GEOMETRY_FIELDS if rec.get(field) is None])
        if missing:
            raise ValueError(f"missing required fields: {sorted(set(missing))}")

    @staticmethod
    def _bbox4(value: object, field: str) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"{field} must be [x, y, w, h]")
        x, y, w, h = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"{field} must have positive width/height")
        return x, y, w, h

    def _load_array(self, base_dir: Path, path_or_inline: object, *, field: str, channels: int | None = None) -> np.ndarray:
        if isinstance(path_or_inline, list):
            arr = np.asarray(path_or_inline, dtype=np.float32)
        elif isinstance(path_or_inline, str):
            path = Path(path_or_inline)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            if not path.exists():
                raise ValueError(f"{field} path does not exist: {path}")
            if path.suffix.lower() in {".npy", ".npz"}:
                arr = np.asarray(np.load(path), dtype=np.float32)
            else:
                try:
                    from PIL import Image  # type: ignore
                except Exception as exc:  # pragma: no cover
                    raise ValueError(f"{field} requires pillow for image files: {path}") from exc
                arr = np.asarray(Image.open(path), dtype=np.float32) / 255.0
        else:
            raise ValueError(f"{field} must be path string or inline array")
        if arr.ndim == 2:
            arr = arr[..., None]
        if channels is not None:
            if arr.shape[-1] == 1 and channels == 3:
                arr = np.repeat(arr, 3, axis=2)
            if arr.shape[-1] == 3 and channels == 1:
                arr = np.mean(arr, axis=2, keepdims=True)
            if arr.shape[-1] != channels:
                raise ValueError(f"{field} expected {channels} channels, got shape {arr.shape}")
        return np.clip(arr.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _crop_roi(frame: np.ndarray, bbox: tuple[float, float, float, float], *, patch_size: tuple[int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox
        x0 = int(round(max(0.0, min(1.0, x)) * w))
        y0 = int(round(max(0.0, min(1.0, y)) * h))
        x1 = int(round(max(0.0, min(1.0, x + bw)) * w))
        y1 = int(round(max(0.0, min(1.0, y + bh)) * h))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        patch = frame[y0:y1, x0:x1]
        if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
            return patch
        yi = np.linspace(0, max(0, patch.shape[0] - 1), patch_size[0]).astype(int)
        xi = np.linspace(0, max(0, patch.shape[1] - 1), patch_size[1]).astype(int)
        return patch[np.ix_(yi, xi)]

    def _build_sample_from_record(self, *, rec: dict[str, object], manifest_path: str) -> tuple[RendererPathTrainingSample, dict[str, object]]:
        self._validate_required_fields(rec)
        path_type = str(rec["path_type"])
        if path_type not in self._SUPPORTED_PATHS:
            raise ValueError(f"path_type must be one of {self._SUPPORTED_PATHS}, got {path_type}")
        typed_path: TorchPathName = path_type  # type: ignore[assignment]
        reveal_type = str(rec.get("reveal_type", "none"))
        insertion_type = str(rec.get("insertion_type", "none"))
        hidden_mode = str(rec.get("hidden_mode", "not_hidden"))
        transition_mode = str(rec.get("transition_mode", "stable"))
        transition_family = str(rec.get("transition_family", "unknown_transition"))
        lifecycle = str(rec.get("lifecycle", "already_existing"))
        region_role = str(rec.get("region_role", "primary"))
        region_type = str(rec.get("region_type", "generic"))
        entity_type = str(rec.get("entity_type", "generic_entity"))
        pose_role = str(rec.get("pose_role", "neutral"))
        missing_optional_fields: list[str] = []

        before_bbox = self._bbox4(rec.get("roi_bbox_before"), "roi_bbox_before")
        after_bbox = self._bbox4(rec.get("roi_bbox_after"), "roi_bbox_after")
        context_bbox_raw = rec.get("context_bbox")
        context_bbox = self._bbox4(context_bbox_raw, "context_bbox") if context_bbox_raw is not None else before_bbox
        if context_bbox_raw is None:
            missing_optional_fields.append("context_bbox")

        base_dir = Path(manifest_path).parent
        frame_before = self._load_array(base_dir, rec["frame_before_path"], field="frame_before_path", channels=3)
        frame_after = self._load_array(base_dir, rec["frame_after_path"], field="frame_after_path", channels=3)
        if frame_before.shape != frame_after.shape:
            raise ValueError("frame_before_path and frame_after_path must resolve to tensors with equal shapes")

        roi_before = self._crop_roi(frame_before, before_bbox, patch_size=self.config.patch_size)
        roi_after = self._crop_roi(frame_after, after_bbox, patch_size=self.config.patch_size)
        supervision = rec.get("supervision", {})
        if not isinstance(supervision, dict):
            raise ValueError("supervision must be object when present")
        explicit_rgb = supervision.get("target_rgb_patch")
        explicit_alpha = supervision.get("target_alpha_mask")
        explicit_changed = supervision.get("changed_mask")
        explicit_unc = supervision.get("target_uncertainty_proxy")
        explicit_mode = all(v is not None for v in (explicit_rgb, explicit_alpha, explicit_changed, explicit_unc))
        notes: list[str] = []
        if explicit_mode:
            target_rgb = self._load_array(base_dir, explicit_rgb, field="supervision.target_rgb_patch", channels=3)
            target_alpha = self._load_array(base_dir, explicit_alpha, field="supervision.target_alpha_mask", channels=1)
            changed_mask = self._load_array(base_dir, explicit_changed, field="supervision.changed_mask", channels=1)
            target_unc = self._load_array(base_dir, explicit_unc, field="supervision.target_uncertainty_proxy", channels=1)
            notes.append("explicit_supervision_from_manifest_paths")
        else:
            target_rgb = roi_after
            changed_mask = np.clip(np.mean(np.abs(roi_after - roi_before), axis=2, keepdims=True) * 3.0, 0.0, 1.0)
            target_alpha = np.clip(0.1 + 0.9 * changed_mask, 0.0, 1.0)
            target_unc = np.clip(0.2 + 0.7 * (1.0 - changed_mask), 0.0, 1.0)
            notes.append("derived_supervision_from_before_after_roi")
            for k, v in (("target_rgb_patch", explicit_rgb), ("target_alpha_mask", explicit_alpha), ("changed_mask", explicit_changed), ("target_uncertainty_proxy", explicit_unc)):
                if v is None:
                    missing_optional_fields.append(f"supervision.{k}")

        # Path-specific supervision policies.
        if typed_path == "existing_update":
            changed_mask = np.clip(changed_mask * 0.8, 0.0, 1.0)
            target_unc = np.clip(0.15 + 0.45 * (1.0 - changed_mask), 0.0, 1.0)
            notes.append("existing_update_preservation_weighted_supervision")
        elif typed_path == "reveal":
            reveal_strength = float(np.clip(float(rec.get("reveal_memory_strength", 0.6) or 0.6), 0.0, 1.0))
            hidden_boost = 0.25 if hidden_mode == "unknown_hidden" else 0.15
            target_alpha = np.clip(0.3 + 0.7 * target_alpha, 0.0, 1.0)
            target_unc = np.clip(hidden_boost + (1.0 - reveal_strength) * 0.5 + 0.5 * (1.0 - changed_mask), 0.0, 1.0)
            notes.append("reveal_hidden_memory_uncertainty_policy")
        else:
            insertion_strength = float(np.clip(float(rec.get("insertion_context_strength", 0.6) or 0.6), 0.0, 1.0))
            target_alpha = np.clip(0.2 + 0.8 * np.maximum(target_alpha, changed_mask), 0.0, 1.0)
            target_unc = np.clip(0.2 + 0.3 * (1.0 - insertion_strength) + 0.4 * (1.0 - changed_mask), 0.0, 1.0)
            notes.append("insertion_occupancy_alpha_policy")

        retrieval_evidence = float(np.clip(float(rec.get("retrieval_evidence", 0.5) or 0.5), 0.0, 1.0))
        batch = build_renderer_tensor_batch(
            roi=roi_before.tolist(),
            path_type=typed_path,
            transition_mode=transition_mode,
            hidden_mode=hidden_mode,
            retrieval_top_score=retrieval_evidence,
            memory_hint_strength=float(np.clip(float(rec.get("memory_hint_strength", retrieval_evidence) or retrieval_evidence), 0.0, 1.0)),
            lifecycle=lifecycle,
            region_role=region_role,
            region_type=region_type,
            entity_type=entity_type,
            insertion_type=insertion_type,
            reveal_type=reveal_type,
            transition_strength=float(np.clip(float(rec.get("appearance_conditioning_strength", 0.4) or 0.4), 0.0, 1.0)),
            retrieval_evidence=retrieval_evidence,
            reveal_memory_strength=float(np.clip(float(rec.get("reveal_memory_strength", 0.0) or 0.0), 0.0, 1.0)),
            insertion_context_strength=float(np.clip(float(rec.get("insertion_context_strength", 0.0) or 0.0), 0.0, 1.0)),
            appearance_conditioning_strength=float(np.clip(float(rec.get("appearance_conditioning_strength", 0.5) or 0.5), 0.0, 1.0)),
            scene_context_strength=float(np.clip(float(rec.get("scene_context_strength", 0.5) or 0.5), 0.0, 1.0)),
            pose_role=pose_role,
            bbox_summary=tuple(float(x) for x in rec.get("bbox_summary", context_bbox)),
        )
        sample_id = str(rec.get("sample_id"))
        summary = {
            "sample_id": sample_id,
            "video_id": str(rec.get("video_id")),
            "sample_source": "manifest",
            "manifest_path": manifest_path,
            "supervision_mode": "explicit" if explicit_mode else "derived",
            "path_semantics": {
                "path_type": typed_path,
                "transition_family": transition_family,
                "transition_mode": transition_mode,
                "lifecycle": lifecycle,
                "hidden_mode": hidden_mode,
                "reveal_type": reveal_type,
                "insertion_type": insertion_type,
                "region_role": region_role,
                "entity_type": entity_type,
                "region_type": region_type,
                "pose_role": pose_role,
            },
            "missing_optional_fields": sorted(set(missing_optional_fields)),
            "diagnostic_notes": notes,
        }
        return (
            RendererPathTrainingSample(
                tensor_batch=batch,
                target_rgb=target_rgb.astype(np.float32),
                target_alpha=target_alpha.astype(np.float32),
                target_uncertainty_proxy=target_unc.astype(np.float32),
                path_type=typed_path,
                lifecycle=lifecycle,
                hidden_mode=hidden_mode,
                summary=summary,
            ),
            {
                "sample_id": sample_id,
                "path_type": typed_path,
                "supervision_mode": "explicit" if explicit_mode else "derived",
                "missing_optional_fields": sorted(set(missing_optional_fields)),
                "notes": notes,
                "hidden_mode": hidden_mode,
                "reveal_type": reveal_type,
                "insertion_type": insertion_type,
                "transition_mode": transition_mode,
                "transition_family": transition_family,
            },
        )


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
    def build_manifest_surface(
        manifest_path: str,
        *,
        strict: bool = False,
        patch_size: tuple[int, int] = (16, 16),
    ) -> RendererPathDatasetSurface:
        builder = RendererPathManifestDatasetBuilder(config=RendererManifestDatasetConfig(strict=strict, patch_size=patch_size))
        return builder.build_surface(manifest_path)

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
