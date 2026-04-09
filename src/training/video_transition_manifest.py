from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.input_layer import AssetFrame, InputAssetLayer
from perception.pipeline import PerceptionPipeline
from representation.graph_builder import SceneGraphBuilder
from training.datasets import _serialize_graph


@dataclass(slots=True)
class VideoTransitionBuilderConfig:
    fps: int = 8
    duration: float = 4.0
    quality_profile: str = "debug"
    min_motion_threshold: float = 0.015
    roi_size: int = 64


class VideoTransitionManifestBuilder:
    """Primary extraction path: video/frames -> manifest-backed transition supervision."""

    def __init__(self, config: VideoTransitionBuilderConfig | None = None) -> None:
        self.config = config or VideoTransitionBuilderConfig()
        self.input_layer = InputAssetLayer()
        self.perception = PerceptionPipeline()
        self.graph_builder = SceneGraphBuilder()

    @staticmethod
    def _to_float01(frame: object) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"frame must be HxWx3 tensor-like, got {list(arr.shape)}")
        if float(np.max(arr)) > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _infer_transition_family(diff_mean: float, track_id: str, graph_before: dict[str, Any], graph_after: dict[str, Any]) -> str:
        if diff_mean > 0.16:
            return "major_motion"
        before_persons = graph_before.get("persons", []) if isinstance(graph_before, dict) else []
        after_persons = graph_after.get("persons", []) if isinstance(graph_after, dict) else []
        if before_persons and after_persons:
            bp = before_persons[0]
            ap = after_persons[0]
            b_exp = ((bp.get("expression_state") or {}).get("label") if isinstance(bp, dict) else None) or ""
            a_exp = ((ap.get("expression_state") or {}).get("label") if isinstance(ap, dict) else None) or ""
            if b_exp != a_exp:
                return "face_expression"
            b_g = len((bp.get("garments") or [])) if isinstance(bp, dict) else 0
            a_g = len((ap.get("garments") or [])) if isinstance(ap, dict) else 0
            if b_g != a_g:
                return "garment_state"
        return "micro_adjust" if track_id else "scene_context"

    @staticmethod
    def _phase_estimate(index: int, total_pairs: int, diff_mean: float) -> str:
        if total_pairs <= 1:
            return "single"
        ratio = index / max(1, total_pairs - 1)
        if ratio < 0.33:
            return "early"
        if ratio > 0.66:
            return "late"
        if diff_mean > 0.18:
            return "transition_peak"
        return "mid"

    @staticmethod
    def _bbox_from_graph(graph: dict[str, Any], track_id: str) -> tuple[float, float, float, float] | None:
        persons = graph.get("persons", []) if isinstance(graph, dict) else []
        for p in persons:
            if not isinstance(p, dict):
                continue
            if track_id and str(p.get("track_id", "")) != track_id:
                continue
            bb = p.get("bbox")
            if isinstance(bb, dict):
                return float(bb.get("x", 0.25)), float(bb.get("y", 0.25)), float(bb.get("w", 0.5)), float(bb.get("h", 0.5))
        if persons and isinstance(persons[0], dict) and isinstance(persons[0].get("bbox"), dict):
            bb0 = persons[0]["bbox"]
            return float(bb0.get("x", 0.25)), float(bb0.get("y", 0.25)), float(bb0.get("w", 0.5)), float(bb0.get("h", 0.5))
        return None

    def _extract_pair_roi(self, before: np.ndarray, after: np.ndarray, graph_before: dict[str, Any], graph_after: dict[str, Any], track_id: str) -> dict[str, Any]:
        h, w = before.shape[:2]
        bb = self._bbox_from_graph(graph_after, track_id) or self._bbox_from_graph(graph_before, track_id)
        if bb is None:
            bb = (0.2, 0.2, 0.6, 0.6)
        x, y, bw, bh = bb
        x0 = int(max(0, min(w - 1, x * w)))
        y0 = int(max(0, min(h - 1, y * h)))
        x1 = int(max(x0 + 1, min(w, (x + bw) * w)))
        y1 = int(max(y0 + 1, min(h, (y + bh) * h)))

        rb = before[y0:y1, x0:x1, :]
        ra = after[y0:y1, x0:x1, :]

        if rb.size == 0 or ra.size == 0:
            rb = before
            ra = after
            x0, y0, x1, y1 = 0, 0, w, h

        diff = np.mean(np.abs(ra - rb), axis=2, keepdims=True)
        changed = np.clip(diff * 3.5, 0.0, 1.0)
        preservation = np.clip(1.0 - changed, 0.0, 1.0)
        occlusion = np.clip(changed * (np.mean(rb, axis=2, keepdims=True) > np.mean(ra, axis=2, keepdims=True)).astype(np.float32), 0.0, 1.0)
        visibility = np.clip(changed * (np.mean(ra, axis=2, keepdims=True) >= np.mean(rb, axis=2, keepdims=True)).astype(np.float32), 0.0, 1.0)
        return {
            "roi_before": rb.tolist(),
            "roi_after": ra.tolist(),
            "changed_mask": changed.tolist(),
            "preservation_mask": preservation.tolist(),
            "visibility_target": visibility.tolist(),
            "occlusion_target": occlusion.tolist(),
            "roi_manifest": {
                "bbox": {"x": x0 / w, "y": y0 / h, "w": (x1 - x0) / w, "h": (y1 - y0) / h},
                "shape": [int(ra.shape[0]), int(ra.shape[1]), 3],
            },
            "changed_ratio": float(np.mean(changed)),
        }

    @staticmethod
    def _graph_delta_weak(graph_before: dict[str, Any], graph_after: dict[str, Any], family: str, phase: str, changed_ratio: float, track_id: str) -> dict[str, Any]:
        reasons = [family, f"phase:{phase}"]
        pose_delta = {"motion_energy": round(changed_ratio, 6)}
        expr_delta: dict[str, float] = {}
        garment_delta: dict[str, float] = {}
        visibility_delta: dict[str, str] = {}
        interaction_delta: dict[str, float] = {}
        affected_regions = ["person_roi"]
        if family == "face_expression":
            expr_delta = {"smile_intensity": round(min(1.0, changed_ratio * 2.2), 6)}
            affected_regions = ["face"]
        elif family == "garment_state":
            garment_delta = {"coverage_delta": round(changed_ratio, 6)}
            visibility_delta = {"torso": "partially_visible" if changed_ratio > 0.05 else "visible"}
            affected_regions = ["torso", "garments"]
        elif family == "major_motion":
            interaction_delta = {"support_contact": round(min(1.0, 0.2 + changed_ratio), 6)}
            affected_regions = ["body", "environment_contact"]

        return {
            "pose_deltas": pose_delta,
            "garment_deltas": garment_delta,
            "visibility_deltas": visibility_delta,
            "expression_deltas": expr_delta,
            "interaction_deltas": interaction_delta,
            "affected_entities": [track_id or "scene"],
            "affected_regions": affected_regions,
            "semantic_reasons": reasons,
            "region_transition_mode": {r: ("changing" if changed_ratio > 0.02 else "stable") for r in affected_regions},
            "transition_phase": phase,
            "transition_diagnostics": {"weak_supervision": True, "changed_ratio": round(changed_ratio, 6)},
        }

    def build_from_frames(self, frames: list[object], *, source_id: str = "frame_sequence", scene_manifests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if len(frames) < 2:
            raise ValueError("need at least 2 frames to build transition samples")
        frame_arrays = [self._to_float01(f) for f in frames]
        asset_frames = [AssetFrame(frame_id=f"frame_{i}", tensor=np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8), width=int(a.shape[1]), height=int(a.shape[0]), timestamp=float(i) / max(1, self.config.fps), source=source_id) for i, a in enumerate(frame_arrays)]

        if scene_manifests:
            graphs = scene_manifests
        else:
            perception_outs = [self.perception.analyze(f) for f in asset_frames]
            graphs = [_serialize_graph(self.graph_builder.build(p, frame_index=i)) for i, p in enumerate(perception_outs)]

        records: list[dict[str, Any]] = []
        total_pairs = len(frame_arrays) - 1
        for idx in range(total_pairs):
            before = frame_arrays[idx]
            after = frame_arrays[idx + 1]
            graph_before = graphs[idx]
            graph_after = graphs[idx + 1]
            persons = graph_after.get("persons", []) if isinstance(graph_after, dict) else []
            track_id = ""
            if persons and isinstance(persons[0], dict):
                track_id = str(persons[0].get("track_id", ""))
            diff_mean = float(np.mean(np.abs(after - before)))
            phase = self._phase_estimate(idx, total_pairs, diff_mean)
            family = self._infer_transition_family(diff_mean, track_id, graph_before, graph_after)
            roi_pack = self._extract_pair_roi(before, after, graph_before, graph_after, track_id)
            changed_ratio = float(roi_pack["changed_ratio"])
            graph_delta = self._graph_delta_weak(graph_before, graph_after, family, phase, changed_ratio, track_id)
            planner_context = {
                "step_index": idx + 1,
                "total_steps": total_pairs,
                "phase": phase,
                "target_duration": 1.0 / max(1, self.config.fps),
                "progression": round((idx + 1) / max(1, total_pairs), 6),
            }
            fallback_flags = {
                "scene_graph_after_weak": scene_manifests is None,
                "heuristic_priors_used": False,
                "primary_target_source": "paired_video_roi",
            }
            records.append(
                {
                    "record_id": f"{source_id}_transition_{idx:05d}",
                    "source": source_id,
                    "frame_before": before.tolist(),
                    "frame_after": after.tolist(),
                    "scene_graph_before": graph_before,
                    "scene_graph_after": graph_after,
                    "tracked_entity_id": track_id or "scene",
                    "runtime_semantic_transition": {"family": family, "label": family, "weak": True},
                    "transition_family": family,
                    "target_profile": {"primary": family, "secondary": ["context_stability"], "context": "weak_inferred"},
                    "planner_context": planner_context,
                    "phase_estimate": phase,
                    "graph_delta_target": graph_delta,
                    "roi_manifests": [roi_pack["roi_manifest"]],
                    "roi_before": roi_pack["roi_before"],
                    "roi_after": roi_pack["roi_after"],
                    "changed_mask": roi_pack["changed_mask"],
                    "preservation_mask": roi_pack["preservation_mask"],
                    "visibility_target": roi_pack["visibility_target"],
                    "occlusion_target": roi_pack["occlusion_target"],
                    "support_interaction_clues": {"contact_hint": 1.0 if family == "major_motion" else 0.2, "changed_ratio": round(changed_ratio, 6)},
                    "target_transition_context": {
                        "phase": phase,
                        "family": family,
                        "visibility_consequence": "mixed" if changed_ratio > 0.1 else "minor",
                    },
                    "memory_context": {"track_id": track_id or "scene", "weak_temporal_window": [idx, idx + 1]},
                    "heuristic_priors": {
                        "bootstrap_changed_prior": roi_pack["changed_mask"],
                        "bootstrap_visibility_prior": roi_pack["visibility_target"],
                        "role": "auxiliary_only",
                    },
                    "fallback_flags": fallback_flags,
                }
            )

        return {
            "manifest_type": "video_transition_manifest",
            "manifest_version": 1,
            "source": {"id": source_id, "fps": self.config.fps, "duration": self.config.duration},
            "records": records,
            "diagnostics": {
                "total_records": len(records),
                "usable_sample_count": len(records),
                "family_coverage": sorted({r["transition_family"] for r in records}),
                "phase_coverage": sorted({r["phase_estimate"] for r in records}),
                "fallback_free_ratio": 1.0,
                "invalid_records": 0,
            },
        }

    def build_from_video(self, video_path: str, *, text: str = "") -> dict[str, Any]:
        req = self.input_layer.build_request(images=[], video=video_path, text=text, fps=self.config.fps, duration=self.config.duration, quality_profile=self.config.quality_profile)
        frames = [f.tensor for f in (req.unified_asset.frames if req.unified_asset else [])]
        if len(frames) == 1:
            frames = [frames[0], frames[0]]
        manifest = self.build_from_frames(frames, source_id=str(video_path))
        manifest["source"]["input_type"] = "video"
        manifest["source"]["request_metadata"] = (req.unified_asset.metadata if req.unified_asset else {})
        return manifest


def save_video_transition_manifest(payload: dict[str, Any], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
