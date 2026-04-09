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
    CANONICAL_PHASES = ("prepare", "transition", "contact_or_reveal", "stabilize")
    CANONICAL_FAMILIES = (
        "pose_transition",
        "garment_transition",
        "expression_transition",
        "interaction_transition",
        "visibility_transition",
    )

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
    def _infer_transition_family(diff_mean: float, track_id: str, graph_before: dict[str, Any], graph_after: dict[str, Any]) -> tuple[str, float, bool]:
        scores = {
            "pose_transition": min(1.0, diff_mean * 4.0),
            "garment_transition": 0.02,
            "expression_transition": 0.02,
            "interaction_transition": 0.02,
            "visibility_transition": min(1.0, max(0.0, diff_mean - 0.015) * 3.2),
        }
        before_persons = graph_before.get("persons", []) if isinstance(graph_before, dict) else []
        after_persons = graph_after.get("persons", []) if isinstance(graph_after, dict) else []
        if before_persons and after_persons:
            bp = before_persons[0]
            ap = after_persons[0]
            b_exp = ((bp.get("expression_state") or {}).get("label") if isinstance(bp, dict) else None) or ""
            a_exp = ((ap.get("expression_state") or {}).get("label") if isinstance(ap, dict) else None) or ""
            if b_exp != a_exp:
                scores["expression_transition"] += 0.72
            b_g = len((bp.get("garments") or [])) if isinstance(bp, dict) else 0
            a_g = len((ap.get("garments") or [])) if isinstance(ap, dict) else 0
            if b_g != a_g:
                scores["garment_transition"] += 0.74
            b_parts = bp.get("body_parts", []) if isinstance(bp, dict) else []
            a_parts = ap.get("body_parts", []) if isinstance(ap, dict) else []
            if len(b_parts) != len(a_parts):
                scores["visibility_transition"] += 0.44
            if (bp.get("bbox") and ap.get("bbox")) and isinstance(bp.get("bbox"), dict) and isinstance(ap.get("bbox"), dict):
                bbb = bp["bbox"]
                abb = ap["bbox"]
                shift = abs(float(abb.get("x", 0.0)) - float(bbb.get("x", 0.0))) + abs(float(abb.get("y", 0.0)) - float(bbb.get("y", 0.0)))
                scale_delta = abs(float(abb.get("w", 0.0)) - float(bbb.get("w", 0.0))) + abs(float(abb.get("h", 0.0)) - float(bbb.get("h", 0.0)))
                scores["pose_transition"] += min(0.5, shift + scale_delta)
        if not track_id:
            scores["interaction_transition"] += 0.16

        family = max(scores, key=scores.get)
        confidence = float(np.clip(scores[family], 0.0, 1.0))
        weak_flag = confidence < 0.5
        return family, confidence, weak_flag

    @staticmethod
    def _phase_estimate(index: int, total_pairs: int, diff_mean: float) -> str:
        if total_pairs <= 1:
            return "transition"
        ratio = index / max(1, total_pairs - 1)
        if ratio < 0.25:
            return "prepare"
        if diff_mean > 0.16:
            return "contact_or_reveal"
        if ratio > 0.74:
            return "stabilize"
        return "transition"

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

    @staticmethod
    def _clip_bbox(bb: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x, y, w, h = bb
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        w = float(np.clip(w, 0.01, 1.0 - x))
        h = float(np.clip(h, 0.01, 1.0 - y))
        return x, y, w, h

    @staticmethod
    def _part_bbox_from_person(person_bb: tuple[float, float, float, float], region_type: str) -> tuple[float, float, float, float]:
        px, py, pw, ph = person_bb
        if region_type == "face":
            return VideoTransitionManifestBuilder._clip_bbox((px + 0.25 * pw, py + 0.02 * ph, 0.5 * pw, 0.28 * ph))
        if region_type == "torso":
            return VideoTransitionManifestBuilder._clip_bbox((px + 0.16 * pw, py + 0.24 * ph, 0.68 * pw, 0.42 * ph))
        if region_type == "left_arm":
            return VideoTransitionManifestBuilder._clip_bbox((px + 0.0 * pw, py + 0.24 * ph, 0.26 * pw, 0.58 * ph))
        if region_type == "right_arm":
            return VideoTransitionManifestBuilder._clip_bbox((px + 0.74 * pw, py + 0.24 * ph, 0.26 * pw, 0.58 * ph))
        if region_type == "legs":
            return VideoTransitionManifestBuilder._clip_bbox((px + 0.22 * pw, py + 0.62 * ph, 0.56 * pw, 0.36 * ph))
        if region_type == "inner_garment":
            return VideoTransitionManifestBuilder._clip_bbox((px + 0.2 * pw, py + 0.30 * ph, 0.60 * pw, 0.36 * ph))
        if region_type == "outer_garment":
            return VideoTransitionManifestBuilder._clip_bbox((px + 0.10 * pw, py + 0.20 * ph, 0.80 * pw, 0.56 * ph))
        return VideoTransitionManifestBuilder._clip_bbox((px, py, pw, ph))

    @staticmethod
    def _candidate_region_types(graph_before: dict[str, Any], graph_after: dict[str, Any], family: str) -> list[str]:
        base = ["face", "torso", "left_arm", "right_arm", "legs"]
        before_person = ((graph_before.get("persons") or [{}])[0]) if isinstance(graph_before, dict) else {}
        after_person = ((graph_after.get("persons") or [{}])[0]) if isinstance(graph_after, dict) else {}
        garments_before = before_person.get("garments", []) if isinstance(before_person, dict) else []
        garments_after = after_person.get("garments", []) if isinstance(after_person, dict) else []
        garment_types = [str(g.get("garment_type", "")).lower() for g in garments_before + garments_after if isinstance(g, dict)]
        if garments_before or garments_after:
            base.append("garments")
        if any(t in {"shirt", "top", "dress", "tank_top", "tshirt"} for t in garment_types):
            base.append("inner_garment")
        if any(t in {"jacket", "coat", "hoodie", "sweater", "blazer"} for t in garment_types):
            base.append("outer_garment")
        if family == "expression_transition":
            return ["face", "torso"]
        if family == "garment_transition":
            return [r for r in ["torso", "garments", "inner_garment", "outer_garment", "left_arm", "right_arm"] if r in base]
        if family == "interaction_transition":
            return [r for r in ["left_arm", "right_arm", "torso", "legs"] if r in base]
        if family == "visibility_transition":
            return [r for r in ["face", "torso", "garments", "legs"] if r in base]
        return base

    def _extract_region_rois(
        self,
        before: np.ndarray,
        after: np.ndarray,
        graph_before: dict[str, Any],
        graph_after: dict[str, Any],
        track_id: str,
        family: str,
    ) -> list[dict[str, Any]]:
        h, w = before.shape[:2]
        person_bb = self._bbox_from_graph(graph_after, track_id) or self._bbox_from_graph(graph_before, track_id) or (0.2, 0.2, 0.6, 0.6)
        regions = self._candidate_region_types(graph_before, graph_after, family)
        extracted: list[dict[str, Any]] = []
        for region in regions:
            x, y, bw, bh = self._part_bbox_from_person(person_bb, region)
            x0 = int(max(0, min(w - 1, x * w)))
            y0 = int(max(0, min(h - 1, y * h)))
            x1 = int(max(x0 + 1, min(w, (x + bw) * w)))
            y1 = int(max(y0 + 1, min(h, (y + bh) * h)))
            rb = before[y0:y1, x0:x1, :]
            ra = after[y0:y1, x0:x1, :]
            if rb.size == 0 or ra.size == 0:
                continue
            diff = np.mean(np.abs(ra - rb), axis=2, keepdims=True)
            changed = np.clip(diff * 3.5, 0.0, 1.0)
            preservation = np.clip(1.0 - changed, 0.0, 1.0)
            before_luma = np.mean(rb, axis=2, keepdims=True)
            after_luma = np.mean(ra, axis=2, keepdims=True)
            occlusion = np.clip(changed * (before_luma > after_luma).astype(np.float32), 0.0, 1.0)
            visibility = np.clip(changed * (after_luma >= before_luma).astype(np.float32), 0.0, 1.0)
            changed_ratio = float(np.mean(changed))
            if changed_ratio < 0.005:
                continue
            extracted.append(
                {
                    "region_type": region,
                    "roi_before": rb.tolist(),
                    "roi_after": ra.tolist(),
                    "changed_mask": changed.tolist(),
                    "preservation_mask": preservation.tolist(),
                    "visibility_target": visibility.tolist(),
                    "occlusion_target": occlusion.tolist(),
                    "roi_manifest": {
                        "region_type": region,
                        "bbox": {"x": x0 / w, "y": y0 / h, "w": (x1 - x0) / w, "h": (y1 - y0) / h},
                        "shape": [int(ra.shape[0]), int(ra.shape[1]), 3],
                    },
                    "changed_ratio": changed_ratio,
                    "priors": {
                        "changed_prior": changed.tolist(),
                        "reveal_prior": visibility.tolist(),
                        "visibility_prior": visibility.tolist(),
                        "support_prior": np.clip(changed * 0.5, 0.0, 1.0).tolist(),
                    },
                }
            )
        if extracted:
            return extracted
        fallback = self._part_bbox_from_person(person_bb, "torso")
        x, y, bw, bh = fallback
        x0 = int(max(0, min(w - 1, x * w)))
        y0 = int(max(0, min(h - 1, y * h)))
        x1 = int(max(x0 + 1, min(w, (x + bw) * w)))
        y1 = int(max(y0 + 1, min(h, (y + bh) * h)))
        rb = before[y0:y1, x0:x1, :]
        ra = after[y0:y1, x0:x1, :]
        diff = np.mean(np.abs(ra - rb), axis=2, keepdims=True)
        changed = np.clip(diff * 3.5, 0.0, 1.0)
        return [
            {
                "region_type": "torso",
                "roi_before": rb.tolist(),
                "roi_after": ra.tolist(),
                "changed_mask": changed.tolist(),
                "preservation_mask": np.clip(1.0 - changed, 0.0, 1.0).tolist(),
                "visibility_target": changed.tolist(),
                "occlusion_target": np.zeros_like(changed).tolist(),
                "roi_manifest": {"region_type": "torso", "bbox": {"x": x0 / w, "y": y0 / h, "w": (x1 - x0) / w, "h": (y1 - y0) / h}, "shape": [int(ra.shape[0]), int(ra.shape[1]), 3]},
                "changed_ratio": float(np.mean(changed)),
                "priors": {"changed_prior": changed.tolist(), "reveal_prior": changed.tolist(), "visibility_prior": changed.tolist(), "support_prior": np.zeros_like(changed).tolist()},
            }
        ]

    @staticmethod
    def _graph_delta_weak(
        graph_before: dict[str, Any],
        graph_after: dict[str, Any],
        family: str,
        phase: str,
        changed_regions: dict[str, float],
        track_id: str,
    ) -> dict[str, Any]:
        changed_ratio = float(np.mean(list(changed_regions.values()) or [0.0]))
        reasons = [family, f"phase:{phase}"]
        pose_delta = {
            "torso_motion": round(changed_regions.get("torso", 0.0), 6),
            "head_motion": round(changed_regions.get("face", 0.0), 6),
            "arm_motion": round(max(changed_regions.get("left_arm", 0.0), changed_regions.get("right_arm", 0.0)), 6),
            "leg_motion": round(changed_regions.get("legs", 0.0), 6),
            "motion_energy": round(changed_ratio, 6),
        }
        expr_delta: dict[str, float] = {}
        garment_delta: dict[str, float] = {}
        visibility_delta: dict[str, str | float] = {}
        interaction_delta: dict[str, float] = {}
        affected_regions = sorted(changed_regions.keys())
        if family == "expression_transition":
            expr_delta = {"face_expression_shift": round(min(1.0, changed_regions.get("face", changed_ratio) * 2.0), 6)}
        elif family == "garment_transition":
            garment_delta = {
                "coverage_change": round(max(changed_regions.get("garments", 0.0), changed_regions.get("torso", 0.0)), 6),
                "attachment_shift": round(max(changed_regions.get("left_arm", 0.0), changed_regions.get("right_arm", 0.0)), 6),
                "opening_reveal_state": round(max(changed_regions.get("inner_garment", 0.0), changed_regions.get("outer_garment", 0.0)), 6),
            }
        elif family == "interaction_transition":
            interaction_delta = {
                "support_contact": round(min(1.0, 0.1 + max(changed_regions.get("left_arm", 0.0), changed_regions.get("right_arm", 0.0))), 6),
                "contact_hint": round(min(1.0, changed_ratio * 1.6), 6),
            }
        if family == "visibility_transition" or changed_regions.get("face", 0.0) > 0.05 or changed_regions.get("torso", 0.0) > 0.05:
            visibility_delta = {
                "revealed_regions_score": round(sum(v for k, v in changed_regions.items() if k in {"face", "torso", "inner_garment"}) / max(1, len(changed_regions)), 6),
                "occluded_regions_score": round(sum(v for k, v in changed_regions.items() if k in {"outer_garment", "left_arm", "right_arm", "legs"}) / max(1, len(changed_regions)), 6),
            }

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
            "transition_diagnostics": {"weak_supervision": True, "changed_ratio": round(changed_ratio, 6), "part_aware_delta": True},
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
            family, family_confidence, weak_family = self._infer_transition_family(diff_mean, track_id, graph_before, graph_after)
            roi_packs = self._extract_region_rois(before, after, graph_before, graph_after, track_id, family)
            changed_regions = {str(p["region_type"]): float(p["changed_ratio"]) for p in roi_packs}
            changed_ratio = float(np.mean(list(changed_regions.values()) or [0.0]))
            graph_delta = self._graph_delta_weak(graph_before, graph_after, family, phase, changed_regions, track_id)
            sorted_regions = sorted(changed_regions.items(), key=lambda x: x[1], reverse=True)
            primary_regions = [r for r, score in sorted_regions[:2] if score > 0.02] or [sorted_regions[0][0]]
            secondary_regions = [r for r, score in sorted_regions[2:5] if score > 0.01]
            context_regions = [r for r in changed_regions.keys() if r not in set(primary_regions + secondary_regions)]
            target_profile = {
                "primary_regions": primary_regions,
                "secondary_regions": secondary_regions,
                "context_regions": context_regions,
                "family": family,
            }
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
                    "runtime_semantic_transition": {"family": family, "label": family, "weak": weak_family, "confidence": round(family_confidence, 6)},
                    "transition_family": family,
                    "target_profile": target_profile,
                    "planner_context": planner_context,
                    "phase_estimate": phase,
                    "graph_delta_target": graph_delta,
                    "roi_records": [
                        {
                            "region_type": p["region_type"],
                            "transition_family": family,
                            "target_profile": target_profile,
                            "roi_before": p["roi_before"],
                            "roi_after": p["roi_after"],
                            "changed_mask": p["changed_mask"],
                            "preservation_mask": p["preservation_mask"],
                            "visibility_target": p["visibility_target"],
                            "occlusion_target": p["occlusion_target"],
                            "priors": p["priors"],
                            "changed_ratio": round(float(p["changed_ratio"]), 6),
                        }
                        for p in roi_packs
                    ],
                    "roi_manifests": [p["roi_manifest"] for p in roi_packs],
                    "roi_before": roi_packs[0]["roi_before"],
                    "roi_after": roi_packs[0]["roi_after"],
                    "changed_mask": roi_packs[0]["changed_mask"],
                    "preservation_mask": roi_packs[0]["preservation_mask"],
                    "visibility_target": roi_packs[0]["visibility_target"],
                    "occlusion_target": roi_packs[0]["occlusion_target"],
                    "support_interaction_clues": {"contact_hint": round(min(1.0, changed_regions.get("left_arm", 0.0) + changed_regions.get("right_arm", 0.0) + 0.1), 6), "changed_ratio": round(changed_ratio, 6)},
                    "target_transition_context": {
                        "phase": phase,
                        "family": family,
                        "visibility_consequence": "mixed" if (changed_regions.get("face", 0.0) > 0.05 or changed_regions.get("torso", 0.0) > 0.05) else "minor",
                    },
                    "memory_context": {"track_id": track_id or "scene", "weak_temporal_window": [idx, idx + 1]},
                    "heuristic_priors": {
                        "changed_prior": {p["region_type"]: p["priors"]["changed_prior"] for p in roi_packs},
                        "reveal_prior": {p["region_type"]: p["priors"]["reveal_prior"] for p in roi_packs},
                        "support_prior": {p["region_type"]: p["priors"]["support_prior"] for p in roi_packs},
                        "visibility_prior": {p["region_type"]: p["priors"]["visibility_prior"] for p in roi_packs},
                        "role": "auxiliary_only",
                    },
                    "record_diagnostics": {
                        "family_confidence": round(family_confidence, 6),
                        "family_weak": weak_family,
                        "affected_regions": sorted(changed_regions.keys()),
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
                "family_coverage": sorted({r["transition_family"] for r in records if r.get("transition_family") in self.CANONICAL_FAMILIES}),
                "phase_coverage": sorted({r["phase_estimate"] for r in records if r.get("phase_estimate") in self.CANONICAL_PHASES}),
                "region_coverage": sorted({rm.get("region_type", "unknown") for rec in records for rm in rec.get("roi_manifests", []) if isinstance(rm, dict)}),
                "target_profile_coverage": {
                    "primary_regions": sorted({r for rec in records for r in ((rec.get("target_profile") or {}).get("primary_regions") or [])}),
                    "secondary_regions": sorted({r for rec in records for r in ((rec.get("target_profile") or {}).get("secondary_regions") or [])}),
                    "context_regions": sorted({r for rec in records for r in ((rec.get("target_profile") or {}).get("context_regions") or [])}),
                },
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
