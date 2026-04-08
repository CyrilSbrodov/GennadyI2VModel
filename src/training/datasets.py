from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from core.input_layer import InputAssetLayer
from core.schema import (
    ActionStep,
    BBox,
    BodyPartNode,
    ExpressionState,
    GarmentNode,
    GlobalSceneContext,
    GraphDelta,
    Keypoint,
    OrientationState,
    PersonNode,
    PoseState,
    RelationEdge,
    RegionRef,
    SceneGraph,
    SceneObjectNode,
)
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from memory.video_memory import MemoryManager
from perception.pipeline import ObjectFacts, PerceptionOutput, PerceptionPipeline, PersonFacts
from representation.graph_builder import SceneGraphBuilder
from rendering.roi_renderer import ROISelector


class TrainingSample(TypedDict, total=False):
    frames: list[list]
    graphs: list[SceneGraph]
    actions: list[ActionStep]
    deltas: list[GraphDelta]
    roi_pairs: list[tuple[list, list]]
    targets: list[GraphDelta]
    text: str
    text_alignment: dict[str, object]
    memory_records: list[dict[str, object]]
    source: str
    sanity_metrics: dict[str, float]
    delta_contract: dict[str, object]
    text_action_contract: dict[str, object]
    graph_transition_contract: dict[str, object]
    patch_synthesis_contract: dict[str, object]
    temporal_consistency_contract: dict[str, object]


@dataclass(slots=True)
class BaseStageDataset:
    samples: list[TrainingSample]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        return self.samples[index]


class PerceptionDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "PerceptionDataset":
        from utils_tensor import zeros

        return cls(samples=[{"frames": [zeros(64, 64, 3)], "source": "synthetic"} for _ in range(size)])

    @classmethod
    def from_image_manifest(cls, manifest_path: str, quality_profile: str = "balanced") -> "PerceptionDataset":
        manifest = json.loads(Path(manifest_path).read_text())
        layer = InputAssetLayer()
        samples: list[TrainingSample] = []
        for rec in manifest.get("records", []):
            req = layer.build_request(images=[rec["image"]], text=rec.get("text", ""), quality_profile=quality_profile)
            frame = req.unified_asset.frames[0].tensor
            samples.append({"frames": [frame], "source": rec["image"], "text": rec.get("text", "")})
        return cls(samples=samples)


class RepresentationDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "RepresentationDataset":
        from utils_tensor import zeros

        samples: list[TrainingSample] = []
        for idx in range(size):
            graph = SceneGraph(frame_index=idx)
            samples.append({"frames": [zeros(64, 64, 3)], "graphs": [graph], "source": "synthetic"})
        return cls(samples=samples)

    @classmethod
    def from_perception_dataset(cls, source: PerceptionDataset) -> "RepresentationDataset":
        perception = PerceptionPipeline()
        builder = SceneGraphBuilder()
        out: list[TrainingSample] = []
        for idx, sample in enumerate(source.samples):
            text = sample.get("text", "")
            frame = sample.get("frames", [[]])[0]
            p = perception.analyze(frame)
            graph = builder.build(p, frame_index=idx)
            out.append({"frames": sample.get("frames", []), "graphs": [graph], "text": text, "source": sample.get("source", "")})
        return cls(samples=out)

    @classmethod
    def _from_cached_facts(cls, rec: dict[str, object], frame_index: int) -> PerceptionOutput:
        persons: list[PersonFacts] = []
        for idx, p in enumerate(rec.get("persons", [])):
            bbox_data = p.get("bbox")
            if not bbox_data:
                raise ValueError("perception cache person requires bbox")
            bbox = BBox(float(bbox_data["x"]), float(bbox_data["y"]), float(bbox_data["w"]), float(bbox_data["h"]))
            persons.append(
                PersonFacts(
                    bbox=bbox,
                    bbox_confidence=float(p.get("bbox_confidence", 0.0)),
                    bbox_source=str(p.get("bbox_source", "cache")),
                    mask_ref=p.get("mask_ref"),
                    mask_confidence=float(p.get("mask_confidence", 0.0)),
                    mask_source=str(p.get("mask_source", "cache")),
                    pose=PoseState(coarse_pose=str(p.get("coarse_pose", "unknown"))),
                    pose_confidence=float(p.get("pose_confidence", 0.0)),
                    pose_source=str(p.get("pose_source", "cache")),
                    expression=ExpressionState(label=str(p.get("expression_label", "neutral"))),
                    expression_confidence=float(p.get("expression_confidence", 0.0)),
                    expression_source=str(p.get("expression_source", "cache")),
                    orientation=OrientationState(),
                    orientation_confidence=float(p.get("orientation_confidence", 0.0)),
                    orientation_source=str(p.get("orientation_source", "cache")),
                    track_id=str(p.get("track_id", f"track_cached_{idx}")),
                    track_confidence=float(p.get("track_confidence", 0.0)),
                    track_source=str(p.get("track_source", "cache")),
                    garments=list(p.get("garments", [])),
                )
            )

        objects = [
            ObjectFacts(
                object_type=str(o.get("object_type", "unknown")),
                bbox=BBox(float(o["bbox"]["x"]), float(o["bbox"]["y"]), float(o["bbox"]["w"]), float(o["bbox"]["h"])),
                confidence=float(o.get("confidence", 0.0)),
                source=str(o.get("source", "cache")),
            )
            for o in rec.get("objects", [])
            if o.get("bbox")
        ]

        return PerceptionOutput(persons=persons, objects=objects, frame_size=tuple(rec.get("frame_size", [0, 0])), warnings=[], module_confidence={"cache": 1.0}, module_latency_ms={"cache": 0.0}, module_fallbacks={"source": "perception_cache"}, depth_score=rec.get("depth_score"))

    @classmethod
    def from_perception_cache(cls, cache_path: str, strict: bool = True) -> "RepresentationDataset":
        payload = json.loads(Path(cache_path).read_text())
        builder = SceneGraphBuilder()
        out: list[TrainingSample] = []
        for idx, rec in enumerate(payload.get("records", [])):
            if "persons" not in rec:
                if strict:
                    raise ValueError("perception cache record missing 'persons'")
                continue
            p = cls._from_cached_facts(rec, frame_index=int(rec.get("frame_index", idx)))
            graph = builder.build(p, frame_index=int(rec.get("frame_index", idx)))
            out.append({"frames": [rec.get("frame", [])] if isinstance(rec.get("frame"), list) else [], "graphs": [graph], "source": rec.get("source", "perception_cache")})
        return cls(samples=out)


class DynamicsDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "DynamicsDataset":
        from utils_tensor import zeros

        samples: list[TrainingSample] = []
        for idx in range(size):
            action = ActionStep(type="move", priority=1, target_entity=f"person_{idx}")
            delta = GraphDelta(pose_deltas={"torso_pitch": 0.1 * (idx + 1)})
            samples.append({"graphs": [SceneGraph(frame_index=idx)], "actions": [action], "deltas": [delta], "frames": [zeros(64, 64, 3), [[[1.0, 1.0, 1.0] for _ in range(64)] for _ in range(64)]], "source": "synthetic", "delta_contract": _serialize_delta_contract(delta)})
        return cls(samples=samples)

    @classmethod
    def from_graph_sequence(cls, graphs: list[SceneGraph], actions: list[ActionStep]) -> "DynamicsDataset":
        from training.learned_contracts import build_graph_transition_contract

        samples: list[TrainingSample] = []
        for idx in range(max(0, len(graphs) - 1)):
            delta = GraphDelta(pose_deltas={"frame_delta": float(graphs[idx + 1].frame_index - graphs[idx].frame_index)})
            graph_contract = build_graph_transition_contract(
                before=graphs[idx],
                after=graphs[idx + 1],
                delta=delta,
                transition_context={"source": "graph_sequence", "step_index": idx},
            )
            samples.append(
                {
                    "graphs": [graphs[idx], graphs[idx + 1]],
                    "actions": actions,
                    "deltas": [delta],
                    "source": "graph_sequence",
                    "delta_contract": _serialize_delta_contract(delta),
                    "graph_transition_contract": graph_contract,
                }
            )
        return cls(samples=samples)

    @classmethod
    def from_transition_manifest(cls, manifest_path: str) -> "DynamicsDataset":
        records = json.loads(Path(manifest_path).read_text()).get("records", [])
        predictor = GraphDeltaPredictor()
        out: list[TrainingSample] = []
        for idx, rec in enumerate(records):
            person = PersonNode(person_id="person_1", track_id="track_1", bbox=BBox(0.2, 0.1, 0.6, 0.8), mask_ref=None)
            prev = SceneGraph(frame_index=idx, persons=[person])
            nxt = SceneGraph(frame_index=idx + 1, persons=[person])
            action_labels = rec.get("labels", ["micro_adjust"])
            planned_state = type("_PS", (), {"labels": action_labels, "step_index": idx + 1})()
            delta, _ = predictor.predict(prev, planned_state)
            actions = [ActionStep(type=label, priority=i + 1) for i, label in enumerate(action_labels)]
            out.append({"graphs": [prev, nxt], "actions": actions, "deltas": [delta], "source": rec.get("source", "transition_manifest"), "delta_contract": _serialize_delta_contract(delta)})
        return cls(samples=out)


class RendererDataset(BaseStageDataset):
    @staticmethod
    def _as_hw3_tensor(value: object, field_name: str) -> list[list[list[float]]]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3:
            raise ValueError(f"{field_name} must be HxWxC tensor")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[2] != 3:
            raise ValueError(f"{field_name} channel count must be 3 or 1")
        return np.clip(arr, 0.0, 1.0).tolist()

    @staticmethod
    def _as_hw1_tensor(value: object, field_name: str) -> list[list[list[float]]]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3 or arr.shape[2] not in {1, 3}:
            raise ValueError(f"{field_name} must be HxW or HxWx1/HxWx3 tensor")
        if arr.shape[2] == 3:
            arr = np.mean(arr, axis=2, keepdims=True)
        return np.clip(arr, 0.0, 1.0).tolist()

    @staticmethod
    def _semantic_embed_from_family(family: str) -> list[float]:
        family = family.lower().strip()
        if family == "face_expression":
            return [1.0, 0.0, 0.0, 0.9, 0.1, 0.2]
        if family == "torso_reveal":
            return [0.0, 1.0, 0.0, 0.2, 0.85, 0.4]
        return [0.0, 0.0, 1.0, 0.15, 0.45, 0.9]

    @classmethod
    def from_renderer_manifest(cls, manifest_path: str, strict: bool = False) -> "RendererDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        samples: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_paired_roi",
            "manifest_path": manifest_path,
            "total_records": len(records),
            "loaded_records": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
            "family_counts": {"face_expression": 0, "torso_reveal": 0, "sleeve_arm_transition": 0},
        }
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be a json object")
                if "roi_before" not in rec or "roi_after" not in rec:
                    raise ValueError("record requires roi_before and roi_after")

                roi_before = cls._as_hw3_tensor(rec["roi_before"], "roi_before")
                roi_after = cls._as_hw3_tensor(rec["roi_after"], "roi_after")
                b = np.asarray(roi_before, dtype=np.float32)
                a = np.asarray(roi_after, dtype=np.float32)
                if b.shape != a.shape:
                    raise ValueError("roi_before and roi_after shape mismatch")

                family = str(rec.get("semantic_family", rec.get("region_family", ""))).strip().lower()
                region_id = str(rec.get("region_id", rec.get("region", {}).get("region_id", ""))).strip().lower()
                if not family:
                    if "face" in region_id or "head" in region_id:
                        family = "face_expression"
                    elif "torso" in region_id or "inner" in region_id:
                        family = "torso_reveal"
                    elif "arm" in region_id or "sleeve" in region_id or "outer" in region_id:
                        family = "sleeve_arm_transition"
                    else:
                        family = "sleeve_arm_transition"
                if family not in diagnostics["family_counts"]:
                    raise ValueError(f"unsupported semantic_family={family!r}")
                diagnostics["family_counts"][family] = int(diagnostics["family_counts"][family]) + 1

                changed_mask = rec.get("changed_mask")
                if changed_mask is None:
                    changed_mask = np.clip(np.mean(np.abs(a - b), axis=2, keepdims=True) * 3.0, 0.0, 1.0).tolist()
                alpha_target = rec.get("alpha_target")
                if alpha_target is None:
                    alpha_target = np.clip(0.15 + 0.85 * np.asarray(changed_mask, dtype=np.float32), 0.0, 1.0).tolist()
                blend_hint = rec.get("blend_hint")
                if blend_hint is None:
                    blend_hint = np.clip(0.2 + 0.75 * np.asarray(changed_mask, dtype=np.float32), 0.0, 1.0).tolist()

                changed_mask = cls._as_hw1_tensor(changed_mask, "changed_mask")
                alpha_target = cls._as_hw1_tensor(alpha_target, "alpha_target")
                blend_hint = cls._as_hw1_tensor(blend_hint, "blend_hint")

                renderer_contract = {
                    "semantic_embed": rec.get("semantic_embed", cls._semantic_embed_from_family(family)),
                    "delta_cond": rec.get("delta_cond", rec.get("graph_delta_cond", [0.0] * 9)),
                    "planner_cond": rec.get("planner_cond", rec.get("transition_context", {}).get("planner_cond", [0.0] * 8)),
                    "graph_cond": rec.get("graph_cond", rec.get("graph_context", [0.0] * 7)),
                    "memory_cond": rec.get("memory_cond", rec.get("memory_context", [0.0] * 8)),
                    "appearance_cond": rec.get("appearance_cond", np.concatenate([np.mean(b, axis=(0, 1)), np.std(b, axis=(0, 1))]).tolist()),
                    "bbox_cond": rec.get("bbox_cond", rec.get("bbox", [0.2, 0.2, 0.4, 0.4])),
                    "alpha_target": alpha_target,
                    "blend_hint": blend_hint,
                    "changed_mask": changed_mask,
                }

                samples.append(
                    {
                        "frames": [roi_before, roi_after],
                        "roi_pairs": [(roi_before, roi_after)],
                        "source": str(rec.get("source", "manifest_paired_roi")),
                        "region_family": family,
                        "region_id": region_id,
                        "renderer_batch_contract": renderer_contract,
                        "delta_contract": rec.get("graph_delta", rec.get("delta_contract", {})),
                        "graph_transition_contract": rec.get("graph_transition_contract", {}),
                        "memory_records": rec.get("memory_records", []),
                        "patch_synthesis_contract": rec.get("patch_synthesis_contract", {}),
                    }
                )
                diagnostics["loaded_records"] = int(diagnostics["loaded_records"]) + 1
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "error": str(exc)})
                if strict:
                    raise ValueError(f"renderer manifest record {idx} is invalid: {exc}") from exc

        ds = cls(samples=samples)
        ds.diagnostics = diagnostics
        return ds

    @classmethod
    def synthetic(cls, size: int) -> "RendererDataset":
        from utils_tensor import zeros

        samples: list[TrainingSample] = []
        families = ["face_expression", "torso_reveal", "sleeve_arm_transition"]
        for idx in range(size):
            family = families[idx % len(families)]
            before = zeros(16, 16, 3, value=0.25)
            after = zeros(16, 16, 3, value=0.25)
            for y in range(16):
                for x in range(16):
                    if family == "face_expression" and 6 <= y <= 11 and 4 <= x <= 11:
                        after[y][x] = [0.6, 0.35, 0.35]
                    elif family == "torso_reveal" and x in {7, 8}:
                        after[y][x] = [0.8, 0.78, 0.75]
                    elif family == "sleeve_arm_transition" and 3 <= x <= 10 and y >= 6:
                        after[y][x] = [0.45, 0.5, 0.7]
            yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 16), np.linspace(0.0, 1.0, 16), indexing="ij")
            changed = np.clip(np.mean(np.abs(np.asarray(after) - np.asarray(before)), axis=2, keepdims=True) * 3.0, 0.0, 1.0)
            blend_hint = np.clip(0.2 + 0.75 * changed + 0.05 * (1.0 - np.abs(xx - 0.5))[..., None], 0.0, 1.0)
            semantic = [1.0, 0.0, 0.0, 0.9, 0.1, 0.2] if family == "face_expression" else ([0.0, 1.0, 0.0, 0.2, 0.85, 0.4] if family == "torso_reveal" else [0.0, 0.0, 1.0, 0.15, 0.45, 0.9])
            samples.append(
                {
                    "frames": [zeros(64, 64, 3), zeros(64, 64, 3)],
                    "roi_pairs": [(before, after)],
                    "source": "synthetic",
                    "region_family": family,
                    "renderer_batch_contract": {
                        "semantic_embed": semantic,
                        "delta_cond": [0.2 + 0.1 * idx, 0.1, 0.2 if family != "face_expression" else 0.05, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        "planner_cond": [min(1.0, idx / max(1, size)), 0.0, 1.0, 0.0, 1.0 if family == "torso_reveal" else 0.0, 1.0 if family == "sleeve_arm_transition" else 0.0, 1.0 if family == "face_expression" else 0.0, 0.0],
                        "graph_cond": [1.0, 0.2, 0.1, 0.53, 0.5, 0.5, 0.15],
                        "memory_cond": [0.6, 0.4, 0.3, 0.5, 0.2, 0.35, 0.5, 1.0 if family == "torso_reveal" else 0.0],
                        "appearance_cond": [0.25, 0.25, 0.25, 0.02, 0.02, 0.02],
                        "bbox_cond": [0.2, 0.2, 0.4, 0.4],
                        "alpha_target": np.clip(0.15 + 0.85 * changed, 0.0, 1.0).tolist(),
                        "blend_hint": blend_hint.tolist(),
                    },
                }
            )
        ds = cls(samples=samples)
        ds.diagnostics = {
            "source": "synthetic_bootstrap",
            "total_records": size,
            "loaded_records": size,
            "invalid_records": 0,
            "skipped_records": 0,
            "family_counts": {
                "face_expression": sum(1 for s in samples if s["region_family"] == "face_expression"),
                "torso_reveal": sum(1 for s in samples if s["region_family"] == "torso_reveal"),
                "sleeve_arm_transition": sum(1 for s in samples if s["region_family"] == "sleeve_arm_transition"),
            },
            "invalid_examples": [],
        }
        return ds

    @classmethod
    def from_frame_pairs(cls, before_frames: list[list], after_frames: list[list]) -> "RendererDataset":
        pairs = list(zip(before_frames, after_frames))
        samples: list[TrainingSample] = []
        for before, after in pairs:
            samples.append({"frames": [before, after], "roi_pairs": [(before, after)], "source": "real_pair"})
        return cls(samples=samples)

    @classmethod
    def from_video_manifest(cls, manifest_path: str, quality_profile: str = "debug") -> "RendererDataset":
        from training.learned_contracts import build_patch_synthesis_contract, build_temporal_consistency_contract

        layer = InputAssetLayer()
        selector = ROISelector()
        records = json.loads(Path(manifest_path).read_text()).get("records", [])
        out: list[TrainingSample] = []
        for rec in records:
            req = layer.build_request(images=rec.get("images", []), text=rec.get("text", ""), quality_profile=quality_profile)
            frames = [f.tensor for f in (req.unified_asset.frames if req.unified_asset else [])]
            if len(frames) < 2:
                continue
            person = PersonNode(person_id="person_1", track_id="track_1", bbox=BBox(0.2, 0.1, 0.6, 0.8), mask_ref=None)
            graph = SceneGraph(frame_index=0, persons=[person])
            delta = GraphDelta(affected_entities=[person.person_id], semantic_reasons=rec.get("reasons", []), affected_regions=rec.get("affected_regions", []))
            rois = selector.select(graph, delta)
            roi_pairs = [(frames[0], frames[1]) for _ in rois] or [(frames[0], frames[1])]
            patch_contract = None
            temporal_contract = None
            if rois:
                patch_contract = build_patch_synthesis_contract(
                    roi_before=frames[0],
                    roi_after=frames[1],
                    region=rois[0],
                    retrieval_summary=rec.get("retrieval_summary", "manifest"),
                    selected_strategy=rec.get("selected_strategy", "fallback"),
                    hidden_state=rec.get("hidden_lifecycle_state", {}),
                    synthesis_mode=rec.get("synthesis_mode", "deterministic"),
                    transition_context=rec.get("transition_context", {}),
                )
                temporal_contract = build_temporal_consistency_contract(
                    previous_frame=frames[0],
                    composed_frame=frames[1],
                    target_frame=frames[1],
                    changed_regions=rois,
                    region_consistency_metadata=rec.get("region_consistency_metadata", {}),
                    scene_transition_context=rec.get("scene_transition_context", {}),
                    memory_transition_context=rec.get("memory_transition_context", {}),
                )
            out.append(
                {
                    "frames": [frames[0], frames[1]],
                    "roi_pairs": roi_pairs,
                    "patch_synthesis_contract": patch_contract,
                    "temporal_consistency_contract": temporal_contract,
                    "renderer_batch_contract": {
                        "semantic_embed": rec.get("semantic_embed", [0.0, 1.0, 0.0, 0.25, 0.8, 0.35]),
                        "delta_cond": rec.get("delta_cond", [0.3, 0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
                        "planner_cond": rec.get("planner_cond", [0.2, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                        "graph_cond": rec.get("graph_cond", [1.0, 0.2, 0.15, 0.5, 0.5, 0.5, 0.2]),
                        "memory_cond": rec.get("memory_cond", [0.5, 0.4, 0.2, 0.6, 0.3, 0.2, 0.4, 0.0]),
                        "appearance_cond": rec.get("appearance_cond", [0.3, 0.3, 0.3, 0.05, 0.05, 0.05]),
                        "bbox_cond": rec.get("bbox_cond", [0.2, 0.2, 0.4, 0.4]),
                    },
                    "source": rec.get("source", "video_manifest"),
                }
            )
        return cls(samples=out)


class TemporalDataset(BaseStageDataset):
    @staticmethod
    def _as_hw3_tensor(value: object, field_name: str) -> list[list[list[float]]]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3:
            raise ValueError(f"{field_name} must be HxWxC tensor")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[2] != 3:
            raise ValueError(f"{field_name} channel count must be 3 or 1")
        return np.clip(arr, 0.0, 1.0).tolist()

    @classmethod
    def synthetic(cls, size: int) -> "TemporalDataset":
        samples: list[TrainingSample] = []
        for idx in range(size):
            base = 0.22 + 0.04 * (idx % 4)
            prev = np.full((24, 24, 3), base, dtype=np.float32)
            cur = prev.copy()
            cur[5:18, 7:20, :] = np.array([min(1.0, base + 0.2), base + 0.05, base + 0.03], dtype=np.float32)
            target = cur.copy()
            target[6:17, 8:19, :] = np.array([min(1.0, base + 0.17), base + 0.04, base + 0.03], dtype=np.float32)
            rid = RegionRef(region_id=f"p1:region_{idx%3}", bbox=BBox(0.25, 0.2, 0.45, 0.45), reason="temporal_drift")
            samples.append(
                {
                    "frames": [prev.tolist(), cur.tolist(), target.tolist()],
                    "temporal_consistency_contract": {
                        "previous_frame": prev.tolist(),
                        "composed_frame": cur.tolist(),
                        "target_frame": target.tolist(),
                        "changed_regions": [{"region_id": rid.region_id, "reason": rid.reason, "bbox": {"x": rid.bbox.x, "y": rid.bbox.y, "w": rid.bbox.w, "h": rid.bbox.h}}],
                        "region_consistency_metadata": {"alpha_mean": 0.55, "confidence_mean": 0.72},
                        "scene_transition_context": {"transition_phase": "motion", "step_index": idx + 1},
                        "memory_transition_context": {"drift": [0.02, 0.05, 0.1][idx % 3]},
                    },
                    "source": "synthetic_temporal",
                }
            )
        ds = cls(samples=samples)
        ds.diagnostics = {"source": "synthetic_temporal", "loaded_records": len(samples), "invalid_records": 0, "skipped_records": 0}
        return ds

    @classmethod
    def from_temporal_manifest(cls, manifest_path: str, strict: bool = False) -> "TemporalDataset":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        samples: list[TrainingSample] = []
        diagnostics: dict[str, object] = {
            "source": "manifest_temporal_consistency",
            "manifest_path": manifest_path,
            "total_records": len(records),
            "loaded_records": 0,
            "invalid_records": 0,
            "skipped_records": 0,
            "invalid_examples": [],
        }
        for idx, rec in enumerate(records):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be a json object")
                prev = cls._as_hw3_tensor(rec["previous_frame"], "previous_frame")
                cur = cls._as_hw3_tensor(rec["current_composed_frame"], "current_composed_frame")
                target = cls._as_hw3_tensor(rec.get("target_frame", rec["current_composed_frame"]), "target_frame")
                changed_regions_raw = rec.get("changed_regions", [])
                changed_regions: list[dict[str, object]] = []
                for ridx, region in enumerate(changed_regions_raw if isinstance(changed_regions_raw, list) else []):
                    if not isinstance(region, dict):
                        continue
                    bb = region.get("bbox", {}) if isinstance(region.get("bbox", {}), dict) else {}
                    changed_regions.append(
                        {
                            "region_id": str(region.get("region_id", f"scene:region_{ridx}")),
                            "reason": str(region.get("reason", "temporal_drift")),
                            "bbox": {
                                "x": float(bb.get("x", 0.2)),
                                "y": float(bb.get("y", 0.2)),
                                "w": float(bb.get("w", 0.3)),
                                "h": float(bb.get("h", 0.3)),
                            },
                        }
                    )
                if not changed_regions:
                    changed_regions = [{"region_id": "scene:region_0", "reason": "temporal_drift", "bbox": {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}}]
                contract = {
                    "previous_frame": prev,
                    "composed_frame": cur,
                    "target_frame": target,
                    "changed_regions": changed_regions,
                    "region_consistency_metadata": rec.get("region_consistency_metadata", {}),
                    "scene_transition_context": rec.get("scene_transition_context", {}),
                    "memory_transition_context": rec.get("memory_transition_context", {}),
                }
                samples.append({"frames": [prev, cur, target], "temporal_consistency_contract": contract, "source": rec.get("source", "temporal_manifest")})
                diagnostics["loaded_records"] = int(diagnostics["loaded_records"]) + 1
            except Exception as exc:
                diagnostics["invalid_records"] = int(diagnostics["invalid_records"]) + 1
                if len(diagnostics["invalid_examples"]) < 8:
                    diagnostics["invalid_examples"].append({"index": idx, "reason": str(exc)})
                if strict:
                    raise
        ds = cls(samples=samples)
        ds.diagnostics = diagnostics
        return ds



class TextActionDataset(BaseStageDataset):
    @classmethod
    def from_jsonl(cls, path: str) -> "TextActionDataset":
        samples: list[TrainingSample] = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            steps = [ActionStep(type=a["type"], priority=i + 1, target_object=a.get("target_object")) for i, a in enumerate(rec.get("actions", []))]
            samples.append({"text": rec.get("text", ""), "actions": steps, "text_alignment": rec, "source": rec.get("source", "annotation")})
        return cls(samples=samples)

    @classmethod
    def from_annotation_manifest(cls, manifest_path: str) -> "TextActionDataset":
        from text.learned_bridge import BaselineTextEncoderAdapter
        from training.learned_contracts import build_text_action_state_contract

        manifest = json.loads(Path(manifest_path).read_text())
        encoder = BaselineTextEncoderAdapter()
        samples: list[TrainingSample] = []
        for rec in manifest.get("records", []):
            actions = [ActionStep(type=a.get("type", "unknown"), priority=i + 1, target_entity=a.get("target_entity"), target_object=a.get("target_object")) for i, a in enumerate(rec.get("actions", []))]
            encoding = encoder.encode(rec.get("text", ""))
            contract = build_text_action_state_contract(rec.get("text", ""), actions, vars(encoding))
            samples.append(
                {
                    "text": rec.get("text", ""),
                    "actions": actions,
                    "text_alignment": rec,
                    "text_action_contract": contract,
                    "source": rec.get("source", "annotation_manifest"),
                }
            )
        return cls(samples=samples)


class MemoryDataset(BaseStageDataset):
    @classmethod
    def from_representation_dataset(cls, ds: RepresentationDataset) -> "MemoryDataset":
        samples: list[TrainingSample] = []
        for sample in ds.samples:
            graphs = sample.get("graphs", [])
            records = []
            for g in graphs:
                records.append({"frame_index": g.frame_index, "person_count": len(g.persons), "object_count": len(g.objects)})
            samples.append({"memory_records": records, "source": sample.get("source", "representation")})
        return cls(samples=samples)

    @classmethod
    def from_graph_sequence(cls, graphs: list[SceneGraph], frames: list[list] | None = None) -> "MemoryDataset":
        manager = MemoryManager()
        records: list[dict[str, object]] = []
        if not graphs:
            return cls(samples=[])
        memory = manager.initialize(graphs[0])
        for idx, graph in enumerate(graphs):
            memory = manager.update_from_graph(memory, graph)
            if frames and idx < len(frames):
                memory = manager.update_from_frame(memory, frames[idx], graph)
            records.append({"frame_index": graph.frame_index, "regions": len(memory.region_descriptors), "patches": len(memory.texture_patches), "hidden_slots": len(memory.hidden_region_slots)})
        return cls(samples=[{"memory_records": records, "source": "graph_sequence"}])


def _serialize_bbox(b: BBox) -> dict[str, float]:
    return {"x": b.x, "y": b.y, "w": b.w, "h": b.h}


def _serialize_graph(graph: SceneGraph) -> dict[str, object]:
    return {
        "frame_index": graph.frame_index,
        "global_context": {"frame_size": list(graph.global_context.frame_size), "fps": graph.global_context.fps, "source_type": graph.global_context.source_type},
        "persons": [
            {
                "person_id": p.person_id,
                "track_id": p.track_id,
                "bbox": _serialize_bbox(p.bbox),
                "confidence": p.confidence,
                "garments": [{"garment_id": g.garment_id, "garment_type": g.garment_type, "coverage_targets": g.coverage_targets, "visibility": g.visibility, "confidence": g.confidence} for g in p.garments],
                "body_parts": [{"part_id": bp.part_id, "part_type": bp.part_type, "visibility": bp.visibility, "confidence": bp.confidence, "keypoints": [{"name": kp.name, "x": kp.x, "y": kp.y, "confidence": kp.confidence} for kp in bp.keypoints]} for bp in p.body_parts],
            }
            for p in graph.persons
        ],
        "objects": [{"object_id": o.object_id, "object_type": o.object_type, "bbox": _serialize_bbox(o.bbox), "confidence": o.confidence} for o in graph.objects],
        "relations": [{"source": r.source, "relation": r.relation, "target": r.target, "confidence": r.confidence, "provenance": r.provenance} for r in graph.relations],
    }


def _deserialize_graph(payload: dict[str, object]) -> SceneGraph:
    persons = []
    for p in payload.get("persons", []):
        body_parts = []
        for bp in p.get("body_parts", []):
            kps = [Keypoint(name=kp.get("name", "kp"), x=float(kp.get("x", 0.0)), y=float(kp.get("y", 0.0)), confidence=float(kp.get("confidence", 0.0))) for kp in bp.get("keypoints", [])]
            body_parts.append(BodyPartNode(part_id=bp.get("part_id", "bp"), part_type=bp.get("part_type", "unknown"), visibility=bp.get("visibility", "unknown"), confidence=float(bp.get("confidence", 0.0)), keypoints=kps))
        garments = [GarmentNode(garment_id=g.get("garment_id", "garment"), garment_type=g.get("garment_type", "unknown"), coverage_targets=list(g.get("coverage_targets", [])), visibility=g.get("visibility", "unknown"), confidence=float(g.get("confidence", 0.0))) for g in p.get("garments", [])]
        bb = p.get("bbox", {})
        persons.append(PersonNode(person_id=p.get("person_id", "person"), track_id=p.get("track_id"), bbox=BBox(float(bb.get("x", 0.0)), float(bb.get("y", 0.0)), float(bb.get("w", 0.0)), float(bb.get("h", 0.0))), mask_ref=None, body_parts=body_parts, garments=garments, confidence=float(p.get("confidence", 0.0))))

    objects = []
    for o in payload.get("objects", []):
        bb = o.get("bbox", {})
        objects.append(SceneObjectNode(object_id=o.get("object_id", "object"), object_type=o.get("object_type", "unknown"), bbox=BBox(float(bb.get("x", 0.0)), float(bb.get("y", 0.0)), float(bb.get("w", 0.0)), float(bb.get("h", 0.0))), confidence=float(o.get("confidence", 0.0))))

    relations = [RelationEdge(source=r.get("source", ""), relation=r.get("relation", "near"), target=r.get("target", ""), confidence=float(r.get("confidence", 0.0)), provenance=r.get("provenance", "cache")) for r in payload.get("relations", [])]
    gc = payload.get("global_context", {})
    context = GlobalSceneContext(frame_size=tuple(gc.get("frame_size", [0, 0])), fps=int(gc.get("fps", 16)), source_type=str(gc.get("source_type", "single_image")))
    return SceneGraph(frame_index=int(payload.get("frame_index", 0)), persons=persons, objects=objects, relations=relations, global_context=context)


def save_graph_cache(graphs: list[SceneGraph], path: str) -> None:
    payload = {"cache_version": 2, "partial_fields": ["persons.bbox", "objects.bbox", "garments", "body_parts", "relations", "global_context"], "graphs": [_serialize_graph(g) for g in graphs]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_graph_cache(path: str) -> list[SceneGraph]:
    payload = json.loads(Path(path).read_text())
    return [_deserialize_graph(rec) for rec in payload.get("graphs", [])]


def _serialize_delta_contract(delta: GraphDelta) -> dict[str, object]:
    magnitude = 0.0
    for payload in (delta.pose_deltas, delta.interaction_deltas, delta.garment_deltas, delta.expression_deltas):
        magnitude += sum(abs(float(v)) for v in payload.values() if isinstance(v, (int, float)))
    return {
        "state_before": dict(delta.state_before),
        "state_after": dict(delta.state_after),
        "transition_phase": delta.transition_phase,
        "region_transition_mode": dict(delta.region_transition_mode),
        "predicted_visibility_changes": dict(delta.predicted_visibility_changes),
        "semantic_reasons": list(delta.semantic_reasons),
        "affected_entities": list(delta.affected_entities),
        "affected_regions": list(delta.affected_regions),
        "pose_deltas": dict(delta.pose_deltas),
        "interaction_deltas": dict(delta.interaction_deltas),
        "garment_deltas": dict(delta.garment_deltas),
        "expression_deltas": dict(delta.expression_deltas),
        "magnitude": magnitude,
    }
