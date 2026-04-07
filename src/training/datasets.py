from __future__ import annotations

import json
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
        samples: list[TrainingSample] = []
        for idx in range(max(0, len(graphs) - 1)):
            delta = GraphDelta(pose_deltas={"frame_delta": float(graphs[idx + 1].frame_index - graphs[idx].frame_index)})
            samples.append({"graphs": [graphs[idx], graphs[idx + 1]], "actions": actions, "deltas": [delta], "source": "graph_sequence", "delta_contract": _serialize_delta_contract(delta)})
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
    @classmethod
    def synthetic(cls, size: int) -> "RendererDataset":
        from utils_tensor import zeros

        return cls(samples=[{"frames": [zeros(64, 64, 3), [[[1.0, 1.0, 1.0] for _ in range(64)] for _ in range(64)]], "roi_pairs": [(zeros(16, 16, 3), [[[1.0, 1.0, 1.0] for _ in range(16)] for _ in range(16)])], "source": "synthetic"} for _ in range(size)])

    @classmethod
    def from_frame_pairs(cls, before_frames: list[list], after_frames: list[list]) -> "RendererDataset":
        pairs = list(zip(before_frames, after_frames))
        samples: list[TrainingSample] = []
        for before, after in pairs:
            samples.append({"frames": [before, after], "roi_pairs": [(before, after)], "source": "real_pair"})
        return cls(samples=samples)

    @classmethod
    def from_video_manifest(cls, manifest_path: str, quality_profile: str = "debug") -> "RendererDataset":
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
            out.append({"frames": [frames[0], frames[1]], "roi_pairs": roi_pairs, "source": rec.get("source", "video_manifest")})
        return cls(samples=out)


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
        manifest = json.loads(Path(manifest_path).read_text())
        samples: list[TrainingSample] = []
        for rec in manifest.get("records", []):
            actions = [ActionStep(type=a.get("type", "unknown"), priority=i + 1, target_entity=a.get("target_entity"), target_object=a.get("target_object")) for i, a in enumerate(rec.get("actions", []))]
            samples.append({"text": rec.get("text", ""), "actions": actions, "text_alignment": rec, "source": rec.get("source", "annotation_manifest")})
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
    return {
        "state_before": dict(delta.state_before),
        "state_after": dict(delta.state_after),
        "transition_phase": delta.transition_phase,
        "region_transition_mode": dict(delta.region_transition_mode),
        "predicted_visibility_changes": dict(delta.predicted_visibility_changes),
        "semantic_reasons": list(delta.semantic_reasons),
    }
