from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from core.input_layer import InputAssetLayer
from core.schema import ActionStep, GraphDelta, SceneGraph
from perception.pipeline import PerceptionPipeline
from representation.graph_builder import SceneGraphBuilder


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
            source_ref = f"dataset://{sample.get('source', idx)}"
            p = perception.analyze(source_ref)
            graph = builder.build(p, frame_index=idx)
            out.append({"frames": sample.get("frames", []), "graphs": [graph], "text": text, "source": sample.get("source", "")})
        return cls(samples=out)


class DynamicsDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "DynamicsDataset":
        from utils_tensor import zeros

        samples: list[TrainingSample] = []
        for idx in range(size):
            action = ActionStep(type="move", priority=1, target_entity=f"person_{idx}")
            delta = GraphDelta(pose_deltas={"torso_pitch": 0.1 * (idx + 1)})
            samples.append(
                {
                    "graphs": [SceneGraph(frame_index=idx)],
                    "actions": [action],
                    "deltas": [delta],
                    "frames": [zeros(64, 64, 3), [[[1.0, 1.0, 1.0] for _ in range(64)] for _ in range(64)]],
                    "source": "synthetic",
                }
            )
        return cls(samples=samples)

    @classmethod
    def from_graph_sequence(cls, graphs: list[SceneGraph], actions: list[ActionStep]) -> "DynamicsDataset":
        samples: list[TrainingSample] = []
        for idx in range(max(0, len(graphs) - 1)):
            delta = GraphDelta(pose_deltas={"frame_delta": float(graphs[idx + 1].frame_index - graphs[idx].frame_index)})
            samples.append({"graphs": [graphs[idx], graphs[idx + 1]], "actions": actions, "deltas": [delta], "source": "graph_sequence"})
        return cls(samples=samples)


class RendererDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "RendererDataset":
        from utils_tensor import zeros

        return cls(
            samples=[
                {
                    "frames": [zeros(64, 64, 3), [[[1.0, 1.0, 1.0] for _ in range(64)] for _ in range(64)]],
                    "roi_pairs": [(zeros(16, 16, 3), [[[1.0, 1.0, 1.0] for _ in range(16)] for _ in range(16)])],
                    "source": "synthetic",
                }
                for _ in range(size)
            ]
        )

    @classmethod
    def from_frame_pairs(cls, before_frames: list[list], after_frames: list[list]) -> "RendererDataset":
        pairs = list(zip(before_frames, after_frames))
        samples: list[TrainingSample] = []
        for before, after in pairs:
            samples.append({"frames": [before, after], "roi_pairs": [(before, after)], "source": "real_pair"})
        return cls(samples=samples)


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
