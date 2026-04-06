from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from core.schema import ActionStep, GraphDelta, SceneGraph


class TrainingSample(TypedDict, total=False):
    frames: list[str]
    graphs: list[SceneGraph]
    actions: list[ActionStep]
    deltas: list[GraphDelta]
    roi_pairs: list[tuple[str, str]]


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
        return cls(samples=[{"frames": [f"frame_{idx:04d}.png"]} for idx in range(size)])


class RepresentationDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "RepresentationDataset":
        samples: list[TrainingSample] = []
        for idx in range(size):
            graph = SceneGraph(frame_index=idx)
            samples.append({"frames": [f"frame_{idx:04d}.png"], "graphs": [graph]})
        return cls(samples=samples)


class DynamicsDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "DynamicsDataset":
        samples: list[TrainingSample] = []
        for idx in range(size):
            action = ActionStep(type="move", priority=1, target_entity=f"person_{idx}")
            delta = GraphDelta(pose_deltas={"torso_pitch": 0.1 * (idx + 1)})
            samples.append({"graphs": [SceneGraph(frame_index=idx)], "actions": [action], "deltas": [delta]})
        return cls(samples=samples)


class RendererDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "RendererDataset":
        return cls(
            samples=[
                {
                    "frames": [f"before_{idx:04d}.png", f"after_{idx:04d}.png"],
                    "roi_pairs": [(f"roi_before_{idx}", f"roi_after_{idx}")],
                }
                for idx in range(size)
            ]
        )
