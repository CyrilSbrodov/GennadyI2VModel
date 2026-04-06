from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from core.schema import ActionStep, GraphDelta, SceneGraph


class TrainingSample(TypedDict, total=False):
    frames: list[list]
    graphs: list[SceneGraph]
    actions: list[ActionStep]
    deltas: list[GraphDelta]
    roi_pairs: list[tuple[list, list]]
    targets: list[GraphDelta]
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
        return cls(samples=[{"frames": [zeros(64, 64, 3)]} for _ in range(size)])


class RepresentationDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "RepresentationDataset":
        from utils_tensor import zeros
        samples: list[TrainingSample] = []
        for idx in range(size):
            graph = SceneGraph(frame_index=idx)
            samples.append({"frames": [zeros(64, 64, 3)], "graphs": [graph]})
        return cls(samples=samples)


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
                    "frames": [zeros(64, 64, 3), [[[1.0,1.0,1.0] for _ in range(64)] for _ in range(64)]],
                }
            )
        return cls(samples=samples)


class RendererDataset(BaseStageDataset):
    @classmethod
    def synthetic(cls, size: int) -> "RendererDataset":
        from utils_tensor import zeros
        return cls(
            samples=[
                {
                    "frames": [zeros(64, 64, 3), [[[1.0,1.0,1.0] for _ in range(64)] for _ in range(64)]],
                    "roi_pairs": [(zeros(16, 16, 3), [[[1.0,1.0,1.0] for _ in range(16)] for _ in range(16)])],
                }
                for _ in range(size)
            ]
        )
