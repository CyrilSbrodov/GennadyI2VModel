from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import BBox, ExpressionState, Keypoint, OrientationState, PoseState


@dataclass(slots=True)
class PersonFacts:
    bbox: BBox
    mask_ref: str
    pose: PoseState
    expression: ExpressionState
    orientation: OrientationState
    garments: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class ObjectFacts:
    object_type: str
    bbox: BBox
    confidence: float


@dataclass(slots=True)
class PerceptionOutput:
    persons: list[PersonFacts] = field(default_factory=list)
    objects: list[ObjectFacts] = field(default_factory=list)
    frame_size: tuple[int, int] = (1024, 1024)


class PerceptionPipeline:
    """Pluggable perception facade. Current implementation is deterministic bootstrap."""

    def analyze(self, image_ref: str) -> PerceptionOutput:
        default_pose = PoseState(
            keypoints=[
                Keypoint("nose", 0.5, 0.2, 0.9),
                Keypoint("left_shoulder", 0.42, 0.35, 0.88),
                Keypoint("right_shoulder", 0.58, 0.35, 0.88),
            ],
            coarse_pose="standing",
            angles={"left_elbow": 8.0, "right_elbow": 7.0, "left_knee": 2.0, "right_knee": 3.0},
        )
        person = PersonFacts(
            bbox=BBox(0.25, 0.1, 0.5, 0.84),
            mask_ref=f"mask::{image_ref}::person_1",
            pose=default_pose,
            expression=ExpressionState(smile_intensity=0.1, label="neutral"),
            orientation=OrientationState(yaw=0.0, pitch=0.0, roll=0.0),
            garments=[
                {"type": "coat", "confidence": 0.88, "state": "worn"},
                {"type": "shirt", "confidence": 0.61, "state": "covered"},
            ],
        )
        return PerceptionOutput(
            persons=[person],
            objects=[ObjectFacts("chair", BBox(0.62, 0.55, 0.26, 0.4), 0.72)],
            frame_size=(1024, 1024),
        )
