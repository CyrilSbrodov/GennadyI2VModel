from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


VisibilityState = Literal["visible", "partially_visible", "hidden", "unknown"]
RelationType = Literal[
    "part_of",
    "attached_to",
    "covers",
    "occludes",
    "supports",
    "interacts_with",
    "touches",
    "near",
    "inside",
    "in_front_of",
    "behind",
    "held_by",
]


@dataclass(slots=True)
class BBox:
    x: float
    y: float
    w: float
    h: float


@dataclass(slots=True)
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class PoseState:
    keypoints: list[Keypoint] = field(default_factory=list)
    coarse_pose: str = "unknown"
    angles: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExpressionState:
    smile_intensity: float = 0.0
    eye_openness: float = 0.5
    mouth_state: str = "neutral"
    label: str = "neutral"


@dataclass(slots=True)
class OrientationState:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(slots=True)
class BodyPartNode:
    part_id: str
    part_type: str
    keypoints: list[Keypoint] = field(default_factory=list)
    mask_ref: str | None = None
    visibility: VisibilityState = "unknown"
    occluded_by: list[str] = field(default_factory=list)
    depth_order: float = 0.0
    canonical_slot: str = ""
    confidence: float = 0.0
    source: str = "unknown"
    frame_index: int = 0
    timestamp: float | None = None
    alternatives: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GarmentNode:
    garment_id: str
    garment_type: str
    mask_ref: str | None = None
    attachment_targets: list[str] = field(default_factory=list)
    coverage_targets: list[str] = field(default_factory=list)
    garment_state: str = "worn"
    visibility: VisibilityState = "unknown"
    appearance_ref: str | None = None
    confidence: float = 0.0
    source: str = "unknown"
    frame_index: int = 0
    timestamp: float | None = None
    alternatives: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PersonNode:
    person_id: str
    track_id: str | None
    bbox: BBox
    mask_ref: str | None
    pose_state: PoseState = field(default_factory=PoseState)
    expression_state: ExpressionState = field(default_factory=ExpressionState)
    orientation: OrientationState = field(default_factory=OrientationState)
    body_parts: list[BodyPartNode] = field(default_factory=list)
    garments: list[GarmentNode] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "unknown"
    frame_index: int = 0
    timestamp: float | None = None
    alternatives: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SceneObjectNode:
    object_id: str
    object_type: str
    bbox: BBox
    mask_ref: str | None = None
    confidence: float = 0.0
    source: str = "unknown"
    frame_index: int = 0
    timestamp: float | None = None
    alternatives: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RelationEdge:
    source: str
    relation: RelationType
    target: str
    confidence: float = 0.0
    provenance: str = "unknown"
    frame_index: int = 0
    timestamp: float | None = None
    alternatives: list[RelationType] = field(default_factory=list)


@dataclass(slots=True)
class GlobalSceneContext:
    frame_size: tuple[int, int] = (0, 0)
    fps: int = 16
    source_type: str = "single_image"


@dataclass(slots=True)
class SceneGraph:
    frame_index: int
    persons: list[PersonNode] = field(default_factory=list)
    objects: list[SceneObjectNode] = field(default_factory=list)
    relations: list[RelationEdge] = field(default_factory=list)
    global_context: GlobalSceneContext = field(default_factory=GlobalSceneContext)
    timestamp: float | None = None


@dataclass(slots=True)
class ActionStep:
    type: str
    priority: int
    target_entity: str | None = None
    target_object: str | None = None
    body_part: str | None = None
    intensity: float | None = None
    duration_sec: float | None = None
    start_after: list[int] = field(default_factory=list)
    can_run_parallel: bool = False
    constraints: list[str] = field(default_factory=list)
    modifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionPlan:
    actions: list[ActionStep] = field(default_factory=list)
    temporal_ordering: list[int] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    parallel_groups: list[list[int]] = field(default_factory=list)
    global_style: str | None = None


@dataclass(slots=True)
class RegionRef:
    region_id: str
    bbox: BBox
    reason: str


@dataclass(slots=True)
class GraphDelta:
    pose_deltas: dict[str, float] = field(default_factory=dict)
    garment_deltas: dict[str, float | str] = field(default_factory=dict)
    expression_deltas: dict[str, float | str] = field(default_factory=dict)
    interaction_deltas: dict[str, float] = field(default_factory=dict)
    visibility_deltas: dict[str, VisibilityState] = field(default_factory=dict)
    newly_revealed_regions: list[RegionRef] = field(default_factory=list)
    newly_occluded_regions: list[RegionRef] = field(default_factory=list)
    affected_entities: list[str] = field(default_factory=list)
    affected_regions: list[str] = field(default_factory=list)
    semantic_reasons: list[str] = field(default_factory=list)
    predicted_visibility_changes: dict[str, VisibilityState] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryEntry:
    entity_id: str
    entry_type: str
    embedding: list[float]
    confidence: float
    last_seen_frames: list[int] = field(default_factory=list)


@dataclass(slots=True)
class TexturePatchMemory:
    patch_id: str
    region_type: str
    entity_id: str
    source_frame: int
    patch_ref: str
    confidence: float
    descriptor: dict[str, float | list[float]] = field(default_factory=dict)
    evidence_score: float = 0.0


@dataclass(slots=True)
class RegionDescriptor:
    region_id: str
    entity_id: str
    region_type: str
    bbox: BBox
    visibility: VisibilityState
    confidence: float
    last_update_frame: int


@dataclass(slots=True)
class HiddenRegionSlot:
    slot_id: str
    region_type: str
    owner_entity: str
    candidate_patch_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    stale_frames: int = 0


@dataclass(slots=True)
class PlannerDiagnostics:
    skipped_actions: list[str] = field(default_factory=list)
    inserted_objects: list[str] = field(default_factory=list)
    policy_decisions: list[str] = field(default_factory=list)
    constraint_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoMemory:
    identity_memory: dict[str, MemoryEntry] = field(default_factory=dict)
    garment_memory: dict[str, MemoryEntry] = field(default_factory=dict)
    temporal_history: list[SceneGraph] = field(default_factory=list)
    patch_cache: dict[str, Any] = field(default_factory=dict)
    hidden_region_memory: dict[str, MemoryEntry] = field(default_factory=dict)
    texture_patches: dict[str, TexturePatchMemory] = field(default_factory=dict)
    region_descriptors: dict[str, RegionDescriptor] = field(default_factory=dict)
    hidden_region_slots: dict[str, HiddenRegionSlot] = field(default_factory=dict)
