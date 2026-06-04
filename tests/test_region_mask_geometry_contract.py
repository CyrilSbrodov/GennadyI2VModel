from __future__ import annotations

from core.schema import BBox, BodyPartNode, PersonNode, SceneGraph
from perception.mask_store import InMemoryMaskStore
from runtime.region_mask_propagation import seed_input_region_observations


def test_seed_input_missing_body_part_bbox_is_no_geometry_not_person_bbox() -> None:
    person_bbox = BBox(0.1, 0.1, 0.7, 0.8)
    scene = SceneGraph(
        frame_index=0,
        persons=[
            PersonNode(
                person_id="p1",
                track_id="t1",
                bbox=person_bbox,
                mask_ref=None,
                body_parts=[BodyPartNode(part_id="f", part_type="face", visibility="visible", confidence=0.9)],
                confidence=0.9,
            )
        ],
        objects=[],
    )

    observations = seed_input_region_observations(scene, InMemoryMaskStore())

    face = next(o for o in observations if o.region_id == "p1:face")
    assert face.observation_source == "no_geometry"
    assert face.confidence == 0.0
    assert face.bbox != person_bbox
    assert face.bbox == BBox(0.0, 0.0, 0.0, 0.0)
