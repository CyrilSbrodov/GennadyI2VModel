import json
from pathlib import Path

from core.region_ids import is_canonical_region_id, make_region_id
from core.schema import BBox, GraphDelta, Keypoint, PersonNode, RegionRef, SceneGraph
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from memory.video_memory import MemoryManager
from perception.pipeline import PerceptionPipeline
from rendering.compositor import TemporalStabilizer
from rendering.roi_renderer import PatchRenderer, ROISelector
from training.datasets import DynamicsDataset, RepresentationDataset, load_graph_cache, save_graph_cache


def _solid(h: int, w: int, rgb: tuple[float, float, float]) -> list:
    return [[[rgb[0], rgb[1], rgb[2]] for _ in range(w)] for _ in range(h)]


def test_perception_tensor_input_changes_with_visual_content() -> None:
    p = PerceptionPipeline()
    red = _solid(32, 32, (1.0, 0.0, 0.0))
    blue = _solid(32, 32, (0.0, 0.0, 1.0))

    out_red = p.analyze(red)
    out_blue = p.analyze(blue)

    assert out_red.module_fallbacks["input_mode"] == "frame_tensor"
    assert out_blue.persons[0].bbox != out_red.persons[0].bbox


def test_memory_descriptors_differ_for_different_visual_regions() -> None:
    mm = MemoryManager()
    graph = SceneGraph(frame_index=0, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)], objects=[])
    mem = mm.initialize(graph)
    mem = mm.update_from_frame(mem, _solid(64, 64, (1.0, 0.0, 0.0)), graph)
    desc_red = mm.retrieve_for_region(mem, "face", "p1")[0].descriptor
    mem = mm.update_from_frame(mem, _solid(64, 64, (0.0, 1.0, 0.0)), SceneGraph(frame_index=1, persons=graph.persons, objects=[]))
    desc_green = mm.retrieve_for_region(mem, "face", "p1")[0].descriptor
    assert desc_red["mean"] != desc_green["mean"]


def test_canonical_region_ids_across_delta_roi_memory_and_renderer() -> None:
    predictor = GraphDeltaPredictor()
    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)
    graph = SceneGraph(frame_index=0, persons=[person], objects=[])
    planned = type("_PS", (), {"labels": ["sit_down"], "step_index": 1})()
    delta, _ = predictor.predict(graph, planned)
    assert delta.newly_revealed_regions
    assert all(is_canonical_region_id(r.region_id) for r in delta.newly_revealed_regions)

    selector = ROISelector()
    rois = selector.select(graph, delta)
    assert rois and all(is_canonical_region_id(r.region_id) for r in rois)

    mm = MemoryManager()
    memory = mm.initialize(graph)
    memory = mm.update_from_frame(memory, _solid(64, 64, (0.6, 0.2, 0.2)), graph)
    rid = make_region_id("p1", "garments")
    mm.apply_visibility_event(memory, {"region_id": rid, "entity": "p1"}, {}, "hidden")
    assert mm.query_hidden_region(memory, rid) is not None

    renderer = PatchRenderer()
    patch = renderer.render(_solid(64, 64, (0.2, 0.2, 0.2)), graph, delta, memory, rois[0])
    assert any("strategy=" in msg for msg in patch.debug_trace)


def test_roi_selector_uses_graph_semantics_before_fallback() -> None:
    selector = ROISelector()
    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)
    person.body_parts = [
        type("BP", (), {"part_type": "head", "keypoints": [Keypoint("k1", 0.45, 0.2, 0.9), Keypoint("k2", 0.55, 0.22, 0.9)]})(),
        type("BP", (), {"part_type": "left_upper_arm", "keypoints": [Keypoint("a1", 0.25, 0.35, 0.8), Keypoint("a2", 0.2, 0.45, 0.8)]})(),
    ]
    graph = SceneGraph(frame_index=0, persons=[person], objects=[])
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["face", "left_arm"])
    rois = selector.select(graph, delta)
    reasons = {r.reason for r in rois}
    assert any(r.startswith("graph_semantic:") for r in reasons)


def test_temporal_stabilizer_only_updates_rois() -> None:
    st = TemporalStabilizer()
    prev = _solid(16, 16, (0.0, 0.0, 0.0))
    cur = _solid(16, 16, (1.0, 1.0, 1.0))
    mem = MemoryManager().initialize(SceneGraph(frame_index=0))
    roi = RegionRef(region_id="p1:face", bbox=BBox(0.0, 0.0, 0.5, 0.5), reason="test")
    out = st.refine(prev, cur, mem, enabled=True, updated_regions=[roi], region_confidence=1.0)
    assert out[14][14] == [1.0, 1.0, 1.0]
    assert out[1][1] != [1.0, 1.0, 1.0]


def test_graph_cache_roundtrip_keeps_useful_structure(tmp_path: Path) -> None:
    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.6, 0.8), mask_ref=None)
    graph = SceneGraph(frame_index=5, persons=[person], objects=[])
    path = tmp_path / "cache.json"
    save_graph_cache([graph], str(path))
    loaded = load_graph_cache(str(path))
    assert loaded[0].frame_index == 5
    assert loaded[0].persons[0].person_id == "p1"
    assert loaded[0].persons[0].bbox.w > 0


def test_representation_from_perception_cache_uses_cached_facts(tmp_path: Path) -> None:
    cache = tmp_path / "perception_cache.json"
    cache.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "frame_index": 0,
                        "source": "cached_1",
                        "frame_size": [640, 480],
                        "persons": [
                            {
                                "bbox": {"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.7},
                                "bbox_confidence": 0.9,
                                "bbox_source": "cache",
                                "garments": [{"type": "coat", "state": "worn", "confidence": 0.8, "source": "cache"}],
                            }
                        ],
                        "objects": [],
                    }
                ]
            }
        )
    )
    ds = RepresentationDataset.from_perception_cache(str(cache))
    g = ds[0]["graphs"][0]
    assert g.persons and g.persons[0].bbox.w == 0.4


def test_patch_renderer_strategy_for_revealed_hidden_region() -> None:
    mm = MemoryManager()
    renderer = PatchRenderer()
    graph = SceneGraph(frame_index=0, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)], objects=[])
    mem = mm.initialize(graph)
    mem = mm.update_from_frame(mem, _solid(64, 64, (0.9, 0.1, 0.1)), graph)

    rid = make_region_id("p1", "garments")
    slot = mm.query_hidden_region(mem, rid)
    assert slot is not None and slot.candidate_patch_ids
    region = RegionRef(region_id=rid, bbox=BBox(0.2, 0.2, 0.4, 0.4), reason="garment_opening")
    delta = GraphDelta(newly_revealed_regions=[region], semantic_reasons=["garment_change"], affected_entities=["p1"], affected_regions=["garments"])
    patch = renderer.render(_solid(64, 64, (0.2, 0.2, 0.2)), graph, delta, mem, region)
    assert "strategy=known_hidden_reveal" in patch.debug_trace


def test_dynamics_dataset_from_transition_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "transitions.json"
    manifest.write_text(json.dumps({"records": [{"labels": ["sit_down", "smile"], "source": "anno_1"}]}))
    ds = DynamicsDataset.from_transition_manifest(str(manifest))
    sample = ds[0]
    assert sample["deltas"][0].affected_regions
    assert sample["deltas"][0].semantic_reasons
