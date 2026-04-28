from __future__ import annotations

from core.region_ids import make_region_id
from core.routing_contracts import RUNTIME_ROUTING_DECISION_KINDS
from core.schema import BBox, BodyPartNode, GraphDelta, PersonNode, RegionRef, SceneGraph
from memory.video_memory import MemoryManager
from rendering.roi_renderer import ROISelector
from runtime.region_routing import CanonicalRegionRouter


def _payload(name: str, visibility: str = "visible", confidence: float = 0.9) -> dict[str, object]:
    return {
        "canonical_name": name,
        "raw_sources": [name],
        "source_regions": [name],
        "mask_ref": f"mask://{name}",
        "confidence": confidence,
        "visibility_state": visibility,
        "provenance": "canonical_reasoner",
        "attachment_hints": [],
        "ownership_hints": ["person"],
        "coverage_hints": [],
    }


def _scene() -> SceneGraph:
    canonical = {name: _payload(name, visibility="visible", confidence=0.85) for name in CanonicalRegionRouter._CANONICAL_REGIONS}
    canonical["outer_garment"] = _payload("outer_garment", visibility="partially_visible", confidence=0.7)
    person = PersonNode(
        person_id="p1",
        track_id="t1",
        bbox=BBox(0.1, 0.1, 0.7, 0.8),
        mask_ref="mask://person",
        body_parts=[
            BodyPartNode(part_id="p1_face", part_type="face", visibility="visible", confidence=0.9),
            BodyPartNode(part_id="p1_torso", part_type="torso", visibility="visible", confidence=0.87),
            BodyPartNode(part_id="p1_left_arm", part_type="left_arm", visibility="visible", confidence=0.82),
            BodyPartNode(part_id="p1_right_arm", part_type="right_arm", visibility="visible", confidence=0.81),
        ],
        canonical_regions=canonical,
        confidence=0.92,
    )
    return SceneGraph(frame_index=0, persons=[person], objects=[])


def test_reveal_from_memory_vs_synthesis_distinction() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())

    face_entry = mm.get_best_region_memory(memory, "p1", "face")
    assert face_entry is not None
    face_entry.suitable_for_reveal = True
    face_entry.reliable_for_reuse = True

    arm_entry = mm.get_best_region_memory(memory, "p1", "left_arm")
    assert arm_entry is not None
    arm_entry.suitable_for_reveal = False
    arm_entry.reliable_for_reuse = False

    delta = GraphDelta(
        affected_entities=["p1"],
        newly_revealed_regions=[
            RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="reveal"),
            RegionRef(region_id="p1:left_arm", bbox=BBox(0.1, 0.3, 0.2, 0.3), reason="reveal"),
        ],
    )

    plan = router.build_plan(scene_graph=graph, delta=delta, memory=memory, semantic_transition=None)
    face = next(d for d in plan.decisions if d.canonical_region == "face")
    arm = next(d for d in plan.decisions if d.canonical_region == "left_arm")

    assert face.decision == "reveal_from_memory"
    assert face.memory_support_level in {"strong", "medium"}
    assert face.reveal_mode == "from_memory"
    assert face.synthesis_required is False
    assert arm.decision == "reveal_partial_memory_assist"
    assert arm.memory_support_level in {"weak", "medium"}
    assert arm.synthesis_required is True


def test_unchanged_region_not_marked_for_synthesis_and_identity_bias_for_expression() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())

    delta = GraphDelta(affected_entities=["p1"], expression_deltas={"smile_intensity": 0.2})
    plan = router.build_plan(scene_graph=graph, delta=delta, memory=memory, semantic_transition=None)

    face = next(d for d in plan.decisions if d.canonical_region == "face")
    leg = next(d for d in plan.decisions if d.canonical_region == "left_leg")

    assert face.decision == "expression_refine"
    assert face.priority > 80
    assert leg.decision in {"direct_reuse", "temporal_stabilize", "local_deform_or_update"}
    assert not (leg.decision.startswith("reveal_") or leg.synthesis_required)


def test_garment_transition_biases_torso_and_arms() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())

    delta = GraphDelta(
        affected_entities=["p1"],
        garment_deltas={"coat_state": "opening"},
        semantic_reasons=["garment_opening"],
        affected_regions=["outer_garment", "torso", "left_arm"],
    )
    plan = router.build_plan(scene_graph=graph, delta=delta, memory=memory, semantic_transition=None)

    torso = next(d for d in plan.decisions if d.canonical_region == "torso")
    arm = next(d for d in plan.decisions if d.canonical_region == "left_arm")
    outer = next(d for d in plan.decisions if d.canonical_region == "outer_garment")

    assert torso.decision == "garment_transition_update"
    assert arm.decision == "garment_transition_update"
    assert outer.decision == "garment_transition_update"
    assert plan.render_regions


def test_runtime_integration_plan_guides_renderer_mode() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())

    delta = GraphDelta(
        affected_entities=["p1"],
        expression_deltas={"smile_intensity": 0.4},
        region_transition_mode={"face": "stable"},
    )
    plan = router.build_plan(scene_graph=graph, delta=delta, memory=memory, semantic_transition=None)
    delta.transition_diagnostics = {
        "region_routing_plan": {
            f"p1:{d.canonical_region}": {
                "decision": d.decision,
                "priority": d.priority,
                "reveal_mode": d.reveal_mode,
                "synthesis_required": d.synthesis_required,
                "renderer_mode_hint": d.renderer_mode_hint,
            }
            for d in plan.decisions
        }
    }

    from rendering.roi_renderer import PatchRenderer

    renderer = PatchRenderer()
    face_region = next(r for r in plan.render_regions if r.region_id == "p1:face")
    frame = [[[0.2, 0.25, 0.3] for _ in range(64)] for _ in range(64)]

    patch = renderer.render(frame, graph, delta, memory, face_region)

    assert patch.execution_trace["selection"]["selected_render_mode"] == "refine"
    assert any("runtime_route_decision=expression_refine" in reason for reason in patch.execution_trace["selection"]["mode_reasons"])
    assert delta.region_transition_mode["face"] == "stable"
    assert any("mode_source=runtime_plan_authoritative" in reason for reason in patch.execution_trace["selection"]["mode_reasons"])
    policy = patch.execution_trace["selection"]["execution_policy"]
    assert policy["runtime_authoritative"] is True
    assert policy["routing_input_status"] == "authoritative_runtime_plan"
    assert policy["decision_kind"] == "expression_refine"
    assert policy["execution_policy_kind"] == "identity_refine_oriented"
    assert patch.execution_trace["memory_dependency_summary"]["routing_input_status"] == "authoritative_runtime_plan"


def test_mode_router_falls_back_without_runtime_routing_plan() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    from rendering.roi_renderer import PatchRenderer

    renderer = PatchRenderer()
    delta = GraphDelta(affected_entities=["p1"], affected_regions=["left_arm"], region_transition_mode={"left_arm": "pose_deform"})
    region = RegionRef(region_id="p1:left_arm", bbox=BBox(0.2, 0.3, 0.2, 0.25), reason="pose")
    frame = [[[0.2, 0.25, 0.3] for _ in range(64)] for _ in range(64)]

    patch = renderer.render(frame, graph, delta, memory, region)

    assert patch.execution_trace["selection"]["selected_render_mode"] == "deform"
    assert any("mode_source=fallback_heuristic" in reason for reason in patch.execution_trace["selection"]["mode_reasons"])
    assert patch.execution_trace["selection"]["execution_policy"]["runtime_authoritative"] is False
    assert patch.execution_trace["selection"]["execution_policy"]["routing_input_status"] == "no_runtime_plan"


def test_decision_lookup_uses_canonical_region_parser_path() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())
    plan = router.build_plan(scene_graph=graph, delta=GraphDelta(affected_entities=["p1"]), memory=memory, semantic_transition=None)

    face_by_region_id = plan.decision_for_region_id(make_region_id("p1", "face"))
    assert face_by_region_id is not None
    assert face_by_region_id.canonical_region == "face"


def test_reveal_requires_synthesis_when_no_memory_support() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())
    memory.canonical_region_memory.pop("p1:right_hand", None)

    delta = GraphDelta(
        affected_entities=["p1"],
        newly_revealed_regions=[RegionRef(region_id="p1:right_hand", bbox=BBox(0.6, 0.5, 0.1, 0.2), reason="reveal")],
    )
    plan = router.build_plan(scene_graph=graph, delta=delta, memory=memory, semantic_transition=None)
    hand = next(d for d in plan.decisions if d.canonical_region == "right_hand")

    assert hand.decision == "reveal_requires_synthesis"
    assert hand.memory_support_level == "none"
    assert hand.synthesis_required is True


def test_reveal_execution_variants_are_distinct_in_trace() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())
    from rendering.roi_renderer import PatchRenderer

    face_entry = mm.get_best_region_memory(memory, "p1", "face")
    assert face_entry is not None
    face_entry.suitable_for_reveal = True
    face_entry.reliable_for_reuse = True

    arm_entry = mm.get_best_region_memory(memory, "p1", "left_arm")
    assert arm_entry is not None
    arm_entry.suitable_for_reveal = False
    arm_entry.reliable_for_reuse = False

    memory.canonical_region_memory.pop("p1:right_hand", None)
    delta = GraphDelta(
        affected_entities=["p1"],
        newly_revealed_regions=[
            RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="reveal"),
            RegionRef(region_id="p1:left_arm", bbox=BBox(0.2, 0.35, 0.2, 0.2), reason="reveal"),
            RegionRef(region_id="p1:right_hand", bbox=BBox(0.55, 0.55, 0.15, 0.15), reason="reveal"),
        ],
    )
    plan = router.build_plan(scene_graph=graph, delta=delta, memory=memory, semantic_transition=None)
    delta.transition_diagnostics = {
        "region_routing_plan": {
            f"p1:{d.canonical_region}": {
                "decision": d.decision,
                "priority": d.priority,
                "reveal_mode": d.reveal_mode,
                "synthesis_required": d.synthesis_required,
                "renderer_mode_hint": d.renderer_mode_hint,
                "memory_support_level": d.memory_support_level,
            }
            for d in plan.decisions
        }
    }
    renderer = PatchRenderer()
    frame = [[[0.2, 0.25, 0.3] for _ in range(64)] for _ in range(64)]

    variants = {}
    for region_id in ("p1:face", "p1:left_arm", "p1:right_hand"):
        region = next(r for r in plan.render_regions if r.region_id == region_id)
        patch = renderer.render(frame, graph, delta, memory, region)
        variants[region_id] = patch.execution_trace

    assert variants["p1:face"]["selection"]["execution_policy"]["decision_kind"] == "reveal_from_memory"
    assert variants["p1:left_arm"]["selection"]["execution_policy"]["decision_kind"] == "reveal_partial_memory_assist"
    assert variants["p1:right_hand"]["selection"]["execution_policy"]["decision_kind"] == "reveal_requires_synthesis"
    assert variants["p1:left_arm"]["selection"]["execution_policy"]["assist_oriented"] is True
    assert variants["p1:left_arm"]["selected_render_strategy"] == "PARTIAL_MEMORY_ASSIST_REVEAL"
    assert variants["p1:left_arm"]["selection"]["execution_policy"]["execution_strategy"] == variants["p1:left_arm"]["selected_render_strategy"]
    assert variants["p1:right_hand"]["memory_dependency_summary"]["runtime_synthesis_required"] is True


def test_direct_reuse_and_temporal_stabilize_have_distinct_execution_policy() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    from rendering.roi_renderer import PatchRenderer

    renderer = PatchRenderer()
    frame = [[[0.2, 0.25, 0.3] for _ in range(64)] for _ in range(64)]
    base_plan = {
        "p1:face": {"decision": "direct_reuse", "priority": 1, "reveal_mode": "none", "synthesis_required": False, "renderer_mode_hint": "keep", "memory_support_level": "strong"},
        "p1:left_arm": {"decision": "temporal_stabilize", "priority": 1, "reveal_mode": "none", "synthesis_required": False, "renderer_mode_hint": "keep", "memory_support_level": "medium"},
    }
    delta = GraphDelta(affected_entities=["p1"], region_transition_mode={"face": "expression_refine", "left_arm": "pose_deform"}, transition_diagnostics={"region_routing_plan": base_plan})

    face = renderer.render(frame, graph, delta, memory, RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="reuse"))
    arm = renderer.render(frame, graph, delta, memory, RegionRef(region_id="p1:left_arm", bbox=BBox(0.2, 0.3, 0.2, 0.2), reason="stabilize"))

    face_policy = face.execution_trace["selection"]["execution_policy"]
    arm_policy = arm.execution_trace["selection"]["execution_policy"]
    assert face_policy["decision_kind"] == "direct_reuse"
    assert arm_policy["decision_kind"] == "temporal_stabilize"
    assert face_policy["execution_policy_kind"] != arm_policy["execution_policy_kind"]
    assert face.execution_trace["selection"]["selected_render_mode"] == "keep"
    assert arm.execution_trace["selection"]["selected_render_mode"] == "keep"


def test_partial_runtime_plan_is_downgraded_and_marked_partial() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    from rendering.roi_renderer import PatchRenderer

    renderer = PatchRenderer()
    frame = [[[0.2, 0.25, 0.3] for _ in range(64)] for _ in range(64)]
    partial_plan = {
        "p1:face": {
            "decision": "expression_refine",
            "priority": 1,
            # intentionally omit reveal_mode/synthesis_required -> partial
            "renderer_mode_hint": "refine",
        }
    }
    delta = GraphDelta(affected_entities=["p1"], region_transition_mode={"face": "stable"}, transition_diagnostics={"region_routing_plan": partial_plan})
    patch = renderer.render(frame, graph, delta, memory, RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="expr"))

    policy = patch.execution_trace["selection"]["execution_policy"]
    assert policy["routing_input_status"] == "partial_runtime_plan"
    assert policy["runtime_authoritative"] is False
    assert patch.execution_trace["memory_dependency_summary"]["fallback_path"] == "runtime_plan_incomplete"
    assert any("routing_input_status=partial_runtime_plan" in r for r in patch.execution_trace["selection"]["mode_reasons"])
    assert patch.execution_trace["selection"]["execution_policy"]["execution_strategy"] == patch.execution_trace["selected_render_strategy"]


def test_typed_execution_policy_is_single_source_contract() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    from rendering.roi_renderer import PatchRenderer, RenderExecutionPolicy
    from utils_tensor import crop

    renderer = PatchRenderer()
    frame = [[[0.2, 0.25, 0.3] for _ in range(64)] for _ in range(64)]
    region = RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="expr")
    roi = crop(frame, 12, 12, 24, 24)
    delta = GraphDelta(
        affected_entities=["p1"],
        transition_diagnostics={
            "region_routing_plan": {
                "p1:face": {
                    "decision": "expression_refine",
                    "priority": 10,
                    "reveal_mode": "none",
                    "synthesis_required": False,
                    "renderer_mode_hint": "refine",
                    "memory_support_level": "strong",
                }
            }
        },
    )
    route_decision = renderer.mode_router.route(renderer, scene_graph=graph, delta=delta, memory=memory, region=region, roi=roi)

    assert isinstance(route_decision.execution_policy, RenderExecutionPolicy)
    assert route_decision.execution_policy.decision_kind == "expression_refine"
    assert route_decision.execution_policy.execution_policy_kind == "identity_refine_oriented"
    assert route_decision.execution_policy.execution_strategy == "EXISTING_REGION_UPDATE"


def test_single_canonical_runtime_strategy_field_and_planner_separation() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    from rendering.roi_renderer import PatchRenderer

    renderer = PatchRenderer()
    frame = [[[0.2, 0.25, 0.3] for _ in range(64)] for _ in range(64)]
    delta = GraphDelta(
        affected_entities=["p1"],
        transition_diagnostics={
            "region_routing_plan": {
                "p1:face": {
                    "decision": "expression_refine",
                    "priority": 10,
                    "reveal_mode": "none",
                    "synthesis_required": False,
                    "renderer_mode_hint": "refine",
                }
            }
        },
    )
    patch = renderer.render(frame, graph, delta, memory, RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="expr"))
    selection = patch.execution_trace["selection"]

    assert "selected_execution_strategy" not in selection
    assert "planner_selected_strategy" in selection
    assert "selected_render_strategy" in patch.execution_trace
    assert patch.execution_trace["selected_render_strategy"] == selection["execution_policy"]["execution_strategy"]


def test_producer_side_selected_render_strategy_contract_is_always_serialized() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    from rendering.roi_renderer import PatchRenderer

    renderer = PatchRenderer()
    frame = [[[0.2, 0.25, 0.3] for _ in range(64)] for _ in range(64)]
    delta = GraphDelta(
        affected_entities=["p1"],
        transition_diagnostics={
            "region_routing_plan": {
                "p1:left_arm": {
                    "decision": "reveal_partial_memory_assist",
                    "priority": 10,
                    "reveal_mode": "partial_memory_assist",
                    "synthesis_required": True,
                    "renderer_mode_hint": "reveal",
                    "memory_support_level": "weak",
                }
            }
        },
    )
    patch = renderer.render(frame, graph, delta, memory, RegionRef(region_id="p1:left_arm", bbox=BBox(0.2, 0.3, 0.2, 0.2), reason="reveal"))
    trace = patch.execution_trace

    assert "selected_render_strategy" in trace
    assert trace["selected_render_strategy"] == trace["selection"]["execution_policy"]["execution_strategy"]
    assert "selected_execution_strategy" not in trace
    assert "selected_execution_strategy" not in trace["selection"]
    assert "selected_strategy" not in trace["selection"]
    assert "planner_selected_strategy" in trace["selection"]


def test_router_and_renderer_share_decision_vocabulary_without_drift() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())
    plan = router.build_plan(scene_graph=graph, delta=GraphDelta(affected_entities=["p1"], expression_deltas={"smile_intensity": 0.2}), memory=memory, semantic_transition=None)

    for decision in plan.decisions:
        assert decision.decision in RUNTIME_ROUTING_DECISION_KINDS


def test_patch_eval_consumer_accepts_selected_render_strategy_as_canonical_field() -> None:
    import importlib.util
    from pathlib import Path

    contracts_path = Path(__file__).resolve().parents[1] / "src" / "evaluation" / "contracts.py"
    spec = importlib.util.spec_from_file_location("eval_contracts_light", contracts_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["eval_contracts_light"] = module
    spec.loader.exec_module(module)
    payload = module.build_patch_eval_payload({"selected_render_strategy": "PARTIAL_MEMORY_ASSIST_REVEAL", "hidden_lifecycle_state": {}})
    assert payload["patch_quality"] > 0.6


def test_garment_delta_does_not_overly_expand_without_exposure_reason() -> None:
    graph = _scene()
    mm = MemoryManager()
    memory = mm.initialize_from_scene(graph)
    router = CanonicalRegionRouter(mm, ROISelector())

    delta = GraphDelta(
        affected_entities=["p1"],
        garment_deltas={"coat_state": "worn_adjust"},
        semantic_reasons=["garment_adjustment"],
        affected_regions=["outer_garment"],
    )
    plan = router.build_plan(scene_graph=graph, delta=delta, memory=memory, semantic_transition=None)

    changed = {d.canonical_region: d.decision for d in plan.decisions}
    assert changed["outer_garment"] == "garment_transition_update"
    assert changed["torso"] != "garment_transition_update"
    assert changed["left_arm"] != "garment_transition_update"
