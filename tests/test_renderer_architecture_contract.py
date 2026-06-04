from __future__ import annotations

import pytest

from core.pipeline_contract import ContractValidationError, validate_rendering_context
from core.schema import BBox, GraphDelta, GlobalSceneContext, RegionRef, SceneGraph, VideoMemory
from learned.interfaces import PatchSynthesisRequest
from rendering.learned_bridge import LegacyDeterministicPatchSynthesisModel


def _region() -> RegionRef:
    return RegionRef("p1:face", BBox(0.2, 0.2, 0.25, 0.25), "test_roi")


def _frame() -> list[list[list[float]]]:
    return [[[0.25, 0.25, 0.25] for _ in range(32)] for _ in range(32)]


def _delta(region: RegionRef) -> GraphDelta:
    return GraphDelta(
        affected_entities=["p1"],
        affected_regions=["face"],
        semantic_reasons=["expression_refine"],
        transition_diagnostics={
            "region_routing_plan": {
                region.region_id: {
                    "decision": "expression_refine",
                    "renderer_mode_hint": "refine",
                    "reveal_mode": "none",
                    "synthesis_required": False,
                    "memory_support_level": "none",
                }
            }
        },
    )


def _route_context(region: RegionRef) -> dict[str, object]:
    return {
        "region_id": region.region_id,
        "canonical_region_id": region.region_id,
        "canonical_region": "face",
        "decision": "expression_refine",
        "render_mode": "refine",
        "renderer_mode_hint": "refine",
        "reveal_mode": "none",
        "synthesis_required": False,
        "source_provenance": "parser",
        "material_provenance": "observed",
    }


def _request(*, with_route: bool) -> PatchSynthesisRequest:
    region = _region()
    delta = _delta(region)
    memory = VideoMemory()
    context: dict[str, object] = {"graph_delta": delta, "video_memory": memory}
    if with_route:
        context["region_route_decision"] = _route_context(region)
    return PatchSynthesisRequest(
        region=region,
        scene_state=SceneGraph(frame_index=0, global_context=GlobalSceneContext(frame_size=(32, 32))),
        memory_summary={},
        transition_context=context,
        retrieval_summary={},
        current_frame=_frame(),
        region_metadata={"region_id": region.region_id, "source_provenance": "parser"},
    )


def test_roi_rendering_without_routing_decision_fails() -> None:
    with pytest.raises(ContractValidationError):
        LegacyDeterministicPatchSynthesisModel().synthesize_patch(_request(with_route=False))


def test_roi_rendering_with_valid_routing_decision_passes() -> None:
    output = LegacyDeterministicPatchSynthesisModel().synthesize_patch(_request(with_route=True))
    assert output.region.region_id == "p1:face"
    assert output.execution_trace["region_route_decision"]["decision"] == "expression_refine"
    assert output.execution_trace["routing_region_id"] == "p1:face"


def test_rendering_cannot_invent_routing_context() -> None:
    with pytest.raises(ContractValidationError):
        validate_rendering_context(region_id="p1:face", transition_context={"region_route_decision": {"decision": "expression_refine"}})
