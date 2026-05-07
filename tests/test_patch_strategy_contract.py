from __future__ import annotations

from core.schema import BBox, PersonNode, RegionRef, SceneGraph
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from learned.parity import build_parity_result, patch_io_to_contract
from runtime.orchestrator import PATCH_PARITY_REQUIRED_FIELDS
from training.learned_contracts import build_patch_synthesis_contract


PATCH_REQUIRED_FIELDS = ["roi_before", "roi_after", "region_metadata", "selected_render_strategy", "transition_context"]


def _region() -> RegionRef:
    return RegionRef(region_id="p1:face", bbox=BBox(0.1, 0.1, 0.4, 0.4), reason="test")


def _scene() -> SceneGraph:
    return SceneGraph(frame_index=0, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.0, 0.0, 1.0, 1.0), mask_ref=None)])


def _frame(value: float = 0.1) -> list[list[list[float]]]:
    return [[[value, value, value] for _ in range(2)] for _ in range(2)]


def _request() -> PatchSynthesisRequest:
    return PatchSynthesisRequest(
        region=_region(),
        scene_state=_scene(),
        memory_summary={},
        transition_context={"stage": "patch"},
        retrieval_summary={"source": "test"},
        current_frame=_frame(0.2),
        memory_channels={},
    )


def _output(execution_trace: dict[str, object]) -> PatchSynthesisOutput:
    return PatchSynthesisOutput(
        region=_region(),
        rgb_patch=_frame(0.8),
        alpha_mask=[[1.0, 1.0], [1.0, 1.0]],
        height=2,
        width=2,
        channels=3,
        confidence=0.9,
        execution_trace=execution_trace,
    )


def test_patch_contract_uses_selected_render_strategy_as_required_field() -> None:
    contract = patch_io_to_contract(_request(), _output({"selected_render_strategy": "LEARNED_PRIMARY"}))

    assert "selected_render_strategy" in contract
    assert contract["selected_render_strategy"] == "LEARNED_PRIMARY"


def test_patch_parity_requires_selected_render_strategy_not_legacy_selected_strategy() -> None:
    contract = patch_io_to_contract(_request(), _output({"selected_render_strategy": "LEARNED_PRIMARY"}))
    parity = build_parity_result(
        contract=contract,
        required_fields=PATCH_REQUIRED_FIELDS,
        stage="patch",
        request=_request(),
        output=_output({"selected_render_strategy": "LEARNED_PRIMARY"}),
    )

    assert parity["missing_fields"] == []
    assert "selected_strategy" not in PATCH_REQUIRED_FIELDS


def test_patch_contract_does_not_require_selected_execution_strategy() -> None:
    request = _request()
    output = _output({"selected_render_strategy": "LEARNED_PRIMARY"})
    contract = patch_io_to_contract(request, output)

    assert "selected_execution_strategy" not in contract
    parity = build_parity_result(
        contract=contract,
        required_fields=PATCH_REQUIRED_FIELDS,
        stage="patch",
        request=request,
        output=output,
    )

    assert parity["missing_fields"] == []


def test_patch_contract_legacy_selected_strategy_is_alias_only() -> None:
    request = _request()
    output = _output({"selected_strategy": "LEGACY_ONLY"})
    contract = patch_io_to_contract(request, output)

    assert contract["selected_render_strategy"] == "unknown"
    assert contract.get("selected_strategy") == "unknown"
    parity = build_parity_result(
        contract=contract,
        required_fields=PATCH_REQUIRED_FIELDS,
        stage="patch",
        request=request,
        output=output,
    )

    assert "selected_strategy" not in PATCH_REQUIRED_FIELDS
    assert "missing_field:selected_strategy" not in parity["errors"]
    assert parity["missing_fields"] == []


def test_build_patch_synthesis_contract_parameter_uses_selected_render_strategy_name() -> None:
    contract = build_patch_synthesis_contract(
        roi_before=_frame(0.2),
        roi_after=_frame(0.8),
        region=_region(),
        retrieval_summary="test",
        selected_render_strategy="LEARNED_PRIMARY",
        hidden_state={"visible": True},
        synthesis_mode="learned_primary",
        transition_context={"stage": "patch"},
    )

    assert contract["selected_render_strategy"] == "LEARNED_PRIMARY"
    assert contract.get("selected_strategy") == "LEARNED_PRIMARY"
    assert "selected_execution_strategy" not in contract


def test_runtime_patch_parity_required_fields_are_canonical() -> None:
    assert "selected_render_strategy" in PATCH_PARITY_REQUIRED_FIELDS
    assert "selected_strategy" not in PATCH_PARITY_REQUIRED_FIELDS
    assert "selected_execution_strategy" not in PATCH_PARITY_REQUIRED_FIELDS
