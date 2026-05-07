from __future__ import annotations

import numpy as np

from core.schema import BBox, RegionRef, SceneGraph
from evaluation.contracts import build_patch_eval_payload
from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest
from runtime.orchestrator import PATCH_PARITY_REQUIRED_FIELDS
from training.learned_contracts import build_patch_synthesis_contract
from training.renderer_manifest_exporter import RendererManifestRecordExporter


def _region() -> RegionRef:
    return RegionRef(region_id="person_0:left_arm", bbox=BBox(0.0, 0.0, 1.0, 1.0), reason="audit")


def _frame(value: float = 0.0) -> list[list[list[float]]]:
    return np.full((2, 2, 3), value, dtype=np.float32).tolist()


def _request(*, transition_context: dict[str, object] | None = None) -> PatchSynthesisRequest:
    return PatchSynthesisRequest(
        region=_region(),
        scene_state=SceneGraph(frame_index=0),
        memory_summary={},
        transition_context=transition_context or {},
        retrieval_summary={"source": "audit"},
        current_frame=_frame(0.1),
        memory_channels={},
    )


def _output(trace: dict[str, object]) -> PatchSynthesisOutput:
    return PatchSynthesisOutput(
        region=_region(),
        rgb_patch=_frame(0.8),
        alpha_mask=np.ones((2, 2), dtype=np.float32).tolist(),
        height=2,
        width=2,
        channels=3,
        confidence=0.9,
        execution_trace=trace,
    )


def test_no_runtime_patch_parity_requires_selected_strategy() -> None:
    assert "selected_render_strategy" in PATCH_PARITY_REQUIRED_FIELDS
    assert "selected_strategy" not in PATCH_PARITY_REQUIRED_FIELDS
    assert "selected_execution_strategy" not in PATCH_PARITY_REQUIRED_FIELDS


def test_patch_contract_alias_is_not_required() -> None:
    contract = build_patch_synthesis_contract(
        roi_before=_frame(0.1),
        roi_after=_frame(0.8),
        region=_region(),
        retrieval_summary="audit",
        selected_render_strategy="X",
        hidden_state={},
        synthesis_mode="audit_mode",
        transition_context={},
    )

    assert contract["selected_render_strategy"] == "X"
    assert contract["selected_strategy"] == "X"
    assert "selected_strategy" not in PATCH_PARITY_REQUIRED_FIELDS


def test_exporter_ignores_legacy_selected_execution_strategy() -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output({"selected_execution_strategy": "LEGACY_EXECUTION_ONLY"}),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert record["selected_render_strategy"] == "unknown"
    assert "selected_execution_strategy" not in record
    assert "selected_execution_strategy" not in record["patch_synthesis_contract"]
    assert "selected_execution_strategy" not in record["patch_output_contract"]


def test_planner_strategy_not_used_as_renderer_strategy() -> None:
    record = RendererManifestRecordExporter().build_record(
        request=_request(),
        output=_output({"planner_selected_strategy": "PLANNER_ROUTE_ONLY"}),
        roi_before=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert record["selected_render_strategy"] == "unknown"
    assert record["planner_selected_strategy"] == "PLANNER_ROUTE_ONLY"


def test_selected_render_strategy_wins_over_selected_strategy() -> None:
    canonical_payload = build_patch_eval_payload(
        {"selected_render_strategy": "CANONICAL", "selected_strategy": "LEGACY", "hidden_lifecycle_state": {}}
    )
    canonical_only_payload = build_patch_eval_payload(
        {"selected_render_strategy": "CANONICAL", "hidden_lifecycle_state": {}}
    )

    assert canonical_payload == canonical_only_payload


def test_selected_render_strategy_unknown_not_upgraded_by_legacy_selected_strategy() -> None:
    payload = build_patch_eval_payload(
        {"selected_render_strategy": "unknown", "selected_strategy": "LEGACY", "hidden_lifecycle_state": {}}
    )
    canonical_unknown_payload = build_patch_eval_payload(
        {"selected_render_strategy": "unknown", "hidden_lifecycle_state": {}}
    )

    assert payload == canonical_unknown_payload


def test_legacy_selected_strategy_fallback_is_compatibility_only() -> None:
    legacy_payload = build_patch_eval_payload({"selected_strategy": "LEGACY", "hidden_lifecycle_state": {}})
    legacy_equivalent_payload = build_patch_eval_payload(
        {"selected_render_strategy": "LEGACY", "hidden_lifecycle_state": {}}
    )

    # This covers old payload compatibility only; new/canonical payloads must use selected_render_strategy.
    assert legacy_payload == legacy_equivalent_payload
