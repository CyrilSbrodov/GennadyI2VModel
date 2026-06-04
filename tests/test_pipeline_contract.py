from __future__ import annotations

import pytest

from core.pipeline_contract import (
    CANONICAL_STAGE_NAMES,
    ContractValidationError,
    PipelineStage,
    validate_runtime_trace,
    validate_stage_name,
    validate_transition,
)


def test_known_canonical_stages_pass() -> None:
    assert [validate_stage_name(stage) for stage in CANONICAL_STAGE_NAMES] == list(CANONICAL_STAGE_NAMES)


def test_unknown_stage_fails() -> None:
    with pytest.raises(ContractValidationError):
        validate_stage_name("runtime_renderer_shortcut")


def test_valid_canonical_order_passes() -> None:
    assert validate_runtime_trace([{"stage": stage} for stage in CANONICAL_STAGE_NAMES]) == list(CANONICAL_STAGE_NAMES)


def test_invalid_order_fails() -> None:
    invalid = list(CANONICAL_STAGE_NAMES)
    routing_idx = invalid.index(PipelineStage.REGION_ROUTING.value)
    rendering_idx = invalid.index(PipelineStage.RENDERING.value)
    invalid[routing_idx], invalid[rendering_idx] = invalid[rendering_idx], invalid[routing_idx]
    with pytest.raises(ContractValidationError):
        validate_runtime_trace(invalid)


def test_invalid_transition_fails() -> None:
    with pytest.raises(ContractValidationError):
        validate_transition("input", "rendering")


def test_missing_mandatory_stage_fails() -> None:
    trace = [stage for stage in CANONICAL_STAGE_NAMES if stage != "memory"]
    with pytest.raises(ContractValidationError):
        validate_runtime_trace(trace)
