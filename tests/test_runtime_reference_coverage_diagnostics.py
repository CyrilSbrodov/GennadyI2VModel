from __future__ import annotations

from runtime.orchestrator import GennadyEngine


def _patch(region_id: str, **trace: object) -> dict[str, object]:
    return {"region_id": region_id, "execution_trace": trace}


def test_expected_family_classification() -> None:
    assert GennadyEngine._reference_family_expected_for_region("p1:face") == "identity"
    assert GennadyEngine._reference_family_expected_for_region("p1:head") == "identity"
    assert GennadyEngine._reference_family_expected_for_region("p1:hair") == "identity"
    assert GennadyEngine._reference_family_expected_for_region("p1:neck") == "skin"
    assert GennadyEngine._reference_family_expected_for_region("p1:left_hand") == "skin"
    assert GennadyEngine._reference_family_expected_for_region("p1:torso") == "body_shape"
    assert GennadyEngine._reference_family_expected_for_region("p1:left_arm") == "body_shape"
    assert GennadyEngine._reference_family_expected_for_region("p1:outer_garment") == "garment"
    assert GennadyEngine._reference_family_expected_for_region("p1:accessories") == "accessory"
    assert GennadyEngine._reference_family_expected_for_region("invalid") == "unknown"


def test_step_coverage_counts_strong_identity_reference() -> None:
    summary = GennadyEngine._summarize_step_reference_coverage(
        [
            _patch(
                "p1:face",
                identity_reference_used=True,
                identity_reference_strength=1.0,
                memory_bundle_present=True,
                memory_support_level="strong",
            )
        ]
    )

    assert summary["expected_family_counts"]["identity"] == 1
    assert summary["used_counts"]["identity"] == 1
    assert summary["strong_counts"]["identity"] == 1
    assert summary["missing_counts"]["identity"] == 0
    assert summary["blocked_counts"]["identity"] == 0
    assert summary["critical_warnings"] == []


def test_face_patch_without_identity_reference_creates_warning() -> None:
    summary = GennadyEngine._summarize_step_reference_coverage(
        [
            _patch(
                "p1:face",
                identity_reference_used=False,
                identity_reference_strength=0.0,
                identity_reference_blocked=False,
                memory_bundle_present=True,
            )
        ]
    )

    assert summary["missing_counts"]["identity"] == 1
    assert "identity_region_without_identity_reference:p1:face" in summary["critical_warnings"]


def test_blocked_identity_is_counted_separately_from_missing() -> None:
    summary = GennadyEngine._summarize_step_reference_coverage(
        [
            _patch(
                "p1:face",
                identity_reference_used=False,
                identity_reference_strength=0.0,
                identity_reference_blocked=True,
                identity_reference_block_reasons=["identity_reference_blocked_generated"],
                memory_bundle_present=True,
            )
        ]
    )

    assert summary["blocked_counts"]["identity"] == 1
    assert summary["missing_counts"]["identity"] == 0
    assert "identity_region_reference_blocked:p1:face" in summary["critical_warnings"]


def test_garment_reference_coverage() -> None:
    summary = GennadyEngine._summarize_step_reference_coverage(
        [_patch("p1:outer_garment", garment_reference_used=True, garment_reference_strength=1.0)]
    )

    assert summary["expected_family_counts"]["garment"] == 1
    assert summary["used_counts"]["garment"] == 1
    assert summary["strong_counts"]["garment"] == 1


def test_run_level_aggregation_ratios() -> None:
    step_1 = GennadyEngine._summarize_step_reference_coverage(
        [_patch("p1:face", identity_reference_used=True, identity_reference_strength=1.0)]
    )
    step_2 = GennadyEngine._summarize_step_reference_coverage(
        [
            _patch(
                "p1:face",
                identity_reference_used=False,
                identity_reference_strength=0.0,
                identity_reference_blocked=False,
            ),
            _patch("p1:outer_garment", garment_reference_used=True, garment_reference_strength=1.0),
        ]
    )

    aggregate = GennadyEngine._aggregate_reference_coverage([step_1, step_2])

    assert aggregate["total_patch_count"] == 3
    assert aggregate["identity_reference_coverage_ratio"] == 0.5
    assert aggregate["garment_reference_coverage_ratio"] == 1.0
    assert aggregate["critical_warning_count"] == 1
