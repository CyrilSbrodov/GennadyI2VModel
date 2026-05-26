from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from core.input_layer import InputAssetLayer
from core.schema import BBox, GraphDelta, ReferencePatchPayload, RegionMemoryBundle, RegionRef, SceneGraph
from dynamics.state_update import apply_delta
from learned.factory import BackendBundle, BackendConfig, LearnedBackendFactory
from learned.interfaces import DynamicsTransitionRequest, PatchSynthesisRequest, TemporalRefinementRequest
from learned.parity import (
    build_parity_result,
    text_output_to_contract,
    dynamics_io_to_contract,
    patch_io_to_contract,
    temporal_io_to_contract,
)
from memory.summaries import AppearanceMemorySummarizer
from memory.video_memory import MemoryManager
from perception.pipeline import PerceptionBackendsConfig, PerceptionPipeline
from planning.transition_engine import StatePlan, TransitionPlanner
from rendering.compositor import Compositor
from rendering.roi_renderer import ROISelector, RenderedPatch
from representation.graph_builder import SceneGraphBuilder
from representation.learned_bridge import summarize_memory
from runtime.profiles import PROFILES, RuntimeProfile
from runtime.region_metadata import build_region_metadata
from runtime.region_routing import CanonicalRegionRouter
from runtime.i2v_frame_planner import I2VFramePlanEntry, plan_i2v_frames
from text.intent_parser import IntentParser
from utils_tensor import shape, zeros
from core.region_ids import make_region_id, parse_region_id
from core.reference_families import reference_family_for_region, reference_kind_for_region


def _payload_to_dict(payload: ReferencePatchPayload | None) -> dict[str, object] | None:
    return asdict(payload) if payload is not None else None


def _region_type_from_region_id(region_id: str) -> str:
    try:
        _, region_type = parse_region_id(region_id)
    except ValueError:
        region_type = region_id
    return str(region_type or "").strip()


def _expected_reference_payload_for_region(region_id: str, region_memory_bundle: RegionMemoryBundle) -> ReferencePatchPayload | None:
    region_type = _region_type_from_region_id(region_id)
    family = reference_family_for_region(region_type)
    if family == "identity":
        return region_memory_bundle.identity_reference_payload
    if family == "skin":
        return region_memory_bundle.skin_reference_payload
    if family == "body_shape":
        return region_memory_bundle.body_shape_reference_payload
    if family == "garment":
        return region_memory_bundle.garment_reference_payload
    if family == "accessory":
        return region_memory_bundle.accessory_reference_payload
    return region_memory_bundle.appearance_reference_payload


def _payload_trusted_for_runtime_selection(payload: ReferencePatchPayload) -> bool:
    min_confidence, min_evidence = (0.65, 0.70) if payload.reference_kind == "identity_reference" else (0.58, 0.58)
    return bool(
        payload.observed_directly
        and not payload.generated
        and not payload.inferred
        and payload.confidence >= min_confidence
        and payload.evidence_score >= min_evidence
    )


def _trusted_matching_payloads(region_id: str, region_memory_bundle: RegionMemoryBundle) -> list[ReferencePatchPayload]:
    region_type = _region_type_from_region_id(region_id)
    expected_kind = reference_kind_for_region(region_type)
    matches: list[ReferencePatchPayload] = []
    for payload in getattr(region_memory_bundle, "reference_payloads", []):
        if not isinstance(payload, ReferencePatchPayload):
            continue
        same_region = payload.region_id == region_id or (payload.entity_id == region_memory_bundle.entity_id and payload.canonical_region == region_type)
        if (
            same_region
            and payload.reference_kind == expected_kind
            and payload.canonical_region == region_type
            and _payload_trusted_for_runtime_selection(payload)
        ):
            matches.append(payload)
    return matches


def _serialize_reference_payload_context(region_id: str, region_memory_bundle: RegionMemoryBundle) -> dict[str, object]:
    payloads = [_payload_to_dict(payload) for payload in getattr(region_memory_bundle, "reference_payloads", [])]
    payload_dicts = [payload for payload in payloads if payload is not None]
    expected_payload = _expected_reference_payload_for_region(region_id, region_memory_bundle)
    reasons: list[str] = []
    if expected_payload is None:
        matches = _trusted_matching_payloads(region_id, region_memory_bundle)
        if len(matches) == 1:
            expected_payload = matches[0]
            reasons.append("expected_reference_payload_resolved_from_matching_payload")
    expected_payload_dict = _payload_to_dict(expected_payload)
    if expected_payload_dict is None:
        reasons.append("expected_reference_payload_missing")
    return {
        "reference_patch_payloads": payload_dicts,
        "expected_reference_payload": expected_payload_dict,
        "reference_payload_trace_reasons": reasons,
    }


PATCH_PARITY_REQUIRED_FIELDS = ["roi_before", "roi_after", "region_metadata", "selected_render_strategy", "transition_context"]


@dataclass(slots=True)
class InferenceArtifacts:
    frames: list[list]
    scene_graphs: list[SceneGraph]
    state_plan: StatePlan
    debug: dict[str, list[str] | str | dict[str, object] | list[dict[str, object]]] = field(default_factory=dict)


class GennadyEngine:
    def __init__(self, backend_config: BackendConfig | None = None, backend_bundle: BackendBundle | None = None, perception_config: PerceptionBackendsConfig | None = None) -> None:
        self.input_layer = InputAssetLayer()
        self.perception = PerceptionPipeline(backends=perception_config)
        self.graph_builder = SceneGraphBuilder()
        self.memory_manager = MemoryManager()
        self.intent_parser = IntentParser()
        self.planner = TransitionPlanner()
        self.roi_selector = ROISelector()
        self.region_router = CanonicalRegionRouter(self.memory_manager, self.roi_selector)
        self.compositor = Compositor()
        self.memory_summarizer = AppearanceMemorySummarizer()
        self.backend_config = backend_config or BackendConfig()
        self.backends = backend_bundle or LearnedBackendFactory(self.backend_config).build()

    @staticmethod
    def _frame_plan_entry_to_dict(entry: I2VFramePlanEntry | None) -> dict[str, object]:
        return asdict(entry) if entry is not None else {}

    @staticmethod
    def _resolve_planned_region(scene_graph: SceneGraph, roi_selector: ROISelector, region_id: str) -> RegionRef | None:
        if not isinstance(region_id, str) or ":" not in region_id:
            return None
        try:
            entity_id, region_type = parse_region_id(region_id)
        except ValueError:
            return None
        semantic_resolver = getattr(roi_selector, "semantic_roi_from_graph", None)
        if callable(semantic_resolver):
            resolved = semantic_resolver(scene_graph, entity_id, region_type)
            if resolved is not None:
                return resolved
        fallback_resolver = getattr(roi_selector, "fallback_roi_from_person_bbox", None)
        if callable(fallback_resolver):
            resolved = fallback_resolver(scene_graph, entity_id, region_type)
            if resolved is not None:
                return resolved
        person = next((p for p in scene_graph.persons if p.person_id == entity_id), None)
        if person is None:
            return None
        x, y, w, h = person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h
        layout = {
            "face": (x + 0.30 * w, y + 0.05 * h, 0.40 * w, 0.24 * h),
            "hair": (x + 0.24 * w, y + 0.00 * h, 0.52 * w, 0.16 * h),
            "torso": (x + 0.20 * w, y + 0.30 * h, 0.60 * w, 0.48 * h),
            "left_arm": (x + 0.00 * w, y + 0.30 * h, 0.22 * w, 0.52 * h),
            "right_arm": (x + 0.78 * w, y + 0.30 * h, 0.22 * w, 0.52 * h),
            "upper_clothes": (x + 0.20 * w, y + 0.30 * h, 0.60 * w, 0.32 * h),
        }
        if region_type not in layout:
            return None
        rx, ry, rw, rh = layout[region_type]
        return RegionRef(region_id=region_id, bbox=BBox(max(0.0, rx), max(0.0, ry), max(0.05, min(1.0 - rx, rw)), max(0.05, min(1.0 - ry, rh))), reason="frame_plan_debug_layout_fallback")

    @staticmethod
    def _apply_frame_plan_to_delta(delta: GraphDelta, frame_plan_entry: I2VFramePlanEntry | None) -> tuple[bool, list[str]]:
        if frame_plan_entry is None:
            return False, []
        applied: list[str] = []
        if not isinstance(delta.region_transition_mode, dict):
            delta.region_transition_mode = {}
        for region_id, mode in frame_plan_entry.region_transition_mode.items():
            if not isinstance(region_id, str) or not isinstance(mode, str):
                continue
            try:
                _, region_type = parse_region_id(region_id)
            except ValueError:
                region_type = region_id
            if not delta.region_transition_mode.get(region_type):
                delta.region_transition_mode[region_type] = mode
                applied.append(region_type)
            if not delta.region_transition_mode.get(region_id):
                delta.region_transition_mode[region_id] = mode
                applied.append(region_id)
        return bool(applied), applied

    @staticmethod
    def _normalize_frame_tensor(frame: object, *, field_name: str) -> list[list[list[float]]]:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"{field_name} must be HxWx3 tensor-like, got shape={list(arr.shape)}")
        if arr.size == 0:
            raise ValueError(f"{field_name} must not be empty")
        max_val = float(np.max(arr))
        if max_val > 1.0:
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        else:
            arr = np.clip(arr, 0.0, 1.0)
        return arr.tolist()

    @staticmethod
    def _validate_patch_output_contract(patch_out: object, *, expected_region_id: str) -> dict[str, object]:
        issues: list[str] = []
        path = str(getattr(patch_out, "execution_trace", {}).get("renderer_path", "unknown"))
        region = getattr(getattr(patch_out, "region", None), "region_id", "")
        if region != expected_region_id:
            issues.append(f"region_mismatch:{region}->{expected_region_id}")
        rgb = np.asarray(getattr(patch_out, "rgb_patch", None), dtype=np.float32)
        alpha = np.asarray(getattr(patch_out, "alpha_mask", None), dtype=np.float32)
        uncertainty = getattr(patch_out, "uncertainty_map", None)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            issues.append(f"rgb_patch_invalid_shape:{list(rgb.shape)}")
        if alpha.ndim != 2:
            issues.append(f"alpha_mask_invalid_shape:{list(alpha.shape)}")
        if rgb.ndim == 3 and alpha.ndim == 2 and rgb.shape[:2] != alpha.shape:
            issues.append(f"rgb_alpha_shape_mismatch:{list(rgb.shape)} vs {list(alpha.shape)}")
        if isinstance(uncertainty, list):
            unc = np.asarray(uncertainty, dtype=np.float32)
            if unc.ndim != 2 or (rgb.ndim == 3 and unc.shape != rgb.shape[:2]):
                issues.append(f"uncertainty_shape_mismatch:{list(unc.shape)}")
        return {"issues": issues, "renderer_path": path, "is_learned_primary": path == "learned_primary"}

    @staticmethod
    def _validate_temporal_output_contract(temporal_out: object, *, expected_shape: tuple[int, int, int]) -> dict[str, object]:
        issues: list[str] = []
        refined = np.asarray(getattr(temporal_out, "refined_frame", None), dtype=np.float32)
        if refined.ndim != 3 or refined.shape[2] != 3:
            issues.append(f"refined_frame_invalid_shape:{list(refined.shape)}")
        elif tuple(refined.shape) != expected_shape:
            issues.append(f"refined_frame_shape_mismatch:{list(refined.shape)} vs {list(expected_shape)}")
        meta = getattr(temporal_out, "metadata", {}) if isinstance(getattr(temporal_out, "metadata", {}), dict) else {}
        temporal_path = str(meta.get("temporal_path", "unknown"))
        return {"issues": issues, "temporal_path": temporal_path, "is_learned_primary": temporal_path == "learned_primary"}

    @staticmethod
    def _memory_update_context_for_generated_frame(
        delta: GraphDelta,
        *,
        temporal_refinement_enabled: bool,
    ) -> dict[str, object]:
        """Build provenance-aware context for memory updates from runtime outputs.

        The stable frame at this point is renderer/compositor output, optionally
        passed through temporal refinement, not an externally observed input
        frame. Marking this explicitly lets MemoryManager keep identity-sensitive
        observed references from being refreshed by generated pixels while still
        preserving the transition phase semantics used by memory scoring.
        """

        update_source = (
            "renderer_temporal_refined_generated_output"
            if temporal_refinement_enabled
            else "renderer_composited_generated_output"
        )
        context: dict[str, object] = {
            "transition_phase": delta.transition_phase,
            "visibility_phase": delta.state_after.get("visibility_phase", "stable"),
            "garment_phase": delta.state_after.get("garment_phase", "worn"),
            "pose_phase": delta.state_after.get("pose_phase", "stable"),
            "generated": True,
            "is_generated": True,
            "frame_source": "runtime_generated_stable_frame",
            "update_source": update_source,
            "source_frame_kind": "generated_runtime_frame",
            "source_is_input_frame": False,
            "immutable_i2v_anchor": False,
        }
        if delta.affected_regions:
            primary_region = delta.affected_regions[0]
            context["region_transition_mode"] = delta.region_transition_mode.get(primary_region, "stable")
        return context

    @staticmethod
    def build_dynamics_memory_channels(memory_channels: dict[str, object]) -> dict[str, object]:
        keep = ("identity", "garments", "hidden_regions", "body_regions")
        return {k: memory_channels.get(k, {}) for k in keep}

    @staticmethod
    def build_patch_memory_channels(memory_channels: dict[str, object]) -> dict[str, object]:
        keep = ("identity", "garments", "hidden_regions")
        return {k: memory_channels.get(k, {}) for k in keep}

    @staticmethod
    def build_temporal_memory_channels(memory_channels: dict[str, object]) -> dict[str, object]:
        keep = ("identity", "body_regions", "hidden_regions")
        return {k: memory_channels.get(k, {}) for k in keep}

    @staticmethod
    def _reference_family_expected_for_region(region_id: str) -> str:
        if not isinstance(region_id, str) or ":" not in region_id:
            return "unknown"
        region_type = _region_type_from_region_id(region_id)
        if not region_type or region_type == "unknown":
            return "unknown"
        return reference_family_for_region(region_type)

    @staticmethod
    def _extract_patch_reference_usage(patch_debug: dict[str, object]) -> dict[str, object]:
        patch_output = patch_debug.get("patch_output") if isinstance(patch_debug, dict) else None
        trace = patch_debug.get("execution_trace", {}) if isinstance(patch_debug, dict) else {}
        if (not isinstance(trace, dict) or not trace) and patch_output is not None:
            trace = getattr(patch_output, "execution_trace", {})
        if not isinstance(trace, dict):
            trace = {}

        def _get_value(key: str, default: object = None) -> object:
            if key in trace:
                return trace.get(key, default)
            return patch_debug.get(key, default)

        def _as_bool(value: object) -> bool:
            return bool(value) if value is not None else False

        def _as_float(value: object) -> float:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        def _as_reasons(value: object) -> list[str]:
            if isinstance(value, list):
                return [str(v) for v in value]
            if isinstance(value, tuple):
                return [str(v) for v in value]
            if value:
                return [str(value)]
            return []

        region_id = str(patch_debug.get("region_id") or trace.get("region_id") or "")
        if not region_id and patch_output is not None:
            region = getattr(patch_output, "region", None)
            region_id = str(getattr(region, "region_id", "") or "")
        if ":" in region_id:
            try:
                _, region_type = parse_region_id(region_id)
            except ValueError:
                region_type = "unknown"
        else:
            region_type = "unknown"
        expected_family = GennadyEngine._reference_family_expected_for_region(region_id)

        usage: dict[str, object] = {
            "region_id": region_id,
            "region_type": region_type,
            "expected_reference_family": expected_family,
            "memory_bundle_present": _as_bool(_get_value("memory_bundle_present", False)),
            "memory_support_level": str(_get_value("memory_support_level", "unknown") or "unknown"),
        }
        for family in ("identity", "skin", "body_shape", "garment", "accessory"):
            prefix = f"{family}_reference"
            usage[f"{prefix}_used"] = _as_bool(_get_value(f"{prefix}_used", False))
            usage[f"{prefix}_strength"] = _as_float(_get_value(f"{prefix}_strength", 0.0))
            usage[f"{prefix}_blocked"] = _as_bool(_get_value(f"{prefix}_blocked", False))
            usage[f"{prefix}_source"] = str(_get_value(f"{prefix}_source", "unknown") or "unknown")
            usage[f"{prefix}_block_reasons"] = _as_reasons(_get_value(f"{prefix}_block_reasons", []))

        expected_prefix = f"{expected_family}_reference" if expected_family != "unknown" else ""
        expected_used = bool(usage.get(f"{expected_prefix}_used", False)) if expected_prefix else False
        expected_strength = float(usage.get(f"{expected_prefix}_strength", 0.0) or 0.0) if expected_prefix else 0.0
        expected_blocked = bool(usage.get(f"{expected_prefix}_blocked", False)) if expected_prefix else False
        rendered = _as_bool(patch_debug.get("rendered", True))
        usage["expected_reference_used"] = expected_used
        usage["expected_reference_strength"] = expected_strength
        usage["expected_reference_blocked"] = expected_family != "unknown" and expected_blocked
        usage["expected_reference_missing"] = expected_family != "unknown" and rendered and not expected_used and expected_strength == 0.0 and not expected_blocked
        material_used = _as_bool(_get_value("reference_patch_material_used", False))
        material_present = _as_bool(_get_value("reference_patch_material_present", False))
        material_trusted = _as_bool(_get_value("reference_patch_material_trusted", False))
        material_reason = str(_get_value("reference_patch_material_missing_reason", "") or "")
        material_from_input = _as_bool(_get_value("reference_material_from_input_frame", False))
        material_from_generated = _as_bool(_get_value("reference_material_from_generated_frame", False))
        usage["expected_material_used"] = expected_family != "unknown" and material_used
        usage["expected_input_frame_material_used"] = expected_family != "unknown" and material_used and material_from_input
        usage["reference_material_from_generated_frame"] = material_from_generated
        usage["reference_patch_material_present"] = material_present
        usage["reference_patch_material_trusted"] = material_trusted
        usage["reference_patch_material_missing_reason"] = material_reason
        usage["expected_material_missing"] = expected_family != "unknown" and rendered and not material_used
        return usage

    @staticmethod
    def _summarize_step_reference_coverage(patch_step_debug: list[dict[str, object]]) -> dict[str, object]:
        families = ("identity", "skin", "body_shape", "garment", "accessory")
        expected_families = (*families, "unknown")
        expected_family_counts = {family: 0 for family in expected_families}
        used_counts = {family: 0 for family in families}
        blocked_counts = {family: 0 for family in families}
        missing_counts = {family: 0 for family in families}
        strong_counts = {family: 0 for family in families}
        medium_counts = {family: 0 for family in families}
        material_used_counts = {family: 0 for family in families}
        input_frame_material_used_counts = {family: 0 for family in families}
        generated_material_rejected_counts = {family: 0 for family in families}
        non_input_material_rejected_counts = {family: 0 for family in families}
        missing_i2v_anchor_counts = {family: 0 for family in families}
        material_missing_counts = {family: 0 for family in families}
        material_missing_reasons: dict[str, dict[str, int]] = {family: {} for family in families}
        critical_warnings: list[str] = []
        patches: list[dict[str, object]] = []
        memory_bundle_present_count = 0
        memory_bundle_absent_count = 0
        warning_prefix = {
            "identity": "identity_region",
            "skin": "skin_region",
            "body_shape": "body_region",
            "garment": "garment_region",
            "accessory": "accessory_region",
        }

        for patch_debug in patch_step_debug:
            usage = GennadyEngine._extract_patch_reference_usage(patch_debug)
            patches.append(usage)
            family = str(usage.get("expected_reference_family", "unknown"))
            if family not in expected_family_counts:
                family = "unknown"
            expected_family_counts[family] += 1
            if usage.get("memory_bundle_present"):
                memory_bundle_present_count += 1
            else:
                memory_bundle_absent_count += 1
            if family in families:
                if usage.get("expected_reference_used"):
                    used_counts[family] += 1
                if usage.get("expected_reference_blocked"):
                    blocked_counts[family] += 1
                    critical_warnings.append(f"{warning_prefix[family]}_reference_blocked:{usage.get('region_id', '')}")
                if usage.get("expected_reference_missing"):
                    missing_counts[family] += 1
                    critical_warnings.append(f"{warning_prefix[family]}_without_{family}_reference:{usage.get('region_id', '')}")
                if usage.get("expected_material_used"):
                    material_used_counts[family] += 1
                if usage.get("expected_input_frame_material_used"):
                    input_frame_material_used_counts[family] += 1
                elif usage.get("expected_material_missing"):
                    material_missing_counts[family] += 1
                    reason = str(usage.get("reference_patch_material_missing_reason", "material_missing") or "material_missing")
                    material_missing_reasons[family][reason] = material_missing_reasons[family].get(reason, 0) + 1
                    critical_warnings.append(f"{warning_prefix[family]}_without_visual_material:{usage.get('region_id', '')}:{reason}")
                    critical_warnings.append(f"{warning_prefix[family]}_without_input_frame_material:{usage.get('region_id', '')}:{reason}")
                    if reason == "generated_runtime_material_rejected":
                        generated_material_rejected_counts[family] += 1
                        critical_warnings.append(f"generated_material_rejected_for_reference:{usage.get('region_id', '')}:{reason}")
                    if reason == "non_input_frame_material_rejected":
                        non_input_material_rejected_counts[family] += 1
                    if reason == "missing_i2v_anchor":
                        missing_i2v_anchor_counts[family] += 1
                strength = float(usage.get("expected_reference_strength", 0.0) or 0.0)
                if usage.get("expected_reference_used") and strength >= 0.9:
                    strong_counts[family] += 1
                elif usage.get("expected_reference_used") and 0.0 < strength < 0.9:
                    medium_counts[family] += 1

        return {
            "patch_count": len(patch_step_debug),
            "expected_family_counts": expected_family_counts,
            "used_counts": used_counts,
            "blocked_counts": blocked_counts,
            "missing_counts": missing_counts,
            "strong_counts": strong_counts,
            "medium_counts": medium_counts,
            "material_used_counts": material_used_counts,
            "input_frame_material_used_counts": input_frame_material_used_counts,
            "generated_material_rejected_counts": generated_material_rejected_counts,
            "non_input_material_rejected_counts": non_input_material_rejected_counts,
            "missing_i2v_anchor_counts": missing_i2v_anchor_counts,
            "material_missing_counts": material_missing_counts,
            "material_missing_reasons": material_missing_reasons,
            "memory_bundle_present_count": memory_bundle_present_count,
            "memory_bundle_absent_count": memory_bundle_absent_count,
            "critical_warnings": critical_warnings,
            "patches": patches,
        }

    @staticmethod
    def _aggregate_reference_coverage(step_coverages: list[dict[str, object]]) -> dict[str, object]:
        families = ("identity", "skin", "body_shape", "garment", "accessory")
        expected_families = (*families, "unknown")
        expected_family_counts = {family: 0 for family in expected_families}
        used_counts = {family: 0 for family in families}
        blocked_counts = {family: 0 for family in families}
        missing_counts = {family: 0 for family in families}
        strong_counts = {family: 0 for family in families}
        medium_counts = {family: 0 for family in families}
        material_used_counts = {family: 0 for family in families}
        input_frame_material_used_counts = {family: 0 for family in families}
        generated_material_rejected_counts = {family: 0 for family in families}
        non_input_material_rejected_counts = {family: 0 for family in families}
        missing_i2v_anchor_counts = {family: 0 for family in families}
        material_missing_counts = {family: 0 for family in families}
        material_missing_reasons: dict[str, dict[str, int]] = {family: {} for family in families}
        total_patch_count = 0
        memory_bundle_present_count = 0
        memory_bundle_absent_count = 0
        critical_warnings: list[str] = []

        def _add_counts(target: dict[str, int], source: object) -> None:
            if not isinstance(source, dict):
                return
            for key in target:
                target[key] += int(source.get(key, 0) or 0)

        def _add_nested_counts(target: dict[str, dict[str, int]], source: object) -> None:
            if not isinstance(source, dict):
                return
            for family, reasons in source.items():
                if family not in target or not isinstance(reasons, dict):
                    continue
                for reason, count in reasons.items():
                    r = str(reason)
                    target[family][r] = target[family].get(r, 0) + int(count or 0)

        for coverage in step_coverages:
            total_patch_count += int(coverage.get("patch_count", 0) or 0)
            _add_counts(expected_family_counts, coverage.get("expected_family_counts", {}))
            _add_counts(used_counts, coverage.get("used_counts", {}))
            _add_counts(blocked_counts, coverage.get("blocked_counts", {}))
            _add_counts(missing_counts, coverage.get("missing_counts", {}))
            _add_counts(strong_counts, coverage.get("strong_counts", {}))
            _add_counts(medium_counts, coverage.get("medium_counts", {}))
            _add_counts(material_used_counts, coverage.get("material_used_counts", {}))
            _add_counts(input_frame_material_used_counts, coverage.get("input_frame_material_used_counts", {}))
            _add_counts(generated_material_rejected_counts, coverage.get("generated_material_rejected_counts", {}))
            _add_counts(non_input_material_rejected_counts, coverage.get("non_input_material_rejected_counts", {}))
            _add_counts(missing_i2v_anchor_counts, coverage.get("missing_i2v_anchor_counts", {}))
            _add_counts(material_missing_counts, coverage.get("material_missing_counts", {}))
            _add_nested_counts(material_missing_reasons, coverage.get("material_missing_reasons", {}))
            memory_bundle_present_count += int(coverage.get("memory_bundle_present_count", 0) or 0)
            memory_bundle_absent_count += int(coverage.get("memory_bundle_absent_count", 0) or 0)
            warnings = coverage.get("critical_warnings", [])
            if isinstance(warnings, list):
                critical_warnings.extend(str(w) for w in warnings)

        def _ratio(family: str) -> float:
            return used_counts[family] / max(1, expected_family_counts[family])

        def _material_ratio(family: str) -> float:
            return material_used_counts[family] / max(1, expected_family_counts[family])
        def _input_ratio(family: str) -> float:
            return input_frame_material_used_counts[family] / max(1, expected_family_counts[family])

        total_expected_known = sum(expected_family_counts[family] for family in families)
        total_used_known = sum(used_counts.values())
        total_material_used_known = sum(material_used_counts.values())
        total_input_material_used_known = sum(input_frame_material_used_counts.values())
        return {
            "total_patch_count": total_patch_count,
            "expected_family_counts": expected_family_counts,
            "used_counts": used_counts,
            "blocked_counts": blocked_counts,
            "missing_counts": missing_counts,
            "strong_counts": strong_counts,
            "medium_counts": medium_counts,
            "material_used_counts": material_used_counts,
            "input_frame_material_used_counts": input_frame_material_used_counts,
            "generated_material_rejected_counts": generated_material_rejected_counts,
            "non_input_material_rejected_counts": non_input_material_rejected_counts,
            "missing_i2v_anchor_counts": missing_i2v_anchor_counts,
            "material_missing_counts": material_missing_counts,
            "material_missing_reasons": material_missing_reasons,
            "memory_bundle_present_count": memory_bundle_present_count,
            "memory_bundle_absent_count": memory_bundle_absent_count,
            "critical_warning_count": len(critical_warnings),
            "top_critical_warnings": critical_warnings[:20],
            "identity_reference_coverage_ratio": _ratio("identity"),
            "body_shape_reference_coverage_ratio": _ratio("body_shape"),
            "garment_reference_coverage_ratio": _ratio("garment"),
            "skin_reference_coverage_ratio": _ratio("skin"),
            "identity_material_coverage_ratio": _material_ratio("identity"),
            "body_shape_material_coverage_ratio": _material_ratio("body_shape"),
            "garment_material_coverage_ratio": _material_ratio("garment"),
            "skin_material_coverage_ratio": _material_ratio("skin"),
            "identity_input_frame_material_coverage_ratio": _input_ratio("identity"),
            "body_shape_input_frame_material_coverage_ratio": _input_ratio("body_shape"),
            "garment_input_frame_material_coverage_ratio": _input_ratio("garment"),
            "skin_input_frame_material_coverage_ratio": _input_ratio("skin"),
            "overall_expected_reference_coverage_ratio": total_used_known / max(1, total_expected_known),
            "overall_expected_material_coverage_ratio": total_material_used_known / max(1, total_expected_known),
            "overall_input_frame_material_coverage_ratio": total_input_material_used_known / max(1, total_expected_known),
            "input_frame_material_coverage_source": "observed_directly_not_generated_v1",
        }

    def run(
        self,
        images: list[str],
        text: str,
        fps: int = 16,
        duration: float = 4.0,
        quality_profile: str = "balanced",
        export_renderer_manifest_path: str | None = None,
    ) -> InferenceArtifacts:
        profile = self._resolve_profile(quality_profile)
        request = self.input_layer.build_request(
            images=images,
            text=text,
            fps=fps,
            duration=duration,
            quality_profile=quality_profile,
        )

        first_frame = request.unified_asset.frames[0] if request.unified_asset and request.unified_asset.frames else None
        current_frame = self._normalize_frame_tensor(first_frame.tensor, field_name="input_frame") if first_frame else self._debug_seed_frame_tensor(profile)
        perception_input = first_frame if first_frame else current_frame
        perception_output = self.perception.analyze(perception_input)
        perception_output.frame_size = (shape(current_frame)[1], shape(current_frame)[0])
        scene_graph = self.graph_builder.build(perception_output, frame_index=0)
        scene_graph.global_context.fps = fps
        scene_graph.global_context.frame_size = perception_output.frame_size
        scene_graph.global_context.source_type = request.input_type

        memory = self.memory_manager.initialize_from_scene(scene_graph)
        memory = self.memory_manager.update_from_frame(
            memory,
            current_frame,
            scene_graph,
            transition_context={
                "frame_source": "observed_input_frame",
                "generated": False,
                "source_frame_kind": "observed_input_frame",
                "source_is_input_frame": True,
                "immutable_i2v_anchor": True,
            },
        )
        graph_encoding = self.backends.graph_encoder.encode(scene_graph)
        fallback_log: list[str] = []
        renderer_manifest_records: list[dict[str, object]] = []
        renderer_manifest_export_warnings: list[str] = []
        renderer_manifest_export_error: str | None = None
        renderer_manifest_export_contract_version: str | None = None
        renderer_manifest_exporter = None
        renderer_manifest_export_strict = bool(self.backend_config.patch_strict_mode)
        if export_renderer_manifest_path:
            from training.renderer_manifest_exporter import RendererManifestRecordExporter

            renderer_manifest_exporter = RendererManifestRecordExporter()
            renderer_manifest_export_contract_version = "renderer_patch_manifest_v2"

        action_plan = self.intent_parser.parse(request.text, scene_graph=scene_graph)
        text_encoding = self.backends.text_encoder.encode(request.text, scene_graph=scene_graph, action_plan=action_plan)
        text_contract = text_output_to_contract(request.text, text_encoding)
        text_parity = build_parity_result(
            contract=text_contract,
            required_fields=["text", "parsed_actions", "action_embedding", "target_entities", "target_objects", "temporal_decomposition", "constraints"],
            stage="text",
            request={"text": request.text, "actions": action_plan.actions},
            output=text_encoding,
        )
        if text_parity["missing_fields"]:
            fallback_log.append(f"step=0:text_parity_missing={text_parity['missing_fields']}")
        for severity in ("errors", "warnings", "traces"):
            for issue in text_parity.get(severity, []):
                fallback_log.append(f"step=0:text_semantic_{severity}={issue}")
        state_plan = self.planner.expand(
            scene_graph,
            action_plan,
            runtime_profile={
                "fps": fps,
                "max_transition_steps": profile.max_transition_steps,
            },
            target_duration_sec=duration,
            policy="insert" if quality_profile == "debug" else "use_existing",
        )
        frame_plan = plan_i2v_frames(
            request.text,
            max(1, len(state_plan.steps)),
            scene_graph.persons[0].person_id if scene_graph.persons else "scene",
        )

        frames: list[list[list[list[float]]]] = [current_frame]
        graphs = [scene_graph]
        overlay_log: list[str] = []
        dynamics_metrics_log: list[str] = []
        channel_usage_log: list[dict[str, object]] = []
        step_debug: list[dict[str, object]] = []
        step_reference_coverages: list[dict[str, object]] = []
        hidden_recon_stats = {"known_hidden": 0, "unknown_hidden": 0, "hidden_reveal": 0, "steps_with_hidden_reconstruction": 0}
        hidden_recon_quality = {
            "confidence_sum": 0.0,
            "quality_hint_sum": 0.0,
            "refinement_strength_sum": 0.0,
            "count": 0,
            "synthesis_mode_counts": {},
            "strategy_counts": {},
            "by_strategy": {},
            "by_synthesis_mode": {},
            "by_family": {"known_hidden": {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0}, "unknown_hidden": {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0}, "hidden_reveal": {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0}},
        }

        for planned_state in state_plan.steps[1 : profile.max_transition_steps + 1]:
            frame_plan_entry = frame_plan[min(len(frame_plan) - 1, planned_state.step_index)] if frame_plan else None
            memory_summary = self.memory_summarizer.summarize(memory).as_dict()
            memory_channels = summarize_memory(memory)
            dynamics_channels = self.build_dynamics_memory_channels(memory_channels)
            entity_id = scene_graph.persons[0].person_id if scene_graph.persons else "scene"
            identity_embedding = self.backends.identity_encoder.encode_identity(memory_channels, entity_id)

            transition_request = DynamicsTransitionRequest(
                graph_state=scene_graph,
                memory_summary=memory_summary,
                memory_channels=dynamics_channels,
                text_action_summary=text_encoding,
                graph_encoding=graph_encoding,
                identity_embeddings={entity_id: identity_embedding},
                step_context={"step_index": planned_state.step_index, "memory": memory, "semantic_transition": planned_state.semantic_transition},
            )
            transition_output = self.backends.dynamics_backend.predict_transition(transition_request)
            parity_contract = dynamics_io_to_contract(transition_request, transition_output)
            dynamics_parity = build_parity_result(
                contract=parity_contract,
                required_fields=["graph_before", "graph_after", "delta_contract", "transition_context"],
                stage="dynamics",
                request=transition_request,
                output=transition_output,
            )
            if dynamics_parity["missing_fields"]:
                fallback_log.append(f"step={planned_state.step_index}:dynamics_parity_missing={dynamics_parity['missing_fields']}")
            for severity in ("errors", "warnings", "traces"):
                for issue in dynamics_parity.get(severity, []):
                    fallback_log.append(f"step={planned_state.step_index}:dynamics_semantic_{severity}={issue}")

            delta = transition_output.delta
            frame_plan_modes_applied, frame_plan_transition_modes_applied = self._apply_frame_plan_to_delta(delta, frame_plan_entry)
            region_plan = self.region_router.build_plan(
                scene_graph=scene_graph,
                delta=delta,
                memory=memory,
                semantic_transition=planned_state.semantic_transition,
            )
            transition_diag = delta.transition_diagnostics if isinstance(delta.transition_diagnostics, dict) else {}
            transition_diag["region_routing_plan"] = {
                make_region_id(region_plan.entity_id, d.canonical_region): {
                    "decision": d.decision,
                    "priority": d.priority,
                    "reveal_mode": d.reveal_mode,
                    "synthesis_required": d.synthesis_required,
                    "renderer_mode_hint": d.renderer_mode_hint,
                    "confidence": d.confidence,
                    "reasons": d.reasons,
                    "memory_source_available": d.memory_source_available,
                    "memory_support_level": d.memory_support_level,
                }
                for d in region_plan.decisions
            }
            transition_diag["region_transition_semantics"] = region_plan.as_debug_dict()["transition_semantics"]
            delta.transition_diagnostics = transition_diag
            changed_regions = region_plan.render_regions or self.roi_selector.select(scene_graph, delta)
            frame_plan_added_regions: list[str] = []
            frame_plan_missing_regions: list[str] = []
            if frame_plan_entry and frame_plan_entry.affected_regions:
                existing_ids = {r.region_id for r in changed_regions}
                for planned_region_id in frame_plan_entry.affected_regions:
                    if planned_region_id in existing_ids:
                        continue
                    resolved = self._resolve_planned_region(scene_graph, self.roi_selector, planned_region_id)
                    if resolved is None:
                        frame_plan_missing_regions.append(planned_region_id)
                        continue
                    changed_regions.append(resolved)
                    existing_ids.add(planned_region_id)
                    frame_plan_added_regions.append(planned_region_id)
            frame_plan_applied = bool(frame_plan_modes_applied or frame_plan_added_regions)
            patches: list[RenderedPatch] = []
            patch_step_debug: list[dict[str, object]] = []
            step_hidden_reconstruction = False
            step_hidden_cases = 0
            for region in changed_regions[: profile.max_roi_count]:
                patch_channels = self.build_patch_memory_channels(memory_channels)
                region_route = region_plan.decision_for_region_id(region.region_id)
                try:
                    region_entity_id, canonical_region = parse_region_id(region.region_id)
                except ValueError:
                    region_entity_id = entity_id
                    canonical_region = region.region_id
                region_memory_bundle = self.memory_manager.get_region_memory_bundle(memory, region_entity_id, canonical_region)
                region_metadata = build_region_metadata(
                    scene_graph=scene_graph,
                    memory=memory,
                    region=region,
                    route_decision=region_route,
                    delta=delta,
                )
                transition_metadata = transition_output.metadata if isinstance(transition_output.metadata, dict) else {}
                learned_temporal_contract = transition_metadata.get("temporal_transition_contract", {})
                learned_human_state_contract = transition_metadata.get("human_state_contract", {})
                learned_target_profile = {}
                if isinstance(learned_human_state_contract, dict):
                    learned_target_profile = learned_human_state_contract.get("target_profile", {}) if isinstance(learned_human_state_contract.get("target_profile", {}), dict) else {}
                if not learned_target_profile and isinstance(learned_temporal_contract, dict):
                    learned_target_profile = learned_temporal_contract.get("target_profile", {}) if isinstance(learned_temporal_contract.get("target_profile", {}), dict) else {}
                reference_payload_context = _serialize_reference_payload_context(region.region_id, region_memory_bundle)
                expected_reference_payload = _expected_reference_payload_for_region(region.region_id, region_memory_bundle)
                resolved_reference_payload_from_matching = False
                if expected_reference_payload is None:
                    matches = _trusted_matching_payloads(region.region_id, region_memory_bundle)
                    if len(matches) == 1:
                        expected_reference_payload = matches[0]
                        resolved_reference_payload_from_matching = True
                if expected_reference_payload is not None:
                    reference_payload_context = dict(reference_payload_context)
                    reference_payload_context["expected_reference_payload"] = asdict(expected_reference_payload)
                    if resolved_reference_payload_from_matching:
                        reasons = reference_payload_context.get("reference_payload_trace_reasons", [])
                        reason_list = list(reasons) if isinstance(reasons, list) else []
                        if "expected_reference_payload_resolved_from_matching_payload" not in reason_list:
                            reason_list.append("expected_reference_payload_resolved_from_matching_payload")
                        reference_payload_context["reference_payload_trace_reasons"] = reason_list
                reference_material = self.memory_manager.build_reference_patch_material(memory, expected_reference_payload)
                reference_material_reason = ""
                if reference_material is None:
                    reference_material_reason = "payload_missing"
                elif not reference_material.material_trusted:
                    reference_material_reason = reference_material.material_missing_reason or "payload_untrusted"
                reference_material_context = {
                    "expected_reference_patch_material": asdict(reference_material) if reference_material is not None else None,
                    "reference_patch_material_trace_reasons": [reference_material_reason] if reference_material_reason else [],
                }
                patch_request = PatchSynthesisRequest(
                    region=region,
                    scene_state=scene_graph,
                    memory_summary=memory_summary,
                    transition_context={
                        "graph_delta": delta,
                        "video_memory": memory,
                        "transition_phase": delta.transition_phase,
                        "step_index": planned_state.step_index,
                        "frame_plan": self._frame_plan_entry_to_dict(frame_plan_entry),
                        "i2v_action_phase": frame_plan_entry.action_phase if frame_plan_entry is not None else "stable_idle",
                        "i2v_region_transition_mode": frame_plan_entry.region_transition_mode if frame_plan_entry is not None else {},
                        "target_profile": learned_target_profile or transition_metadata.get("target_profile", {}),
                        "learned_temporal_contract": learned_temporal_contract,
                        "learned_human_state_contract": learned_human_state_contract,
                        "region_selection_rationale": transition_metadata.get("region_selection_rationale", {}),
                        "semantic_families": transition_metadata.get("semantic_families", []),
                        "region_route_decision": {
                            "decision": region_route.decision if region_route else "unknown",
                            "reveal_mode": region_route.reveal_mode if region_route else "none",
                            "renderer_mode_hint": region_route.renderer_mode_hint if region_route else "keep",
                            "synthesis_required": region_route.synthesis_required if region_route else False,
                        },
                        "region_memory_bundle": region_memory_bundle,
                        "region_memory_bundle_serialized": asdict(region_memory_bundle),
                        "region_memory_support_level": region_memory_bundle.memory_support_level,
                        "region_memory_retrieval_reasons": list(region_memory_bundle.retrieval_reasons),
                        **reference_payload_context,
                        **reference_material_context,
                    },
                    retrieval_summary={
                        "backend": "learned_primary",
                        "identity_entity": entity_id,
                        "target_profile": learned_target_profile or transition_metadata.get("target_profile", {}),
                    },
                    current_frame=current_frame,
                    memory_channels=patch_channels,
                    graph_encoding=graph_encoding,
                    identity_embedding=identity_embedding,
                    region_metadata=region_metadata,
                )
                patch_out = self.backends.patch_backend.synthesize_patch(patch_request)
                patch_contract_validation = self._validate_patch_output_contract(patch_out, expected_region_id=region.region_id)
                if patch_contract_validation["issues"]:
                    raise ValueError(f"Patch contract violation at step={planned_state.step_index}, region={region.region_id}: {patch_contract_validation['issues']}")
                if renderer_manifest_exporter is not None:
                    export_frame_index = len(frames)
                    try:
                        roi_before_export = self._extract_region_roi_for_export(current_frame, region, patch_out.height, patch_out.width)
                        roi_after_export = self._resolve_observed_roi_after_for_export(patch_request, patch_out)
                        renderer_manifest_records.append(
                            renderer_manifest_exporter.build_record(
                                request=patch_request,
                                output=patch_out,
                                roi_before=roi_before_export,
                                roi_after=roi_after_export,
                                step_index=planned_state.step_index,
                                frame_index=export_frame_index,
                            )
                        )
                    except Exception as exc:
                        warning = (
                            f"step_index={planned_state.step_index};"
                            f"frame_index={export_frame_index};"
                            f"region_id={region.region_id};"
                            f"error={type(exc).__name__}: {exc}"
                        )
                        renderer_manifest_export_warnings.append(warning)
                        if renderer_manifest_export_strict:
                            raise
                patch_contract = patch_io_to_contract(patch_request, patch_out)
                patch_parity = build_parity_result(
                    contract=patch_contract,
                    required_fields=PATCH_PARITY_REQUIRED_FIELDS,
                    stage="patch",
                    request=patch_request,
                    output=patch_out,
                )
                strategy = str(patch_out.execution_trace.get("selected_render_strategy", ""))
                synth_mode = str(patch_out.execution_trace.get("synthesis_mode", "deterministic"))
                patch_hidden_case = False
                if "KNOWN_HIDDEN_REVEAL" in strategy:
                    hidden_recon_stats["known_hidden"] += 1
                    hidden_recon_stats["hidden_reveal"] += 1
                    patch_hidden_case = True
                elif "UNKNOWN_HIDDEN_SYNTHESIS" in strategy:
                    hidden_recon_stats["unknown_hidden"] += 1
                    patch_hidden_case = True
                if patch_hidden_case:
                    step_hidden_reconstruction = True
                    step_hidden_cases += 1
                    hint = float(patch_out.execution_trace.get("patch_refinement_strength", 0.0) or 0.0)
                    hidden_recon_quality["count"] += 1
                    hidden_recon_quality["confidence_sum"] += float(patch_out.confidence)
                    hidden_recon_quality["quality_hint_sum"] += hint
                    hidden_recon_quality["refinement_strength_sum"] += hint
                    hidden_recon_quality["synthesis_mode_counts"][synth_mode] = int(hidden_recon_quality["synthesis_mode_counts"].get(synth_mode, 0)) + 1
                    hidden_recon_quality["strategy_counts"][strategy] = int(hidden_recon_quality["strategy_counts"].get(strategy, 0)) + 1
                    family = "hidden_reveal" if "REVEAL" in strategy else ("unknown_hidden" if "UNKNOWN_HIDDEN" in strategy else "known_hidden")
                    fam = hidden_recon_quality["by_family"][family]
                    fam["count"] += 1
                    fam["confidence_sum"] += float(patch_out.confidence)
                    fam["quality_hint_sum"] += hint
                    strat = hidden_recon_quality["by_strategy"].setdefault(strategy, {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0})
                    strat["count"] += 1
                    strat["confidence_sum"] += float(patch_out.confidence)
                    strat["quality_hint_sum"] += hint
                    mode = hidden_recon_quality["by_synthesis_mode"].setdefault(synth_mode, {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0})
                    mode["count"] += 1
                    mode["confidence_sum"] += float(patch_out.confidence)
                    mode["quality_hint_sum"] += hint
                if patch_parity["missing_fields"]:
                    fallback_log.append(f"step={planned_state.step_index}:patch_parity_missing={patch_parity['missing_fields']}")
                for severity in ("errors", "warnings", "traces"):
                    fallback_log.extend([f"step={planned_state.step_index}:patch_semantic_{severity}={x}" for x in patch_parity.get(severity, [])])
                patch_step_debug.append(
                    {
                        "region_id": region.region_id,
                        "selected_render_strategy": patch_out.execution_trace.get("selected_render_strategy", patch_contract.get("selected_render_strategy", "unknown")),
                        "execution_policy": patch_out.execution_trace.get("selection", {}).get("execution_policy", {}),
                        "mode_source": (patch_out.execution_trace.get("memory_dependency_summary", {}) or {}).get("mode_source", "unknown"),
                        "runtime_plan_authoritative": (patch_out.execution_trace.get("memory_dependency_summary", {}) or {}).get("runtime_plan_authoritative", False),
                        "confidence": patch_out.confidence,
                        "synthesis_mode": synth_mode,
                        "i2v_action_phase": frame_plan_entry.action_phase if frame_plan_entry is not None else "stable_idle",
                        "i2v_region_transition_mode": frame_plan_entry.region_transition_mode if frame_plan_entry is not None else {},
                        "retrieval_summary": str(patch_request.retrieval_summary)[:120],
                        "learned_ready": patch_out.metadata.get("learned_ready_usage", {}),
                        "region_metadata_completeness_score": region_metadata.get("metadata_completeness_score", 0.0),
                        "region_metadata_evidence_strength_score": region_metadata.get("evidence_strength_score", 0.0),
                        "region_metadata_source_node_type": region_metadata.get("source_node_type", "unknown"),
                        "region_metadata_mask_kind": region_metadata.get("mask_kind", ""),
                        "region_metadata_roi_source": region_metadata.get("roi_source", "unknown"),
                        "region_metadata_missing_fields": region_metadata.get("missing_fields", []),
                        "region_metadata_source_trace": region_metadata.get("metadata_source_trace", []),
                        "hidden_reconstruction": {
                            "patch_hidden_reconstruction_case": patch_hidden_case,
                            "strategy": strategy,
                            "mode": synth_mode,
                            "quality_hint": patch_out.execution_trace.get("patch_refinement_strength", 0.0),
                        },
                        "parity": patch_parity,
                        "contract_validation": patch_contract_validation,
                        "execution_trace": {
                            key: patch_out.execution_trace.get(key)
                            for key in (
                                "identity_reference_used",
                                "identity_reference_strength",
                                "identity_reference_source",
                                "identity_reference_blocked",
                                "identity_reference_block_reasons",
                                "skin_reference_used",
                                "skin_reference_strength",
                                "skin_reference_source",
                                "skin_reference_blocked",
                                "skin_reference_block_reasons",
                                "body_shape_reference_used",
                                "body_shape_reference_strength",
                                "body_shape_reference_source",
                                "body_shape_reference_blocked",
                                "body_shape_reference_block_reasons",
                                "garment_reference_used",
                                "garment_reference_strength",
                                "garment_reference_source",
                                "garment_reference_blocked",
                                "garment_reference_block_reasons",
                                "accessory_reference_used",
                                "accessory_reference_strength",
                                "accessory_reference_source",
                                "accessory_reference_blocked",
                                "accessory_reference_block_reasons",
                                "memory_bundle_present",
                                "memory_support_level",
                                "reference_patch_material_present",
                                "reference_patch_material_validated",
                                "reference_patch_material_trusted",
                                "reference_patch_material_used",
                                "reference_patch_material_source",
                                "reference_patch_material_shape",
                                "reference_patch_material_kind",
                                "reference_patch_material_missing_reason",
                                "reference_tensor_input_used",
                                "reference_tensor_zero_fallback",
                                "reference_tensor_input_channels",
                                "region_id",
                                "transition_mode",
                                "reference_payload_trusted",
                                "material_gate_mean",
                                "material_gate_max",
                                "preservation_drift",
                                "alpha_mean",
                                "uncertainty_mean",
                            )
                            if key in patch_out.execution_trace
                        },
                    }
                )
                patches.append(
                    RenderedPatch(
                        region=patch_out.region,
                        rgb_patch=patch_out.rgb_patch,
                        alpha_mask=patch_out.alpha_mask,
                        height=patch_out.height,
                        width=patch_out.width,
                        channels=patch_out.channels,
                        uncertainty_map=patch_out.uncertainty_map,
                        confidence=patch_out.confidence,
                        z_index=patch_out.z_index,
                        debug_trace=patch_out.debug_trace,
                        execution_trace=patch_out.execution_trace,
                    )
                )

            composed = self.compositor.compose(current_frame, patches, delta)
            temporal_channels = self.build_temporal_memory_channels(memory_channels)
            if patches:
                patch_conf = [float(p.confidence) for p in patches]
                patch_alpha_mean = []
                patch_alpha_edge = []
                for p in patches:
                    alpha_vals = [float(v) for row in p.alpha_mask for v in row] if p.alpha_mask else [0.0]
                    patch_alpha_mean.append(sum(alpha_vals) / max(1, len(alpha_vals)))
                    edge_vals = []
                    if p.alpha_mask:
                        edge_vals.extend([float(v) for v in p.alpha_mask[0]])
                        edge_vals.extend([float(v) for v in p.alpha_mask[-1]])
                        edge_vals.extend([float(row[0]) for row in p.alpha_mask])
                        edge_vals.extend([float(row[-1]) for row in p.alpha_mask])
                    patch_alpha_edge.append(sum(edge_vals) / max(1, len(edge_vals)))
                temporal_channels = {
                    **temporal_channels,
                    "patch_confidence": {
                        "mean_confidence": sum(patch_conf) / max(1, len(patch_conf)),
                        "min_confidence": min(patch_conf),
                        "max_confidence": max(patch_conf),
                    },
                    "patch_alpha": {
                        "mean_alpha": sum(patch_alpha_mean) / max(1, len(patch_alpha_mean)),
                        "edge_alpha": sum(patch_alpha_edge) / max(1, len(patch_alpha_edge)),
                    },
                    "patch_history": {
                        "count": len(patches),
                        "region_ids": [p.region.region_id for p in patches],
                    },
                }
            temporal_request = TemporalRefinementRequest(
                previous_frame=frames[-1],
                current_composed_frame=composed,
                changed_regions=[p.region for p in patches],
                scene_state=scene_graph,
                memory_state=memory,
                memory_channels=temporal_channels,
            )
            temporal_out = self.backends.temporal_backend.refine_temporal(temporal_request)
            temporal_contract_validation = self._validate_temporal_output_contract(
                temporal_out,
                expected_shape=tuple(np.asarray(composed, dtype=np.float32).shape),
            )
            if temporal_contract_validation["issues"]:
                raise ValueError(f"Temporal contract violation at step={planned_state.step_index}: {temporal_contract_validation['issues']}")
            temporal_contract = temporal_io_to_contract(temporal_request, temporal_out)
            temporal_parity = build_parity_result(
                contract=temporal_contract,
                required_fields=["previous_frame", "composed_frame", "target_frame", "changed_regions", "scene_transition_context"],
                stage="temporal",
                request=temporal_request,
                output=temporal_out,
                changed_regions_count=len(patches),
            )
            if temporal_parity["missing_fields"]:
                fallback_log.append(f"step={planned_state.step_index}:temporal_parity_missing={temporal_parity['missing_fields']}")
            for severity in ("errors", "warnings", "traces"):
                fallback_log.extend([f"step={planned_state.step_index}:temporal_semantic_{severity}={x}" for x in temporal_parity.get(severity, [])])
            stable_frame = temporal_out.refined_frame if profile.temporal_refinement else composed
            stable_frame = self._normalize_frame_tensor(stable_frame, field_name="stable_frame")

            scene_graph = apply_delta(scene_graph, delta)
            graph_encoding = self.backends.graph_encoder.encode(scene_graph)
            memory = self.memory_manager.update_from_graph(memory, scene_graph)
            transition_context = self._memory_update_context_for_generated_frame(
                delta,
                temporal_refinement_enabled=profile.temporal_refinement,
            )
            memory_update_provenance = {
                "generated": transition_context["generated"],
                "frame_source": transition_context["frame_source"],
                "update_source": transition_context["update_source"],
            }
            memory.last_transition_context = transition_context
            memory = self.memory_manager.update_from_frame(memory, stable_frame, scene_graph, transition_context=transition_context)

            frames.append(stable_frame)
            current_frame = stable_frame
            graphs.append(scene_graph)
            overlay_log.append(f"step={planned_state.step_index}, regions={len(changed_regions)}")
            step_reference_coverage = self._summarize_step_reference_coverage(patch_step_debug)
            step_reference_coverages.append(step_reference_coverage)
            dynamics_metrics_log.append(
                f"delta={transition_output.diagnostics.get('delta_magnitude', 0.0):.3f}, smooth={transition_output.diagnostics.get('temporal_smoothness_proxy', 0.0):.3f}, violations={transition_output.diagnostics.get('constraint_violations', 0.0):.0f}"
            )
            step_debug.append(
                {
                    "step_index": planned_state.step_index,
                    "frame_plan": self._frame_plan_entry_to_dict(frame_plan_entry),
                    "frame_plan_applied": frame_plan_applied,
                    "frame_plan_added_regions": frame_plan_added_regions,
                    "frame_plan_missing_regions": frame_plan_missing_regions,
                    "frame_plan_transition_modes_applied": frame_plan_transition_modes_applied,
                    "region_routing": region_plan.as_debug_dict(),
                    "region_render_order": [r.region_id for r in changed_regions[: profile.max_roi_count]],
                    "dynamics": {
                        "backend": self.backends.backend_names.get("dynamics_backend", "unknown"),
                        "confidence": transition_output.confidence,
                        "supervision_mode": parity_contract.get("transition_context", {}).get("supervision_mode", "inference"),
                            "diagnostics_summary": {
                                "delta_magnitude": transition_output.diagnostics.get("delta_magnitude"),
                                "smoothness": transition_output.diagnostics.get("temporal_smoothness_proxy"),
                                "violations": transition_output.diagnostics.get("constraint_violations"),
                                "learned_ready": transition_output.metadata.get("learned_ready_usage", {}),
                                "temporal_contract_alignment": transition_output.metadata.get("temporal_contract_alignment", {}),
                            },
                        },
                    "patch": patch_step_debug,
                    "reference_coverage": step_reference_coverage,
                    "hidden_reconstruction": {
                        "step_has_hidden_reconstruction": step_hidden_reconstruction,
                        "step_hidden_reconstruction_case_count": step_hidden_cases,
                        "known_hidden_count": hidden_recon_stats["known_hidden"],
                        "unknown_hidden_count": hidden_recon_stats["unknown_hidden"],
                        "hidden_reveal_count": hidden_recon_stats["hidden_reveal"],
                    },
                    "memory_update_provenance": memory_update_provenance,
                    "temporal": {
                        "backend": self.backends.backend_names.get("temporal_backend", "unknown"),
                        "temporal_path": temporal_out.metadata.get("temporal_path", "unknown"),
                        "fallback_reason": temporal_out.metadata.get("fallback_reason"),
                        "region_consistency_summary": temporal_out.region_consistency_scores,
                        "learned_ready": temporal_out.metadata.get("learned_ready_usage", {}),
                        "drift_consistency": {
                            "changed_regions": len(temporal_request.changed_regions),
                            "drift_proxy": 1.0 - (sum(temporal_out.region_consistency_scores.values()) / max(1.0, float(len(temporal_out.region_consistency_scores) or 1))),
                        },
                        "contract_validation": temporal_contract_validation,
                    },
                    "parity": {
                        "missing_fields": {
                            "dynamics": dynamics_parity["missing_fields"],
                            "temporal": temporal_parity["missing_fields"],
                            "patch": [p["parity"]["missing_fields"] for p in patch_step_debug],
                        },
                        "semantic_issues": {
                            "dynamics": dynamics_parity,
                            "temporal": temporal_parity,
                            "patch": [p["parity"] for p in patch_step_debug],
                        },
                    },
                }
            )
            channel_usage_log.append(
                {
                    "step_index": planned_state.step_index,
                    "frame_plan": self._frame_plan_entry_to_dict(frame_plan_entry),
                    "dynamics_channels": list(transition_request.memory_channels.keys()),
                    "patch_channels": list(self.build_patch_memory_channels(memory_channels).keys()) if changed_regions else [],
                    "temporal_channels": list(temporal_request.memory_channels.keys()),
                    "identity_encoder_used": bool(identity_embedding),
                    "graph_embedding_dim": len(graph_encoding.graph_embedding),
                    "dynamics_backend_usage": transition_output.metadata.get("learned_ready_usage", {}),
                    "memory_update_provenance": memory_update_provenance,
                    "reference_coverage": {
                        "identity_ratio": step_reference_coverage["used_counts"]["identity"] / max(1, step_reference_coverage["expected_family_counts"]["identity"]),
                        "body_shape_ratio": step_reference_coverage["used_counts"]["body_shape"] / max(1, step_reference_coverage["expected_family_counts"]["body_shape"]),
                        "garment_ratio": step_reference_coverage["used_counts"]["garment"] / max(1, step_reference_coverage["expected_family_counts"]["garment"]),
                        "critical_warning_count": len(step_reference_coverage["critical_warnings"]),
                    },
                }
            )
            if step_hidden_reconstruction:
                hidden_recon_stats["steps_with_hidden_reconstruction"] += 1

        if export_renderer_manifest_path and not renderer_manifest_records:
            renderer_manifest_export_warnings.append("renderer manifest export enabled but no patch records were exported")
        if renderer_manifest_exporter is not None and export_renderer_manifest_path:
            try:
                Path(export_renderer_manifest_path).parent.mkdir(parents=True, exist_ok=True)
                renderer_manifest_exporter.write_manifest(renderer_manifest_records, export_renderer_manifest_path)
            except Exception as exc:
                renderer_manifest_export_error = f"{type(exc).__name__}: {exc}"
                if renderer_manifest_export_strict:
                    raise
        reference_coverage_summary = self._aggregate_reference_coverage(step_reference_coverages)
        renderer_manifest_export_debug = {
            "enabled": bool(export_renderer_manifest_path),
            "path": export_renderer_manifest_path,
            "record_count": len(renderer_manifest_records),
            "contract_version": renderer_manifest_export_contract_version if export_renderer_manifest_path else None,
            "warnings": list(renderer_manifest_export_warnings),
            "error": renderer_manifest_export_error,
        }

        video_uri = self._export_video(frames, fps)
        return InferenceArtifacts(
            frames=frames,
            scene_graphs=graphs,
            state_plan=state_plan,
            debug={
                "overlay_log": overlay_log,
                "dynamics_metrics": dynamics_metrics_log,
                "step_execution": step_debug,
                "profile": {
                    "name": profile.name,
                    "internal_resolution": profile.internal_resolution,
                    "max_transition_steps": profile.max_transition_steps,
                    "temporal_refinement": profile.temporal_refinement,
                    "backend": profile.backend,
                },
                "video_export": video_uri,
                "renderer_manifest_export": renderer_manifest_export_debug,
                "reference_coverage_summary": reference_coverage_summary,
                "renderer_manifest_export_enabled": bool(export_renderer_manifest_path),
                "renderer_manifest_export_path": export_renderer_manifest_path,
                "renderer_manifest_export_record_count": len(renderer_manifest_records),
                "renderer_manifest_export_contract_version": renderer_manifest_export_contract_version if export_renderer_manifest_path else None,
                "renderer_manifest_export_warnings": list(renderer_manifest_export_warnings),
                "renderer_manifest_export_error": renderer_manifest_export_error,
                "i2v_frame_plan": [self._frame_plan_entry_to_dict(entry) for entry in frame_plan],
                "input_metadata": {
                    "input_type": request.input_type,
                    "orig_size": request.orig_size,
                    "normalized_size": request.normalized_size,
                    "frame_count": request.frame_count,
                    "timestamps_preview": request.timestamps[: min(10, len(request.timestamps))],
                    "reference_set": request.reference_set,
                    "source_mode": "image_grounded" if first_frame else "debug_fallback",
                },
                "learned_ready": {
                    "backend_selection": self.backends.backend_names,
                    "backend_config": asdict(self.backend_config),
                    "graph_encoder_used": True,
                    "identity_encoder_used": True,
                    "graph_encoding_confidence": graph_encoding.confidence,
                    "memory_channel_usage": channel_usage_log,
                    "fallbacks": fallback_log,
                    "contract_types": {
                        "text": "TextEncodingOutput->TextActionStateContract",
                        "dynamics": "DynamicsTransitionRequest/Output->GraphTransitionContract",
                        "patch": "PatchSynthesisRequest/Output->PatchSynthesisContract",
                        "temporal": "TemporalRefinementRequest/Output->TemporalConsistencyContract",
                    },
                    "text_parity": text_parity,
                    "hidden_reconstruction_summary": {
                        **hidden_recon_stats,
                        "average_hidden_reconstruction_confidence": (hidden_recon_quality["confidence_sum"] / hidden_recon_quality["count"]) if hidden_recon_quality["count"] else 0.0,
                        "average_refinement_strength": (hidden_recon_quality["refinement_strength_sum"] / hidden_recon_quality["count"]) if hidden_recon_quality["count"] else 0.0,
                        "average_quality_hint": (hidden_recon_quality["quality_hint_sum"] / hidden_recon_quality["count"]) if hidden_recon_quality["count"] else 0.0,
                        "count_by_synthesis_mode": hidden_recon_quality["synthesis_mode_counts"],
                        "count_by_selected_strategy": hidden_recon_quality["strategy_counts"],
                        "quality_by_selected_strategy": hidden_recon_quality["by_strategy"],
                        "quality_by_synthesis_mode": hidden_recon_quality["by_synthesis_mode"],
                        "quality_by_hidden_family": hidden_recon_quality["by_family"],
                    },
                },
            },
        )


    @staticmethod
    def _resolve_observed_roi_after_for_export(request: PatchSynthesisRequest, output: object) -> object | None:
        """Return explicit observed post-transition ROI when runtime context provides one.

        Normal inference usually has no ground-truth future crop, so callers pass
        ``None`` and the manifest exporter marks ``output.rgb_patch`` as a
        self-generated runtime target. Tests/advanced orchestrators may attach an
        observed ROI to the request context or output metadata, in which case the
        same exporter records it as external/observed supervision.
        """

        ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
        metadata = getattr(output, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        for source in (ctx, metadata):
            for key in (
                "roi_after",
                "observed_roi_after",
                "ground_truth_roi_after",
                "post_transition_roi",
                "post_transition_roi_observed",
                "target_roi_after",
            ):
                value = source.get(key)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _extract_region_roi_for_export(frame: list, region: object, height: int, width: int) -> list:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"renderer export roi_before requires HxWx3 frame, got shape={list(arr.shape)}")
        h, w = arr.shape[:2]
        bbox = getattr(region, "bbox", None)
        if bbox is not None:
            x = float(getattr(bbox, "x", 0.0))
            y = float(getattr(bbox, "y", 0.0))
            bw = float(getattr(bbox, "w", 1.0))
            bh = float(getattr(bbox, "h", 1.0))
            if max(abs(x), abs(y), abs(bw), abs(bh)) <= 1.5:
                x0, y0 = int(round(x * w)), int(round(y * h))
                x1, y1 = int(round((x + bw) * w)), int(round((y + bh) * h))
            else:
                x0, y0 = int(round(x)), int(round(y))
                x1, y1 = int(round(x + bw)), int(round(y + bh))
            x0, y0 = max(0, min(w - 1, x0)), max(0, min(h - 1, y0))
            x1, y1 = max(x0 + 1, min(w, x1)), max(y0 + 1, min(h, y1))
            crop = arr[y0:y1, x0:x1]
        else:
            crop = arr
        target_h = max(1, int(height))
        target_w = max(1, int(width))
        if crop.shape[0] == target_h and crop.shape[1] == target_w:
            return np.clip(crop, 0.0, 1.0).tolist()
        yy = np.linspace(0, crop.shape[0] - 1, target_h).round().astype(int)
        xx = np.linspace(0, crop.shape[1] - 1, target_w).round().astype(int)
        resized = crop[yy][:, xx]
        return np.clip(resized, 0.0, 1.0).tolist()

    def _debug_seed_frame_tensor(self, profile: RuntimeProfile) -> list:
        h, w = profile.internal_resolution
        return zeros(h, w, 3, value=0.5)

    def _resolve_profile(self, name: str) -> RuntimeProfile:
        return PROFILES.get(name, PROFILES["balanced"])

    def _export_video(self, frames: list[list], fps: int) -> str:
        if not frames:
            return "video://empty"

        out_dir = Path(tempfile.gettempdir()) / "gennady_exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "rendered_sequence.mp4"
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            uint8_frames = [np.array([[[(int(max(0, min(1, ch)) * 255)) for ch in px] for px in row] for row in f], dtype=np.uint8) for f in frames]
            h, w = uint8_frames[0].shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for frame in uint8_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception:
            out_path.write_bytes(b"mp4-export-unavailable")

        return str(out_path)
