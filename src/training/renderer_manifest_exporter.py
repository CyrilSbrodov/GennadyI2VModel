from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from learned.interfaces import PatchSynthesisOutput, PatchSynthesisRequest


@dataclass(slots=True)
class RendererManifestRecordExporter:
    """Strict serializer from runtime patch IO into renderer-training manifest records.

    The exporter does not generate targets or claim supervision quality.  When an
    explicit post-transition ROI is not supplied, it serializes the runtime patch
    output as a self-generated target and marks that provenance in the record.
    """

    def build_record(
        self,
        *,
        request: PatchSynthesisRequest,
        output: PatchSynthesisOutput,
        roi_before: list | np.ndarray,
        roi_after: list | np.ndarray | None = None,
        step_index: int | None = None,
        frame_index: int | None = None,
    ) -> dict[str, object]:
        before = _as_hw3_list(roi_before, "roi_before")
        if roi_after is None:
            after = _as_hw3_list(output.rgb_patch, "output.rgb_patch")
            target_source = "runtime_output_patch"
            training_target_quality = "self_generated_runtime_target"
        else:
            after = _as_hw3_list(roi_after, "roi_after")
            target_source = "provided_ground_truth_roi"
            training_target_quality = "external_or_observed_target"

        ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
        trace = output.execution_trace if isinstance(output.execution_trace, dict) else {}
        memory_bundle = _memory_bundle_serialized(ctx)
        selected_render_strategy = str(trace.get("selected_render_strategy", "unknown"))
        synthesis_mode = str(trace.get("synthesis_mode", trace.get("renderer_path", "unknown")))
        renderer_path = str(trace.get("renderer_path", output.metadata.get("renderer_path", "unknown") if isinstance(output.metadata, dict) else "unknown"))
        family = _infer_semantic_family(request, ctx)
        delta_contract = _graph_delta_summary(ctx.get("graph_delta", ctx.get("delta_contract", {})))
        transition_summary = _transition_context_summary(ctx, step_index=step_index, memory_bundle=memory_bundle)

        alpha_target = _alpha_target(output, before)
        changed_mask = _changed_mask(before, after)
        blend_hint = alpha_target

        renderer_contract = _renderer_batch_contract(
            ctx=ctx,
            trace=trace,
            memory_bundle=memory_bundle,
            alpha_target=alpha_target,
            blend_hint=blend_hint,
            changed_mask=changed_mask,
        )
        execution_trace_summary = _execution_trace_summary(
            trace=trace,
            ctx=ctx,
            memory_bundle=memory_bundle,
            selected_render_strategy=selected_render_strategy,
            renderer_path=renderer_path,
            synthesis_mode=synthesis_mode,
        )

        patch_output_contract = {
            "region_id": request.region.region_id,
            "height": int(output.height),
            "width": int(output.width),
            "channels": int(output.channels),
            "confidence": float(output.confidence),
            "renderer_path": renderer_path,
            "selected_render_strategy": selected_render_strategy,
            "synthesis_mode": synthesis_mode,
            "target_source": target_source,
            "training_target_quality": training_target_quality,
        }
        region_metadata = _region_metadata_v2(request)
        metadata_completeness_score = _safe_float(region_metadata.get("metadata_completeness_score"), 0.0)
        evidence_strength_score = _safe_float(region_metadata.get("evidence_strength_score"), 0.0)
        roi_source = str(region_metadata.get("roi_source", "unknown"))
        source_node_type = str(region_metadata.get("source_node_type", "unknown"))
        mask_kind = str(region_metadata.get("mask_kind", ""))
        mask_ref_present = bool(region_metadata.get("mask_ref"))

        patch_synthesis_contract = {
            "region_metadata": region_metadata,
            "retrieval_explanation_summary": _safe_json(request.retrieval_summary),
            "selected_render_strategy": selected_render_strategy,
            "synthesis_mode": synthesis_mode,
            "transition_context": transition_summary,
            "target_source": target_source,
            "training_target_quality": training_target_quality,
        }

        record_id = _record_id(request, frame_index, step_index)
        bbox = request.region.bbox
        canonical_region = str(region_metadata.get("canonical_region", _region_type(request.region.region_id)))
        entity_id = str(region_metadata.get("entity_id", _entity_id(request.region.region_id)))
        record: dict[str, object] = {
            "contract_version": "renderer_patch_manifest_v2",
            "record_id": record_id,
            "frame_index": int(frame_index if frame_index is not None else ctx.get("frame_index", -1)),
            "step_index": int(step_index if step_index is not None else ctx.get("step_index", -1)),
            "region_id": request.region.region_id,
            "canonical_region": canonical_region,
            "entity_id": entity_id,
            "roi_before": before,
            "roi_after": after,
            "alpha_mask": alpha_target,
            "alpha_target": alpha_target,
            "changed_mask": changed_mask,
            "preservation_mask": _mask_from_contract_or_none(ctx, trace, "preservation_mask"),
            "uncertainty_target": _mask_from_contract_or_none(ctx, trace, "uncertainty_target"),
            "seam_prior": _mask_from_contract_or_none(ctx, trace, "seam_prior"),
            "region_metadata": region_metadata,
            "transition_context_summary": transition_summary,
            "metadata_completeness_score": metadata_completeness_score,
            "evidence_strength_score": evidence_strength_score,
            "roi_source": roi_source,
            "source_node_type": source_node_type,
            "mask_kind": mask_kind,
            "mask_ref_present": mask_ref_present,
            "canonical_region_bbox_xywh": [float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)],
            "source_frame_ref": {"frame_index": int(frame_index if frame_index is not None else ctx.get("frame_index", -1)), "current_frame_shape": _shape_of(request.current_frame)},
            "semantic_family": family,
            "region_family": family,
            "source": "runtime_patch_export",
            "renderer_batch_contract": renderer_contract,
            "patch_synthesis_contract": patch_synthesis_contract,
            "patch_output_contract": patch_output_contract,
            "graph_delta": delta_contract,
            "delta_contract": delta_contract,
            "transition_context": transition_summary,
            "selected_render_strategy": selected_render_strategy,
            "renderer_path": renderer_path,
            "synthesis_mode": synthesis_mode,
            "target_source": target_source,
            "training_target_quality": training_target_quality,
            "renderer_memory_bundle": memory_bundle,
            "reference_patch_material_present": bool(execution_trace_summary.get("reference_patch_material_present", False)),
            "reference_patch_material_validated": bool(execution_trace_summary.get("reference_patch_material_validated", False)),
            "reference_patch_material_trusted": bool(execution_trace_summary.get("reference_patch_material_trusted", False)),
            "reference_patch_material_used": bool(execution_trace_summary.get("reference_patch_material_used", False)),
            "reference_patch_material_shape": execution_trace_summary.get("reference_patch_material_shape", []),
            "reference_patch_material_missing_reason": str(execution_trace_summary.get("reference_patch_material_missing_reason", "")),
            "reference_tensor_input_used": bool(execution_trace_summary.get("reference_tensor_input_used", False)),
            "reference_tensor_zero_fallback": bool(execution_trace_summary.get("reference_tensor_zero_fallback", True)),
            "memory_retrieval_evidence": _memory_summary(memory_bundle, request.retrieval_summary),
            "execution_trace_summary": execution_trace_summary,
        }
        planner_selected = trace.get("planner_selected_strategy") or ctx.get("planner_selected_strategy")
        if planner_selected is not None:
            record["planner_selected_strategy"] = _safe_json(planner_selected)
        safe = _safe_json(record)
        validate_renderer_manifest_v2_record(safe, strict=True)
        return safe

    def write_manifest(
        self,
        records: list[dict[str, object]],
        path: str,
        *,
        manifest_type: str = "renderer_patch_manifest",
    ) -> None:
        safe_records = [_safe_json(record) for record in records]
        target_quality_counts = {
            "self_generated_runtime_target": sum(
                1 for record in safe_records if isinstance(record, dict) and record.get("training_target_quality") == "self_generated_runtime_target"
            ),
            "external_or_observed_target": sum(
                1 for record in safe_records if isinstance(record, dict) and record.get("training_target_quality") == "external_or_observed_target"
            ),
        }
        payload = {
            "manifest_type": manifest_type,
            "source": "runtime_patch_export",
            "contract_version": "renderer_patch_manifest_v2",
            "record_count": len(safe_records),
            "target_quality_counts": target_quality_counts,
            "contains_self_generated_targets": target_quality_counts["self_generated_runtime_target"] > 0,
            "contains_external_targets": target_quality_counts["external_or_observed_target"] > 0,
            "records": safe_records,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_hw3_list(value: object, field_name: str) -> list[list[list[float]]]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{field_name} must be HxWx3-compatible, got shape={list(arr.shape)}")
    return np.clip(arr, 0.0, 1.0).tolist()


def _as_hw1_list(value: object, field_name: str) -> list[list[list[float]]]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.mean(arr, axis=2, keepdims=True)
    if arr.ndim != 3 or arr.shape[2] != 1:
        raise ValueError(f"{field_name} must be HxW/HxWx1-compatible, got shape={list(arr.shape)}")
    return np.clip(arr, 0.0, 1.0).tolist()


def _alpha_target(output: PatchSynthesisOutput, before: list[list[list[float]]]) -> list[list[list[float]]]:
    if output.alpha_mask is not None:
        return _as_hw1_list(output.alpha_mask, "output.alpha_mask")
    h = len(before)
    w = len(before[0]) if h else 0
    return np.ones((h, w, 1), dtype=np.float32).tolist()


def _changed_mask(before: list[list[list[float]]], after: list[list[list[float]]]) -> list[list[list[float]]]:
    b = np.asarray(before, dtype=np.float32)
    a = np.asarray(after, dtype=np.float32)
    if b.shape == a.shape:
        return np.clip(np.mean(np.abs(a - b), axis=2, keepdims=True) * 3.0, 0.0, 1.0).tolist()
    h = min(b.shape[0], a.shape[0]) if b.ndim == 3 and a.ndim == 3 else 0
    w = min(b.shape[1], a.shape[1]) if b.ndim == 3 and a.ndim == 3 else 0
    return np.zeros((h, w, 1), dtype=np.float32).tolist()


def _memory_bundle_serialized(ctx: dict[str, object]) -> dict[str, object]:
    serialized = ctx.get("region_memory_bundle_serialized")
    if isinstance(serialized, dict) and serialized:
        return _safe_json(serialized)
    bundle = ctx.get("region_memory_bundle")
    if bundle is not None:
        if is_dataclass(bundle):
            data = asdict(bundle)
        elif isinstance(bundle, dict):
            data = bundle
        else:
            data = {"memory_bundle_present": True, "memory_support_level": str(getattr(bundle, "memory_support_level", "none"))}
        return _safe_json(data)
    return {"memory_bundle_present": False, "memory_support_level": "none", "memory_support_level_reason": "no_runtime_region_memory_bundle"}



def _fallback_reference_material_summary(ctx: dict[str, object], trace: dict[str, object]) -> dict[str, object]:
    material = ctx.get("expected_reference_patch_material") if isinstance(ctx, dict) else None
    expected_payload = ctx.get("expected_reference_payload") if isinstance(ctx, dict) else None
    shape = _shape_of(material.get("rgb_patch")) if isinstance(material, dict) else []
    validated = bool(
        isinstance(material, dict)
        and bool(material.get("material_trusted", False))
        and len(shape) == 3
        and shape[2] == 3
        and shape[0] > 0
        and shape[1] > 0
    )
    payload_present = isinstance(expected_payload, dict)
    payload_safe = bool(
        payload_present
        and bool(expected_payload.get("observed_directly", False))
        and not bool(expected_payload.get("generated", False))
        and not bool(expected_payload.get("inferred", False))
    )
    trusted = bool(validated and payload_safe)
    used = bool(trusted)
    if used:
        missing_reason = ""
    elif material is None:
        missing_reason = "material_missing"
    elif not validated:
        missing_reason = str(material.get("material_missing_reason", "material_untrusted") or "material_untrusted")
    elif not payload_safe:
        missing_reason = "expected_reference_payload_missing" if not payload_present else "payload_untrusted"
    else:
        missing_reason = "material_untrusted"
    fallback = {
        "reference_patch_material_present": isinstance(material, dict),
        "reference_patch_material_validated": validated,
        "reference_patch_material_trusted": trusted,
        "reference_patch_material_used": used,
        "reference_patch_material_missing_reason": missing_reason,
    }
    if isinstance(material, dict):
        fallback["reference_patch_material_source"] = str(material.get("material_source", "unknown"))
        fallback["reference_patch_material_shape"] = shape
    return fallback

def _execution_trace_summary(
    *,
    trace: dict[str, object],
    ctx: dict[str, object],
    memory_bundle: dict[str, object],
    selected_render_strategy: str,
    renderer_path: str,
    synthesis_mode: str,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "selected_render_strategy": selected_render_strategy,
        "renderer_path": renderer_path,
        "synthesis_mode": synthesis_mode,
    }
    planner_selected = trace.get("planner_selected_strategy") or ctx.get("planner_selected_strategy")
    if planner_selected is not None:
        summary["planner_selected_strategy"] = _safe_json(planner_selected)

    memory_defaults = {
        "memory_bundle_present": False,
        "memory_support_level": "none",
        "memory_bundle_has_current_reuse": False,
        "memory_bundle_has_identity_reference": False,
        "memory_bundle_has_appearance_reference": False,
        "memory_bundle_has_garment_reference": False,
        "memory_bundle_has_hidden_slot": False,
        "memory_bundle_hidden_type": "none",
        "memory_bundle_hidden_support_active": False,
        "memory_bundle_reveal_lifecycle": "unknown",
        "memory_bundle_retrieval_reasons": [],
    }
    summary.update(memory_defaults)
    summary.update(
        {
            "memory_bundle_present": bool(memory_bundle.get("memory_bundle_present", bool(memory_bundle))),
            "memory_support_level": str(memory_bundle.get("memory_support_level", "none")),
            "memory_bundle_has_current_reuse": bool(memory_bundle.get("has_current_reuse", memory_bundle.get("memory_bundle_has_current_reuse", False))),
            "memory_bundle_has_identity_reference": bool(memory_bundle.get("has_identity_reference", memory_bundle.get("memory_bundle_has_identity_reference", False))),
            "memory_bundle_has_appearance_reference": bool(memory_bundle.get("has_appearance_reference", memory_bundle.get("memory_bundle_has_appearance_reference", False))),
            "memory_bundle_has_garment_reference": bool(memory_bundle.get("has_garment_reference", memory_bundle.get("memory_bundle_has_garment_reference", False))),
            "memory_bundle_has_hidden_slot": bool(memory_bundle.get("has_hidden_slot", memory_bundle.get("memory_bundle_has_hidden_slot", False))),
            "memory_bundle_hidden_type": str(memory_bundle.get("hidden_type", memory_bundle.get("memory_bundle_hidden_type", "none"))),
            "memory_bundle_hidden_support_active": bool(memory_bundle.get("hidden_support_active", memory_bundle.get("memory_bundle_hidden_support_active", False))),
            "memory_bundle_reveal_lifecycle": str(memory_bundle.get("reveal_lifecycle", memory_bundle.get("memory_bundle_reveal_lifecycle", "unknown"))),
            "memory_bundle_retrieval_reasons": _safe_json(memory_bundle.get("retrieval_reasons", memory_bundle.get("memory_bundle_retrieval_reasons", []))),
        }
    )
    hidden_slot = memory_bundle.get("hidden_slot")
    if isinstance(hidden_slot, dict):
        summary["memory_bundle_hidden_type"] = str(hidden_slot.get("hidden_type", summary["memory_bundle_hidden_type"]))

    for key in (
        "expected_reference_payload_kind",
        "reference_patch_material_present",
        "reference_patch_material_validated",
        "reference_patch_material_trusted",
        "reference_patch_material_source",
        "reference_patch_material_shape",
        "reference_patch_material_missing_reason",
        "reference_patch_material_used",
        "i2v_reference_contract_version",
        "reference_material_from_input_frame",
        "reference_material_from_generated_frame",
        "reference_patch_material_source_frame_kind",
        "reference_patch_material_source_frame_index",
        "reference_patch_material_immutable_i2v_anchor",
        "reference_patch_material_source_is_input_frame",
        "reference_tensor_input_used",
        "reference_tensor_zero_fallback",
        "reference_tensor_input_channels",
        "local_tensor_input_channels",
        "material_gate_mean",
        "material_gate_max",
        "material_gate_cap",
        "reference_validity_mean",
        "reference_mask_mean",
        "material_consistency_loss",
        "material_gate_regularization",
        "material_gate_preservation_penalty",
        "material_gate_invalidity_penalty",
        "material_gate_area_penalty",
        "material_gate_suppressed_by_preservation",
    ):
        if key in trace:
            summary[key] = _safe_json(trace[key])
    expected_payload = ctx.get("expected_reference_payload")
    if isinstance(expected_payload, dict):
        summary.setdefault("expected_reference_payload_kind", str(expected_payload.get("reference_kind", "")))
    material = ctx.get("expected_reference_patch_material")
    fallback_material = _fallback_reference_material_summary(ctx, trace)
    for key, value in fallback_material.items():
        summary.setdefault(key, _safe_json(value))
    if isinstance(material, dict):
        summary.setdefault("reference_patch_material_patch_id", str(material.get("source_patch_id", "")))
        descriptor = material.get("descriptor")
        if isinstance(descriptor, dict):
            summary.setdefault("reference_patch_material_descriptor_keys", sorted(str(k) for k in descriptor.keys()))
    elif "reference_patch_material_present" not in summary:
        summary["reference_patch_material_present"] = False
        reasons = ctx.get("reference_patch_material_trace_reasons", [])
        if isinstance(reasons, list) and reasons:
            summary["reference_patch_material_missing_reason"] = str(reasons[0])

    for key in ("confidence_semantics_by_mode", "uncertainty_semantics_by_mode"):
        if isinstance(trace.get(key), dict):
            summary[key] = _safe_json(trace[key])
    learnable = trace.get("learnable_mode_surface")
    if isinstance(learnable, dict):
        summary["learnable_mode_surface"] = {"keys": sorted(str(k) for k in learnable.keys())}
    return _safe_json(summary)


def _renderer_batch_contract(
    *,
    ctx: dict[str, object],
    trace: dict[str, object],
    memory_bundle: dict[str, object],
    alpha_target: list[list[list[float]]],
    blend_hint: list[list[list[float]]],
    changed_mask: list[list[list[float]]],
) -> dict[str, object]:
    existing = ctx.get("renderer_batch_contract") if isinstance(ctx.get("renderer_batch_contract"), dict) else {}
    contract: dict[str, object] = {}
    for key in (
        "semantic_embed",
        "delta_cond",
        "planner_cond",
        "graph_cond",
        "memory_cond",
        "appearance_cond",
        "bbox_cond",
        "conditioning_summary",
    ):
        if key in existing:
            contract[key] = _safe_json(existing[key])
        elif key in ctx:
            contract[key] = _safe_json(ctx[key])
        elif key in trace:
            contract[key] = _safe_json(trace[key])
    contract["alpha_target"] = alpha_target
    contract["blend_hint"] = blend_hint
    contract["changed_mask"] = changed_mask
    contract["region_memory_bundle_serialized"] = memory_bundle
    if isinstance(ctx.get("expected_reference_payload"), dict):
        contract["expected_reference_payload"] = _safe_json(ctx.get("expected_reference_payload"))
    material = ctx.get("expected_reference_patch_material")
    if isinstance(material, dict):
        contract["expected_reference_patch_material"] = _safe_json(material)
    elif "expected_reference_patch_material" in ctx:
        contract["expected_reference_patch_material"] = None
    if "reference_patch_material_trace_reasons" in ctx:
        contract["reference_patch_material_trace_reasons"] = _safe_json(ctx.get("reference_patch_material_trace_reasons"))
    for key in (
        "reference_patch_material_present",
        "reference_patch_material_validated",
        "reference_patch_material_trusted",
        "reference_patch_material_used",
        "reference_patch_material_shape",
        "reference_patch_material_missing_reason",
        "i2v_reference_contract_version",
        "reference_material_from_input_frame",
        "reference_material_from_generated_frame",
        "reference_patch_material_source_frame_kind",
        "reference_patch_material_source_frame_index",
        "reference_patch_material_immutable_i2v_anchor",
        "reference_patch_material_source_is_input_frame",
        "reference_tensor_input_used",
        "reference_tensor_zero_fallback",
        "reference_tensor_input_channels",
        "local_tensor_input_channels",
        "material_gate_mean",
        "material_gate_max",
        "material_gate_cap",
        "reference_validity_mean",
        "reference_mask_mean",
        "material_consistency_loss",
        "material_gate_regularization",
        "material_gate_preservation_penalty",
        "material_gate_invalidity_penalty",
        "material_gate_area_penalty",
        "material_gate_suppressed_by_preservation",
    ):
        if key in trace:
            contract[key] = _safe_json(trace[key])
    if "conditioning_summary" not in contract:
        summary = trace.get("conditioning_summary") or ctx.get("conditioning_summary")
        if summary is not None:
            contract["conditioning_summary"] = _safe_json(summary)
    return contract


def _compact_reference_material_summary(material: object) -> dict[str, object] | None:
    if not isinstance(material, dict):
        return None
    source_frame_kind = str(material.get("source_frame_kind", "unknown") or "unknown")
    source_is_input_frame = bool(material.get("source_is_input_frame", False))
    immutable_i2v_anchor = bool(material.get("immutable_i2v_anchor", False))
    generated = bool(material.get("generated", False))
    inferred = bool(material.get("inferred", False))
    from_input = bool(source_is_input_frame and immutable_i2v_anchor and source_frame_kind == "observed_input_frame")
    from_generated = bool(source_frame_kind == "generated_runtime_frame" or generated or inferred or not source_is_input_frame)
    descriptor = material.get("descriptor")
    return {
        "reference_kind": str(material.get("reference_kind", "")),
        "canonical_region": str(material.get("canonical_region", "")),
        "entity_id": str(material.get("entity_id", "")),
        "source_patch_id": material.get("source_patch_id"),
        "source_patch_ref": material.get("source_patch_ref"),
        "material_source": str(material.get("material_source", "unknown")),
        "material_trusted": bool(material.get("material_trusted", False)),
        "material_missing_reason": str(material.get("material_missing_reason", "")),
        "i2v_reference_contract_version": str(material.get("i2v_reference_contract_version", "i2v_first_frame_reference_v1")),
        "reference_material_from_input_frame": from_input,
        "reference_material_from_generated_frame": from_generated,
        "source_frame_kind": source_frame_kind,
        "source_frame_index": int(material.get("source_frame_index", 0) or 0),
        "immutable_i2v_anchor": immutable_i2v_anchor,
        "source_is_input_frame": source_is_input_frame,
        "rgb_patch_shape": _shape_of(material.get("rgb_patch")),
        "descriptor_keys": sorted(str(k) for k in descriptor.keys()) if isinstance(descriptor, dict) else [],
    }


def _transition_context_summary(ctx: dict[str, object], *, step_index: int | None, memory_bundle: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key in ("transition_phase", "target_profile", "semantic_families", "region_route_decision", "expected_reference_payload", "reference_patch_material_trace_reasons"):
        if key in ctx:
            summary[key] = _safe_json(ctx[key])
    if "expected_reference_patch_material" in ctx:
        summary["expected_reference_patch_material"] = _safe_json(_compact_reference_material_summary(ctx.get("expected_reference_patch_material")))
    summary["step_index"] = int(step_index if step_index is not None else ctx.get("step_index", -1)) if (step_index is not None or "step_index" in ctx) else None
    summary["memory_bundle_present"] = bool(memory_bundle.get("memory_bundle_present", bool(memory_bundle)))
    summary["memory_support_level"] = str(memory_bundle.get("memory_support_level", "none"))
    if "retrieval_reasons" in memory_bundle:
        summary["region_memory_retrieval_reasons"] = _safe_json(memory_bundle.get("retrieval_reasons", []))
    if "region_memory_support_level" in ctx:
        summary["region_memory_support_level"] = _safe_json(ctx["region_memory_support_level"])
    if "region_memory_retrieval_reasons" in ctx:
        summary["region_memory_retrieval_reasons"] = _safe_json(ctx["region_memory_retrieval_reasons"])
    graph_delta = ctx.get("graph_delta") or ctx.get("delta_contract")
    if graph_delta is not None:
        summary["graph_delta_summary"] = _graph_delta_summary(graph_delta)
    return summary


def _graph_delta_summary(delta: object) -> dict[str, object]:
    if delta is None:
        return {}
    if is_dataclass(delta):
        data = asdict(delta)
    elif isinstance(delta, dict):
        data = delta
    else:
        return {"type": type(delta).__name__, "summary": str(delta)[:240]}
    summary: dict[str, object] = {}
    for key in (
        "transition_phase",
        "affected_entities",
        "affected_regions",
        "semantic_reasons",
        "region_transition_mode",
        "state_before",
        "state_after",
        "predicted_visibility_changes",
    ):
        if key in data:
            summary[key] = _safe_json(data[key])
    for key in ("pose_deltas", "garment_deltas", "expression_deltas", "interaction_deltas", "visibility_deltas"):
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, dict) and value:
            summary[key] = _safe_json(value)
    return summary


def _infer_semantic_family(request: PatchSynthesisRequest, ctx: dict[str, object]) -> str:
    for key in ("semantic_family", "region_family", "transition_family"):
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_family(value)
    region_id = request.region.region_id.lower()
    reason = request.region.reason.lower()
    text = f"{region_id} {reason}"
    if "face" in text or "head" in text or "expression" in text:
        return "face_expression"
    if "torso" in text or "inner" in text:
        return "torso_reveal"
    return "sleeve_arm_transition"


def _normalize_family(value: str) -> str:
    v = value.strip().lower()
    aliases = {
        "expression_transition": "face_expression",
        "visibility_transition": "torso_reveal",
        "garment_transition": "sleeve_arm_transition",
        "pose_transition": "sleeve_arm_transition",
        "interaction_transition": "sleeve_arm_transition",
    }
    return aliases.get(v, v)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_json(value: Any) -> Any:
    if is_dataclass(value):
        return _safe_json(asdict(value))
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return _safe_json(value.detach().cpu().tolist())
    return {"type": type(value).__name__, "summary": str(value)[:240]}

RENDERER_PATCH_MANIFEST_V2 = "renderer_patch_manifest_v2"
RENDERER_MANIFEST_V2_REQUIRED_FIELDS = (
    "record_id",
    "frame_index",
    "step_index",
    "region_id",
    "canonical_region",
    "entity_id",
    "roi_before",
    "roi_after",
    "alpha_mask",
    "region_metadata",
    "transition_context_summary",
    "selected_render_strategy",
    "synthesis_mode",
    "execution_trace_summary",
    "metadata_completeness_score",
    "evidence_strength_score",
    "roi_source",
    "source_node_type",
    "mask_kind",
    "mask_ref_present",
)


def validate_renderer_manifest_v2_record(record: dict[str, object], *, strict: bool = True) -> list[str]:
    """Validate the non-optional v2 patch-renderer training record surface."""

    issues: list[str] = []
    if record.get("contract_version") != RENDERER_PATCH_MANIFEST_V2:
        issues.append("contract_version must be renderer_patch_manifest_v2")
    for field in RENDERER_MANIFEST_V2_REQUIRED_FIELDS:
        if field not in record:
            issues.append(f"missing required field: {field}")
    metadata = record.get("region_metadata")
    if not isinstance(metadata, dict) or not metadata:
        issues.append("region_metadata must be a non-empty object")
    elif "metadata_completeness_score" not in metadata:
        issues.append("region_metadata.metadata_completeness_score missing")
    if "metadata_completeness_score" not in record:
        issues.append("metadata_completeness_score missing")
    try:
        json.dumps(record)
    except TypeError as exc:
        issues.append(f"record is not JSON serializable: {exc}")
    if issues and strict:
        raise ValueError("invalid renderer manifest v2 record: " + "; ".join(issues))
    return issues


def _region_metadata_v2(request: PatchSynthesisRequest) -> dict[str, object]:
    metadata = dict(request.region_metadata) if isinstance(request.region_metadata, dict) and request.region_metadata else {}
    metadata.setdefault("region_id", request.region.region_id)
    metadata.setdefault("entity_id", _entity_id(request.region.region_id))
    metadata.setdefault("canonical_region", _region_type(request.region.region_id))
    metadata.setdefault("bbox_xywh", [float(request.region.bbox.x), float(request.region.bbox.y), float(request.region.bbox.w), float(request.region.bbox.h)])
    metadata.setdefault("roi_reason", request.region.reason)
    metadata.setdefault("roi_source", "unknown")
    metadata.setdefault("source_node_type", "unknown")
    metadata.setdefault("mask_kind", "")
    metadata.setdefault("metadata_completeness_score", 0.0)
    metadata.setdefault("evidence_strength_score", 0.0)
    metadata.setdefault("missing_fields", ["region_metadata"] if not request.region_metadata else [])
    return _safe_json(metadata)


def _entity_id(region_id: str) -> str:
    return str(region_id).split(":", 1)[0] if ":" in str(region_id) else "unknown"


def _region_type(region_id: str) -> str:
    return str(region_id).split(":", 1)[1] if ":" in str(region_id) else str(region_id)


def _record_id(request: PatchSynthesisRequest, frame_index: int | None, step_index: int | None) -> str:
    ctx = request.transition_context if isinstance(request.transition_context, dict) else {}
    for key in ("record_id", "patch_record_id", "runtime_record_id"):
        if ctx.get(key):
            return str(ctx[key])
    return f"renderer_patch:{frame_index if frame_index is not None else ctx.get('frame_index', -1)}:{step_index if step_index is not None else ctx.get('step_index', -1)}:{request.region.region_id}:{uuid4().hex[:8]}"


def _mask_from_contract_or_none(ctx: dict[str, object], trace: dict[str, object], key: str) -> object:
    contract = ctx.get("renderer_batch_contract") if isinstance(ctx.get("renderer_batch_contract"), dict) else {}
    value = contract.get(key) if isinstance(contract, dict) else None
    if value is None:
        value = ctx.get(key, trace.get(key))
    return _safe_json(value) if value is not None else None


def _shape_of(value: object) -> list[int]:
    try:
        return [int(x) for x in np.asarray(value).shape]
    except Exception:
        return []


def _memory_summary(memory_bundle: dict[str, object], retrieval_summary: object) -> dict[str, object]:
    retrieval = retrieval_summary if isinstance(retrieval_summary, dict) else {}
    return _safe_json(
        {
            "memory_support_level": str(memory_bundle.get("memory_support_level", "none")),
            "reliable_for_reuse": bool(memory_bundle.get("reliable_for_reuse", memory_bundle.get("has_current_reuse", False))),
            "suitable_for_reveal": bool(memory_bundle.get("suitable_for_reveal", memory_bundle.get("has_hidden_slot", False))),
            "retrieval_reasons": memory_bundle.get("retrieval_reasons", retrieval.get("reasons", [])),
            "retrieval_summary": retrieval,
        }
    )
