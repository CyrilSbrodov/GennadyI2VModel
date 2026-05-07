from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

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
        patch_synthesis_contract = {
            "region_metadata": {"region_id": request.region.region_id, "reason": request.region.reason},
            "retrieval_explanation_summary": _safe_json(request.retrieval_summary),
            "selected_render_strategy": selected_render_strategy,
            "synthesis_mode": synthesis_mode,
            "transition_context": transition_summary,
            "target_source": target_source,
            "training_target_quality": training_target_quality,
        }

        record: dict[str, object] = {
            "roi_before": before,
            "roi_after": after,
            "region_id": request.region.region_id,
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
        }
        if step_index is not None:
            record["step_index"] = int(step_index)
        if frame_index is not None:
            record["frame_index"] = int(frame_index)
        planner_selected = trace.get("planner_selected_strategy") or ctx.get("planner_selected_strategy")
        if planner_selected is not None:
            record["planner_selected_strategy"] = _safe_json(planner_selected)
        return _safe_json(record)

    def write_manifest(
        self,
        records: list[dict[str, object]],
        path: str,
        *,
        manifest_type: str = "renderer_patch_manifest",
    ) -> None:
        payload = {
            "manifest_type": manifest_type,
            "source": "runtime_patch_export",
            "records": [_safe_json(record) for record in records],
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
    if "conditioning_summary" not in contract:
        summary = trace.get("conditioning_summary") or ctx.get("conditioning_summary")
        if summary is not None:
            contract["conditioning_summary"] = _safe_json(summary)
    return contract


def _transition_context_summary(ctx: dict[str, object], *, step_index: int | None, memory_bundle: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key in ("transition_phase", "target_profile", "semantic_families", "region_route_decision"):
        if key in ctx:
            summary[key] = _safe_json(ctx[key])
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
