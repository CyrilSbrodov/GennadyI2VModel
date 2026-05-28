from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

_REQUIRED_PROVENANCE = {
    "target_source": "provided_ground_truth_graph_transition",
    "training_target_quality": "external_or_observed_graph_transition",
    "target_training_role": "supervised_dynamics_external",
}
_FORBIDDEN_SOURCES = {"self_generated_runtime_target", "heuristic_runtime_prediction", "legacy_graph_delta_fallback"}

@dataclass(slots=True)
class DynamicsObservedBuildOutput:
    manifest_path: str
    diagnostics: dict[str, object]

def _inc(m: dict[str, int], key: str) -> None:
    m[key] = int(m.get(key, 0)) + 1

def _first_person(graph: dict[str, object]) -> dict[str, object]:
    persons = graph.get("persons", []) if isinstance(graph.get("persons", []), list) else []
    return persons[0] if persons and isinstance(persons[0], dict) else {}

def _derive_target_delta(graph_before: dict[str, object], graph_after: dict[str, object], family: str, affected_regions: list[str]) -> dict[str, object]:
    before_p = _first_person(graph_before)
    after_p = _first_person(graph_after)
    expression_deltas: dict[str, float] = {}
    visibility_deltas: dict[str, str] = {}

    b_expr = before_p.get("expression_state", {}) if isinstance(before_p.get("expression_state", {}), dict) else {}
    a_expr = after_p.get("expression_state", {}) if isinstance(after_p.get("expression_state", {}), dict) else {}
    b_smile = float(b_expr.get("smile_intensity", 0.0))
    a_smile = float(a_expr.get("smile_intensity", 0.0))
    if abs(a_smile - b_smile) > 1e-6:
        expression_deltas["smile_intensity"] = a_smile - b_smile

    def _vis_map(p: dict[str, object]) -> dict[str, str]:
        out: dict[str, str] = {}
        for bp in p.get("body_parts", []) if isinstance(p.get("body_parts", []), list) else []:
            if isinstance(bp, dict):
                out[str(bp.get("part_id", bp.get("part_type", "part")))] = str(bp.get("visibility", "unknown"))
        return out
    b_vis, a_vis = _vis_map(before_p), _vis_map(after_p)
    for k, av in a_vis.items():
        bv = b_vis.get(k)
        if bv is not None and bv != av:
            visibility_deltas[k] = av

    if not affected_regions:
        pid = str(after_p.get("person_id", before_p.get("person_id", "person_1")))
        affected_regions = [f"{pid}:face"] if expression_deltas else [f"{pid}:body"]

    return {
        "expression_deltas": expression_deltas,
        "visibility_deltas": visibility_deltas,
        "affected_entities": [str(after_p.get("person_id", before_p.get("person_id", "person_1")))],
        "affected_regions": affected_regions,
        "semantic_reasons": [family],
        "region_transition_mode": {r: "deform" for r in affected_regions},
    }

def _has_delta_groups(td: dict[str, object]) -> bool:
    return any(bool(td.get(k)) for k in ("pose_deltas", "garment_deltas", "visibility_deltas", "expression_deltas", "interaction_deltas"))

def build_dynamics_manifest_from_observed_transitions(observed_transitions_path: str, output_path: str, strict: bool = True) -> DynamicsObservedBuildOutput:
    payload = json.loads(Path(observed_transitions_path).read_text(encoding="utf-8"))
    if payload.get("contract_version") != "dynamics_observed_transition_manifest_input_v1":
        raise ValueError("invalid contract_version for observed transitions")
    transitions = payload.get("transitions", [])
    diagnostics: dict[str, object] = {"total_records": len(transitions), "exported_records": 0, "skipped_records": 0, "skipped_by_reason": {}, "transition_family_counts": {}, "affected_region_counts": {}, "rejected_runtime_or_heuristic_targets": 0, "strict": bool(strict)}
    records: list[dict[str, object]] = []
    for idx, rec in enumerate(transitions):
        try:
            if not isinstance(rec, dict): raise ValueError("record must be object")
            gb = rec.get("graph_before"); ga = rec.get("graph_after"); td = rec.get("target_delta")
            if not isinstance(gb, dict): raise ValueError("missing graph_before")
            if not isinstance(ga, dict) and not isinstance(td, dict): raise ValueError("missing graph_after and target_delta")
            for k, v in _REQUIRED_PROVENANCE.items():
                if rec.get(k) != v: raise ValueError(f"invalid {k}")
            if str(rec.get("target_source")) in _FORBIDDEN_SOURCES:
                diagnostics["rejected_runtime_or_heuristic_targets"] = int(diagnostics["rejected_runtime_or_heuristic_targets"]) + 1
                raise ValueError("forbidden target_source")
            ctx = rec.get("transition_context", {})
            if not isinstance(ctx, dict) or not str(ctx.get("family", "")).strip(): raise ValueError("missing transition_context.family")
            affected = rec.get("affected_regions", [])
            if affected is not None and not isinstance(affected, list): raise ValueError("affected_regions must be list")
            family = str(ctx.get("family"))
            if not isinstance(td, dict):
                td = _derive_target_delta(gb, ga, family, [str(x) for x in (affected or [])])
            if not _has_delta_groups(td):
                raise ValueError("empty_target_delta_groups")
            out = {"record_id": str(rec.get("record_id", f"observed_{idx}")), "graph_before": gb, "graph_after": ga if isinstance(ga, dict) else {}, "target_delta": td, "prompt": str(rec.get("prompt", "")), "transition_context": ctx, "affected_regions": [str(x) for x in (affected or td.get("affected_regions", []))], **_REQUIRED_PROVENANCE, "diagnostics": {"source_contract": "dynamics_observed_transition_manifest_input_v1"}}
            _inc(diagnostics["transition_family_counts"], family)
            for r in out["affected_regions"]: _inc(diagnostics["affected_region_counts"], r)
            records.append(out)
        except Exception as exc:
            if strict: raise
            diagnostics["skipped_records"] = int(diagnostics["skipped_records"]) + 1
            _inc(diagnostics["skipped_by_reason"], str(exc))
    if not records: raise ValueError("no valid supervised dynamics observed transitions")
    diagnostics["exported_records"] = len(records)
    manifest = {"contract_version": "dynamics_graph_delta_manifest_v1", "manifest_type": "dynamics_graph_delta_manifest", "record_count": len(records), "records": records, "diagnostics": diagnostics}
    out_path = Path(output_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return DynamicsObservedBuildOutput(manifest_path=str(out_path), diagnostics=diagnostics)
