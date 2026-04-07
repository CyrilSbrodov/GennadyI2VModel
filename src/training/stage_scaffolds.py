from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.schema import ActionStep, BBox, GraphDelta, PersonNode, RegionRef, SceneGraph
from dynamics.state_update import apply_delta
from evaluation.contracts import (
    build_graph_eval_payload,
    build_hidden_reconstruction_payload,
    build_patch_eval_payload,
    build_temporal_eval_payload,
    build_text_eval_payload,
    graph_transition_eval,
    hidden_region_reconstruction_eval,
    patch_synthesis_eval,
    temporal_consistency_eval,
    text_action_alignment_eval,
)
from learned.factory import BackendBundle, BackendConfig, LearnedBackendFactory
from learned.interfaces import DynamicsTransitionRequest, PatchSynthesisRequest, TemporalRefinementRequest
from memory.video_memory import MemoryManager
from training.learned_contracts import (
    build_graph_transition_contract,
    build_patch_synthesis_contract,
    build_temporal_consistency_contract,
    build_text_action_state_contract,
)


@dataclass(slots=True)
class StageScaffoldConfig:
    stage_name: str
    model_backend: str = "baseline"
    dataset_path: str = ""
    batch_size: int = 2
    learning_rate: float = 1e-4
    epochs: int = 1
    checkpoint_path: str = "artifacts/checkpoints/stage.ckpt"
    backend_config: BackendConfig | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StageScaffoldResult:
    stage_name: str
    backend: str
    checkpoint_path: str
    expected_inputs: list[str]
    expected_outputs: list[str]
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    samples_processed: int = 0


class LearnedStageDatasetRouter:
    @staticmethod
    def _coerce_bbox(data: object, fallback: BBox | None = None) -> BBox:
        if isinstance(data, dict):
            return BBox(float(data.get("x", 0.1)), float(data.get("y", 0.1)), float(data.get("w", 0.6)), float(data.get("h", 0.8)))
        return fallback or BBox(0.1, 0.1, 0.6, 0.8)

    @staticmethod
    def _coerce_region(region_id: str, payload: object, reason: str = "motion") -> RegionRef:
        bbox = LearnedStageDatasetRouter._coerce_bbox(payload if isinstance(payload, dict) else {})
        resolved_reason = reason
        if isinstance(payload, dict):
            resolved_reason = str(payload.get("reason", reason))
            region_id = str(payload.get("region_id", region_id))
        return RegionRef(region_id=region_id, bbox=bbox, reason=resolved_reason)

    @staticmethod
    def _coerce_action_steps(payload: object, fallback_token: str = "micro_adjust") -> list[ActionStep]:
        if isinstance(payload, list) and payload:
            steps: list[ActionStep] = []
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                steps.append(
                    ActionStep(
                        type=str(item.get("type", fallback_token)),
                        priority=int(item.get("priority", idx + 1)),
                        target_entity=item.get("target_entity"),
                        target_object=item.get("target_object"),
                        body_part=item.get("body_part"),
                        can_run_parallel=bool(item.get("can_run_parallel", False)),
                        start_after=[int(v) for v in item.get("start_after", []) if isinstance(v, int)],
                        constraints=[str(c) for c in item.get("constraints", []) if isinstance(c, str)],
                    )
                )
            if steps:
                return steps
        return [ActionStep(type=fallback_token, priority=1)]

    @staticmethod
    def _coerce_graph(record: dict[str, object], key: str, frame_index: int, track: str) -> SceneGraph:
        raw = record.get(key)
        if isinstance(raw, dict):
            persons: list[PersonNode] = []
            for idx, p in enumerate(raw.get("persons", []) if isinstance(raw.get("persons"), list) else []):
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("person_id", f"p{track}_{idx}"))
                persons.append(PersonNode(person_id=pid, track_id=str(p.get("track_id", f"t{track}_{idx}")), bbox=LearnedStageDatasetRouter._coerce_bbox(p.get("bbox")), mask_ref=p.get("mask_ref")))
            graph = SceneGraph(frame_index=int(raw.get("frame_index", frame_index)), persons=persons)
            return graph
        return LearnedStageDatasetRouter._base_graph(frame_index, track)

    @staticmethod
    def _coerce_delta(record: dict[str, object], before: SceneGraph, action_tokens: list[str], phase: str = "motion") -> GraphDelta:
        payload = record.get("delta", {})
        if isinstance(payload, dict):
            revealed = [LearnedStageDatasetRouter._coerce_region(region_id=f"{before.persons[0].person_id}:revealed_{idx}", payload=item, reason="reveal") for idx, item in enumerate(payload.get("newly_revealed_regions", [])) if isinstance(item, dict)] if before.persons else []
            occluded = [LearnedStageDatasetRouter._coerce_region(region_id=f"{before.persons[0].person_id}:occluded_{idx}", payload=item, reason="occlude") for idx, item in enumerate(payload.get("newly_occluded_regions", [])) if isinstance(item, dict)] if before.persons else []
            return GraphDelta(
                pose_deltas={str(k): float(v) for k, v in payload.get("pose_deltas", {}).items()} if isinstance(payload.get("pose_deltas"), dict) else {},
                interaction_deltas={str(k): float(v) for k, v in payload.get("interaction_deltas", {}).items()} if isinstance(payload.get("interaction_deltas"), dict) else {},
                semantic_reasons=[str(v) for v in payload.get("semantic_reasons", action_tokens)],
                affected_entities=[str(v) for v in payload.get("affected_entities", [before.persons[0].person_id] if before.persons else [])],
                affected_regions=[str(v) for v in payload.get("affected_regions", [])],
                region_transition_mode={str(k): str(v) for k, v in payload.get("region_transition_mode", {}).items()} if isinstance(payload.get("region_transition_mode"), dict) else {},
                predicted_visibility_changes={str(k): str(v) for k, v in payload.get("predicted_visibility_changes", {}).items()} if isinstance(payload.get("predicted_visibility_changes"), dict) else {},
                state_before={str(k): str(v) for k, v in payload.get("state_before", {}).items()} if isinstance(payload.get("state_before"), dict) else {},
                state_after={str(k): str(v) for k, v in payload.get("state_after", {}).items()} if isinstance(payload.get("state_after"), dict) else {},
                transition_phase=str(payload.get("transition_phase", phase)),
                newly_revealed_regions=revealed,
                newly_occluded_regions=occluded,
            )
        return GraphDelta(
            pose_deltas={"torso_pitch": 0.04},
            semantic_reasons=action_tokens or ["micro_adjust"],
            affected_entities=[before.persons[0].person_id] if before.persons else [],
            affected_regions=["torso"],
            region_transition_mode={"torso": "deform"},
            state_before={"pose_phase": "stable"},
            state_after={"pose_phase": "transition"},
            transition_phase=phase,
        )

    @staticmethod
    def _base_graph(frame_index: int, track: str, with_relation: bool = False) -> SceneGraph:
        person = PersonNode(person_id=f"p{track}", track_id=f"t{track}", bbox=BBox(0.1, 0.1, 0.7, 0.8), mask_ref=None)
        graph = SceneGraph(frame_index=frame_index, persons=[person])
        if with_relation:
            graph.relations = []
        return graph

    @staticmethod
    def _synthetic_text(index: int) -> dict[str, object]:
        actions = [
            "sit down",
            "wave hand then turn left",
            "stand up while holding balance constraint",
            "turn left and raise right hand in parallel",
            "micro adjust torso with stability constraint",
        ]
        steps = [
            ActionStep(type="sit_down", priority=1),
            ActionStep(type="wave", priority=1, body_part="right_hand", target_entity="person_1", start_after=[], can_run_parallel=False),
            ActionStep(type="stand_up", priority=1, constraints=["keep_balance"]),
            ActionStep(type="turn", priority=1, start_after=[], can_run_parallel=True, target_object="left_side"),
            ActionStep(type="micro_adjust", priority=1, constraints=["preserve_identity"], body_part="torso"),
        ]
        idx = index % len(actions)
        return {"text": actions[idx], "actions": [steps[idx]]}

    @staticmethod
    def _synthetic_dynamics(index: int) -> dict[str, object]:
        before = LearnedStageDatasetRouter._base_graph(index, str(index))
        token = ["sit_down", "wave", "turn", "stand_up", "garment_change", "micro_adjust"][index % 6]
        region_pool = {
            "sit_down": ["torso", "hips"],
            "wave": ["right_hand", "right_arm"],
            "turn": ["torso", "left_arm"],
            "stand_up": ["torso", "legs"],
            "garment_change": ["coat", "torso"],
            "micro_adjust": ["neck"],
        }
        delta = GraphDelta(
            pose_deltas={"torso_pitch": 0.08 * (index + 1)},
            interaction_deltas={"chair_contact": 1.0 if token == "sit_down" else 0.2},
            semantic_reasons=[token],
            affected_entities=[before.persons[0].person_id],
            affected_regions=region_pool[token],
            region_transition_mode={region: ("appearance" if "coat" in region else "motion") for region in region_pool[token]},
            predicted_visibility_changes={"torso": "visible"},
            state_before={"pose_phase": "stable"},
            state_after={"pose_phase": "transition", "gesture_phase": "executing"},
            transition_phase=["prepare", "motion", "settle"][index % 3],
        )
        return {"graph_before": before, "text_tokens": [token], "delta": delta}

    @staticmethod
    def _synthetic_patch(index: int) -> dict[str, object]:
        graph = LearnedStageDatasetRouter._base_graph(index, str(index))
        reasons = ["face_expression", "garment_adjustment", "arm_motion", "occlusion", "hidden_reveal", "known_hidden_refresh"]
        rid = ["face", "coat", "right_hand", "left_arm", "torso", "left_leg"][index % 6]
        region = RegionRef(region_id=f"{graph.persons[0].person_id}:{rid}", bbox=BBox(0.2, 0.1 + 0.05 * (index % 3), 0.25, 0.2), reason=reasons[index % len(reasons)])
        v = 0.2 + 0.05 * index
        frame = [[[v + ((x + y) % 5) * 0.01, 0.2, 0.1] for x in range(32)] for y in range(32)]
        hidden = {
            "state": "emerging" if index % 2 else "stable",
            "slot": region.region_id,
            "hidden_type": "unknown_hidden" if index % 2 else "known_hidden",
            "retrieval_profile": "rich" if index % 3 else "poor",
            "lifecycle": ["known_hidden", "unknown_hidden_synthesis", "known_hidden_reveal"][index % 3],
        }
        return {"region": region, "graph": graph, "frame": frame, "hidden_state": hidden, "retrieval_summary": {"profile": hidden["retrieval_profile"]}}

    @staticmethod
    def _synthetic_temporal(index: int) -> dict[str, object]:
        graph = LearnedStageDatasetRouter._base_graph(index, str(index))
        roi_count = 1 + (index % 3)
        regions = [
            RegionRef(
                region_id=f"{graph.persons[0].person_id}:region_{r}",
                bbox=BBox(0.15 + 0.1 * r, 0.1 + 0.05 * (index % 2), 0.2, 0.2),
                reason="temporal_drift" if r else "pose_update",
            )
            for r in range(roi_count)
        ]
        base = 0.25 + 0.03 * index
        prev = [[[base, base, base] for _ in range(32)] for _ in range(32)]
        cur = [[[min(1.0, base + 0.02 + (x % 3) * 0.005), base, base] for x in range(32)] for _ in range(32)]
        return {
            "graph": graph,
            "frame_prev": prev,
            "frame_cur": cur,
            "regions": regions,
            "drift": [0.01, 0.05, 0.11][index % 3],
            "temporal_profile": ["reveal", "occlusion", "multi_roi_sync"][index % 3],
        }

    @staticmethod
    def _from_dataset_path(dataset_path: str, stage_name: str, size: int) -> list[dict[str, object]]:
        path = Path(dataset_path)
        if not path.exists():
            return []
        payload = json.loads(path.read_text())
        records = payload.get("records", []) if isinstance(payload, dict) else []
        filtered = [r for r in records if isinstance(r, dict) and (not stage_name or r.get("stage") in {None, stage_name})][:size]
        return [LearnedStageDatasetRouter._parse_stage_record(stage_name, rec, idx) for idx, rec in enumerate(filtered)]

    @staticmethod
    def _parse_stage_record(stage_name: str, record: dict[str, object], index: int) -> dict[str, object]:
        if stage_name == "text_encoder":
            text = str(record.get("text", record.get("prompt", f"micro adjust #{index}")))
            actions = LearnedStageDatasetRouter._coerce_action_steps(record.get("actions"), fallback_token="micro_adjust")
            return {"text": text, "actions": actions, "metadata": record.get("metadata", {})}
        if stage_name == "dynamics_transition":
            before = LearnedStageDatasetRouter._coerce_graph(record, "graph_before", frame_index=index, track=f"dataset_{index}")
            text_tokens = [a.type for a in LearnedStageDatasetRouter._coerce_action_steps(record.get("actions"), fallback_token="micro_adjust")]
            delta = LearnedStageDatasetRouter._coerce_delta(record, before=before, action_tokens=text_tokens, phase=str(record.get("transition_phase", "motion")))
            return {"graph_before": before, "text_tokens": text_tokens, "delta": delta, "metadata": record.get("metadata", {})}
        if stage_name == "patch_synthesis":
            graph = LearnedStageDatasetRouter._coerce_graph(record, "graph", frame_index=index, track=f"dataset_{index}")
            region_data = record.get("region", {})
            region_id = f"{graph.persons[0].person_id if graph.persons else 'scene'}:{record.get('region_name', 'torso')}"
            region = LearnedStageDatasetRouter._coerce_region(region_id, region_data, reason=str(record.get("reason", "motion")))
            frame = record.get("frame")
            if not isinstance(frame, list):
                base = 0.2 + 0.03 * index
                frame = [[[base, base, base] for _ in range(32)] for _ in range(32)]
            return {
                "region": region,
                "graph": graph,
                "frame": frame,
                "hidden_state": record.get("hidden_state", {}),
                "retrieval_summary": record.get("retrieval_summary", {"source": "dataset_manifest"}),
                "metadata": record.get("metadata", {}),
            }
        if stage_name == "temporal_refinement":
            graph = LearnedStageDatasetRouter._coerce_graph(record, "graph", frame_index=index, track=f"dataset_{index}")
            regions_raw = record.get("regions")
            if isinstance(regions_raw, list):
                regions = [LearnedStageDatasetRouter._coerce_region(f"{graph.persons[0].person_id if graph.persons else 'scene'}:region_{i}", rr, reason="temporal_drift") for i, rr in enumerate(regions_raw) if isinstance(rr, dict)]
            else:
                regions = []
            if not regions:
                regions = [RegionRef(region_id=f"{graph.persons[0].person_id if graph.persons else 'scene'}:region_0", bbox=BBox(0.2, 0.2, 0.2, 0.2), reason="temporal_drift")]
            frame_prev = record.get("frame_prev") if isinstance(record.get("frame_prev"), list) else [[[0.2, 0.2, 0.2] for _ in range(32)] for _ in range(32)]
            frame_cur = record.get("frame_cur") if isinstance(record.get("frame_cur"), list) else [[[0.23, 0.2, 0.2] for _ in range(32)] for _ in range(32)]
            return {"graph": graph, "frame_prev": frame_prev, "frame_cur": frame_cur, "regions": regions, "drift": float(record.get("drift", 0.04)), "temporal_profile": str(record.get("temporal_profile", "dataset"))}
        return record

    @staticmethod
    def build(stage_name: str, size: int = 3, dataset_path: str = "") -> list[dict[str, object]]:
        if dataset_path:
            loaded = LearnedStageDatasetRouter._from_dataset_path(dataset_path, stage_name, size)
            if loaded:
                return loaded
        if stage_name == "text_encoder":
            return [LearnedStageDatasetRouter._synthetic_text(i) for i in range(size)]
        if stage_name == "dynamics_transition":
            return [LearnedStageDatasetRouter._synthetic_dynamics(i) for i in range(size)]
        if stage_name == "patch_synthesis":
            return [LearnedStageDatasetRouter._synthetic_patch(i) for i in range(size)]
        if stage_name == "temporal_refinement":
            return [LearnedStageDatasetRouter._synthetic_temporal(i) for i in range(size)]
        raise ValueError(f"Unknown learned stage {stage_name}")


class _BaseStageRunner:
    def __init__(self, backend: str = "baseline", backend_config: BackendConfig | None = None, backends: BackendBundle | None = None) -> None:
        self.backend = backend
        self.backends = backends or LearnedBackendFactory(backend_config or BackendConfig()).build()

    def _write_checkpoint(self, config: StageScaffoldConfig, payload: dict[str, object]) -> str:
        out = Path(config.checkpoint_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return str(out)


def _validate_contract_fields(payload: dict[str, object], required_fields: list[str]) -> list[str]:
    return [name for name in required_fields if name not in payload]


def _text_semantic_checks(contract: dict[str, object]) -> list[str]:
    issues: list[str] = []
    parsed = contract.get("parsed_actions", [])
    embedding = contract.get("action_embedding", [])
    if not parsed:
        issues.append("structured_action_tokens_empty")
    if parsed and not embedding:
        issues.append("embedding_empty_for_non_empty_actions")
    if isinstance(parsed, list) and len(parsed) > 1 and not contract.get("temporal_decomposition"):
        issues.append("decomposition_hints_missing_for_multi_action_input")
    has_targets = isinstance(parsed, list) and any(isinstance(a, dict) and (a.get("target_entity") or a.get("target_object")) for a in parsed)
    if has_targets and not (contract.get("target_entities") or contract.get("target_objects")):
        issues.append("target_hints_missing_despite_action_targets")
    return issues


class TextEncoderStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("text_encoder", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        text_parity_log: list[dict[str, object]] = []
        for sample in samples:
            encoded = self.backends.text_encoder.encode(str(sample["text"]))
            actions = sample.get("actions") or [ActionStep(type="micro_adjust", priority=1)]
            contract = build_text_action_state_contract(str(sample["text"]), actions, vars(encoded))
            last_contract = contract
            eval_payload = build_text_eval_payload(contract)
            scores.append(text_action_alignment_eval(eval_payload).metrics["alignment_score"])
            missing = _validate_contract_fields(contract, ["text", "parsed_actions", "action_embedding", "target_entities", "target_objects", "temporal_decomposition", "constraints"])
            semantic_text = _text_semantic_checks(contract)
            text_parity_log.append({"missing_fields": missing, "semantic_issues": semantic_text})
        ckpt = self._write_checkpoint(
            config,
            {
                "stage_name": "text_encoder",
                "backend": self.backends.backend_names.get("text_encoder", self.backend),
                "backend_config": self.backends.backend_names,
                "schema_version": "learned_ready.v2",
                "contract_version": "text_action_state.v1",
                "train_metrics": {"progress": 1.0, "score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "val_metrics": {"alignment_score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "samples_processed": len(samples),
                "expected_inputs": ["text", "scene_graph(optional)", "action_plan(optional)"],
                "expected_outputs": ["action_embedding", "structured_action_tokens", "alignment"],
                "contract_payload_shape": {"keys": sorted(last_contract.keys())},
                "eval_summary": {"text_alignment_score": (sum(scores) / len(scores)) if scores else 0.0, "text_parity": text_parity_log[-1] if text_parity_log else {}},
            },
        )
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="text_encoder",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["text", "scene_graph(optional)", "action_plan(optional)"],
            expected_outputs=["action_embedding", "structured_action_tokens", "alignment"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract, "parity": text_parity_log},
            samples_processed=len(samples),
        )


class DynamicsTransitionStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("dynamics_transition", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        for idx, sample in enumerate(samples):
            before = sample["graph_before"]
            memory = mm.initialize(before)
            text = self.backends.text_encoder.encode(" ".join(sample.get("text_tokens", ["micro_adjust"])))
            graph_enc = self.backends.graph_encoder.encode(before)
            entity_id = before.persons[0].person_id if before.persons else "scene"
            request = DynamicsTransitionRequest(
                graph_state=before,
                memory_summary={},
                memory_channels={},
                text_action_summary=text,
                graph_encoding=graph_enc,
                identity_embeddings={entity_id: [0.05 * (idx + 1)] * 8},
                step_context={"step_index": idx + 1, "memory": memory},
            )
            out = self.backends.dynamics_backend.predict_transition(request)
            predicted_after = out.metadata.get("predicted_graph_after") if isinstance(out.metadata, dict) else None
            if not predicted_after:
                from copy import deepcopy
                predicted_after = apply_delta(deepcopy(before), out.delta)
            contract = build_graph_transition_contract(before, predicted_after, out.delta, {"step_index": idx, "diagnostics": out.diagnostics})
            last_contract = contract
            eval_payload = build_graph_eval_payload(contract)
            scores.append(graph_transition_eval(eval_payload).metrics["transition_correctness"])
        ckpt = self._write_checkpoint(
            config,
            {
                "stage_name": "dynamics_transition",
                "backend": self.backends.backend_names.get("dynamics_backend", self.backend),
                "backend_config": self.backends.backend_names,
                "schema_version": "learned_ready.v2",
                "contract_version": "graph_transition.v2",
                "train_metrics": {"progress": 1.0, "score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "val_metrics": {"transition_correctness_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "samples_processed": len(samples),
                "expected_inputs": ["graph_state", "memory_summary", "text_action_summary", "step_context"],
                "expected_outputs": ["graph_delta", "confidence", "transition_metadata"],
                "contract_payload_shape": {"keys": sorted(last_contract.keys())},
                "eval_summary": {"transition_correctness": (sum(scores) / len(scores)) if scores else 0.0},
            },
        )
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="dynamics_transition",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["graph_state", "memory_summary", "text_action_summary", "step_context"],
            expected_outputs=["graph_delta", "confidence", "transition_metadata"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract},
            samples_processed=len(samples),
        )


class PatchSynthesisStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("patch_synthesis", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        for idx, sample in enumerate(samples):
            memory = mm.initialize(sample["graph"])
            graph_enc = self.backends.graph_encoder.encode(sample["graph"])
            req = PatchSynthesisRequest(
                region=sample["region"],
                scene_state=sample["graph"],
                memory_summary={},
                transition_context={"graph_delta": GraphDelta(affected_entities=["p1"], affected_regions=[sample["region"].region_id], region_transition_mode={sample["region"].region_id: "motion"}), "video_memory": memory},
                retrieval_summary=sample.get("retrieval_summary", {"stage": "patch", "case": idx}),
                current_frame=sample["frame"],
                memory_channels={"identity": {"requested": True}, "garments": {}, "hidden_regions": sample.get("hidden_state", {})},
                graph_encoding=graph_enc,
                identity_embedding=[0.1 + 0.01 * idx] * 8,
            )
            out = self.backends.patch_backend.synthesize_patch(req)
            contract = build_patch_synthesis_contract(sample["frame"], out.rgb_patch, sample["region"], "baseline", str(out.execution_trace.get("selected_render_strategy", "unknown")), sample.get("hidden_state", {}), str(out.execution_trace.get("synthesis_mode", "deterministic")), req.transition_context)
            last_contract = contract
            eval_payload = build_patch_eval_payload(contract)
            patch_eval = patch_synthesis_eval(eval_payload)
            hidden_eval = hidden_region_reconstruction_eval(build_hidden_reconstruction_payload(contract))
            scores.append((patch_eval.metrics["patch_quality"] + hidden_eval.metrics["reconstruction_quality"]) / 2.0)
        ckpt = self._write_checkpoint(
            config,
            {
                "stage_name": "patch_synthesis",
                "backend": self.backends.backend_names.get("patch_backend", self.backend),
                "backend_config": self.backends.backend_names,
                "schema_version": "learned_ready.v2",
                "contract_version": "patch_synthesis.v2",
                "train_metrics": {"progress": 1.0, "score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "val_metrics": {"patch_quality_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "samples_processed": len(samples),
                "expected_inputs": ["region", "scene_state", "memory_summary", "transition_context", "retrieval_summary"],
                "expected_outputs": ["rgb_patch", "confidence", "uncertainty_map"],
                "contract_payload_shape": {"keys": sorted(last_contract.keys())},
                "eval_summary": {"composite_patch_hidden_score": (sum(scores) / len(scores)) if scores else 0.0},
            },
        )
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="patch_synthesis",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["region", "scene_state", "memory_summary", "transition_context", "retrieval_summary"],
            expected_outputs=["rgb_patch", "confidence", "uncertainty_map"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract, "hidden_reconstruction_payload": build_hidden_reconstruction_payload(last_contract)},
            samples_processed=len(samples),
        )


class TemporalRefinementStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("temporal_refinement", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        for idx, sample in enumerate(samples):
            memory = mm.initialize(sample["graph"])
            req = TemporalRefinementRequest(
                previous_frame=sample["frame_prev"],
                current_composed_frame=sample["frame_cur"],
                changed_regions=sample["regions"],
                scene_state=sample["graph"],
                memory_state=memory,
                memory_channels={"identity": {}, "body_regions": {"roi_count": len(sample["regions"])}, "hidden_regions": {"drift": sample["drift"]}},
            )
            out = self.backends.temporal_backend.refine_temporal(req)
            contract = build_temporal_consistency_contract(
                sample["frame_prev"],
                sample["frame_cur"],
                out.refined_frame,
                sample["regions"],
                {"scores": out.region_consistency_scores, "temporal_drift": sample["drift"], "roi_count": len(sample["regions"])},
                {"stage": "temporal", "step_index": idx + 1},
                {"channels": list(req.memory_channels.keys())},
            )
            last_contract = contract
            eval_payload = build_temporal_eval_payload(contract)
            scores.append(temporal_consistency_eval(eval_payload).metrics["temporal_consistency"])
        ckpt = self._write_checkpoint(
            config,
            {
                "stage_name": "temporal_refinement",
                "backend": self.backends.backend_names.get("temporal_backend", self.backend),
                "backend_config": self.backends.backend_names,
                "schema_version": "learned_ready.v2",
                "contract_version": "temporal_consistency.v2",
                "train_metrics": {"progress": 1.0, "score_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "val_metrics": {"temporal_consistency_mean": (sum(scores) / len(scores)) if scores else 0.0},
                "samples_processed": len(samples),
                "expected_inputs": ["previous_frame", "current_composed_frame", "changed_regions", "scene_state", "memory_state"],
                "expected_outputs": ["refined_frame", "region_consistency_scores"],
                "contract_payload_shape": {"keys": sorted(last_contract.keys())},
                "eval_summary": {"temporal_consistency": (sum(scores) / len(scores)) if scores else 0.0},
            },
        )
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="temporal_refinement",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["previous_frame", "current_composed_frame", "changed_regions", "scene_state", "memory_state"],
            expected_outputs=["refined_frame", "region_consistency_scores"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract},
            samples_processed=len(samples),
        )


def build_stage_runner(stage_name: str, backend: str = "baseline", backend_config: BackendConfig | None = None, backends: BackendBundle | None = None):
    mapping = {
        "text_encoder": TextEncoderStageRunner,
        "dynamics_transition": DynamicsTransitionStageRunner,
        "patch_synthesis": PatchSynthesisStageRunner,
        "temporal_refinement": TemporalRefinementStageRunner,
    }
    if stage_name not in mapping:
        known = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown learned stage {stage_name}. Known: {known}")
    return mapping[stage_name](backend=backend, backend_config=backend_config, backends=backends)
