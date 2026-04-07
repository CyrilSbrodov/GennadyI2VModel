from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.schema import ActionStep, BBox, GraphDelta, PersonNode, RegionRef, SceneGraph
from dynamics.state_update import apply_delta
from evaluation.contracts import (
    build_graph_eval_payload,
    build_patch_eval_payload,
    build_temporal_eval_payload,
    build_text_eval_payload,
    graph_transition_eval,
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
    def _base_graph(frame_index: int, track: str, with_relation: bool = False) -> SceneGraph:
        person = PersonNode(person_id=f"p{track}", track_id=f"t{track}", bbox=BBox(0.1, 0.1, 0.7, 0.8), mask_ref=None)
        graph = SceneGraph(frame_index=frame_index, persons=[person])
        if with_relation:
            graph.relations = []
        return graph

    @staticmethod
    def _synthetic_text(index: int) -> dict[str, object]:
        actions = ["sit down", "wave hand", "stand up", "turn left"]
        steps = [
            ActionStep(type="sit_down", priority=1),
            ActionStep(type="wave", priority=1, body_part="right_hand"),
            ActionStep(type="stand_up", priority=1),
            ActionStep(type="turn", priority=1),
        ]
        idx = index % len(actions)
        return {"text": actions[idx], "actions": [steps[idx]]}

    @staticmethod
    def _synthetic_dynamics(index: int) -> dict[str, object]:
        before = LearnedStageDatasetRouter._base_graph(index, str(index))
        token = ["sit_down", "wave", "turn", "stand_up"][index % 4]
        delta = GraphDelta(
            pose_deltas={"torso_pitch": 0.08 * (index + 1)},
            interaction_deltas={"chair_contact": 1.0 if token == "sit_down" else 0.2},
            semantic_reasons=[token],
            affected_entities=[before.persons[0].person_id],
            affected_regions=["torso", "right_hand"] if token in {"wave", "turn"} else ["torso"],
            region_transition_mode={"torso": "deform", "right_hand": "motion"},
            predicted_visibility_changes={"torso": "visible"},
            state_before={"pose_phase": "stable"},
            state_after={"pose_phase": "transition"},
            transition_phase="motion",
        )
        return {"graph_before": before, "text_tokens": [token], "delta": delta}

    @staticmethod
    def _synthetic_patch(index: int) -> dict[str, object]:
        graph = LearnedStageDatasetRouter._base_graph(index, str(index))
        reasons = ["face_expression", "garment_adjustment", "arm_motion", "occlusion"]
        rid = ["face", "coat", "right_hand", "left_arm"][index % 4]
        region = RegionRef(region_id=f"{graph.persons[0].person_id}:{rid}", bbox=BBox(0.2, 0.1 + 0.05 * (index % 3), 0.25, 0.2), reason=reasons[index % 4])
        v = 0.2 + 0.05 * index
        frame = [[[v + ((x + y) % 5) * 0.01, 0.2, 0.1] for x in range(32)] for y in range(32)]
        hidden = {"state": "emerging" if index % 2 else "stable", "slot": region.region_id}
        return {"region": region, "graph": graph, "frame": frame, "hidden_state": hidden}

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
        return {"graph": graph, "frame_prev": prev, "frame_cur": cur, "regions": regions, "drift": 0.02 + 0.01 * index}

    @staticmethod
    def _from_dataset_path(dataset_path: str, stage_name: str, size: int) -> list[dict[str, object]]:
        path = Path(dataset_path)
        if not path.exists():
            return []
        payload = json.loads(path.read_text())
        records = payload.get("records", []) if isinstance(payload, dict) else []
        return [r for r in records if isinstance(r, dict) and (not stage_name or r.get("stage") in {None, stage_name})][:size]

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


class TextEncoderStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("text_encoder", size=max(1, config.batch_size), dataset_path=config.dataset_path)
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        for sample in samples:
            encoded = self.backends.text_encoder.encode(str(sample["text"]))
            actions = sample.get("actions") or [ActionStep(type="micro_adjust", priority=1)]
            contract = build_text_action_state_contract(str(sample["text"]), actions, vars(encoded))
            last_contract = contract
            eval_payload = build_text_eval_payload(contract)
            scores.append(text_action_alignment_eval(eval_payload).metrics["alignment_score"])
        ckpt = self._write_checkpoint(config, {"stage": "text_encoder", "scores": scores})
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="text_encoder",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["text", "scene_graph(optional)", "action_plan(optional)"],
            expected_outputs=["action_embedding", "structured_action_tokens", "alignment"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract},
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
        ckpt = self._write_checkpoint(config, {"stage": "dynamics_transition", "scores": scores})
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
                retrieval_summary={"stage": "patch", "case": idx},
                current_frame=sample["frame"],
                memory_channels={"identity": {"requested": True}, "garments": {}, "hidden_regions": sample.get("hidden_state", {})},
                graph_encoding=graph_enc,
                identity_embedding=[0.1 + 0.01 * idx] * 8,
            )
            out = self.backends.patch_backend.synthesize_patch(req)
            contract = build_patch_synthesis_contract(sample["frame"], out.rgb_patch, sample["region"], "baseline", str(out.execution_trace.get("selected_render_strategy", "unknown")), sample.get("hidden_state", {}), str(out.execution_trace.get("synthesis_mode", "deterministic")), req.transition_context)
            last_contract = contract
            eval_payload = build_patch_eval_payload(contract)
            scores.append(patch_synthesis_eval(eval_payload).metrics["patch_quality"])
        ckpt = self._write_checkpoint(config, {"stage": "patch_synthesis", "scores": scores})
        mean = sum(scores) / len(scores)
        return StageScaffoldResult(
            stage_name="patch_synthesis",
            backend=self.backend,
            checkpoint_path=ckpt,
            expected_inputs=["region", "scene_state", "memory_summary", "transition_context", "retrieval_summary"],
            expected_outputs=["rgb_patch", "confidence", "uncertainty_map"],
            train_metrics={"progress": 1.0, "loss": max(0.0, 1.0 - mean)},
            val_metrics={"score": mean, "contract_payload": last_contract},
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
        ckpt = self._write_checkpoint(config, {"stage": "temporal_refinement", "scores": scores})
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
