from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.schema import ActionStep, BBox, GraphDelta, PersonNode, RegionRef, SceneGraph
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
from learned.factory import BackendConfig, LearnedBackendFactory
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
    def build(stage_name: str, size: int = 3) -> list[dict[str, object]]:
        graph_before = SceneGraph(frame_index=0, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)])
        graph_after = SceneGraph(frame_index=1, persons=[PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.8, 0.8), mask_ref=None)])
        region = RegionRef(region_id="p1:face", bbox=BBox(0.2, 0.1, 0.2, 0.2), reason="contract")
        frame = [[[0.3, 0.2, 0.1] for _ in range(32)] for _ in range(32)]
        if stage_name == "text_encoder":
            return [{"text": "sit down", "actions": [ActionStep(type="sit_down", priority=1)]} for _ in range(size)]
        if stage_name == "dynamics_transition":
            return [{"graph_before": graph_before, "graph_after": graph_after, "text_tokens": ["sit_down"]} for _ in range(size)]
        if stage_name == "patch_synthesis":
            return [{"region": region, "graph": graph_before, "frame": frame} for _ in range(size)]
        if stage_name == "temporal_refinement":
            return [{"graph": graph_before, "frame_prev": frame, "frame_cur": frame, "region": region} for _ in range(size)]
        raise ValueError(f"Unknown learned stage {stage_name}")


class _BaseStageRunner:
    def __init__(self, backend: str = "baseline") -> None:
        self.backend = backend
        self.backends = LearnedBackendFactory(BackendConfig()).build()

    def _write_checkpoint(self, config: StageScaffoldConfig, payload: dict[str, object]) -> str:
        out = Path(config.checkpoint_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return str(out)


class TextEncoderStageRunner(_BaseStageRunner):
    def run(self, config: StageScaffoldConfig) -> StageScaffoldResult:
        samples = LearnedStageDatasetRouter.build("text_encoder", size=max(1, config.batch_size))
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        for sample in samples:
            encoded = self.backends.text_encoder.encode(sample["text"])
            contract = build_text_action_state_contract(sample["text"], sample["actions"], vars(encoded))
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
        samples = LearnedStageDatasetRouter.build("dynamics_transition", size=max(1, config.batch_size))
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        for idx, sample in enumerate(samples):
            memory = mm.initialize(sample["graph_before"])
            text = self.backends.text_encoder.encode(" ".join(sample["text_tokens"]))
            graph_enc = self.backends.graph_encoder.encode(sample["graph_before"])
            request = DynamicsTransitionRequest(
                graph_state=sample["graph_before"],
                memory_summary={},
                memory_channels={},
                text_action_summary=text,
                graph_encoding=graph_enc,
                identity_embeddings={"p1": [0.0] * 8},
                step_context={"step_index": idx + 1, "memory": memory},
            )
            out = self.backends.dynamics_backend.predict_transition(request)
            contract = build_graph_transition_contract(sample["graph_before"], sample["graph_after"], out.delta, {"step_index": idx})
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
        samples = LearnedStageDatasetRouter.build("patch_synthesis", size=max(1, config.batch_size))
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        for sample in samples:
            memory = mm.initialize(sample["graph"])
            graph_enc = self.backends.graph_encoder.encode(sample["graph"])
            req = PatchSynthesisRequest(
                region=sample["region"],
                scene_state=sample["graph"],
                memory_summary={},
                transition_context={"graph_delta": GraphDelta(affected_entities=["p1"], affected_regions=["face"]), "video_memory": memory},
                retrieval_summary={"stage": "patch"},
                current_frame=sample["frame"],
                memory_channels={"identity": {}, "garments": {}, "hidden_regions": {}},
                graph_encoding=graph_enc,
                identity_embedding=[0.0] * 8,
            )
            out = self.backends.patch_backend.synthesize_patch(req)
            contract = build_patch_synthesis_contract(sample["frame"], out.rgb_patch, sample["region"], "baseline", str(out.execution_trace.get("selected_render_strategy", "unknown")), {}, "deterministic", req.transition_context)
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
        samples = LearnedStageDatasetRouter.build("temporal_refinement", size=max(1, config.batch_size))
        mm = MemoryManager()
        scores: list[float] = []
        last_contract: dict[str, object] = {}
        for sample in samples:
            memory = mm.initialize(sample["graph"])
            req = TemporalRefinementRequest(
                previous_frame=sample["frame_prev"],
                current_composed_frame=sample["frame_cur"],
                changed_regions=[sample["region"]],
                scene_state=sample["graph"],
                memory_state=memory,
                memory_channels={"identity": {}, "body_regions": {}, "hidden_regions": {}},
            )
            out = self.backends.temporal_backend.refine_temporal(req)
            contract = build_temporal_consistency_contract(
                sample["frame_prev"],
                sample["frame_cur"],
                out.refined_frame,
                [sample["region"]],
                {"scores": out.region_consistency_scores},
                {"stage": "temporal"},
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


def build_stage_runner(stage_name: str, backend: str = "baseline"):
    mapping = {
        "text_encoder": TextEncoderStageRunner,
        "dynamics_transition": DynamicsTransitionStageRunner,
        "patch_synthesis": PatchSynthesisStageRunner,
        "temporal_refinement": TemporalRefinementStageRunner,
    }
    if stage_name not in mapping:
        known = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown learned stage {stage_name}. Known: {known}")
    return mapping[stage_name](backend=backend)
