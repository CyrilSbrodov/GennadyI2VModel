from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.input_layer import InputAssetLayer
from core.schema import SceneGraph
from dynamics.state_update import apply_delta
from learned.factory import BackendBundle, BackendConfig, LearnedBackendFactory
from learned.interfaces import DynamicsTransitionRequest, PatchSynthesisRequest, TemporalRefinementRequest
from learned.parity import (
    text_output_to_contract,
    dynamics_io_to_contract,
    patch_io_to_contract,
    semantic_parity_checks,
    temporal_io_to_contract,
    validate_parity,
)
from memory.summaries import AppearanceMemorySummarizer
from memory.video_memory import MemoryManager
from perception.pipeline import PerceptionPipeline
from planning.transition_engine import StatePlan, TransitionPlanner
from rendering.compositor import Compositor
from rendering.roi_renderer import ROISelector, RenderedPatch
from representation.graph_builder import SceneGraphBuilder
from representation.learned_bridge import summarize_memory
from runtime.profiles import PROFILES, RuntimeProfile
from text.intent_parser import IntentParser
from utils_tensor import shape, zeros


@dataclass(slots=True)
class InferenceArtifacts:
    frames: list[list]
    scene_graphs: list[SceneGraph]
    state_plan: StatePlan
    debug: dict[str, list[str] | str | dict[str, object] | list[dict[str, object]]] = field(default_factory=dict)


class GennadyEngine:
    def __init__(self, backend_config: BackendConfig | None = None, backend_bundle: BackendBundle | None = None) -> None:
        self.input_layer = InputAssetLayer()
        self.perception = PerceptionPipeline()
        self.graph_builder = SceneGraphBuilder()
        self.memory_manager = MemoryManager()
        self.intent_parser = IntentParser()
        self.planner = TransitionPlanner()
        self.roi_selector = ROISelector()
        self.compositor = Compositor()
        self.memory_summarizer = AppearanceMemorySummarizer()
        self.backend_config = backend_config or BackendConfig()
        self.backends = backend_bundle or LearnedBackendFactory(self.backend_config).build()

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

    def run(
        self,
        images: list[str],
        text: str,
        fps: int = 16,
        duration: float = 4.0,
        quality_profile: str = "balanced",
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
        current_frame = first_frame.tensor if first_frame else self._debug_seed_frame_tensor(profile)
        perception_input = first_frame if first_frame else current_frame
        perception_output = self.perception.analyze(perception_input)
        perception_output.frame_size = (shape(current_frame)[1], shape(current_frame)[0])
        scene_graph = self.graph_builder.build(perception_output, frame_index=0)
        scene_graph.global_context.fps = fps
        scene_graph.global_context.frame_size = perception_output.frame_size
        scene_graph.global_context.source_type = request.input_type

        memory = self.memory_manager.initialize_from_scene(scene_graph)
        graph_encoding = self.backends.graph_encoder.encode(scene_graph)
        fallback_log: list[str] = []

        action_plan = self.intent_parser.parse(request.text, scene_graph=scene_graph)
        text_encoding = self.backends.text_encoder.encode(request.text, scene_graph=scene_graph, action_plan=action_plan)
        text_contract = text_output_to_contract(request.text, text_encoding)
        text_missing = validate_parity(text_contract, ["text", "parsed_actions", "action_embedding", "target_entities", "target_objects", "temporal_decomposition", "constraints"])
        text_semantic = semantic_parity_checks(stage="text", contract=text_contract, request={"text": request.text, "actions": action_plan.actions}, output=text_encoding)
        if text_missing:
            fallback_log.append(f"step=0:text_parity_missing={text_missing}")
        if text_semantic:
            fallback_log.extend([f"step=0:text_semantic={x}" for x in text_semantic])
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

        frames = [current_frame]
        graphs = [scene_graph]
        overlay_log: list[str] = []
        dynamics_metrics_log: list[str] = []
        channel_usage_log: list[dict[str, object]] = []
        step_debug: list[dict[str, object]] = []
        hidden_recon_stats = {"known_hidden": 0, "unknown_hidden": 0, "hidden_reveal": 0, "steps_with_hidden_reconstruction": 0}

        for planned_state in state_plan.steps[1 : profile.max_transition_steps + 1]:
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
                step_context={"step_index": planned_state.step_index, "memory": memory},
            )
            transition_output = self.backends.dynamics_backend.predict_transition(transition_request)
            parity_contract = dynamics_io_to_contract(transition_request, transition_output)
            missing_fields = validate_parity(parity_contract, ["graph_before", "graph_after", "delta_contract", "transition_context"])
            semantic_dynamics = semantic_parity_checks(stage="dynamics", contract=parity_contract, request=transition_request, output=transition_output)
            if missing_fields:
                fallback_log.append(f"step={planned_state.step_index}:dynamics_parity_missing={missing_fields}")
            if semantic_dynamics:
                for issue in semantic_dynamics:
                    label = "trace" if issue.endswith("_trace") else "dynamics_semantic"
                    fallback_log.append(f"step={planned_state.step_index}:{label}={issue}")

            delta = transition_output.delta
            changed_regions = self.roi_selector.select(scene_graph, delta)
            patches: list[RenderedPatch] = []
            patch_step_debug: list[dict[str, object]] = []
            step_hidden_reconstruction = False
            for region in changed_regions[: profile.max_roi_count]:
                patch_channels = self.build_patch_memory_channels(memory_channels)
                patch_request = PatchSynthesisRequest(
                    region=region,
                    scene_state=scene_graph,
                    memory_summary=memory_summary,
                    transition_context={"graph_delta": delta, "video_memory": memory},
                    retrieval_summary={"backend": "deterministic", "identity_entity": entity_id},
                    current_frame=current_frame,
                    memory_channels=patch_channels,
                    graph_encoding=graph_encoding,
                    identity_embedding=identity_embedding,
                )
                patch_out = self.backends.patch_backend.synthesize_patch(patch_request)
                patch_contract = patch_io_to_contract(patch_request, patch_out)
                missing_patch_fields = validate_parity(
                    patch_contract,
                    ["roi_before", "roi_after", "region_metadata", "selected_strategy", "transition_context"],
                )
                semantic_patch = semantic_parity_checks(stage="patch", contract=patch_contract, request=patch_request, output=patch_out)
                strategy = str(patch_out.execution_trace.get("selected_render_strategy", ""))
                synth_mode = str(patch_out.execution_trace.get("synthesis_mode", "deterministic"))
                if "KNOWN_HIDDEN_REVEAL" in strategy:
                    hidden_recon_stats["known_hidden"] += 1
                    hidden_recon_stats["hidden_reveal"] += 1
                    step_hidden_reconstruction = True
                elif "UNKNOWN_HIDDEN_SYNTHESIS" in strategy:
                    hidden_recon_stats["unknown_hidden"] += 1
                    step_hidden_reconstruction = True
                if missing_patch_fields:
                    fallback_log.append(f"step={planned_state.step_index}:patch_parity_missing={missing_patch_fields}")
                if semantic_patch:
                    fallback_log.extend([f"step={planned_state.step_index}:patch_semantic={x}" for x in semantic_patch])
                patch_step_debug.append(
                    {
                        "region_id": region.region_id,
                        "selected_strategy": patch_contract.get("selected_strategy", "unknown"),
                        "confidence": patch_out.confidence,
                        "synthesis_mode": synth_mode,
                        "retrieval_summary": str(patch_request.retrieval_summary)[:120],
                        "learned_ready": patch_out.metadata.get("learned_ready_usage", {}),
                        "hidden_reconstruction": {
                            "is_case": step_hidden_reconstruction,
                            "strategy": strategy,
                            "mode": synth_mode,
                            "quality_hint": patch_out.execution_trace.get("patch_refinement_strength", 0.0),
                        },
                        "missing_fields": missing_patch_fields,
                        "semantic_issues": semantic_patch,
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
            temporal_request = TemporalRefinementRequest(
                previous_frame=frames[-1],
                current_composed_frame=composed,
                changed_regions=[p.region for p in patches],
                scene_state=scene_graph,
                memory_state=memory,
                memory_channels=temporal_channels,
            )
            temporal_out = self.backends.temporal_backend.refine_temporal(temporal_request)
            temporal_contract = temporal_io_to_contract(temporal_request, temporal_out)
            missing_temporal_fields = validate_parity(
                temporal_contract,
                ["previous_frame", "composed_frame", "target_frame", "changed_regions", "scene_transition_context"],
            )
            semantic_temporal = semantic_parity_checks(
                stage="temporal",
                contract=temporal_contract,
                request=temporal_request,
                output=temporal_out,
                changed_regions_count=len(patches),
            )
            if missing_temporal_fields:
                fallback_log.append(f"step={planned_state.step_index}:temporal_parity_missing={missing_temporal_fields}")
            if semantic_temporal:
                fallback_log.extend([f"step={planned_state.step_index}:temporal_semantic={x}" for x in semantic_temporal])
            stable_frame = temporal_out.refined_frame if profile.temporal_refinement else composed

            scene_graph = apply_delta(scene_graph, delta)
            graph_encoding = self.backends.graph_encoder.encode(scene_graph)
            memory = self.memory_manager.update_from_graph(memory, scene_graph)
            transition_context = {
                "transition_phase": delta.transition_phase,
                "visibility_phase": delta.state_after.get("visibility_phase", "stable"),
                "garment_phase": delta.state_after.get("garment_phase", "worn"),
                "pose_phase": delta.state_after.get("pose_phase", "stable"),
            }
            if delta.affected_regions:
                primary_region = delta.affected_regions[0]
                transition_context["region_transition_mode"] = delta.region_transition_mode.get(primary_region, "stable")
            memory.last_transition_context = transition_context
            memory = self.memory_manager.update_from_frame(memory, stable_frame, scene_graph, transition_context=transition_context)

            frames.append(stable_frame)
            current_frame = stable_frame
            graphs.append(scene_graph)
            overlay_log.append(f"step={planned_state.step_index}, regions={len(changed_regions)}")
            dynamics_metrics_log.append(
                f"delta={transition_output.diagnostics.get('delta_magnitude', 0.0):.3f}, smooth={transition_output.diagnostics.get('temporal_smoothness_proxy', 0.0):.3f}, violations={transition_output.diagnostics.get('constraint_violations', 0.0):.0f}"
            )
            step_debug.append(
                {
                    "step_index": planned_state.step_index,
                    "dynamics": {
                        "backend": self.backends.backend_names.get("dynamics_backend", "unknown"),
                        "confidence": transition_output.confidence,
                        "diagnostics_summary": {
                            "delta_magnitude": transition_output.diagnostics.get("delta_magnitude"),
                            "smoothness": transition_output.diagnostics.get("temporal_smoothness_proxy"),
                            "violations": transition_output.diagnostics.get("constraint_violations"),
                            "learned_ready": transition_output.metadata.get("learned_ready_usage", {}),
                        },
                    },
                    "patch": patch_step_debug,
                    "hidden_reconstruction": {
                        "step_has_case": step_hidden_reconstruction,
                        "known_hidden_count": hidden_recon_stats["known_hidden"],
                        "unknown_hidden_count": hidden_recon_stats["unknown_hidden"],
                        "hidden_reveal_count": hidden_recon_stats["hidden_reveal"],
                    },
                    "temporal": {
                        "backend": self.backends.backend_names.get("temporal_backend", "unknown"),
                        "region_consistency_summary": temporal_out.region_consistency_scores,
                        "learned_ready": temporal_out.metadata.get("learned_ready_usage", {}),
                        "drift_consistency": {
                            "changed_regions": len(temporal_request.changed_regions),
                            "drift_proxy": 1.0 - (sum(temporal_out.region_consistency_scores.values()) / max(1.0, float(len(temporal_out.region_consistency_scores) or 1))),
                        },
                    },
                    "parity": {
                        "missing_fields": {
                            "dynamics": missing_fields,
                            "temporal": missing_temporal_fields,
                            "patch": [p["missing_fields"] for p in patch_step_debug],
                        },
                        "semantic_issues": {
                            "dynamics": semantic_dynamics,
                            "temporal": semantic_temporal,
                    "patch": [p["semantic_issues"] for p in patch_step_debug],
                        },
                    },
                }
            )
            channel_usage_log.append(
                {
                    "step_index": planned_state.step_index,
                    "dynamics_channels": list(transition_request.memory_channels.keys()),
                    "patch_channels": list(self.build_patch_memory_channels(memory_channels).keys()) if changed_regions else [],
                    "temporal_channels": list(temporal_request.memory_channels.keys()),
                    "identity_encoder_used": bool(identity_embedding),
                    "graph_embedding_dim": len(graph_encoding.graph_embedding),
                    "dynamics_backend_usage": transition_output.metadata.get("learned_ready_usage", {}),
                }
            )
            if step_hidden_reconstruction:
                hidden_recon_stats["steps_with_hidden_reconstruction"] += 1

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
                    "text_parity": {
                        "missing_fields": text_missing,
                        "semantic_issues": text_semantic,
                    },
                    "hidden_reconstruction_summary": hidden_recon_stats,
                },
            },
        )

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
