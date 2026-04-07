from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.input_layer import InputAssetLayer
from core.schema import SceneGraph
from dynamics.state_update import apply_delta
from learned.factory import LearnedBackendFactory
from learned.interfaces import DynamicsTransitionRequest, PatchSynthesisRequest, TemporalRefinementRequest
from learned.parity import dynamics_io_to_contract, patch_io_to_contract, temporal_io_to_contract, validate_parity
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
    debug: dict[str, list[str] | str | dict[str, object]] = field(default_factory=dict)


class GennadyEngine:
    def __init__(self) -> None:
        self.input_layer = InputAssetLayer()
        self.perception = PerceptionPipeline()
        self.graph_builder = SceneGraphBuilder()
        self.memory_manager = MemoryManager()
        self.intent_parser = IntentParser()
        self.planner = TransitionPlanner()
        self.roi_selector = ROISelector()
        self.compositor = Compositor()
        self.memory_summarizer = AppearanceMemorySummarizer()
        self.backends = LearnedBackendFactory().build()

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

        action_plan = self.intent_parser.parse(request.text, scene_graph=scene_graph)
        text_encoding = self.backends.text_encoder.encode(request.text, scene_graph=scene_graph, action_plan=action_plan)
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
        fallback_log: list[str] = []

        for planned_state in state_plan.steps[1 : profile.max_transition_steps + 1]:
            memory_summary = self.memory_summarizer.summarize(memory).as_dict()
            memory_channels = summarize_memory(memory)
            entity_id = scene_graph.persons[0].person_id if scene_graph.persons else "scene"
            identity_embedding = self.backends.identity_encoder.encode_identity(memory_channels, entity_id)

            transition_request = DynamicsTransitionRequest(
                graph_state=scene_graph,
                memory_summary=memory_summary,
                memory_channels=memory_channels,
                text_action_summary=text_encoding,
                graph_encoding=graph_encoding,
                identity_embeddings={entity_id: identity_embedding},
                step_context={"step_index": planned_state.step_index, "memory": memory},
            )
            transition_output = self.backends.dynamics_backend.predict_transition(transition_request)
            parity_contract = dynamics_io_to_contract(transition_request, transition_output)
            missing_fields = validate_parity(parity_contract, ["graph_before", "graph_after", "delta_contract", "transition_context"])
            if missing_fields:
                fallback_log.append(f"dynamics_parity_missing={missing_fields}")

            delta = transition_output.delta
            changed_regions = self.roi_selector.select(scene_graph, delta)
            patches: list[RenderedPatch] = []
            for region in changed_regions[: profile.max_roi_count]:
                patch_request = PatchSynthesisRequest(
                    region=region,
                    scene_state=scene_graph,
                    memory_summary=memory_summary,
                    transition_context={"graph_delta": delta, "video_memory": memory},
                    retrieval_summary={"backend": "deterministic", "identity_entity": entity_id},
                    current_frame=current_frame,
                    memory_channels={
                        "identity": memory_channels.get("identity", {}),
                        "garments": memory_channels.get("garments", {}),
                        "hidden_regions": memory_channels.get("hidden_regions", {}),
                    },
                    graph_encoding=graph_encoding,
                    identity_embedding=identity_embedding,
                )
                patch_out = self.backends.patch_backend.synthesize_patch(patch_request)
                patch_contract = patch_io_to_contract(patch_request, patch_out)
                missing_patch_fields = validate_parity(
                    patch_contract,
                    ["roi_before", "roi_after", "region_metadata", "selected_strategy", "transition_context"],
                )
                if missing_patch_fields:
                    fallback_log.append(f"patch_parity_missing={missing_patch_fields}")
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
            temporal_request = TemporalRefinementRequest(
                previous_frame=frames[-1],
                current_composed_frame=composed,
                changed_regions=[p.region for p in patches],
                scene_state=scene_graph,
                memory_state=memory,
                memory_channels={
                    "identity": memory_channels.get("identity", {}),
                    "body_regions": memory_channels.get("body_regions", {}),
                    "hidden_regions": memory_channels.get("hidden_regions", {}),
                },
            )
            temporal_out = self.backends.temporal_backend.refine_temporal(temporal_request)
            temporal_contract = temporal_io_to_contract(temporal_request, temporal_out)
            missing_temporal_fields = validate_parity(
                temporal_contract,
                ["previous_frame", "composed_frame", "target_frame", "changed_regions", "scene_transition_context"],
            )
            if missing_temporal_fields:
                fallback_log.append(f"temporal_parity_missing={missing_temporal_fields}")
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
            channel_usage_log.append(
                {
                    "step_index": planned_state.step_index,
                    "dynamics_channels": list(transition_request.memory_channels.keys()),
                    "patch_channels": list(patch_request.memory_channels.keys()) if changed_regions else [],
                    "temporal_channels": list(temporal_request.memory_channels.keys()),
                    "identity_encoder_used": bool(identity_embedding),
                    "graph_embedding_dim": len(graph_encoding.graph_embedding),
                }
            )

        video_uri = self._export_video(frames, fps)
        return InferenceArtifacts(
            frames=frames,
            scene_graphs=graphs,
            state_plan=state_plan,
            debug={
                "overlay_log": overlay_log,
                "dynamics_metrics": dynamics_metrics_log,
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
