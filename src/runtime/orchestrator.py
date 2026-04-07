from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.input_layer import InputAssetLayer
from core.schema import SceneGraph
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from dynamics.learned_bridge import BaselineDynamicsTransitionModel
from learned.interfaces import DynamicsTransitionRequest, PatchSynthesisRequest, TemporalRefinementRequest
from dynamics.state_update import apply_delta
from memory.video_memory import MemoryManager
from memory.summaries import AppearanceMemorySummarizer
from perception.pipeline import PerceptionPipeline
from planning.transition_engine import StatePlan, TransitionPlanner
from rendering.compositor import Compositor, TemporalStabilizer
from rendering.learned_bridge import BaselinePatchSynthesisModel
from rendering.roi_renderer import PatchRenderer, ROISelector
from rendering.temporal_bridge import BaselineTemporalConsistencyModel
from representation.graph_builder import SceneGraphBuilder
from runtime.profiles import PROFILES, RuntimeProfile
from text.intent_parser import IntentParser
from text.learned_bridge import BaselineTextEncoderAdapter
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
        self.dynamics = GraphDeltaPredictor()
        self.roi_selector = ROISelector()
        self.patch_renderer = PatchRenderer()
        self.compositor = Compositor()
        self.stabilizer = TemporalStabilizer()
        self.text_encoder = BaselineTextEncoderAdapter()
        self.memory_summarizer = AppearanceMemorySummarizer()
        self.dynamics_backend = BaselineDynamicsTransitionModel()
        self.patch_backend = BaselinePatchSynthesisModel(self.patch_renderer)
        self.temporal_backend = BaselineTemporalConsistencyModel(self.stabilizer)

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

        action_plan = self.intent_parser.parse(request.text, scene_graph=scene_graph)
        text_encoding = self.text_encoder.encode(request.text, scene_graph=scene_graph, action_plan=action_plan)
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

        for planned_state in state_plan.steps[1 : profile.max_transition_steps + 1]:
            memory_summary = self.memory_summarizer.summarize(memory).as_dict()
            transition_output = self.dynamics_backend.predict_transition(
                DynamicsTransitionRequest(
                    graph_state=scene_graph,
                    memory_summary=memory_summary,
                    text_action_summary=text_encoding,
                    step_context={"step_index": planned_state.step_index, "memory": memory},
                )
            )
            delta = transition_output.delta
            _, dyn_metrics = self.dynamics.predict(
                scene_graph=scene_graph,
                target_state=planned_state,
                planner_context={"step_index": planned_state.step_index},
                memory=memory,
            )
            changed_regions = self.roi_selector.select(scene_graph, delta)
            patches = []
            for region in changed_regions[: profile.max_roi_count]:
                patch_out = self.patch_backend.synthesize_patch(
                    PatchSynthesisRequest(
                        region=region,
                        scene_state=scene_graph,
                        memory_summary=memory_summary,
                        transition_context={"graph_delta": delta, "video_memory": memory},
                        retrieval_summary={"backend": "deterministic"},
                        current_frame=current_frame,
                    )
                )
                rendered = self.patch_renderer.render(current_frame, scene_graph, delta, memory, region)
                rendered.rgb_patch = patch_out.rgb_patch
                rendered.confidence = patch_out.confidence
                rendered.uncertainty_map = patch_out.uncertainty_map
                rendered.execution_trace.update({"patch_backend": "deterministic_interface"})
                patches.append(rendered)
            composed = self.compositor.compose(current_frame, patches, delta)
            temporal_out = self.temporal_backend.refine_temporal(
                TemporalRefinementRequest(
                    previous_frame=frames[-1],
                    current_composed_frame=composed,
                    changed_regions=[p.region for p in patches],
                    scene_state=scene_graph,
                    memory_state=memory,
                )
            )
            stable_frame = temporal_out.refined_frame if profile.temporal_refinement else composed

            scene_graph = apply_delta(scene_graph, delta)
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
                f"delta={dyn_metrics.delta_magnitude:.3f}, smooth={dyn_metrics.temporal_smoothness_proxy:.3f}, violations={dyn_metrics.constraint_violations}"
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
                    "text_encoder_backend": "baseline_text_adapter",
                    "dynamics_backend": "deterministic_graph_delta_predictor",
                    "patch_backend": "deterministic_patch_renderer",
                    "temporal_backend": "deterministic_temporal_stabilizer",
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
