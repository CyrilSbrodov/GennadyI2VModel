from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path


from core.input_layer import InputAssetLayer
from core.schema import SceneGraph
from dynamics.graph_delta_predictor import GraphDeltaPredictor
from dynamics.state_update import apply_delta
from memory.video_memory import MemoryManager
from perception.pipeline import PerceptionPipeline
from planning.transition_engine import StatePlan, TransitionPlanner
from rendering.compositor import Compositor, TemporalStabilizer
from rendering.roi_renderer import PatchRenderer, ROISelector
from representation.graph_builder import SceneGraphBuilder
from runtime.profiles import PROFILES, RuntimeProfile
from text.intent_parser import IntentParser


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

    def run(
        self,
        images: list[str],
        text: str,
        fps: int = 16,
        duration: float = 4.0,
        quality_profile: str = "balanced",
    ) -> InferenceArtifacts:
        profile = self._resolve_profile(quality_profile)
        request = self.input_layer.build_request(images=images, text=text, fps=fps, duration=duration)
        source_image = request.images[0] if request.images else (request.reference_set[0] if request.reference_set else "generated://blank")

        perception_output = self.perception.analyze(source_image)
        scene_graph = self.graph_builder.build(perception_output, frame_index=0)
        scene_graph.global_context.fps = fps
        memory = self.memory_manager.initialize(scene_graph)

        action_plan = self.intent_parser.parse(request.text, scene_graph=scene_graph)
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

        current_frame = self._seed_frame_tensor(source_image, profile)
        frames = [current_frame]
        graphs = [scene_graph]
        overlay_log: list[str] = []
        dynamics_metrics_log: list[str] = []

        for planned_state in state_plan.steps[1 : profile.max_transition_steps + 1]:
            delta, dyn_metrics = self.dynamics.predict(
                scene_graph=scene_graph,
                target_state=planned_state,
                planner_context={"step_index": planned_state.step_index},
                memory=memory,
            )
            changed_regions = self.roi_selector.select(scene_graph, delta)
            patches = [
                self.patch_renderer.render(current_frame, scene_graph, delta, memory, region)
                for region in changed_regions[: profile.max_roi_count]
            ]
            composed = self.compositor.compose(current_frame, patches, delta)
            stable_frame = self.stabilizer.refine(frames[-1], composed, memory, enabled=profile.temporal_refinement)

            scene_graph = apply_delta(scene_graph, delta)
            memory = self.memory_manager.update(memory, scene_graph)

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
                },
            },
        )

    def _seed_frame_tensor(self, source_image: str, profile: RuntimeProfile) -> list:
        h, w = profile.internal_resolution
        seed = abs(hash(source_image)) % 255
        from utils_tensor import zeros
        frame = zeros(h, w, 3)
        for y in range(h):
            for x in range(w):
                frame[y][x][0] = (seed % 97) / 96.0
                frame[y][x][1] = (seed % 73) / 72.0
                frame[y][x][2] = (seed % 59) / 58.0
        return frame

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

            uint8_frames = [np.array([[[(int(max(0,min(1,ch))*255)) for ch in px] for px in row] for row in f], dtype=np.uint8) for f in frames]
            h, w = uint8_frames[0].shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for frame in uint8_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception:
            out_path.write_bytes(b"mp4-export-unavailable")

        return str(out_path)
