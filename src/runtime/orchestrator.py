from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from core.input_layer import InputAssetLayer
from core.schema import SceneGraph
from dynamics.state_update import apply_delta
from learned.factory import BackendBundle, BackendConfig, LearnedBackendFactory
from learned.interfaces import DynamicsTransitionRequest, PatchSynthesisRequest, TemporalRefinementRequest
from learned.parity import (
    build_parity_result,
    text_output_to_contract,
    dynamics_io_to_contract,
    patch_io_to_contract,
    temporal_io_to_contract,
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
from runtime.region_routing import CanonicalRegionRouter
from text.intent_parser import IntentParser
from utils_tensor import shape, zeros
from core.region_ids import make_region_id


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
        self.region_router = CanonicalRegionRouter(self.memory_manager, self.roi_selector)
        self.compositor = Compositor()
        self.memory_summarizer = AppearanceMemorySummarizer()
        self.backend_config = backend_config or BackendConfig()
        self.backends = backend_bundle or LearnedBackendFactory(self.backend_config).build()

    @staticmethod
    def _normalize_frame_tensor(frame: object, *, field_name: str) -> list[list[list[float]]]:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"{field_name} must be HxWx3 tensor-like, got shape={list(arr.shape)}")
        if arr.size == 0:
            raise ValueError(f"{field_name} must not be empty")
        max_val = float(np.max(arr))
        if max_val > 1.0:
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        else:
            arr = np.clip(arr, 0.0, 1.0)
        return arr.tolist()

    @staticmethod
    def _validate_patch_output_contract(patch_out: object, *, expected_region_id: str) -> dict[str, object]:
        issues: list[str] = []
        path = str(getattr(patch_out, "execution_trace", {}).get("renderer_path", "unknown"))
        region = getattr(getattr(patch_out, "region", None), "region_id", "")
        if region != expected_region_id:
            issues.append(f"region_mismatch:{region}->{expected_region_id}")
        rgb = np.asarray(getattr(patch_out, "rgb_patch", None), dtype=np.float32)
        alpha = np.asarray(getattr(patch_out, "alpha_mask", None), dtype=np.float32)
        uncertainty = getattr(patch_out, "uncertainty_map", None)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            issues.append(f"rgb_patch_invalid_shape:{list(rgb.shape)}")
        if alpha.ndim != 2:
            issues.append(f"alpha_mask_invalid_shape:{list(alpha.shape)}")
        if rgb.ndim == 3 and alpha.ndim == 2 and rgb.shape[:2] != alpha.shape:
            issues.append(f"rgb_alpha_shape_mismatch:{list(rgb.shape)} vs {list(alpha.shape)}")
        if isinstance(uncertainty, list):
            unc = np.asarray(uncertainty, dtype=np.float32)
            if unc.ndim != 2 or (rgb.ndim == 3 and unc.shape != rgb.shape[:2]):
                issues.append(f"uncertainty_shape_mismatch:{list(unc.shape)}")
        return {"issues": issues, "renderer_path": path, "is_learned_primary": path == "learned_primary"}

    @staticmethod
    def _validate_temporal_output_contract(temporal_out: object, *, expected_shape: tuple[int, int, int]) -> dict[str, object]:
        issues: list[str] = []
        refined = np.asarray(getattr(temporal_out, "refined_frame", None), dtype=np.float32)
        if refined.ndim != 3 or refined.shape[2] != 3:
            issues.append(f"refined_frame_invalid_shape:{list(refined.shape)}")
        elif tuple(refined.shape) != expected_shape:
            issues.append(f"refined_frame_shape_mismatch:{list(refined.shape)} vs {list(expected_shape)}")
        meta = getattr(temporal_out, "metadata", {}) if isinstance(getattr(temporal_out, "metadata", {}), dict) else {}
        temporal_path = str(meta.get("temporal_path", "unknown"))
        return {"issues": issues, "temporal_path": temporal_path, "is_learned_primary": temporal_path == "learned_primary"}

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
        current_frame = self._normalize_frame_tensor(first_frame.tensor, field_name="input_frame") if first_frame else self._debug_seed_frame_tensor(profile)
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
        text_parity = build_parity_result(
            contract=text_contract,
            required_fields=["text", "parsed_actions", "action_embedding", "target_entities", "target_objects", "temporal_decomposition", "constraints"],
            stage="text",
            request={"text": request.text, "actions": action_plan.actions},
            output=text_encoding,
        )
        if text_parity["missing_fields"]:
            fallback_log.append(f"step=0:text_parity_missing={text_parity['missing_fields']}")
        for severity in ("errors", "warnings", "traces"):
            for issue in text_parity.get(severity, []):
                fallback_log.append(f"step=0:text_semantic_{severity}={issue}")
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

        frames: list[list[list[list[float]]]] = [current_frame]
        graphs = [scene_graph]
        overlay_log: list[str] = []
        dynamics_metrics_log: list[str] = []
        channel_usage_log: list[dict[str, object]] = []
        step_debug: list[dict[str, object]] = []
        hidden_recon_stats = {"known_hidden": 0, "unknown_hidden": 0, "hidden_reveal": 0, "steps_with_hidden_reconstruction": 0}
        hidden_recon_quality = {
            "confidence_sum": 0.0,
            "quality_hint_sum": 0.0,
            "refinement_strength_sum": 0.0,
            "count": 0,
            "synthesis_mode_counts": {},
            "strategy_counts": {},
            "by_strategy": {},
            "by_synthesis_mode": {},
            "by_family": {"known_hidden": {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0}, "unknown_hidden": {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0}, "hidden_reveal": {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0}},
        }

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
                step_context={"step_index": planned_state.step_index, "memory": memory, "semantic_transition": planned_state.semantic_transition},
            )
            transition_output = self.backends.dynamics_backend.predict_transition(transition_request)
            parity_contract = dynamics_io_to_contract(transition_request, transition_output)
            dynamics_parity = build_parity_result(
                contract=parity_contract,
                required_fields=["graph_before", "graph_after", "delta_contract", "transition_context"],
                stage="dynamics",
                request=transition_request,
                output=transition_output,
            )
            if dynamics_parity["missing_fields"]:
                fallback_log.append(f"step={planned_state.step_index}:dynamics_parity_missing={dynamics_parity['missing_fields']}")
            for severity in ("errors", "warnings", "traces"):
                for issue in dynamics_parity.get(severity, []):
                    fallback_log.append(f"step={planned_state.step_index}:dynamics_semantic_{severity}={issue}")

            delta = transition_output.delta
            region_plan = self.region_router.build_plan(
                scene_graph=scene_graph,
                delta=delta,
                memory=memory,
                semantic_transition=planned_state.semantic_transition,
            )
            transition_diag = delta.transition_diagnostics if isinstance(delta.transition_diagnostics, dict) else {}
            transition_diag["region_routing_plan"] = {
                make_region_id(region_plan.entity_id, d.canonical_region): {
                    "decision": d.decision,
                    "priority": d.priority,
                    "reveal_mode": d.reveal_mode,
                    "synthesis_required": d.synthesis_required,
                    "renderer_mode_hint": d.renderer_mode_hint,
                    "confidence": d.confidence,
                    "reasons": d.reasons,
                    "memory_source_available": d.memory_source_available,
                    "memory_support_level": d.memory_support_level,
                }
                for d in region_plan.decisions
            }
            transition_diag["region_transition_semantics"] = region_plan.as_debug_dict()["transition_semantics"]
            delta.transition_diagnostics = transition_diag
            changed_regions = region_plan.render_regions or self.roi_selector.select(scene_graph, delta)
            patches: list[RenderedPatch] = []
            patch_step_debug: list[dict[str, object]] = []
            step_hidden_reconstruction = False
            step_hidden_cases = 0
            for region in changed_regions[: profile.max_roi_count]:
                patch_channels = self.build_patch_memory_channels(memory_channels)
                region_route = region_plan.decision_for_region_id(region.region_id)
                transition_metadata = transition_output.metadata if isinstance(transition_output.metadata, dict) else {}
                learned_temporal_contract = transition_metadata.get("temporal_transition_contract", {})
                learned_human_state_contract = transition_metadata.get("human_state_contract", {})
                learned_target_profile = {}
                if isinstance(learned_human_state_contract, dict):
                    learned_target_profile = learned_human_state_contract.get("target_profile", {}) if isinstance(learned_human_state_contract.get("target_profile", {}), dict) else {}
                if not learned_target_profile and isinstance(learned_temporal_contract, dict):
                    learned_target_profile = learned_temporal_contract.get("target_profile", {}) if isinstance(learned_temporal_contract.get("target_profile", {}), dict) else {}
                patch_request = PatchSynthesisRequest(
                    region=region,
                    scene_state=scene_graph,
                    memory_summary=memory_summary,
                    transition_context={
                        "graph_delta": delta,
                        "video_memory": memory,
                        "transition_phase": delta.transition_phase,
                        "step_index": planned_state.step_index,
                        "target_profile": learned_target_profile or transition_metadata.get("target_profile", {}),
                        "learned_temporal_contract": learned_temporal_contract,
                        "learned_human_state_contract": learned_human_state_contract,
                        "region_selection_rationale": transition_metadata.get("region_selection_rationale", {}),
                        "semantic_families": transition_metadata.get("semantic_families", []),
                        "region_route_decision": {
                            "decision": region_route.decision if region_route else "unknown",
                            "reveal_mode": region_route.reveal_mode if region_route else "none",
                            "renderer_mode_hint": region_route.renderer_mode_hint if region_route else "keep",
                            "synthesis_required": region_route.synthesis_required if region_route else False,
                        },
                    },
                    retrieval_summary={
                        "backend": "learned_primary",
                        "identity_entity": entity_id,
                        "target_profile": learned_target_profile or transition_metadata.get("target_profile", {}),
                    },
                    current_frame=current_frame,
                    memory_channels=patch_channels,
                    graph_encoding=graph_encoding,
                    identity_embedding=identity_embedding,
                )
                patch_out = self.backends.patch_backend.synthesize_patch(patch_request)
                patch_contract_validation = self._validate_patch_output_contract(patch_out, expected_region_id=region.region_id)
                if patch_contract_validation["issues"]:
                    raise ValueError(f"Patch contract violation at step={planned_state.step_index}, region={region.region_id}: {patch_contract_validation['issues']}")
                patch_contract = patch_io_to_contract(patch_request, patch_out)
                patch_parity = build_parity_result(
                    contract=patch_contract,
                    required_fields=["roi_before", "roi_after", "region_metadata", "selected_strategy", "transition_context"],
                    stage="patch",
                    request=patch_request,
                    output=patch_out,
                )
                strategy = str(patch_out.execution_trace.get("selected_render_strategy", ""))
                synth_mode = str(patch_out.execution_trace.get("synthesis_mode", "deterministic"))
                patch_hidden_case = False
                if "KNOWN_HIDDEN_REVEAL" in strategy:
                    hidden_recon_stats["known_hidden"] += 1
                    hidden_recon_stats["hidden_reveal"] += 1
                    patch_hidden_case = True
                elif "UNKNOWN_HIDDEN_SYNTHESIS" in strategy:
                    hidden_recon_stats["unknown_hidden"] += 1
                    patch_hidden_case = True
                if patch_hidden_case:
                    step_hidden_reconstruction = True
                    step_hidden_cases += 1
                    hint = float(patch_out.execution_trace.get("patch_refinement_strength", 0.0) or 0.0)
                    hidden_recon_quality["count"] += 1
                    hidden_recon_quality["confidence_sum"] += float(patch_out.confidence)
                    hidden_recon_quality["quality_hint_sum"] += hint
                    hidden_recon_quality["refinement_strength_sum"] += hint
                    hidden_recon_quality["synthesis_mode_counts"][synth_mode] = int(hidden_recon_quality["synthesis_mode_counts"].get(synth_mode, 0)) + 1
                    hidden_recon_quality["strategy_counts"][strategy] = int(hidden_recon_quality["strategy_counts"].get(strategy, 0)) + 1
                    family = "hidden_reveal" if "REVEAL" in strategy else ("unknown_hidden" if "UNKNOWN_HIDDEN" in strategy else "known_hidden")
                    fam = hidden_recon_quality["by_family"][family]
                    fam["count"] += 1
                    fam["confidence_sum"] += float(patch_out.confidence)
                    fam["quality_hint_sum"] += hint
                    strat = hidden_recon_quality["by_strategy"].setdefault(strategy, {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0})
                    strat["count"] += 1
                    strat["confidence_sum"] += float(patch_out.confidence)
                    strat["quality_hint_sum"] += hint
                    mode = hidden_recon_quality["by_synthesis_mode"].setdefault(synth_mode, {"count": 0, "confidence_sum": 0.0, "quality_hint_sum": 0.0})
                    mode["count"] += 1
                    mode["confidence_sum"] += float(patch_out.confidence)
                    mode["quality_hint_sum"] += hint
                if patch_parity["missing_fields"]:
                    fallback_log.append(f"step={planned_state.step_index}:patch_parity_missing={patch_parity['missing_fields']}")
                for severity in ("errors", "warnings", "traces"):
                    fallback_log.extend([f"step={planned_state.step_index}:patch_semantic_{severity}={x}" for x in patch_parity.get(severity, [])])
                patch_step_debug.append(
                    {
                        "region_id": region.region_id,
                        "selected_strategy": patch_out.execution_trace.get("selected_render_strategy", patch_contract.get("selected_render_strategy", patch_contract.get("selected_strategy", "unknown"))),
                        "execution_policy": patch_out.execution_trace.get("selection", {}).get("execution_policy", {}),
                        "mode_source": (patch_out.execution_trace.get("memory_dependency_summary", {}) or {}).get("mode_source", "unknown"),
                        "runtime_plan_authoritative": (patch_out.execution_trace.get("memory_dependency_summary", {}) or {}).get("runtime_plan_authoritative", False),
                        "confidence": patch_out.confidence,
                        "synthesis_mode": synth_mode,
                        "retrieval_summary": str(patch_request.retrieval_summary)[:120],
                        "learned_ready": patch_out.metadata.get("learned_ready_usage", {}),
                        "hidden_reconstruction": {
                            "patch_hidden_reconstruction_case": patch_hidden_case,
                            "strategy": strategy,
                            "mode": synth_mode,
                            "quality_hint": patch_out.execution_trace.get("patch_refinement_strength", 0.0),
                        },
                        "parity": patch_parity,
                        "contract_validation": patch_contract_validation,
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
            if patches:
                patch_conf = [float(p.confidence) for p in patches]
                patch_alpha_mean = []
                patch_alpha_edge = []
                for p in patches:
                    alpha_vals = [float(v) for row in p.alpha_mask for v in row] if p.alpha_mask else [0.0]
                    patch_alpha_mean.append(sum(alpha_vals) / max(1, len(alpha_vals)))
                    edge_vals = []
                    if p.alpha_mask:
                        edge_vals.extend([float(v) for v in p.alpha_mask[0]])
                        edge_vals.extend([float(v) for v in p.alpha_mask[-1]])
                        edge_vals.extend([float(row[0]) for row in p.alpha_mask])
                        edge_vals.extend([float(row[-1]) for row in p.alpha_mask])
                    patch_alpha_edge.append(sum(edge_vals) / max(1, len(edge_vals)))
                temporal_channels = {
                    **temporal_channels,
                    "patch_confidence": {
                        "mean_confidence": sum(patch_conf) / max(1, len(patch_conf)),
                        "min_confidence": min(patch_conf),
                        "max_confidence": max(patch_conf),
                    },
                    "patch_alpha": {
                        "mean_alpha": sum(patch_alpha_mean) / max(1, len(patch_alpha_mean)),
                        "edge_alpha": sum(patch_alpha_edge) / max(1, len(patch_alpha_edge)),
                    },
                    "patch_history": {
                        "count": len(patches),
                        "region_ids": [p.region.region_id for p in patches],
                    },
                }
            temporal_request = TemporalRefinementRequest(
                previous_frame=frames[-1],
                current_composed_frame=composed,
                changed_regions=[p.region for p in patches],
                scene_state=scene_graph,
                memory_state=memory,
                memory_channels=temporal_channels,
            )
            temporal_out = self.backends.temporal_backend.refine_temporal(temporal_request)
            temporal_contract_validation = self._validate_temporal_output_contract(
                temporal_out,
                expected_shape=tuple(np.asarray(composed, dtype=np.float32).shape),
            )
            if temporal_contract_validation["issues"]:
                raise ValueError(f"Temporal contract violation at step={planned_state.step_index}: {temporal_contract_validation['issues']}")
            temporal_contract = temporal_io_to_contract(temporal_request, temporal_out)
            temporal_parity = build_parity_result(
                contract=temporal_contract,
                required_fields=["previous_frame", "composed_frame", "target_frame", "changed_regions", "scene_transition_context"],
                stage="temporal",
                request=temporal_request,
                output=temporal_out,
                changed_regions_count=len(patches),
            )
            if temporal_parity["missing_fields"]:
                fallback_log.append(f"step={planned_state.step_index}:temporal_parity_missing={temporal_parity['missing_fields']}")
            for severity in ("errors", "warnings", "traces"):
                fallback_log.extend([f"step={planned_state.step_index}:temporal_semantic_{severity}={x}" for x in temporal_parity.get(severity, [])])
            stable_frame = temporal_out.refined_frame if profile.temporal_refinement else composed
            stable_frame = self._normalize_frame_tensor(stable_frame, field_name="stable_frame")

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
                    "region_routing": region_plan.as_debug_dict(),
                    "region_render_order": [r.region_id for r in changed_regions[: profile.max_roi_count]],
                    "dynamics": {
                        "backend": self.backends.backend_names.get("dynamics_backend", "unknown"),
                        "confidence": transition_output.confidence,
                        "supervision_mode": parity_contract.get("transition_context", {}).get("supervision_mode", "inference"),
                            "diagnostics_summary": {
                                "delta_magnitude": transition_output.diagnostics.get("delta_magnitude"),
                                "smoothness": transition_output.diagnostics.get("temporal_smoothness_proxy"),
                                "violations": transition_output.diagnostics.get("constraint_violations"),
                                "learned_ready": transition_output.metadata.get("learned_ready_usage", {}),
                                "temporal_contract_alignment": transition_output.metadata.get("temporal_contract_alignment", {}),
                            },
                        },
                    "patch": patch_step_debug,
                    "hidden_reconstruction": {
                        "step_has_hidden_reconstruction": step_hidden_reconstruction,
                        "step_hidden_reconstruction_case_count": step_hidden_cases,
                        "known_hidden_count": hidden_recon_stats["known_hidden"],
                        "unknown_hidden_count": hidden_recon_stats["unknown_hidden"],
                        "hidden_reveal_count": hidden_recon_stats["hidden_reveal"],
                    },
                    "temporal": {
                        "backend": self.backends.backend_names.get("temporal_backend", "unknown"),
                        "temporal_path": temporal_out.metadata.get("temporal_path", "unknown"),
                        "fallback_reason": temporal_out.metadata.get("fallback_reason"),
                        "region_consistency_summary": temporal_out.region_consistency_scores,
                        "learned_ready": temporal_out.metadata.get("learned_ready_usage", {}),
                        "drift_consistency": {
                            "changed_regions": len(temporal_request.changed_regions),
                            "drift_proxy": 1.0 - (sum(temporal_out.region_consistency_scores.values()) / max(1.0, float(len(temporal_out.region_consistency_scores) or 1))),
                        },
                        "contract_validation": temporal_contract_validation,
                    },
                    "parity": {
                        "missing_fields": {
                            "dynamics": dynamics_parity["missing_fields"],
                            "temporal": temporal_parity["missing_fields"],
                            "patch": [p["parity"]["missing_fields"] for p in patch_step_debug],
                        },
                        "semantic_issues": {
                            "dynamics": dynamics_parity,
                            "temporal": temporal_parity,
                            "patch": [p["parity"] for p in patch_step_debug],
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
                    "text_parity": text_parity,
                    "hidden_reconstruction_summary": {
                        **hidden_recon_stats,
                        "average_hidden_reconstruction_confidence": (hidden_recon_quality["confidence_sum"] / hidden_recon_quality["count"]) if hidden_recon_quality["count"] else 0.0,
                        "average_refinement_strength": (hidden_recon_quality["refinement_strength_sum"] / hidden_recon_quality["count"]) if hidden_recon_quality["count"] else 0.0,
                        "average_quality_hint": (hidden_recon_quality["quality_hint_sum"] / hidden_recon_quality["count"]) if hidden_recon_quality["count"] else 0.0,
                        "count_by_synthesis_mode": hidden_recon_quality["synthesis_mode_counts"],
                        "count_by_selected_strategy": hidden_recon_quality["strategy_counts"],
                        "quality_by_selected_strategy": hidden_recon_quality["by_strategy"],
                        "quality_by_synthesis_mode": hidden_recon_quality["by_synthesis_mode"],
                        "quality_by_hidden_family": hidden_recon_quality["by_family"],
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
