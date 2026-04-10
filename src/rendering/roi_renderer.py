from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from core.region_ids import make_region_id, parse_region_id
from core.schema import BBox, GarmentSemanticProfile, GraphDelta, RegionDescriptor, RegionRef, SceneGraph, VideoMemory
from core.semantic_roi import SemanticROIHelper
from memory.video_memory import MemoryManager
from rendering.confidence import PatchConfidenceEstimator
from rendering.contracts.mode_surfaces import InsertionPathContract, RevealPathContract, UpdatePathContract
from rendering.contracts.synthesis_plan import PatchSynthesisPlan
from rendering.torch_backends import TorchRendererBackendBundle, build_renderer_tensor_batch, serialize_tensor_batch
from representation.scene_graph_queries import SceneGraphQueries
from utils_tensor import alpha_radial, crop, mean_color, shape, zeros


@dataclass(slots=True)
class RenderedPatch:
    region: RegionRef
    rgb_patch: list[list[list[float]]]
    alpha_mask: list[list[float]]
    height: int
    width: int
    channels: int
    uncertainty_map: list[list[float]] | None = None
    confidence: float = 0.0
    z_index: int = 0
    debug_trace: list[str] = field(default_factory=list)
    execution_trace: dict[str, object] = field(default_factory=dict)


RenderMode = Literal["keep", "warp", "deform", "refine", "reveal", "insert_new"]
EntityLifecycle = Literal["already_existing", "newly_inserted", "previously_hidden_now_revealed", "still_hidden", "interaction_boundary", "stable_context"]


@dataclass(slots=True)
class RenderRouteDecision:
    mode: RenderMode
    lifecycle: EntityLifecycle
    reason_trace: list[str] = field(default_factory=list)
    memory_dependency: dict[str, object] = field(default_factory=dict)
    plan: PatchSynthesisPlan | None = None
    insertion_context: dict[str, object] = field(default_factory=dict)


class ROISelector:
    """Graph-aware ROI selector без деградации coverage."""

    def __init__(self) -> None:
        self.roi = SemanticROIHelper()

    def _bbox_from_keypoints(self, keypoints: list, margin: float = 0.03) -> BBox | None:
        """Строит bbox из набора keypoints, если confidence достаточный."""
        pts = [(k.x, k.y) for k in keypoints if getattr(k, "confidence", 0.0) > 0.2]
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0.0, min(xs) - margin)
        y0 = max(0.0, min(ys) - margin)
        x1 = min(1.0, max(xs) + margin)
        y1 = min(1.0, max(ys) + margin)
        return BBox(x0, y0, max(0.05, x1 - x0), max(0.05, y1 - y0))

    def semantic_roi_from_graph(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        return self.roi.region_from_graph(scene_graph, entity_id, region_type)

    def fallback_roi_from_person_bbox(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        person = next((p for p in scene_graph.persons if p.person_id == entity_id), None)
        if person is None:
            return None
        return RegionRef(
            make_region_id(entity_id, region_type),
            self.roi.fallback_person_bbox(person.bbox, region_type),
            "fallback:person_bbox_template",
        )

    def _resolve_region(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> RegionRef | None:
        return self.roi.resolve_region(scene_graph, entity_id, region_type)

    def select(self, scene_graph: SceneGraph, delta: GraphDelta) -> list[RegionRef]:
        selected = list(delta.newly_revealed_regions)
        selected.extend(delta.newly_occluded_regions)

        entity = delta.affected_entities[0] if delta.affected_entities else (scene_graph.persons[0].person_id if scene_graph.persons else "scene")
        requested: list[str] = list(delta.affected_regions)

        if delta.expression_deltas:
            requested.append("face")
        if delta.garment_deltas:
            requested.append("outer_garment")
        if "pose_transition" in delta.semantic_reasons:
            requested.extend(["legs", "pelvis"])

        for region_type in requested:
            resolved = self._resolve_region(scene_graph, entity, region_type)
            if resolved is not None:
                selected.append(resolved)
                continue
            fallback = self.fallback_roi_from_person_bbox(scene_graph, entity, region_type)
            if fallback is not None:
                selected.append(fallback)

        dedup: dict[str, RegionRef] = {}
        for region in selected:
            region_entity, region_type = parse_region_id(region.region_id)
            canonical = make_region_id(region_entity, region_type)
            dedup[canonical] = RegionRef(canonical, region.bbox, region.reason)

        if not dedup and scene_graph.persons:
            person = scene_graph.persons[0]
            fallback = self.fallback_roi_from_person_bbox(scene_graph, person.person_id, "fallback")
            if fallback is not None:
                dedup[fallback.region_id] = fallback
        return list(dedup.values())


def _is_empty_roi(value) -> bool:
    if value is None:
        return True
    dims = shape(value)
    return len(dims) < 3 or dims[0] <= 0 or dims[1] <= 0 or dims[2] <= 0


class RegionModeRouter:
    """Structured роутер режимов рендера."""

    def route(self, renderer: "PatchRenderer", *, scene_graph: SceneGraph, delta: GraphDelta, memory: VideoMemory, region: RegionRef, roi: list[list[list[float]]]) -> RenderRouteDecision:
        entity_id, region_type = parse_region_id(region.region_id)
        transition_mode = str(delta.region_transition_mode.get(region_type, "stable") or "stable")
        in_reveal = any(r.region_id == region.region_id for r in delta.newly_revealed_regions)
        slot = memory.hidden_region_slots.get(region.region_id)
        slot_hidden = bool(slot and slot.hidden_type in {"known_hidden", "unknown_hidden", "decayed_unknown"})
        exists_in_graph = any(p.person_id == entity_id for p in scene_graph.persons) or any(o.object_id == entity_id for o in scene_graph.objects)
        exists_in_memory = entity_id in memory.identity_memory or entity_id in memory.garment_memory
        insertion_ctx = delta.transition_diagnostics.get("insertion_context", {}) if isinstance(delta.transition_diagnostics, dict) else {}
        explicit_insert = bool(insertion_ctx.get(region.region_id) or insertion_ctx.get(entity_id))

        if explicit_insert or (not exists_in_graph and not exists_in_memory):
            mode: RenderMode = "insert_new"
            lifecycle: EntityLifecycle = "newly_inserted"
            reasons = ["entity_absent_in_scene_and_memory"]
        elif in_reveal or (slot_hidden and transition_mode in {"garment_reveal", "visibility_reveal", "pose_exposure"}):
            mode = "reveal"
            lifecycle = "previously_hidden_now_revealed"
            reasons = ["region_marked_revealed_or_hidden_slot_exists"]
        else:
            lifecycle = "already_existing"
            if transition_mode in {"stable", ""}:
                mode = "keep"
            elif transition_mode in {"expression_refine"} or region_type in {"face", "head"}:
                mode = "refine"
            elif transition_mode in {"pose_exposure", "pose_deform", "deform_relation_aware"}:
                mode = "deform"
            elif transition_mode in {"garment_surface", "open_front", "remove_outer"}:
                mode = "warp"
            else:
                mode = "refine"
            reasons = [f"existing_region_mode={mode}"]

        plan = renderer._build_plan(scene_graph, delta, memory, region, roi)
        reasons.append(f"transition_mode={transition_mode}")
        return RenderRouteDecision(
            mode=mode,
            lifecycle=lifecycle,
            reason_trace=reasons,
            memory_dependency={
                "has_hidden_slot": bool(slot),
                "hidden_type": slot.hidden_type if slot else "none",
                "texture_patch_count": len(memory.texture_patches),
                "region_descriptor_exists": region.region_id in memory.region_descriptors,
                "retrieval_top_score": float(plan.retrieval_summary.get("top_score", 0.0)),
            },
            plan=plan,
            insertion_context=insertion_ctx.get(region.region_id, insertion_ctx.get(entity_id, {})),
        )


class ExistingRegionUpdater:
    def update(self, renderer: "PatchRenderer", *, roi: list[list[list[float]]], delta: GraphDelta, plan: PatchSynthesisPlan, mode: RenderMode) -> tuple[list[list[list[float]]], dict[str, object], UpdatePathContract]:
        backend_used = "bootstrap_fallback"
        fallback_reason = "mode_not_torch_eligible"
        runtime: dict[str, object] = {"runtime_status": "bootstrap_mode"}
        if renderer.learnable_backend_mode == "torch_learned":
            runtime = renderer.torch_backends.backend_runtime_status("existing_update")
            if runtime["usable_for_inference"] and renderer.torch_backends.existing is not None:
                batch = build_renderer_tensor_batch(
                    roi=roi,
                    path_type="existing_update",
                    transition_mode=plan.transition_mode,
                    hidden_mode=plan.hidden_reconstruction_mode,
                    retrieval_top_score=float(plan.retrieval_summary.get("top_score", 0.0)),
                    memory_hint_strength=float(plan.retrieval_summary.get("top_score", 0.0)),
                    lifecycle="already_existing",
                    region_role="primary",
                    region_type=plan.region_type,
                    entity_type=plan.entity_class or "generic_entity",
                    transition_strength=float(plan.risk_profile.get("retrieval_dependency", 0.0)),
                    retrieval_evidence=float(plan.retrieval_summary.get("top_score", 0.0)),
                    scene_context_strength=float(plan.confidence_prior),
                )
                out = renderer.torch_backends.existing.forward(batch)
                contract = UpdatePathContract(mode=mode, reuse_fraction=0.58, synth_fraction=0.24, refinement_fraction=0.18, target_consistency=0.88, training_tags=["update", "torch_learned", mode])
                return out.rgb, {
                    "module": "existing_updater",
                    "operation": mode,
                    "backend_selection": "torch_learned_primary",
                    "backend_type": "torch_learned",
                    "backend_fallback_reason": "none",
                    "tensor_batch_surface": serialize_tensor_batch(batch),
                    "backend_runtime_status": runtime,
                    **out.backend_trace,
                }, contract
            backend_used = "torch_learned_unusable_fallback"
            fallback_reason = str(runtime.get("checkpoint_status", "torch_unavailable"))
        if mode == "keep":
            contract = UpdatePathContract(mode="keep", reuse_fraction=1.0, synth_fraction=0.0, refinement_fraction=0.0, target_consistency=0.98, training_tags=["update", "preserve"])
            return roi, {"module": "existing_updater", "operation": "keep", "reuse_ratio": 1.0, "backend_selection": backend_used, "backend_type": "bootstrap", "backend_fallback_reason": fallback_reason, "backend_runtime_status": runtime}, contract
        proposal, proposal_trace = renderer._build_proposal(roi, renderer._active_memory or VideoMemory(), plan, delta)
        if mode == "warp":
            contract = UpdatePathContract(mode="warp", reuse_fraction=0.82, synth_fraction=0.1, refinement_fraction=0.08, target_consistency=0.9, training_tags=["update", "warp"])
            return proposal, {"module": "existing_updater", "operation": "warp", "reuse_ratio": 0.82, "backend_selection": backend_used, "backend_type": "bootstrap", "backend_fallback_reason": fallback_reason, "backend_runtime_status": runtime, **proposal_trace}, contract
        if mode == "deform":
            deform_trace = renderer._apply_pose_deform(proposal, plan.region_type)
            if "proposal" in deform_trace:
                proposal = deform_trace.pop("proposal")
            contract = UpdatePathContract(mode="deform", reuse_fraction=0.74, synth_fraction=0.16, refinement_fraction=0.1, target_consistency=0.86, training_tags=["update", "deform"])
            return proposal, {"module": "existing_updater", "operation": "deform", "reuse_ratio": 0.74, "backend_selection": backend_used, "backend_type": "bootstrap", "backend_fallback_reason": fallback_reason, "backend_runtime_status": runtime, **proposal_trace, **deform_trace}, contract
        refined, refine_trace = renderer._refine_proposal(proposal, plan)
        contract = UpdatePathContract(mode="refine", reuse_fraction=0.66, synth_fraction=0.12, refinement_fraction=0.22, target_consistency=0.91, training_tags=["update", "refine"])
        return refined, {"module": "existing_updater", "operation": "refine", "reuse_ratio": 0.66, "backend_selection": backend_used, "backend_type": "bootstrap", "backend_fallback_reason": fallback_reason, "backend_runtime_status": runtime, **proposal_trace, **refine_trace}, contract


class RevealRegionSynthesizer:
    def _reveal_type(self, delta: GraphDelta, plan: PatchSynthesisPlan) -> str:
        mode = str(plan.transition_mode)
        if mode in {"garment_reveal", "open_front", "remove_outer"}:
            return "garment_change_reveal"
        if mode in {"pose_exposure", "pose_deform"}:
            return "pose_exposure_reveal"
        if "occlusion" in " ".join(delta.semantic_reasons):
            return "occlusion_reveal"
        return "generic_reveal"

    def synthesize(self, renderer: "PatchRenderer", *, roi: list[list[list[float]]], delta: GraphDelta, plan: PatchSynthesisPlan) -> tuple[list[list[list[float]]], dict[str, object], RevealPathContract, float, list[list[float]]]:
        reveal_type = self._reveal_type(delta, plan)
        if renderer.learnable_backend_mode == "torch_learned":
            runtime = renderer.torch_backends.backend_runtime_status("reveal")
        else:
            runtime = {"runtime_status": "bootstrap_mode"}
        if renderer.learnable_backend_mode == "torch_learned" and runtime.get("usable_for_inference") and renderer.torch_backends.reveal is not None:
            batch = build_renderer_tensor_batch(
                roi=roi,
                path_type="reveal",
                transition_mode=plan.transition_mode,
                hidden_mode=plan.hidden_reconstruction_mode,
                retrieval_top_score=float(plan.retrieval_summary.get("top_score", 0.0)),
                memory_hint_strength=float(plan.retrieval_summary.get("top_score", 0.0)),
                lifecycle="previously_hidden_now_revealed",
                region_role="primary",
                region_type=plan.region_type,
                entity_type=plan.entity_class or "generic_entity",
                reveal_type=reveal_type,
                retrieval_evidence=float(plan.retrieval_summary.get("top_score", 0.0)),
                reveal_memory_strength=float(plan.retrieval_summary.get("top_score", 0.0)),
                transition_strength=float(plan.risk_profile.get("retrieval_dependency", 0.0)),
            )
            out = renderer.torch_backends.reveal.forward(batch)
            contract = RevealPathContract(
                reveal_type=reveal_type,
                hidden_mode=plan.hidden_reconstruction_mode,
                memory_usage_ratio=0.72,
                reconstruction_bias=0.73,
                hallucination_budget=0.22,
                training_tags=["reveal", "torch_learned", reveal_type],
            )
            return out.rgb, {
                "module": "reveal_synthesizer",
                "operation": "reveal",
                "backend_selection": "torch_learned_primary",
                "backend_type": "torch_learned",
                "backend_fallback_reason": "none",
                "reveal_type": reveal_type,
                "hidden_mode": plan.hidden_reconstruction_mode,
                "tensor_batch_surface": serialize_tensor_batch(batch),
                "backend_runtime_status": runtime,
                **out.backend_trace,
            }, contract, out.confidence, out.uncertainty
        memory = renderer._active_memory or VideoMemory()
        proposal, proposal_trace = renderer._build_proposal(roi, memory, plan, delta)
        hidden_mode = plan.hidden_reconstruction_mode
        retrieval_top = float(plan.retrieval_summary.get("top_score", 0.0))
        h, w, c = shape(proposal)
        out = zeros(h, w, c)

        if hidden_mode == "known_hidden":
            blend = min(0.78, 0.45 + 0.35 * retrieval_top)
            for y in range(h):
                for x in range(w):
                    for k in range(c):
                        out[y][x][k] = max(0.0, min(1.0, proposal[y][x][k] * blend + roi[y][x][k] * (1.0 - blend)))
            uncertainty = [[max(0.08, 0.28 - 0.18 * retrieval_top) for _ in range(w)] for _ in range(h)]
        else:
            mc = mean_color(roi)
            for y in range(h):
                for x in range(w):
                    edge = min(x, w - 1 - x, y, h - 1 - y) / max(1, min(h, w))
                    noise = (((x * 92821) ^ (y * 68917)) % 23) / 22.0
                    for k in range(c):
                        out[y][x][k] = max(0.0, min(1.0, proposal[y][x][k] * 0.56 + mc[k] * 0.3 + 0.08 * noise + 0.06 * edge))
            uncertainty = [[0.42 + 0.18 * (1.0 - min(x, w - 1 - x, y, h - 1 - y) / max(1, min(h, w) / 2)) for x in range(w)] for y in range(h)]

        refined, refinement_trace = renderer._refine_proposal(out, plan)
        contract = RevealPathContract(
            reveal_type=reveal_type,
            hidden_mode=hidden_mode,
            memory_usage_ratio=1.0 if hidden_mode == "known_hidden" else 0.45,
            reconstruction_bias=0.78 if hidden_mode == "known_hidden" else 0.52,
            hallucination_budget=0.14 if hidden_mode == "known_hidden" else 0.42,
            training_tags=["reveal", hidden_mode, reveal_type],
        )
        confidence = max(0.35, min(0.89, 0.52 + 0.32 * retrieval_top - (0.16 if hidden_mode != "known_hidden" else 0.0)))
        return refined, {
            "module": "reveal_synthesizer",
            "operation": "reveal",
            "backend_selection": "bootstrap_fallback" if renderer.learnable_backend_mode != "torch_learned" else "torch_learned_missing_checkpoint_fallback",
            "backend_type": "bootstrap",
            "backend_fallback_reason": "legacy_reveal_path" if renderer.learnable_backend_mode != "torch_learned" else str(runtime.get("checkpoint_status", "torch_unavailable")),
            "hidden_region_slots_used": bool(memory.hidden_region_slots.get(plan.region_id)),
            "texture_patch_memory_used": bool(plan.retrieval_summary.get("candidates")),
            "reveal_type": reveal_type,
            "hidden_mode": hidden_mode,
            "reveal_confidence_semantics": "retrieval_weighted" if hidden_mode == "known_hidden" else "hypothesis_weighted",
            "reveal_uncertainty_semantics": "low_center_known_hidden" if hidden_mode == "known_hidden" else "high_edge_unknown_hidden",
            "backend_runtime_status": runtime,
            **proposal_trace,
            **refinement_trace,
        }, contract, confidence, uncertainty


class NewEntityInserter:
    def _ctx_color(self, ctx: dict[str, object], fallback: list[float]) -> list[float]:
        app = ctx.get("appearance_conditioning", {}) if isinstance(ctx.get("appearance_conditioning", {}), dict) else {}
        palette = app.get("palette_rgb", [])
        if isinstance(palette, list) and palette:
            seed = palette[0]
            if isinstance(seed, list) and len(seed) >= 3:
                return [max(0.0, min(1.0, float(seed[0]))), max(0.0, min(1.0, float(seed[1]))), max(0.0, min(1.0, float(seed[2])))]
        return fallback

    def insert(self, *, roi: list[list[list[float]]], decision: RenderRouteDecision, region: RegionRef) -> tuple[list[list[list[float]]], list[list[float]], list[list[float]], dict[str, object], float, InsertionPathContract]:
        renderer = decision.insertion_context.get("_renderer")
        ctx = decision.insertion_context if isinstance(decision.insertion_context, dict) else {}
        plan = decision.plan
        entity_type = str(ctx.get("entity_type", "generic_entity"))
        pose_role = str(ctx.get("initial_pose_role", "neutral"))
        rel = ctx.get("relation_context", {}) if isinstance(ctx.get("relation_context", {}), dict) else {}
        app = ctx.get("appearance_conditioning", {}) if isinstance(ctx.get("appearance_conditioning", {}), dict) else {}
        insertion_score = float(max(0.0, min(1.0, 0.22 + 0.22 * bool(app) + 0.22 * bool(rel) + 0.18 * bool(ctx.get("scene_context")) + 0.16 * bool(entity_type))))
        if isinstance(renderer, PatchRenderer) and renderer.learnable_backend_mode == "torch_learned":
            runtime = renderer.torch_backends.backend_runtime_status("insertion")
        else:
            runtime = {"runtime_status": "bootstrap_mode"}
        if isinstance(renderer, PatchRenderer) and renderer.learnable_backend_mode == "torch_learned" and runtime.get("usable_for_inference") and renderer.torch_backends.insertion is not None:
            plan = decision.plan
            batch = build_renderer_tensor_batch(
                roi=roi,
                path_type="insertion",
                transition_mode=plan.transition_mode if plan else "stable",
                hidden_mode=plan.hidden_reconstruction_mode if plan else "not_hidden",
                retrieval_top_score=float((plan.retrieval_summary if plan else {}).get("top_score", 0.0)),
                memory_hint_strength=float((plan.retrieval_summary if plan else {}).get("top_score", 0.0)),
                lifecycle=decision.lifecycle,
                region_role="primary",
                region_type=plan.region_type if plan else "generic",
                entity_type=entity_type,
                insertion_type="new_entity",
                insertion_context_strength=insertion_score,
                appearance_conditioning_strength=0.9 if app else 0.1,
                scene_context_strength=0.85 if ctx.get("scene_context") else 0.2,
                pose_role=pose_role,
            )
            out = renderer.torch_backends.insertion.forward(batch)
            contract = InsertionPathContract(
                entity_type=entity_type,
                pose_role=pose_role,
                context_conditioning_score=insertion_score,
                reusable_artifact_expected=True,
                alpha_semantics="torch_predicted_alpha",
                uncertainty_semantics="torch_predicted_uncertainty",
                training_tags=["insert", "torch_learned", entity_type, pose_role],
            )
            trace = {
                "module": "new_entity_inserter",
                "operation": "insert_new",
                "backend_selection": "torch_learned_primary",
                "backend_type": "torch_learned",
                "backend_fallback_reason": "none",
                "learned_ready_interface": True,
                "tensor_batch_surface": serialize_tensor_batch(batch),
                "backend_runtime_status": runtime,
                "insertion_metadata": {
                    "target_region": region.region_id,
                    "entity_type": entity_type,
                    "pose_role": pose_role,
                    "context_score": insertion_score,
                    "relation_context": rel,
                    "appearance_conditioning": app,
                },
                **out.backend_trace,
            }
            return out.rgb, out.alpha, out.uncertainty, trace, out.confidence, contract
        h, w, c = shape(roi)
        out = zeros(h, w, c)
        base = mean_color(roi)
        primary = self._ctx_color(ctx, base)
        accent = [max(0.0, min(1.0, primary[2] + 0.1)), max(0.0, min(1.0, primary[0] + 0.08)), max(0.0, min(1.0, primary[1] + 0.05))]
        silhouette_alpha = [[0.0 for _ in range(w)] for _ in range(h)]
        for y in range(h):
            yn = y / max(1, h - 1)
            for x in range(w):
                xn = x / max(1, w - 1)
                cx = abs(xn - 0.5)
                if entity_type == "person":
                    body = 1.0 - min(1.0, ((cx / 0.30) ** 2 + ((yn - 0.60) / 0.42) ** 2))
                    head = 1.0 - min(1.0, ((cx / 0.16) ** 2 + ((yn - 0.20) / 0.18) ** 2))
                    mask = max(0.0, max(body, head))
                else:
                    rect = 1.0 if (0.18 <= xn <= 0.82 and 0.16 <= yn <= 0.84) else 0.0
                    round_corner = max(0.0, 1.0 - min(1.0, ((cx / 0.42) ** 4 + (abs(yn - 0.5) / 0.42) ** 4)))
                    mask = max(rect * 0.8, round_corner * 0.75)
                stripe = 0.15 * (1.0 if ((x // max(1, w // 8)) % 2 == 0) else -1.0)
                pose_bias = 0.06 if pose_role in {"standing", "active"} else 0.02
                rel_bias = 0.05 if rel.get("supported_by") else 0.0
                texture = max(0.0, mask) * (0.82 + pose_bias + rel_bias)
                silhouette_alpha[y][x] = max(0.0, min(1.0, texture))
                for k in range(c):
                    col = primary[k] * (0.68 + 0.2 * (1.0 - yn)) + accent[k] * 0.22 + stripe * 0.04
                    out[y][x][k] = max(0.0, min(1.0, roi[y][x][k] * (1.0 - silhouette_alpha[y][x]) + col * silhouette_alpha[y][x]))
        radial = alpha_radial(h, w)
        alpha = [[max(0.0, min(1.0, silhouette_alpha[y][x] * 0.92 + radial[y][x] * 0.08)) for x in range(w)] for y in range(h)]
        uncertainty = [[0.22 + 0.48 * (1.0 - silhouette_alpha[y][x]) for x in range(w)] for y in range(h)]
        ctx_score = insertion_score
        confidence = max(0.38, min(0.86, ctx_score))
        contract = InsertionPathContract(
            entity_type=entity_type,
            pose_role=pose_role,
            context_conditioning_score=ctx_score,
            reusable_artifact_expected=True,
            alpha_semantics="silhouette_composite_alpha",
            uncertainty_semantics="background_high_entity_core_low",
            training_tags=["insert", entity_type, pose_role],
        )
        trace = {
            "module": "new_entity_inserter",
            "operation": "insert_new",
            "backend_selection": "bootstrap_fallback" if not isinstance(renderer, PatchRenderer) or renderer.learnable_backend_mode != "torch_learned" else "torch_learned_missing_checkpoint_fallback",
            "backend_type": "bootstrap",
            "backend_fallback_reason": "legacy_insert_path" if not isinstance(renderer, PatchRenderer) or renderer.learnable_backend_mode != "torch_learned" else str(runtime.get("checkpoint_status", "torch_unavailable")),
            "bootstrap_mode": "renderer_local_insert_bootstrap",
            "learned_ready_interface": True,
            "insert_confidence_semantics": "context_conditioned_local_renderer",
            "insert_uncertainty_semantics": "silhouette_aware",
            "backend_runtime_status": runtime,
            "insertion_metadata": {
                "entity_type": entity_type,
                "target_region": region.region_id,
                "relation_context": rel,
                "initial_pose_role": pose_role,
                "appearance_conditioning": ctx.get("appearance_conditioning", {}),
                "context_score": ctx_score,
            },
        }
        return out, alpha, uncertainty, trace, confidence, contract


class LayeredCompositingSupport:
    def z_index_for_mode(self, mode: RenderMode) -> int:
        return {"keep": 1, "warp": 2, "deform": 2, "refine": 3, "reveal": 4, "insert_new": 5}.get(mode, 1)


class PatchRenderer:
    """Plan-driven ROI renderer с property-driven hidden policy и explainability."""

    def __init__(
        self,
        *,
        learnable_backend_mode: str = "bootstrap",
        torch_checkpoint_dir: str | None = None,
        torch_device: str = "cpu",
        torch_allow_random_init_for_dev: bool = False,
    ) -> None:
        self.memory_manager = MemoryManager()
        self.confidence_estimator = PatchConfidenceEstimator()
        self.mode_router = RegionModeRouter()
        self.existing_updater = ExistingRegionUpdater()
        self.reveal_synthesizer = RevealRegionSynthesizer()
        self.new_entity_inserter = NewEntityInserter()
        self.layering = LayeredCompositingSupport()
        self.learnable_backend_mode = learnable_backend_mode
        self.torch_backends = TorchRendererBackendBundle(device=torch_device, allow_random_init_for_dev=torch_allow_random_init_for_dev)
        self.torch_checkpoint_trace = {"loaded": [], "missing": [], "invalid": [], "mode": learnable_backend_mode}
        if learnable_backend_mode == "torch_learned" and torch_checkpoint_dir:
            self.torch_checkpoint_trace = self.torch_backends.load_checkpoint(torch_checkpoint_dir)
        self._active_memory: VideoMemory | None = None

    def _bbox_to_pixels(self, bbox: BBox, frame: list) -> tuple[int, int, int, int]:
        h, w, _ = shape(frame)
        x0 = max(0, min(w - 1, int(bbox.x * w)))
        y0 = max(0, min(h - 1, int(bbox.y * h)))
        x1 = max(x0 + 1, min(w, int((bbox.x + bbox.w) * w)))
        y1 = max(y0 + 1, min(h, int((bbox.y + bbox.h) * h)))
        return x0, y0, x1, y1

    def _blend_memory_patch(self, roi: list[list[list[float]]], memory_patch: list[list[list[float]]], weight: float) -> list[list[list[float]]]:
        h, w, c = shape(roi)
        mh, mw, _ = shape(memory_patch)
        out = zeros(h, w, c)
        for y in range(h):
            for x in range(w):
                my = min(mh - 1, int((y / max(1, h - 1)) * max(0, mh - 1))) if mh else 0
                mx = min(mw - 1, int((x / max(1, w - 1)) * max(0, mw - 1))) if mw else 0
                for k in range(c):
                    mv = memory_patch[my][mx][k] if mh and mw else roi[y][x][k]
                    out[y][x][k] = max(0.0, min(1.0, roi[y][x][k] * (1.0 - weight) + mv * weight))
        return out

    def _normalize_semantics(self, scene_graph: SceneGraph, entity_id: str, region_type: str) -> GarmentSemanticProfile:
        return SceneGraphQueries.normalize_garment_semantics(scene_graph, entity_id, region_type)

    def _select_family(self, region_type: str, delta: GraphDelta, hidden_mode: str, garment: GarmentSemanticProfile) -> \
    tuple[str, str]:
        mode = delta.region_transition_mode.get(region_type, "")
        has_garment_delta = bool(delta.garment_deltas)
        has_reveal_signal = mode == "garment_reveal" or region_type in {"inner_garment", "outer_garment"}
        if region_type in {"face", "head"}:
            return "identity_preserving_face_refinement", "face_refine"
        # Не загоняем torso в hidden reconstruction, если нет реального reveal/remove сигнала.
        if hidden_mode in {"known_hidden", "unknown_hidden"}:
            if region_type == "torso" and not has_garment_delta and not has_reveal_signal and mode in {"", "stable"}:
                return "conservative_body_update", "body_direct_update"
            return "hidden_region_reconstruction", f"hidden_{hidden_mode}"
        if mode == "garment_reveal" and region_type in {"inner_garment", "torso"}:
            return "inner_region_reveal_synthesis", "garment_reveal"
        if region_type in {"outer_garment", "garments"}:
            return "garment_surface_transition", "garment_surface_update"
        if garment.entity_class == "garment":
            if mode in {"garment_surface", "open_front", "remove_outer"}:
                return "garment_surface_transition", "garment_surface_update"
            if garment.semantic_confidence >= 0.18:
                return "garment_surface_transition", "garment_surface_update"
        if mode in {"pose_exposure", "pose_deform", "visibility_occlusion"} or region_type in {"left_arm", "right_arm",
                                                                                               "legs", "pelvis"}:
            return "pose_driven_body_garment_local_deformation", "pose_local_deform"
        if region_type in {"torso", "pelvis", "legs", "left_arm", "right_arm", "arm"}:
            return "conservative_body_update", "body_direct_update"
        return "conservative_fallback_repair", "fallback_repair"

    def _pre_hidden_mode(self, memory: VideoMemory, region_id: str) -> str:
        slot = memory.hidden_region_slots.get(region_id)
        if not slot:
            return "not_hidden"
        if slot.hidden_type == "known_hidden":
            return "known_hidden"
        if slot.hidden_type in {"unknown_hidden", "decayed_unknown"}:
            return "unknown_hidden"
        return "not_hidden"

    def _detect_hidden_mode(self, memory: VideoMemory, region: RegionRef, retrieval_summary: dict[str, object]) -> tuple[str, dict[str, object]]:
        """Жёсткий gating known/unknown на основе evidence качества retrieval."""
        slot = memory.hidden_region_slots.get(region.region_id)
        top = retrieval_summary.get("top_score_breakdown", {})
        top_score = float(retrieval_summary.get("top_score", 0.0))

        similarity = float(top.get("similarity", 0.0))
        same_entity = float(top.get("same_entity_bonus", 0.0))
        reveal = float(top.get("reveal_compatibility", 0.0))
        lifecycle = float(top.get("visibility_lifecycle_compatibility", 0.0))
        semantic = float(
            top.get("semantic_family_bonus", 0.0)
            + top.get("coverage_compatibility", 0.0)
            + top.get("attachment_compatibility", 0.0)
        )

        evidence = {
            "slot_exists": bool(slot),
            "slot_hidden_type": slot.hidden_type if slot else "none",
            "candidate_patch_count": len(slot.candidate_patch_ids) if slot else 0,
            "slot_confidence": slot.confidence if slot else 0.0,
            "slot_evidence_score": slot.evidence_score if slot else 0.0,
            "retrieval_top_score": top_score,
            "descriptor_similarity": similarity,
            "same_entity_evidence": same_entity,
            "semantic_compatibility": semantic,
            "reveal_suitability": reveal,
            "lifecycle_consistency": lifecycle,
        }

        if not slot:
            return "not_hidden", evidence

        known_gate = all(
            [
                slot.hidden_type == "known_hidden",
                len(slot.candidate_patch_ids) > 0,
                slot.confidence >= 0.35,
                slot.evidence_score >= 0.3,
                top_score >= 0.6,
                similarity >= 0.04,
                same_entity >= 0.12,
                semantic >= 0.08,
                lifecycle >= 0.03,
            ]
        )
        if known_gate:
            return "known_hidden", evidence
        if slot.hidden_type in {"unknown_hidden", "decayed_unknown", "known_hidden"}:
            return "unknown_hidden", evidence
        return "not_hidden", evidence

    def _build_plan(
            self,
            scene_graph: SceneGraph,
            delta: GraphDelta,
            memory: VideoMemory,
            region: RegionRef,
            roi: list[list[list[float]]],
    ) -> PatchSynthesisPlan:
        entity_id, region_type = parse_region_id(region.region_id)
        garment = self._normalize_semantics(scene_graph, entity_id, region_type)
        query_desc = self.memory_manager._patch_descriptor(roi)

        retrieval_pre = self.memory_manager.route_region_retrieval(
            memory,
            region.region_id,
            region_type,
            entity_id,
            query_descriptor=query_desc,
            garment_semantics=garment,
            transition_context={
                "transition_phase": delta.transition_phase,
                "visibility_phase": delta.state_after.get("visibility_phase", "stable"),
                "region_transition_mode": delta.region_transition_mode.get(region_type, ""),
            },
            hidden_mode=self._pre_hidden_mode(memory, region.region_id),
        )
        hidden_mode, hidden_evidence = self._detect_hidden_mode(memory, region, retrieval_pre)

        retrieval = self.memory_manager.route_region_retrieval(
            memory,
            region.region_id,
            region_type,
            entity_id,
            query_descriptor=query_desc,
            garment_semantics=garment,
            transition_context={
                "transition_phase": delta.transition_phase,
                "visibility_phase": delta.state_after.get("visibility_phase", "stable"),
                "region_transition_mode": delta.region_transition_mode.get(region_type, ""),
            },
            hidden_mode=hidden_mode,
        )
        family, strategy = self._select_family(region_type, delta, hidden_mode, garment)

        # Если стратегия уже не hidden, не тащим hidden mode дальше в plan.
        effective_hidden_mode = hidden_mode
        if strategy in {"body_direct_update", "pose_local_deform", "face_refine"}:
            effective_hidden_mode = "not_hidden"

        retrieval_mode = "retrieval_first" if effective_hidden_mode == "known_hidden" else (
            "hypothesis_first" if effective_hidden_mode == "unknown_hidden" else "evidence_aware"
        )

        candidate_count = int((retrieval.get("summary") or {}).get("candidate_count", 0))
        top_score = float(retrieval.get("top_score", 0.0))

        proposal_mode = "deterministic"
        if strategy == "face_refine":
            proposal_mode = "identity_expression_refine"
        elif strategy == "hidden_known_hidden":
            proposal_mode = "recall_reconstruction"
        elif strategy == "hidden_unknown_hidden":
            proposal_mode = "hypothesis_reconstruction"
        elif strategy == "garment_reveal":
            proposal_mode = "semantic_reveal"
        elif strategy == "garment_surface_update":
            proposal_mode = "semantic_surface_transition"
        elif strategy == "pose_local_deform":
            proposal_mode = "pose_guided_deformation"
        elif strategy == "body_direct_update":
            proposal_mode = "body_region_update"

        if candidate_count > 0 and top_score > 0.7 and effective_hidden_mode == "known_hidden":
            proposal_mode = "retrieval_guided_recall"

        return PatchSynthesisPlan(
            region_id=region.region_id,
            region_type=region_type,
            entity_id=entity_id,
            entity_class=garment.entity_class,
            selected_family=family,
            selected_strategy=strategy,
            retrieval_mode=retrieval_mode,
            retrieval_summary=retrieval,
            hidden_reconstruction_mode=effective_hidden_mode,
            transition_mode=delta.region_transition_mode.get(region_type, "stable"),
            garment_semantics=garment,
            risk_profile={
                "seam_sensitivity": 0.55 if region_type in {"face", "torso"} else 0.42,
                "retrieval_dependency": 0.75 if effective_hidden_mode == "known_hidden" else 0.25,
            },
            confidence_prior=0.58 if strategy not in {"fallback_repair", "hidden_unknown_hidden"} else 0.34,
            seam_sensitivity=0.64 if region_type in {"face", "head"} else (
                0.52 if region_type in {"torso", "pelvis"} else 0.43),
            proposal_mode=proposal_mode,
            refinement_mode="identity_safe" if strategy == "face_refine" else (
                "seam_preserving" if effective_hidden_mode in {"known_hidden", "unknown_hidden"} else "conservative"
            ),
            explainability={"hidden_evidence": hidden_evidence, "reason": region.reason},
        )

    def _apply_face_refinement(self, proposal: list[list[list[float]]], expr_delta: float) -> dict[str, object]:
        """Мягкий identity-preserving путь с mouth-area bias."""
        h, w, c = shape(proposal)
        for y in range(h):
            yn = y / max(1, h - 1)
            mouth_bias = max(0.0, (yn - 0.55) / 0.45)
            for x in range(w):
                xn = x / max(1, w - 1)
                center_bias = 1.0 - abs(xn - 0.5)
                local = 0.01 + 0.018 * expr_delta * mouth_bias * center_bias
                for k in range(c):
                    scale = 1.0 + local * (0.85 if k == 0 else 0.65)
                    proposal[y][x][k] = max(0.0, min(1.0, proposal[y][x][k] * scale))
        return {"face_expression_delta": expr_delta, "mouth_area_bias": True}

    def _apply_garment_surface(self, proposal: list[list[list[float]]], plan: PatchSynthesisPlan) -> dict[str, object]:
        h, w, _ = shape(proposal)
        gs = plan.garment_semantics
        openness = 0.03 if gs.front_openable else 0.01
        tension = 0.02 if gs.fit_hint in {"tight", "fitted"} else 0.008
        looseness = 0.018 if gs.fit_hint in {"loose", "relaxed"} else 0.006
        for y in range(h):
            yn = y / max(1, h - 1)
            for x in range(w):
                xn = x / max(1, w - 1)
                center_split = 1.0 - abs(xn - 0.5) * 2.0
                surface_shift = openness * center_split + (tension - looseness) * (1.0 - yn)
                proposal[y][x][1] = max(0.0, min(1.0, proposal[y][x][1] + surface_shift))
        return {"surface_openness": openness, "tension": tension, "looseness": looseness}

    def _apply_inner_reveal(self, proposal: list[list[list[float]]], plan: PatchSynthesisPlan) -> dict[str, object]:
        h, w, _ = shape(proposal)
        gs = plan.garment_semantics
        reveal_strength = 0.08 if gs.exposure_behavior == "revealing" else 0.04
        for y in range(h):
            yn = y / max(1, h - 1)
            for x in range(w):
                xn = x / max(1, w - 1)
                split = 1.0 - abs(xn - 0.5) * 2.0
                gain = reveal_strength * max(0.0, split) * (0.6 + 0.4 * yn)
                proposal[y][x][0] = max(0.0, min(1.0, proposal[y][x][0] + gain))
                proposal[y][x][2] = max(0.0, min(1.0, proposal[y][x][2] - gain * 0.25))
        return {"inner_reveal_strength": reveal_strength, "coverage_targets": gs.coverage_targets}

    def _apply_pose_deform(self, proposal: list[list[list[float]]], region_type: str) -> dict[str, object]:
        h, w, c = shape(proposal)
        out = zeros(h, w, c)

        bend_strength = {
            "left_arm": 0.04,
            "right_arm": 0.04,
            "sleeves": 0.035,
            "torso": 0.025,
            "pelvis": 0.05,
            "legs": 0.07,
        }.get(region_type, 0.02)

        for y in range(h):
            yn = y / max(1, h - 1)

            if region_type == "legs":
                shift_x = int(round((0.5 - abs(yn - 0.55)) * bend_strength * w))
                shift_y = int(round(yn * bend_strength * h * 0.35))
            elif region_type == "pelvis":
                shift_x = int(round((0.5 - abs(yn - 0.5)) * bend_strength * w * 0.6))
                shift_y = int(round(yn * bend_strength * h * 0.2))
            else:
                shift_x = int(round((0.5 - abs(yn - 0.5)) * bend_strength * w * 0.4))
                shift_y = 0

            for x in range(w):
                src_x = min(w - 1, max(0, x - shift_x))
                src_y = min(h - 1, max(0, y - shift_y))
                for k in range(c):
                    out[y][x][k] = proposal[src_y][src_x][k]

        return {
            "pose_region_scale": bend_strength,
            "region_type": region_type,
            "warp_like_deform": True,
            "proposal": out,
        }

    def _build_proposal(
            self,
            roi: list[list[list[float]]],
            memory: VideoMemory,
            plan: PatchSynthesisPlan,
            delta: GraphDelta,
    ) -> tuple[list[list[list[float]]], dict[str, object]]:
        proposal = [[px[:] for px in row] for row in roi]
        retrieval_candidates = plan.retrieval_summary.get("candidates") or []
        top = retrieval_candidates[0] if retrieval_candidates else None
        trace: dict[str, object] = {"proposal_mode": plan.proposal_mode}

        allow_retrieval = True

        # Для pose регионов запрещаем retrieval, если region mismatch.
        if plan.selected_strategy == "pose_local_deform" and top is not None:
            if plan.region_type in {"legs", "pelvis"} and top.region_type not in {"legs", "pelvis"}:
                allow_retrieval = False
                trace["retrieval_rejected"] = f"pose_region_mismatch:{top.region_type}"
            elif plan.region_type in {"left_arm", "right_arm", "arm", "left_hand", "right_hand",
                                      "hand"} and top.region_type not in {
                "left_arm",
                "right_arm",
                "arm",
                "left_hand",
                "right_hand",
                "hand",
            }:
                allow_retrieval = False
                trace["retrieval_rejected"] = f"pose_region_mismatch:{top.region_type}"

        # Для body direct update тоже не даём грязный retrieval с чужого региона.
        if plan.selected_strategy == "body_direct_update" and top is not None:
            if plan.region_type == "torso" and top.region_type != "torso":
                allow_retrieval = False
                trace["retrieval_rejected"] = f"body_region_mismatch:{top.region_type}"

        if top and allow_retrieval:
            mem_patch = memory.patch_cache.get(top.patch_id)
            if mem_patch:
                if plan.selected_strategy == "face_refine":
                    blend = 0.18
                elif plan.selected_strategy == "garment_surface_update":
                    blend = 0.24
                elif plan.selected_strategy == "garment_reveal":
                    blend = 0.18
                elif plan.selected_strategy == "body_direct_update":
                    blend = 0.12
                elif plan.selected_strategy == "pose_local_deform":
                    blend = 0.0
                elif plan.hidden_reconstruction_mode == "known_hidden":
                    blend = 0.40
                elif plan.hidden_reconstruction_mode == "unknown_hidden":
                    blend = 0.14
                else:
                    blend = 0.10

                if blend > 0.0:
                    proposal = self._blend_memory_patch(proposal, mem_patch, blend)
                    trace["retrieval_patch"] = top.patch_id
                    trace["retrieval_blend"] = blend
                else:
                    trace["retrieval_patch_skipped"] = top.patch_id
                    trace["retrieval_blend"] = 0.0

        if plan.selected_strategy == "face_refine":
            expr = float(delta.expression_deltas.get("smile_intensity", 0.0) if delta.expression_deltas else 0.0)
            trace.update(self._apply_face_refinement(proposal, expr))

        elif plan.selected_strategy == "garment_surface_update":
            trace.update(self._apply_garment_surface(proposal, plan))

        elif plan.selected_strategy == "garment_reveal":
            trace.update(self._apply_inner_reveal(proposal, plan))

        elif plan.selected_strategy == "hidden_known_hidden":
            trace["known_hidden_recall"] = True

        elif plan.selected_strategy == "hidden_unknown_hidden":
            h, w, c = shape(proposal)
            mc = mean_color(roi)
            for y in range(h):
                for x in range(w):
                    spatial = ((x * 73856093) ^ (y * 19349663)) % 17 / 16.0
                    for k in range(c):
                        jitter = (spatial - 0.5) * 0.04
                        proposal[y][x][k] = max(0.0, min(1.0, proposal[y][x][k] * 0.82 + mc[k] * 0.18 + jitter))
            trace["hypothesis_based"] = True

        elif plan.selected_strategy == "pose_local_deform":
            deform_trace = self._apply_pose_deform(proposal, plan.region_type)
            if "proposal" in deform_trace:
                proposal = deform_trace.pop("proposal")
            trace.update(deform_trace)

        elif plan.selected_strategy == "body_direct_update":
            h, w, c = shape(proposal)
            out = zeros(h, w, c)
            for y in range(h):
                yn = y / max(1, h - 1)
                for x in range(w):
                    xn = x / max(1, w - 1)
                    center = 1.0 - abs(xn - 0.5)
                    vertical = 1.0 - yn
                    scale = 1.0 + 0.006 * center * vertical
                    for k in range(c):
                        out[y][x][k] = max(0.0, min(1.0, proposal[y][x][k] * scale))
            proposal = out
            trace["body_direct_update"] = True

        else:
            trace["fallback_reason"] = "strategy_default"

        return proposal, trace

    def _refine_proposal(self, proposal: list[list[list[float]]], plan: PatchSynthesisPlan) -> tuple[list[list[list[float]]], dict[str, object]]:
        h, w, c = shape(proposal)
        out = zeros(h, w, c)
        for y in range(h):
            for x in range(w):
                edge = min(x, w - 1 - x, y, h - 1 - y) / max(1, min(w, h) / 2)
                seam_weight = 0.85 + 0.15 * max(0.0, min(1.0, edge))
                for k in range(c):
                    v = proposal[y][x][k]
                    if plan.refinement_mode == "identity_safe" and k == 0:
                        v *= 0.99
                    elif plan.refinement_mode == "seam_preserving":
                        v *= seam_weight
                    out[y][x][k] = max(0.0, min(1.0, v))
        return out, {"refinement_mode": plan.refinement_mode}

    def _estimate_confidence(self, plan: PatchSynthesisPlan) -> dict[str, object]:
        top_score = float(plan.retrieval_summary.get("top_score", 0.0))
        top_breakdown = plan.retrieval_summary.get("top_score_breakdown", {})
        garment = plan.garment_semantics

        semantic_completeness = (
                0.35
                + 0.15 * bool(garment.coverage_targets)
                + 0.15 * bool(garment.attachment_targets)
                + 0.15 * bool(garment.deformation_mode != "unknown")
        )

        semantic_compatibility = float(plan.retrieval_summary.get("top_semantic_compatibility", 0.0)) + 0.25 * float(
            top_breakdown.get("similarity", 0.0)
        )

        if plan.selected_strategy in {"pose_local_deform", "body_direct_update", "face_refine"}:
            hidden_prior = 0.5
            hallucination_risk = 0.18 if plan.selected_strategy != "face_refine" else 0.12
        else:
            hidden_prior = 0.86 if plan.hidden_reconstruction_mode == "known_hidden" else (
                0.22 if plan.hidden_reconstruction_mode == "unknown_hidden" else 0.5
            )
            hallucination_risk = 0.82 if plan.hidden_reconstruction_mode == "unknown_hidden" else 0.24

        missing_evidence_penalty = 0.34 if not plan.retrieval_summary.get("candidates") else 0.0

        # Если у pose/body retrieval слабый или region-mismatch, уменьшаем влияние retrieval.
        if plan.selected_strategy in {"pose_local_deform", "body_direct_update"}:
            retrieval_evidence = min(0.35, max(0.0, min(1.0, top_score)))
        else:
            retrieval_evidence = max(0.0, min(1.0, top_score))

        return self.confidence_estimator.estimate(
            {
                "strategy_prior": plan.confidence_prior,
                "retrieval_evidence": retrieval_evidence,
                "semantic_compatibility": max(0.0, min(1.0, semantic_compatibility)),
                "garment_semantics_completeness": max(0.0, min(1.0, semantic_completeness)),
                "hidden_prior": hidden_prior,
                "transition_difficulty": 0.67 if plan.transition_mode in {"remove_outer", "open_front",
                                                                          "pose_exposure"} else 0.42,
                "region_difficulty": 0.76 if plan.region_type in {"face", "head"} else (
                    0.54 if plan.region_type in {"torso", "pelvis"} else 0.45),
                "seam_risk": plan.seam_sensitivity,
                "hallucination_risk": hallucination_risk,
                "fallback_penalty": 0.28 if plan.selected_strategy == "fallback_repair" else 0.0,
                "missing_evidence_penalty": missing_evidence_penalty,
            },
            strategy=plan.selected_strategy,
        )

    def _register_inserted_entity(
        self,
        *,
        memory: VideoMemory,
        scene_graph: SceneGraph,
        region: RegionRef,
        patch: list[list[list[float]]],
        confidence: float,
        insertion_meta: dict[str, object],
    ) -> dict[str, object]:
        entity_id, region_type = parse_region_id(region.region_id)
        memory.region_descriptors[region.region_id] = RegionDescriptor(
            region_id=region.region_id,
            entity_id=entity_id,
            region_type=region_type,
            bbox=region.bbox,
            visibility="visible",
            confidence=confidence,
            last_update_frame=scene_graph.frame_index,
        )
        patch_id = f"insert::{region.region_id}:{scene_graph.frame_index}"
        desc = self.memory_manager._patch_descriptor(patch)
        memory.patch_cache[patch_id] = patch
        from core.schema import TexturePatchMemory

        memory.texture_patches[patch_id] = TexturePatchMemory(
            patch_id=patch_id,
            region_type=region_type,
            entity_id=entity_id,
            source_frame=scene_graph.frame_index,
            patch_ref=f"insert://{region.region_id}",
            confidence=confidence,
            descriptor=desc,
            evidence_score=confidence,
            semantic_family="inserted_entity",
            coverage_targets=[region_type],
            attachment_targets=[],
            suitable_for_reveal=True,
        )
        slot = memory.hidden_region_slots.get(region.region_id)
        if slot is None:
            from core.schema import HiddenRegionSlot

            slot = HiddenRegionSlot(slot_id=region.region_id, region_type=region_type, owner_entity=entity_id, hidden_type="known_hidden")
            memory.hidden_region_slots[region.region_id] = slot
        slot.candidate_patch_ids = [patch_id] + [pid for pid in slot.candidate_patch_ids if pid != patch_id][:4]
        slot.confidence = max(slot.confidence, confidence)
        slot.evidence_score = max(slot.evidence_score, confidence)
        slot.last_transition_reason = "inserted_entity_seed"
        if entity_id not in memory.identity_memory:
            from core.schema import MemoryEntry

            memory.identity_memory[entity_id] = MemoryEntry(
                entity_id=entity_id,
                entry_type="inserted_entity_identity_seed",
                embedding=self.memory_manager._descriptor_to_embedding(desc),
                confidence=max(0.35, confidence),
                last_seen_frames=[scene_graph.frame_index],
            )
        memory.last_transition_context["last_inserted_entity"] = entity_id
        return {
            "patch_id": patch_id,
            "registered_region_descriptor": True,
            "registered_texture_patch": True,
            "registered_hidden_slot": True,
            "registered_identity_seed": True,
            "provenance": insertion_meta,
        }

    def render(
        self,
        current_frame: list,
        scene_graph: SceneGraph,
        delta: GraphDelta,
        memory: VideoMemory,
        region: RegionRef,
        image_tensor: str | list | None = None,
        crop_tensor: str | list | None = None,
    ) -> RenderedPatch:
        _ = (image_tensor, crop_tensor)
        x0, y0, x1, y1 = self._bbox_to_pixels(region.bbox, current_frame)
        roi = crop(current_frame, x0, y0, x1, y1)
        if _is_empty_roi(roi):
            roi = zeros(32, 32, 3)
        self._active_memory = memory
        decision = self.mode_router.route(self, scene_graph=scene_graph, delta=delta, memory=memory, region=region, roi=roi)
        decision.insertion_context["_renderer"] = self
        plan = decision.plan or self._build_plan(scene_graph, delta, memory, region, roi)
        confidence_payload = self._estimate_confidence(plan)
        learnable_surface: dict[str, object] = {}
        registration_summary: dict[str, object] = {"registered": False}

        if decision.mode == "insert_new":
            refined, alpha, uncertainty, module_trace, confidence, insert_contract = self.new_entity_inserter.insert(roi=roi, decision=decision, region=region)
            registration_summary = self._register_inserted_entity(
                memory=memory,
                scene_graph=scene_graph,
                region=region,
                patch=refined,
                confidence=confidence,
                insertion_meta=module_trace.get("insertion_metadata", {}),
            )
            registration_summary["registered"] = True
            learnable_surface = {"insertion_path_contract": asdict(insert_contract)}
        elif decision.mode == "reveal":
            refined, module_trace, reveal_contract, reveal_confidence, uncertainty = self.reveal_synthesizer.synthesize(self, roi=roi, delta=delta, plan=plan)
            rh, rw, _ = shape(refined)
            alpha = alpha_radial(rh, rw)
            confidence = max(float(confidence_payload["confidence"]) * 0.55 + reveal_confidence * 0.45, reveal_confidence * 0.85)
            learnable_surface = {"reveal_path_contract": asdict(reveal_contract)}
        else:
            refined, module_trace, update_contract = self.existing_updater.update(self, roi=roi, delta=delta, plan=plan, mode=decision.mode)
            rh, rw, _ = shape(refined)
            alpha = alpha_radial(rh, rw)
            confidence = float(confidence_payload["confidence"])
            uncertainty = [[1.0 - confidence for _ in range(rw)] for _ in range(rh)]
            learnable_surface = {"update_path_contract": asdict(update_contract)}

        selected_strategy = "EXISTING_REGION_UPDATE"
        if decision.mode == "reveal":
            selected_strategy = "KNOWN_HIDDEN_REVEAL" if plan.hidden_reconstruction_mode == "known_hidden" else "UNKNOWN_HIDDEN_SYNTHESIS"
        elif decision.mode == "insert_new":
            selected_strategy = "NEW_ENTITY_INSERTION"

        h, w, ch = shape(refined)
        structured_trace = {
            "selection": {
                "selected_family": plan.selected_family,
                "selected_strategy": plan.selected_strategy,
                "selected_render_mode": decision.mode,
                "entity_lifecycle": decision.lifecycle,
                "mode_reasons": decision.reason_trace,
                "retrieval_mode": plan.retrieval_mode,
                "transition_mode": plan.transition_mode,
                "proposal_mode": plan.proposal_mode,
                "refinement_mode": plan.refinement_mode,
            },
            "garment_semantics": asdict(plan.garment_semantics),
            "hidden_state": {
                "hidden_reconstruction_mode": plan.hidden_reconstruction_mode,
                "policy": "recall_based" if plan.hidden_reconstruction_mode == "known_hidden" else ("hypothesis_based" if plan.hidden_reconstruction_mode == "unknown_hidden" else "direct_update"),
                "evidence": plan.explainability.get("hidden_evidence", {}),
            },
            "retrieval": {
                "summary": plan.retrieval_summary.get("summary", {}),
                "candidates": plan.retrieval_summary.get("candidate_summaries", []),
                "top_candidate_score_breakdown": plan.retrieval_summary.get("top_score_breakdown", {}),
            },
            "module_trace": module_trace,
            "learnable_backend": {
                "requested_mode": self.learnable_backend_mode,
                "available": self.torch_backends.available,
                "checkpoint_trace": self.torch_checkpoint_trace,
                "backend_requested_mode": self.learnable_backend_mode,
                "backend_runtime_status": module_trace.get("backend_runtime_status", {}),
                "checkpoint_status": (module_trace.get("backend_runtime_status") or {}).get("checkpoint_status", "n/a"),
                "usable_for_inference": bool((module_trace.get("backend_runtime_status") or {}).get("usable_for_inference", False)),
                "selected_backend": module_trace.get("backend_selection", "bootstrap_fallback"),
                "backend_type": module_trace.get("backend_type", "bootstrap"),
                "fallback_reason": module_trace.get("backend_fallback_reason", "none"),
                "tensor_conditioning_summary": (module_trace.get("tensor_batch_surface") or {}).get("conditioning_summary", {}),
            },
            "memory_dependency_summary": decision.memory_dependency,
            "entity_registration_summary": registration_summary,
            "confidence": confidence_payload["decomposition"],
            "risks": confidence_payload["risks"],
            "selected_render_strategy": selected_strategy,
            "fallback_reason": plan.retrieval_summary.get("fallback_reason", "none"),
            "synthesis_mode": "insertion" if decision.mode == "insert_new" else ("retrieval" if decision.mode == "reveal" else "deterministic"),
            "layer_priority": self.layering.z_index_for_mode(decision.mode),
            "confidence_semantics_by_mode": {
                "mode": decision.mode,
                "confidence": confidence,
                "semantic": "context_conditioned_insert" if decision.mode == "insert_new" else ("reveal_memory_conditioned" if decision.mode == "reveal" else "update_consistency_conditioned"),
            },
            "uncertainty_semantics_by_mode": {
                "mode": decision.mode,
                "mean_uncertainty": sum(sum(row) for row in uncertainty) / max(1, len(uncertainty) * len(uncertainty[0]) if uncertainty else 1),
                "semantic": "entity_core_low_uncertainty" if decision.mode == "insert_new" else ("reveal_hidden_uncertainty" if decision.mode == "reveal" else "confidence_inverse_uniform"),
            },
            "reusable_output": {
                "reusable_next_frame": True,
                "reuse_reason": "seeded_texture_memory" if decision.mode == "insert_new" else ("hidden_reveal_candidate" if decision.mode == "reveal" else "existing_region_carry"),
            },
            "learnable_mode_surface": learnable_surface,
        }

        debug_trace = [
            f"region={region.region_id}",
            f"family={plan.selected_family}",
            f"strategy={selected_strategy.lower()}",
            f"render_mode={decision.mode}",
            f"lifecycle={decision.lifecycle}",
            f"hidden_mode={plan.hidden_reconstruction_mode}",
            f"retrieval_mode={plan.retrieval_mode}",
            f"proposal_mode={plan.proposal_mode}",
            f"retrieval_debug={plan.retrieval_summary.get('debug', plan.retrieval_summary.get('summary', {}))}",
        ]
        return RenderedPatch(region, refined, alpha, h, w, ch, uncertainty, confidence, self.layering.z_index_for_mode(decision.mode), debug_trace, structured_trace)
