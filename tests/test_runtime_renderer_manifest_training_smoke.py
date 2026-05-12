from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.schema import GraphDelta
from learned.interfaces import DynamicsTransitionOutput, PatchSynthesisOutput, TemporalRefinementOutput
from representation.learned_bridge import BaselineGraphEncoder, BaselineIdentityAppearanceEncoder
from runtime.orchestrator import GennadyEngine
from text.learned_bridge import BaselineTextEncoderAdapter
from training.renderer_trainer import RendererTrainer
from training.types import StageResult, TrainingConfig


def _write_ppm(path: Path, w: int = 16, h: int = 16, rgb: tuple[int, int, int] = (90, 100, 130)) -> None:
    pixels = "\n".join(" ".join(map(str, rgb)) for _ in range(w * h))
    path.write_text(f"P3\n{w} {h}\n255\n{pixels}\n", encoding="utf-8")


class _DynamicsBackend:
    def predict_transition(self, request):
        entity_id = request.graph_state.persons[0].person_id
        delta = GraphDelta(
            affected_entities=[entity_id],
            affected_regions=["face"],
            expression_deltas={"smile_intensity": 0.4},
            region_transition_mode={"face": "expression_refine"},
            transition_phase="motion",
            semantic_reasons=["expression_delta"],
        )
        return DynamicsTransitionOutput(
            delta=delta,
            confidence=0.9,
            metadata={"semantic_families": ["face_expression"], "target_profile": {"primary_regions": ["face"]}},
        )


class _PatchBackend:
    def synthesize_patch(self, request):
        return PatchSynthesisOutput(
            region=request.region,
            rgb_patch=[[[0.2, 0.25, 0.3] for _ in range(2)] for _ in range(2)],
            alpha_mask=[[0.5, 0.5], [0.5, 0.5]],
            height=2,
            width=2,
            channels=3,
            confidence=0.8,
            execution_trace={
                "renderer_path": "learned_primary",
                "selected_render_strategy": "LEARNED_EXPRESSION_REFINE_PRIMARY",
                "synthesis_mode": "learned_expression_micro_edit",
                "metadata_used": bool(request.region_metadata),
            },
            metadata={"renderer_path": "learned_primary"},
        )


class _TemporalBackend:
    def refine_temporal(self, request):
        return TemporalRefinementOutput(
            refined_frame=request.current_composed_frame,
            region_consistency_scores={r.region_id: 0.8 for r in request.changed_regions},
            metadata={"temporal_path": "learned_primary"},
        )


def _engine() -> GennadyEngine:
    bundle = SimpleNamespace(
        graph_encoder=BaselineGraphEncoder(),
        identity_encoder=BaselineIdentityAppearanceEncoder(),
        text_encoder=BaselineTextEncoderAdapter(),
        dynamics_backend=_DynamicsBackend(),
        patch_backend=_PatchBackend(),
        temporal_backend=_TemporalBackend(),
        backend_names={"patch_backend": "unit_test"},
    )
    return GennadyEngine(backend_bundle=bundle)


def test_runtime_exported_renderer_manifest_trains_renderer_smoke(tmp_path: Path) -> None:
    img = tmp_path / "ref.ppm"
    manifest_path = tmp_path / "runtime_renderer_manifest.json"
    _write_ppm(img)

    _engine().run(
        [str(img)],
        "Улыбается.",
        quality_profile="debug",
        export_renderer_manifest_path=str(manifest_path),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "renderer_patch_manifest_v2"
    assert manifest["record_count"] > 0

    trainer = RendererTrainer()
    result = trainer.train(
        TrainingConfig(
            learned_dataset_path=str(manifest_path),
            epochs=1,
            train_size=1,
            val_size=1,
            renderer_backend="numpy_local",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
    )

    assert isinstance(result, StageResult)
    assert result.stage_name == "renderer"
    assert "manifest" in trainer.dataset_source
    assert not trainer.dataset_source.startswith("synthetic")
    assert trainer.dataset_diagnostics["manifest_path"] == str(manifest_path)
    assert trainer.dataset_diagnostics["loaded_records"] > 0
    assert result.train_metrics["train_post_filter_sample_count"] > 0
    assert result.val_metrics["val_post_filter_sample_count"] > 0
    assert result.val_metrics["usable_sample_count"] > 0.0
