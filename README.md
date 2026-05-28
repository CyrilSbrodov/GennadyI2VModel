# Gennady I2V Model (Modular MVP)

Модульный прототип **image-to-video scene engine**: входное изображение + текстовая инструкция → sequence of scene states → graph deltas → ROI patch rendering → composited video.

## Статус проекта (честно)

### Реально working baseline (сейчас)
- **Perception реально получает frame tensor / `AssetFrame`** и строит признаки от пикселей (mean/std luminance, edge density, aspect), а не от строкового `frame://...mean=...` как primary path.
- **Runtime действительно image-grounded**: `GennadyEngine` передаёт первый реальный кадр в perception.
- **Visual memory baseline стал image-driven**:
  - region mean/std,
  - coarse histogram,
  - edge density / energy,
  - patch cache по нескольким semantic regions (face/torso/sleeves/garments/pelvis/legs).
- **PatchRenderer реально использует memory**: region-specific retrieval (face / garments / generic), hidden-region candidates и debug trace о причинах выбора.
- **ROISelector стал graph/delta-semantics-aware**: учитывает `semantic_reasons`, `affected_regions`, sit/arm/face/garment сценарии.
- **GraphDelta расширен**: `affected_entities`, `affected_regions`, `semantic_reasons`, `predicted_visibility_changes`.
- **Canonical region-id contract**: единый формат `entity_id:region_type` во всех модулях (`GraphDelta`, `ROISelector`, `MemoryManager`, `PatchRenderer`).
- **TemporalStabilizer weak baseline**: стабилизация только внутри обновлённых ROI (без константного fake flow по всему кадру).
- **Training data builders расширены**:
  - `RepresentationDataset.from_perception_cache(...)`
  - `DynamicsDataset.from_transition_manifest(...)`
  - `RendererDataset.from_video_manifest(...)`
  - `TextActionDataset.from_annotation_manifest(...)`
  - `MemoryDataset.from_graph_sequence(...)`
  - graph cache `save_graph_cache(...)` / `load_graph_cache(...)`

### Deterministic baseline (и это явно deterministic)
- Перцепция и рендер по-прежнему heuristic/детерминированные адаптеры и hand-crafted processing, без полноценного learned visual backbone.
- ROI rendering — memory-aware compositional baseline, не diffusion/video model.
- Dynamics predictor использует rule-guided deltas + tiny learned-ready scaling stub.

### Future learned hooks (ещё не реализовано как learned)
- Замена perception adapters на полноценно обученные детекторы/парсеры/pose/face.
- Learned patch synthesis вместо deterministic blend/refine.
- Learned temporal model (optical-flow/transformer-level), а не weak ROI smoothing baseline.
- Полноценное обучение на production-scale аннотациях (текущий pipeline mostly pseudo/manifest-driven).

---

## Архитектура (сохранена модульность)
1. `core/input_layer.py` — unified assets, image/video loading.
2. `perception/*` — detector/pose/parser/face/objects/tracker adapters.
3. `representation/graph_builder.py` — scene graph build + relation reasoning.
4. `memory/video_memory.py` — appearance/garment/hidden region memory.
5. `text/intent_parser.py` — intent normalization + structured action plan.
6. `planning/transition_engine.py` — multi-step transition plan.
7. `dynamics/graph_delta_predictor.py` + `dynamics/state_update.py` — graph delta lifecycle.
8. `rendering/roi_renderer.py` + `rendering/compositor.py` — ROI render + compose + stabilization.
9. `runtime/orchestrator.py` — end-to-end orchestration.
10. `training/datasets.py` + `training/*` — stage datasets + training scaffold.

---

## Training data contracts
- `PerceptionDataset.from_image_manifest(...)` — ingestion из image manifest.
- `RepresentationDataset.from_perception_dataset(...)` / `from_perception_cache(...)` — graph samples.
- `DynamicsDataset.from_graph_sequence(...)` / `from_transition_manifest(...)` — transition/delta samples.
- `RendererDataset.from_frame_pairs(...)` / `from_video_manifest(...)` — ROI before/after pairs.
- `TemporalDataset.from_temporal_manifest(...)` — runtime-aligned temporal sequence samples (previous/current/target + region/hints/history context).
- `TextActionDataset.from_jsonl(...)` / `from_annotation_manifest(...)` — text-action alignment.
- `MemoryDataset.from_representation_dataset(...)` / `from_graph_sequence(...)` — memory records from visible regions.
- `save_graph_cache(...)` / `load_graph_cache(...)` — cache serialization helpers.

### Graph cache contract (partial but useful)
- `frame_index`
- `global_context` (`frame_size`, `fps`, `source_type`)
- `persons`: `person_id`, `track_id`, `bbox`, `confidence`, `garments`, `body_parts` (включая keypoint summary)
- `objects`: `object_id`, `object_type`, `bbox`, `confidence`
- `relations`: `source`, `relation`, `target`, `confidence`, `provenance`

### Perception cache contract for `RepresentationDataset.from_perception_cache(...)`
- ожидается `records[]` с cached perception facts:
  - `frame_index`
  - `frame_size`
  - `persons[]` (обязательно, включая `bbox`)
  - `objects[]` (опционально)
- builder строит graph из cached facts напрямую; если `persons` отсутствует и `strict=True`, выбрасывается ошибка.

> Важно: часть текущих датасетов остаётся pseudo-labeled baseline для smoke/debug. Контракты и структуры данных уже пригодны как learned-ready входы.

### Dynamics manifest-backed dataset (primary path when provided)

`DynamicsTrainer` теперь использует `DynamicsDataset.from_transition_manifest(...)` как **primary data path**, если задан `TrainingConfig.learned_dataset_path` (или `--learned-dataset-path` в CLI). Synthetic путь сохранён только как bootstrap fallback (single-sample/empty-manifest cases).

Минимальный contract записи `records[]`:
- `scene_graph` — текущий graph-like runtime input (`frame_index`, `persons[]`, optional `relations[]`).
- `actions` (или `labels`) — conditioning для planner/action tokens.
- `planner_context` — `step_index`, `total_steps`, `phase`, `target_duration`.
- `graph_delta_target` — целевой `GraphDelta` supervision:
  - `pose_deltas`
  - `garment_deltas`
  - `visibility_deltas`
  - `expression_deltas`
  - `interaction_deltas`
  - optional `newly_revealed_regions` / `newly_occluded_regions` / `region_transition_mode`
- optional runtime-aligned context:
  - `target_transition_context`
  - `memory_context`
  - `tags`, `notes`, `scenario`, `record_id`

Диагностика загрузчика:
- `total_records`, `loaded_records`, `usable_records`, `invalid_records`, `skipped_records`
- `invalid_examples`
- `family_counts` по meaningful delta groups
- `tag_counts`

Train/eval запуск:

```bash
python -m training.cli --stage dynamics --epochs 2 --learned-dataset-path /path/to/dynamics_manifest.json
python -m training.cli --eval-dynamics --weights-path artifacts/checkpoints/dynamics/dynamics_weights.json --learned-dataset-path /path/to/dynamics_manifest.json
```

Dynamics eval report считает component MSE по head-группам + `contract_valid_ratio`, `conditioning_sensitivity`, `fallback_free_ratio`, delta-group coverage (`*_group_coverage`), dataset diagnostics (`usable/invalid/skipped`) и regression `summary_score`.

### Temporal manifest-backed dataset (primary path when provided)

`TemporalTrainer` использует `TemporalDataset.from_temporal_manifest(...)` как **primary source**, если задан `TrainingConfig.learned_dataset_path` / `--learned-dataset-path`.
Synthetic temporal dataset остаётся только bootstrap fallback для empty/invalid manifest.

Минимальный manifest record (`records[]`) для runtime-aligned temporal contract:
- `previous_frame` (HxWx3)
- `current_composed_frame` (или `current_frame`) (HxWx3)
- `target_refined_frame` (или `target_frame`) (HxWx3)
- `changed_regions[]` (region_id/reason/bbox) **или** `changed_region_mask` (HxW/HxWx1)

Поддерживаемые conditioning/supervision поля:
- `alpha_hint`, `confidence_hint` (HxW/HxWx1)
- `patch_history` (список кадров/срезов для history-conditioning)
- `transition_context` (или `scene_transition_context`)
- `memory_context` (или `memory_transition_context`)
- `region_consistency_metadata`
- `record_id`, `scenario`/`scenario_id`, `tags`, `notes`

Loader diagnostics:
- `total_records`, `loaded_records`, `usable_records`, `invalid_records`, `skipped_records`
- `invalid_examples`
- `family_counts`:
  `frame_refinement`, `flicker`, `region_stability`, `alpha_consistency`, `confidence_calibration`,
  `history_conditioning`, `memory_conditioning`, `transition_conditioning`
- `tag_counts`, `scenario_counts`

Temporal train/eval запуск:

```bash
python -m training.cli --stage temporal_refinement --epochs 2 --learned-dataset-path /path/to/temporal_manifest.json
```

`TemporalTrainer` eval summary теперь возвращает:
- fidelity/reconstruction (`reconstruction_mae`)
- flicker consistency (`flicker_delta_mae`)
- changed-region stability (`region_consistency_mae`)
- seam/alpha proxy (`seam_temporal_mae`)
- confidence alignment proxy (`confidence_alignment_mae`)
- contract + dataset diagnostics (`contract_validity`, `usable_sample_count`, `invalid_records`, `skipped_records`, `tag_coverage`, `scenario_coverage`)
- `score` для regression tracking.

## Perception parser runtime stack (real backend pass)

`SegFormerHumanParserAdapter` теперь использует реальные runtime-paths, а не только adapter shells:

1. **FASHN (`backend="fashn"`)**: real HF image-segmentation inference (`fashn-ai/fashn-human-parser` по умолчанию), с runtime format detection (`label_map` или list-of-segments) и postprocess в binary masks.
2. **SCHP Pascal (`backend="schp_pascal"`)**: зарезервирован под SCHP-specific runtime path (не generic HF).
3. **SCHP ATR (`backend="schp_atr"`)**: зарезервирован под SCHP-specific runtime path (не generic HF).
4. **Generic HF fallback backends**: `generic_hf_structural` / `generic_hf_garment` для тех случаев, когда используется обычный HF image-segmentation runtime.
5. **FACER (`backend="facer"`)**: real FACER runtime path (face detector + face parser) с face-region extraction.

### Запуск parser smoke/debug

```bash
python scripts/parser_stack_smoke.py   --image ./assets/example.png   --out-dir artifacts/parser_smoke   --fashn-backend fashn   --fashn-model fashn-ai/fashn-human-parser   --schp-pascal-backend generic_hf_structural --schp-pascal-model <hf_model_id_or_checkpoint>   --schp-atr-backend generic_hf_garment --schp-atr-model <hf_model_id_or_checkpoint>   --facer-backend facer --facer-model farl/lapa/448
```

Скрипт сохраняет:
- `primary_fashn_masks/`
- `schp_pascal_masks/`
- `schp_atr_masks/`
- `facer_masks/`
- `fused_masks_overlay.png`
- `fused_summary.json`

### Optional/runtime dependencies

Для real parser runtime нужны optional пакеты:
- `transformers`, `torch`, `numpy`, `pillow` (FASHN + SCHP paths)
- `facer` (FACER path)

Если backend/веса недоступны, `PerceptionPipeline` остаётся fail-safe: warning + fallback на builtin parser path.

## Unified evaluation / benchmark

Добавлен единый runnable evaluation слой для assembled single-image pipeline.

### Stage-level eval

```bash
python -m evaluation.cli stage \
  --image /path/to/ref.ppm \
  --text "Снимает куртку и улыбается" \
  --backend-mode learned_primary \
  --output artifacts/eval/stage_eval.json
```

### End-to-end scenario benchmark + learned vs legacy comparison

По умолчанию benchmark теперь использует curated dataset manifest:
- `benchmark_assets/single_image_curated/manifest.json`
- `benchmark_assets/single_image_curated/images/*.ppm`

```bash
python -m evaluation.cli benchmark   --backend-modes learned_primary,legacy   --output artifacts/eval/benchmark_report.json
```

Фильтрация по subset сценариев/ассетов:

```bash
python -m evaluation.cli benchmark   --scenario-filter torso_garment_reveal,mixed_multi_step   --asset-filter torso_outerwear_front_ref   --output artifacts/eval/benchmark_subset.json
```

Явный manifest path (если нужен альтернативный curated pack):

```bash
python -m evaluation.cli benchmark   --dataset-manifest benchmark_assets/single_image_curated/manifest.json   --output artifacts/eval/benchmark_report.json
```

Legacy-режим с одним seed image сохранился через `--image`.

Manifest record contract (обязательные поля):
- `record_id`, `asset_id`, `asset_path`
- `scenario_id`, `canonical_prompt`
- `action_family`, `transition_family`
- `expected_region_families`, `expected_runtime_conditions`

Опционально: `tags`, `notes`, `weak_expectations`.

Benchmark report теперь содержит:
- dataset diagnostics (`total/valid/invalid/missing assets`),
- scenario + asset-level results,
- coverage по scenario families,
- missing/invalid dataset diagnostics,
- regression-oriented warnings и comparison deltas между `learned_primary` и `legacy`.

## Region Metadata Bridge

`PatchSynthesisRequest` now carries a structured `region_metadata` bridge so the renderer is conditioned on semantic evidence rather than only a `RegionRef` and broad transition context. The bridge is learned-ready contract data, not a claim of final image-quality improvement.

Flow:

```text
Perception masks/parser stats → SceneGraph canonical/body/garment nodes → SemanticROI reason/bbox → RegionRouter decision → runtime.region_metadata.build_region_metadata → PatchSynthesisRequest.region_metadata → trainable patch renderer conditioning + debug/audit traces
```

The metadata contract includes:

- core ROI identity: `region_id`, `entity_id`, `canonical_region`, normalized `bbox_xywh`, `roi_reason`, and `roi_source` (`parser_mask_bbox`, `body_part_keypoints`, `garment_coverage`, `person_bbox_fallback`, or `unknown`);
- graph source evidence: `source_node_type`, source node id/confidence/provenance, visibility/lifecycle, parser class, alternatives, and canonical region hints;
- mask evidence from `DEFAULT_MASK_STORE`: `mask_ref`, kind/confidence/source/backend, pixel count, bbox, frame size, and tags;
- routing/memory/transition evidence: route decision, renderer mode/reveal hints, synthesis requirement, memory reuse/reveal reliability, lifecycle, newly revealed/occluded flags, and semantic reasons;
- diagnostics: `metadata_completeness_score`, `evidence_strength_score`, `missing_fields`, `metadata_source_trace`, and `parse_error` for malformed region ids.

Runtime patch debug records expose a compact summary (`region_metadata_completeness_score`, source node type, mask kind, ROI source, missing fields, source trace), while `PatchSynthesisOutput.execution_trace` records whether metadata was used and which metadata-derived feature keys affected conditioning. Fallback ROIs remain valid, but they carry lower completeness and explicit missing mask evidence instead of silently pretending to be parser-derived regions.

## Renderer Training Manifest v2

Renderer patch training can now use a strict manifest-backed contract instead of ad-hoc ROI JSON. Runtime exports use `contract_version: "renderer_patch_manifest_v2"` at both the manifest and record level, preserving `region_metadata` as a structured dict all the way into renderer dataset loading and `PatchSynthesisRequest` reconstruction.

Required v2 record fields are: `record_id`, `frame_index`, `step_index`, `region_id`, `canonical_region`, `entity_id`, `roi_before`, `roi_after`, `alpha_mask`, `region_metadata`, `transition_context_summary`, `selected_render_strategy`, `synthesis_mode`, `execution_trace_summary`, `metadata_completeness_score`, `evidence_strength_score`, `roi_source`, `source_node_type`, `mask_kind`, and `mask_ref_present`. Optional supervision/conditioning tensors such as `changed_mask`, `preservation_mask`, `uncertainty_target`, and `seam_prior` are included when available.

The exporter deliberately stores compact summaries rather than raw runtime objects: graph delta affected entities/regions and reveal/occlusion semantics, memory support and retrieval reasons, route decisions, learned target profile, execution trace, source image/frame refs, and target provenance. Invalid v2 records raise clear validation errors in strict dataset mode; legacy/minimal manifests still load through the versioned compatibility path with explicit low-completeness metadata diagnostics.

Export runtime renderer records by passing `export_renderer_manifest_path` to `GennadyEngine.run(...)`. Smoke training/eval can read the resulting manifest through the existing renderer trainer path, for example:

```bash
python -m training.cli --stage renderer --epochs 1 --learned-dataset-path /path/to/renderer_manifest.json
python -m training.cli --eval-renderer --weights-path artifacts/checkpoints/renderer/renderer_weights.json --learned-dataset-path /path/to/renderer_manifest.json
```

Useful checks:

```bash
pytest tests/test_region_metadata_bridge.py
pytest tests/test_renderer_manifest_v2.py
pytest tests/test_real_human_parsing_integration.py
```

### Supervised renderer manifests from observed frame pairs

Use `training.renderer_video_manifest_builder.RendererVideoManifestBuilder` when the target is an observed video/paired-frame ROI rather than a runtime-generated patch. The builder crops `roi_before` from the source frame and `roi_after` from the observed target frame using each `RegionRef` bbox (or an explicit target bbox), then delegates serialization to `RendererManifestRecordExporter` so v2 field names and provenance stay unchanged.

```python
from core.schema import BBox, RegionRef, SceneGraph
from training.renderer_video_manifest_builder import build_renderer_video_manifest

build_renderer_video_manifest(
    output_path="artifacts/renderer_observed_pairs.json",
    source_frame=source_frame,  # list or np.ndarray, HxWx3
    target_frame=target_frame,  # observed/ground-truth paired frame
    scene_graph=SceneGraph(frame_index=0),
    regions=[RegionRef("person_1:face", BBox(0.25, 0.20, 0.30, 0.25), "parser_mask_bbox")],
    region_metadata={
        "person_1:face": {
            "roi_source": "parser_mask_bbox",
            "source_node_type": "body_part",
            "metadata_completeness_score": 0.9,
            "evidence_strength_score": 0.8,
        }
    },
    transition_context={"summary": "observed paired-frame expression transition"},
    strict=True,
)
```

Every exported observed-pair record is marked as supervised target provenance (`target_source: provided_ground_truth_roi`, `training_target_quality: external_or_observed_target`), which the existing renderer dataset policy maps to `target_training_role: supervised_external`. Missing region metadata still produces a valid low-completeness v2 record with diagnostics, while fallback `person_bbox` records remain valid but carry low metadata/evidence scores. In `strict=True`, the builder raises on the first invalid region or if no valid supervised records can be exported, rather than silently falling back to synthetic targets.

Train on the resulting manifest the same way as runtime v2 exports:

```bash
PYTHONPATH=src pytest tests/test_renderer_video_manifest_builder.py
python -m training.cli --stage renderer --epochs 1 --learned-dataset-path artifacts/renderer_observed_pairs.json
```

## Runtime readiness modes

- `debug_stub`: допускает builtin/bootstrap/fallback пути; только для smoke/debug.
- `trainable_stub`: trainable backend'ы могут стартовать с bootstrap весами без production-checkpoint, это **не** claim learned production runtime.
- `strict_learned`: requires valid **explicit checkpoint paths** for dynamics + renderer patch + temporal; при missing/invalid/unloadable checkpoint runtime падает fail-fast без silent fallback.
- `production_eval`: strict runtime policy for honest evaluation and also requires explicit checkpoint paths (silent fallback запрещен).

### Checkpoint honesty contract

- Если `runtime_mode` = `strict_learned` или `production_eval`, fallback forbidden (`fallback_forbidden=true`).
- Отсутствие checkpoints означает отсутствие learned-runtime claim.
- Bootstrap/legacy paths сохраняются только для `debug_stub`/`trainable_stub` и отмечаются в metadata как fallback/bootstrap.


## Observed-pair renderer dataset builder

Primary supervised renderer path uses **observed source/target frame pairs** and exports `renderer_patch_manifest_v2` with external ground-truth ROI targets.

- Input contract: `renderer_observed_pair_manifest_input_v1`
- Required per pair: `record_id`, `source_frame`, `target_frame`, `regions[]` (`region_id`, normalized `bbox`), optional `prompt`, `transition_context`, `region_metadata`.
- Every exported supervised record is enforced as:
  - `target_source = "provided_ground_truth_roi"`
  - `training_target_quality = "external_or_observed_target"`
  - `target_training_role = "supervised_external"`

Runtime-generated renderer outputs are debug/distillation artifacts only and are **not supervised ground truth**.

CLI:

```bash
python -m training.cli --stage renderer_manifest_from_observed_pairs \
  --observed-pairs-path data/observed_pairs.json \
  --output-path artifacts/renderer_observed_pairs.json \
  --strict
```

Strict mode fail-fast behavior:
- missing/unreadable `source_frame` or `target_frame`
- source/target shape mismatch
- missing/invalid regions or invalid bbox
- zero exported supervised records

Non-strict mode skips invalid pairs and writes diagnostics, but still never emits self-generated runtime targets as supervised records.

Recommended tests:

```bash
pytest -q tests/test_renderer_video_manifest_builder.py tests/test_renderer_observed_pairs_builder.py
```

## Supervised renderer training baseline

Observed-pair manifests are the **primary supervised renderer training path** in this baseline.
Runtime-generated targets are debug/distillation-only signals and **not supervised ground truth**.
This sprint trains the current baseline renderer, not production-quality video generation.
Saved renderer checkpoints are only as good as observed-pair data quality and current baseline model capacity.

1. Prepare observed pairs (template): `examples/observed_pairs.example.json`.
2. Build observed-pair manifest:

```bash
python -m training.cli --stage renderer_manifest_from_observed_pairs \
  --observed-pairs-path examples/observed_pairs.example.json \
  --output-path artifacts/renderer_observed_pairs.json \
  --strict
```

3. Train renderer:

```bash
python -m training.cli --stage renderer \
  --epochs 1 \
  --learned-dataset-path artifacts/renderer_observed_pairs.json \
  --checkpoint-dir artifacts/checkpoints \
  --strict-dataset
```

4. Run strict runtime with explicit checkpoints:

```bash
python -m runtime.cli \
  --runtime-mode strict_learned \
  --dynamics-checkpoint-path artifacts/checkpoints/dynamics/dynamics_weights.json \
  --patch-checkpoint-path artifacts/checkpoints/renderer/<saved_renderer_checkpoint>.json \
  --temporal-checkpoint-path artifacts/checkpoints/temporal_refinement/latest.json
```

## Supervised temporal refinement training baseline

Observed temporal sequences are the **primary supervised temporal refinement training path**. Runtime/generated frames are **not** temporal supervised ground truth.

1. Prepare temporal observed sequences:

`examples/temporal_sequences.example.json`

2. Build temporal manifest:

```bash
python -m training.cli --stage temporal_manifest_from_observed_sequences \
  --observed-sequences-path examples/temporal_sequences.example.json \
  --output-path artifacts/temporal_observed_sequences.json \
  --strict
```

3. Train temporal refinement baseline model:

```bash
python -m training.cli --stage temporal_refinement \
  --epochs 1 \
  --learned-dataset-path artifacts/temporal_observed_sequences.json \
  --checkpoint-dir artifacts/checkpoints \
  --strict-dataset
```

4. Use checkpoint in strict runtime (`strict_learned` requires explicit temporal checkpoint path):

```bash
python -m runtime.cli \
  --runtime-mode strict_learned \
  --dynamics-checkpoint-path artifacts/checkpoints/dynamics/dynamics_weights.json \
  --patch-checkpoint-path artifacts/checkpoints/renderer/<saved_renderer_checkpoint>.json \
  --temporal-checkpoint-path artifacts/checkpoints/temporal_refinement/latest.json
```

This trains the current baseline temporal refinement model and is **not** production-quality video generation.
