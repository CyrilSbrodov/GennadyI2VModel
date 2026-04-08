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

