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

## Perception backends (builtin vs real)

`PerceptionPipeline` поддерживает backend-конфиг для perception модулей. Для parser теперь используется **human-centric stack** (`SegFormerHumanParserAdapter`) из 4 подпарсеров + fusion:

- person segmentation (`mediapipe` SelfieSegmentation или `builtin`),
- body part parser (`hf` human parsing model или `builtin`),
- garment parser (`hf` clothing/human parsing model или `builtin`),
- face region parser (`mediapipe` FaceMesh или `builtin`).

Важно: parser backend `hf` ожидает **модель human/clothing parsing**, а не generic scene segmentation.

Пример builtin режима:

```python
from perception.pipeline import PerceptionPipeline, PerceptionBackendsConfig

pipe = PerceptionPipeline(backends=PerceptionBackendsConfig())
out = pipe.analyze(frame_tensor)
# module_fallbacks['parser'] == 'builtin'
```

Пример real parser stack:

```python
from perception.pipeline import PerceptionBackendsConfig, PerceptionPipeline
from perception.parser import ParserBackendConfig, ParserStackConfig

parser_cfg = ParserStackConfig(
    person_segmentation=ParserBackendConfig(backend="mediapipe"),
    body_parts=ParserBackendConfig(backend="hf", model_id="<human-parsing-model>"),
    garments=ParserBackendConfig(backend="hf", model_id="<clothing-parsing-model>"),
    face_regions=ParserBackendConfig(backend="mediapipe"),
)

pipe = PerceptionPipeline(backends=PerceptionBackendsConfig(parser=parser_cfg))
out = pipe.analyze(frame_tensor)
```

Что реально умеет parser stack сейчас:
- person mask и mask_ref с payload в in-memory mask store,
- body-part masks (только для меток, которые реально есть в выходе выбранной модели),
- garment predictions (только для реально распознанных garment labels, без scene-label hacks),
- face-region hints (face/eyes/mouth),
- occlusion hints из масок (torso/arms/lower body, fragmented/uncertain).

Что пока ограничено:
- качество и детализация body/garment классов зависят от конкретной выбранной модели на HF,
- нет встроенной поставки весов в репозиторий, нужны внешние model_id/веса.
