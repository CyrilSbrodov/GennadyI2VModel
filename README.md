# Gennady I2V Model (Modular MVP)

Модульный прототип **image-to-video scene engine**: входное изображение + текстовая инструкция → sequence of scene states → graph deltas → ROI patch rendering → composited video.

## Текущее состояние (честно)

### Уже working baseline
- **Real image grounding**: runtime стартует из реального входного изображения (через `InputAssetLayer`), а не из hash-seed кадра.
- **Input layer**: чтение single/multi image, базовое видео-плечо (decode при наличии OpenCV, иначе metadata fallback), unified asset container.
- **Text understanding v2**: нормализация, action lexicon + regex fallback, slot extraction, parser confidence/trace, sequential/parallel decomposition.
- **Scene graph builder**: построение graph из perception facts, relation inference (`part_of`, `covers`, `supports`, `occludes`), confidence/provenance, валидация ссылочной целостности.
- **Graph delta flow**: `graph + planned state -> GraphDelta`, затем `apply_delta(graph, delta)`.
- **ROI patch pipeline**: ROI selector → patch warp/refine baseline → alpha blend compositor → temporal stabilizer.
- **Memory subsystem**: identity/garment entries, texture patches, hidden-region slots, update/retrieve API.
- **Training data pipeline**: synthetic + real/pseudo path для Perception/Representation/Dynamics/Renderer/TextAction/Memory datasets.

### Что пока baseline / future learned hooks
- Perception adapters сейчас lightweight и largely deterministic.
- Dynamics rules + tiny learned-ready stub, без физически корректной симуляции.
- ROI patch renderer — deterministic local transform baseline (warp/refine), не diffusion/video foundation model.
- Continual-learning scaffold: контракты replay/mix/freeze/distillation готовы, но full production CL loop не реализован.

---

## Архитектура

1. `core/input_layer.py` — unified assets, resize profile, image/video loading.
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

## Runtime data flow

```text
input image/video
  -> InputAssetLayer (decode + normalize + unified asset)
  -> PerceptionPipeline (facts)
  -> SceneGraphBuilder (scene graph)
  -> IntentParser (structured action plan)
  -> TransitionPlanner (intermediate states)
  -> GraphDeltaPredictor (state deltas)
  -> ROISelector + PatchRenderer (local patch updates)
  -> Compositor + TemporalStabilizer
  -> frames/video export
```

---

## Быстрый запуск

```python
from runtime.orchestrator import GennadyEngine

engine = GennadyEngine()
artifacts = engine.run(
    images=["/path/to/input.ppm"],
    text="Снимает пальто и садится на стул, затем улыбается",
    quality_profile="balanced",  # debug / balanced / quality
)
print("frames:", len(artifacts.frames))
print("source mode:", artifacts.debug["input_metadata"]["source_mode"])
```

---

## Synthetic test mode

Если изображения не переданы, runtime использует explicit debug fallback frame (`source_mode=debug_fallback`).
Synthetic datasets сохраняются для smoke/integration тестов через `*.synthetic(...)` в `training/datasets.py`.

---

## Real input mode

- Передайте `images=[...]` (single или multi).
- Для видео используйте `video=...` через `InputAssetLayer.build_request(...)`.
- Resize зависит от `quality_profile` (`debug`, `balanced`, `quality`).

---

## Как добавить новый action

1. Добавить phrase/action mapping в `IntentParser._action_lexicon`.
2. Добавить fallback pattern в `_regex_fallback` (опционально).
3. Добавить transition labels в `TransitionPlanner._templates`.
4. Добавить rule/learned hook в `GraphDeltaPredictor._apply_action_rules`.
5. Добавить тест в `tests/test_intent_parser.py` и/или runtime integration tests.

---

## Как расширить graph schema

- Новые relation/fields добавляются в `core/schema.py`.
- Инференс отношений — `SceneGraphBuilder._infer_relations(...)`.
- Консистентность — `SceneGraphBuilder.validate(...)`.
- Обновления памяти/динамики — `memory/video_memory.py`, `dynamics/state_update.py`.

---

## Training flow

- `PerceptionDataset.from_image_manifest(...)` — real image records.
- `RepresentationDataset.from_perception_dataset(...)` — pseudo-labeled scene graphs.
- `DynamicsDataset.from_graph_sequence(...)` — graph delta samples.
- `RendererDataset.from_frame_pairs(...)` — ROI before/after pairs.
- `TextActionDataset.from_jsonl(...)` — text-action alignment.
- `MemoryDataset.from_representation_dataset(...)` — memory training records.

Пример manifest:

```json
{
  "records": [
    {"image": "/data/frame_0001.ppm", "text": "улыбается"}
  ]
}
```

