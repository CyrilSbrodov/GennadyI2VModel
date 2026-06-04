# Architecture Contract

The project is a modular single-image-to-video scene engine, not a monolithic image/video generator.

## Canonical pipeline order

Every runtime trace must use these canonical stage names in this order:

1. `input`
2. `perception`
3. `scene_graph`
4. `memory`
5. `intent`
6. `planning`
7. `dynamics`
8. `region_routing`
9. `rendering`
10. `compositing`
11. `temporal_refinement`
12. `output`

Unknown stage names, out-of-order stages, and skipped mandatory stages are contract violations.

## Runtime trace scope

`runtime_trace` is a canonical stage summary for the whole engine invocation. It is not a per-step event log. Repeated per-step dynamics, region-routing, rendering, compositing, or temporal-refinement events must remain under `step_execution` or another nested per-step event trace so the top-level stage summary stays stable and contract-validated.

## Forbidden bypasses

- `rendering` must never appear before `region_routing`.
- ROI rendering must not run as an independent image-in/image-out shortcut.
- Rendering a patch requires an explicit routing context for the same canonical `entity:region` id.
- Rendering may not invent routing metadata or silently continue with an unknown routing decision.

## Rendering provenance requirements

Each rendering call must be linked to:

- canonical region id,
- routing decision and render mode,
- source provenance,
- material provenance: `observed`, `memory_assisted`, `generated`, `inferred`, `fallback`, or `unknown`.

## Identity memory provenance

Identity-sensitive regions are at least `face`, `head`, and `hair`.
Generated, inferred, or fallback material cannot become authoritative identity memory for these regions. Only observed high-confidence material may become authoritative when the memory policy allows it.

## Training/eval naming

Runtime, training, and evaluation code must use canonical stage semantics. `memory` remains a first-class stage and must not be normalized into `representation`. Legacy renderer names (`renderer`, `patch_synthesis`) normalize to the canonical `rendering` stage rather than diverging.

## Sprint 1 perception evidence contract

Perception is a strict source of observed facts, not a place to manufacture scene evidence. `PerceptionPipeline` now emits explicit person identity, tracking, mask, confidence, provenance, observation-status, and memory-seeding fields. Observed regions must carry observed detector/parser/face provenance; inferred, fallback, generated, synthetic, unknown, or missing evidence must remain labeled as such and must not be normalized into observed evidence.

Mask ownership is instance-local: every `PerceptionPipeline`/`ParserOnlyPipeline` owns an `InMemoryMaskStore`, attaches it to the frame context, snapshots it into `PerceptionOutput.mask_store`, and clears it per request when `reset_mask_store_per_analyze=True`. Production mask writers must fail loudly when no explicit frame-context mask store exists; `DEFAULT_MASK_STORE` is legacy/test-only. Any explicitly referenced legacy masks adopted for compatibility are tagged `adopted_legacy_default_store`/`not_production_observed` and cannot be treated as authoritative production-observed evidence. Video analysis clears once for the sequence and reuses the same store across frames so tracker and mask references stay coherent. Runtime rejects perception objects that do not expose a valid `mask_store`.

Single-image I2V identity is represented as `identity_observation_status="single_frame_anchor"` with `track_provenance="single_frame_observed"`; it is not treated as proven temporal tracking. In video perception, the first occurrence of a tracker id is `tracker_single_frame_observed`; only a repeated tracker id across frames becomes `multi_frame_tracked`.

Face, head, and hair are identity-sensitive regions. Parser/face evidence for those regions is preserved with provenance, confidence, observation status, and mask evidence type. Missing face/head/hair evidence remains unknown/missing; generated or fallback face/head/hair material cannot be marked observed. Parser masks are marked as `parser_mask`; bbox-only projections must remain distinguishable as bbox/fallback/inferred evidence.
