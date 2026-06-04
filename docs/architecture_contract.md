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
