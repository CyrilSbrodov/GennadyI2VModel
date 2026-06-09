# Sprint 5 Dynamics / GraphDelta Contract

The Dynamics stage is a strict contract boundary between the Sprint-4 `ActionPlan` and later Region Routing. It consumes `ActionPlan` / `PlannedAction` / `ActionPhase` and emits a validated `GraphDeltaContract` containing intent-level scene-state transition requests.

Dynamics is **not** a renderer, compositor, router, memory writer, perception system, learned video generator, physics engine, cloth simulator, reveal generator, or private/anatomical renderer.

## Schema

`src/dynamics/graph_delta_contract.py` defines:

- `GraphDeltaContract`: top-level supported/unsupported handoff, ordered `GraphDeltaStep` entries, read-only memory requirements, identity locks, routing candidates, trace, and explicit forbidden-operation flags.
- `GraphDeltaStep`: one action/phase step preserving `action_order`, `phase_order`, `action_type`, `phase_id`, `phase_type`, and the step's `RegionDelta` entries.
- `RegionDelta`: one canonical region transition intent with entity/region/action/phase provenance, delta type, identity/protection flags, routing/render-candidate hints, reveal/occlusion/secondary-motion markers, modification permissions, validation reasons, and optional intent payloads.
- Intent payloads: `PoseDeltaIntent`, `VisibilityDeltaIntent`, `OcclusionDeltaIntent`, `InteractionDeltaIntent`, and `GarmentIntentDelta`.
- `DynamicsTrace`: support level, planner unsupported fragments/reasons, planner trace reasons, and explicit forbidden-operation assertions.
- `DynamicsHandoffResult`: runtime wrapper containing the contract, routing candidates, and trace.

All contract classes serialize with `as_dict()` and contain only JSON-compatible values after serialization.

## Supported delta types

`RegionDeltaType` supports:

- `pose_delta`
- `expression_delta`
- `visibility_delta`
- `occlusion_delta`
- `interaction_delta`
- `garment_intent_delta`
- `secondary_motion_hint`

`visibility_delta` and `occlusion_delta` are emitted when planner dynamics capabilities request `visibility_update` or `occlusion_update`. They are reasoning requirements only: they do not claim final visibility state, create masks, create observed evidence, render, or route.

`secondary_motion_hint` is only a future requirement marker. It does not simulate soft tissue, breast dynamics, cloth, body physics, or final coordinates.

`garment_intent_delta` is only interaction intent. It cannot remove clothing, reveal hidden anatomy, or synthesize under-garment/body content.

## Planner-to-dynamics handoff

`build_dynamics_handoff(action_plan)` runs at the canonical Dynamics stage after runtime Planning is recorded. It validates the Sprint-4 plan and then walks actions in planner order. For each `PlannedAction`, each `ActionPhase` is preserved as a `GraphDeltaStep`; for each affected region, one or more `RegionDelta` entries are emitted.

Dynamics preserves:

- action order and phase order
- `action_type`, `phase_id`, `phase_type`
- region/entity/canonical-region linkage
- planner trace reasons and unsupported fragments
- identity lock regions
- planner memory requirements as **read-only context**

Unsupported `ActionPlan` input produces an unsupported `GraphDeltaContract` with no normal graph-delta steps. Partial unsupported fragments are preserved in `DynamicsTrace` while supported actions still produce deltas.

## Identity lock behavior

Identity-sensitive targets (`face`, `head`, `hair`, `scalp`, and targeted face subregions when `face` is locked) are represented with:

- `identity_locked=True`
- `protected_region=True`
- `allowed_to_modify_identity=False`

Dynamics may request pose or expression intent for identity regions, but it never creates identity evidence, identity embeddings, or authoritative identity memory. Later rendering must use memory policy and region routing; dynamics does not update identity.

## Private / optional restrictions

Private and sex-specific ontology regions are representation addresses only. Dynamics validation fails if such a region appears as a target or delta. These regions cannot become:

- auto-targeted motion regions
- visibility/reveal targets
- rendering candidates
- memory requirements
- identity lock regions

They remain no-reference/no-render by default.

## Routing handoff without routing decisions

Dynamics never calls Region Routing and never selects render modes. It only emits `routing_candidates` with:

- `region_id`
- `canonical_region_id`
- `reason`
- `delta_type`
- `action_type`
- `phase_id`
- `requires_rendering_candidate`

Region Routing consumes these later and remains responsible for decisions.

## Runtime diagnostics

`runtime/orchestrator.py` attaches the serialized contract at:

```python
debug["dynamics_graph_delta_contract"]
```

The canonical runtime trace still includes the Dynamics stage, followed by Region Routing before ROI Rendering.

## Validation boundary

Validation fails loudly with `DynamicsValidationError` for unknown delta types, unknown regions, missing action/phase provenance, empty supported steps, empty step deltas, non-monotonic ordering, private/optional targets, identity modification on locked regions, and forbidden claims such as rendered pixels, observed perception evidence, memory writes, scene graph mutation, learned motion, or region routing calls.
