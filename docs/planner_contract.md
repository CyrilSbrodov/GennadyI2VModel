# Sprint 4 Planner / Action Decomposition Contract

The planner is a strict contract layer between **Intent / Planner** and future **Dynamics / GraphDelta**. It converts a limited explicit user action into a validated `ActionPlan` that dynamics can consume later. It does not generate frames, mutate the scene graph, produce GraphDelta values, reveal hidden anatomy, run physics, call region routing, or select a renderer.

## Schema

The contract lives in `planning.action_plan` and defines:

- `ActionType` — small supported taxonomy.
- `PlannerIntent` — raw text or explicit `action_type`, optional `target_entity_id`, side, and object id.
- `ActionPlan` / `PlannedAction` — validated planner output.
- `ActionPhase` — ordered normalized action phases.
- `RegionPlanTarget` / `PlannerRegionRole` — canonical ontology or allowed garment/object region targets.
- `PlannerMemoryRequirement` — memory family requirements only; no reads or writes.
- `PlannerDynamicsRequirement` — future handoff metadata only; no GraphDelta generation.
- `PlannerVisibilityExpectation` and `PlannerOcclusionExpectation` — expected reasoning categories only.
- `PlannerTrace` — source/provenance/support-level diagnostics.
- `PlannerValidationError` — loud structured failure for unsupported or invalid plans.

Each `ActionPhase` includes `phase_id`, `phase_type`, `normalized_start`, `normalized_end`, `affected_regions`, `expected_motion_role`, visibility and occlusion expectations, required memory families, `dynamics_hint`, `routing_hint`, confidence/support level, and planner provenance.

## Supported action taxonomy

Supported initial action types are:

- Identity / expression: `expression_change`, `gaze_shift`, `head_turn`
- Upper body: `torso_turn`, `arm_raise`, `arm_lower`, `hand_reach`
- Lower body: `sit_down`, `stand_up`, `step_forward`
- Garment / object intent only: `garment_adjust`, `object_reach`

`garment_adjust` and `object_reach` are intent contracts only. They do not implement cloth simulation, reveal generation, object manipulation, or physics.

## Phase decomposition examples

Examples emitted by the contract:

- `head_turn`: `initial_hold` → `rotate_head` → `settle`
- `sit_down`: `initial_stand` → `knees_bend` → `pelvis_lower` → `torso_adjust` → `seated_settle`
- `stand_up`: `seated_start` → `torso_lean` → `pelvis_raise` → `knees_extend` → `standing_settle`
- `arm_raise`: `initial_hold` → `shoulder_lift` → `elbow_adjust` → `hand_settle`
- `hand_reach`: `initial_hold` → `shoulder_extend` → `elbow_extend` → `hand_target_approach` → `settle`
- `expression_change`: `initial_expression` → `expression_transition` → `expression_hold`

All phase times are normalized in `[0, 1]` and strictly monotonic.

## Canonical region targeting

Planner body targets use Sprint 2 `BODY_ONTOLOGY` region ids. The planner does not duplicate the ontology and validation rejects unknown regions.

Examples:

- `head_turn`: `head`, `face`, `hair`, `scalp`, `neck`
- `gaze_shift`: `face`, `left_eye`, `right_eye`
- `torso_turn`: `torso`, `upper_torso`, `lower_torso`, `chest`, `abdomen`, shoulders, with `pelvis` as stabilizer
- `arm_raise`: side-specific arm chain (`left_arm`, `left_upper_arm`, `left_elbow`, `left_forearm`, `left_wrist`, `left_hand`) or right-side equivalent
- `sit_down`: `pelvis`, `hips`, thighs, knees, calves, feet, with `torso` as stabilizer
- `expression_change`: `face`, `mouth`, `lips`, `jaw`, eyes

Private and optional sex-specific regions are ontology addresses only. They are not auto-targeted by generic actions, not reveal targets, not rendered, and not identity memory. Soft-tissue regions can only set future handoff flags such as `secondary_motion_required`; no soft-tissue dynamics are simulated.

## Entity resolution

Entity resolution is intentionally limited:

- Single-person scenes default to the only person.
- Multi-person scenes require explicit `target_entity_id`; otherwise `PlannerValidationError(code="ambiguous_target")` is raised.
- Explicit missing ids fail with `missing_target`.
- The planner does not perform face recognition or infer identity from free text.

## Unsupported action behavior

Intent parsing is limited to controlled phrases or explicit supported `action_type`. Unknown text is not mapped to a nearest action and never falls back to generic motion. If multiple supported controlled phrases appear in one prompt, the planner emits multiple `PlannedAction` entries in text order; each action keeps local normalized phase timing for this sprint.

Matched phrase spans are removed from normalized text, and remaining separator-delimited action-like fragments are reported as `unsupported_fragments` plus `unsupported_intent_fragment:<text>` trace reasons.

- Strict mode raises `PlannerValidationError` for fully unsupported prompts and raises `partial_unsupported_action` when supported actions coexist with unsupported action-like fragments.
- Non-strict mode returns `ActionPlan(supported=False, unsupported_code="unsupported_action", unsupported_reasons=...)` for fully unsupported prompts. For partial unsupported prompts, it returns `supported=True` with detected actions and unsupported-fragment diagnostics.

## Memory requirements

The planner emits memory requirements as references to memory families and canonical regions. It does not read memory, write memory, seed memory, or override Sprint 3 policy.

Examples:

- `head_turn` requires identity references for `face`, `head`, `hair`, and `scalp`, and locks those identity regions.
- `garment_adjust` requires garment references and can flag future reveal/occlusion reasoning, but does not generate reveal.
- `sit_down` may require body-shape and soft-tissue references and sets `secondary_motion_required` for future dynamics.

Private regions remain no-reference by default.

## Dynamics handoff

`PlannerDynamicsRequirement` provides future dynamics with:

- `expected_graph_delta_types`
- `required_dynamics_capabilities`
- `target_regions`
- `secondary_motion_required`
- `reveal_may_be_required`
- `occlusion_reasoning_required`
- `identity_lock_regions`
- `produces_graph_delta=False`

This is metadata only. The planner does not create `GraphDelta`, final coordinates, visibility state changes, or occlusion updates.

## Runtime integration

`runtime/orchestrator.py` invokes the Sprint 4 action planner only during the canonical `planning` stage and attaches the serialized contract to runtime diagnostics as `debug["planner_action_plan"]`. Existing canonical pipeline stages remain unchanged.

## Forbidden planner bypasses

The planner must not:

- Render, composite, refine frames, or select renderer modes.
- Call region routing or bypass routing.
- Mutate `SceneGraph` or update region positions.
- Generate `GraphDelta` or physics/reveal/cloth/soft-tissue simulation.
- Create observed evidence, mask evidence, or authoritative identity memory.
- Auto-target private or optional sex-specific regions.
