# Sprint 7 Region Routing v2 Contract

Region Routing v2 is the renderer-safe handoff layer between Sprint 5 Dynamics / GraphDelta, Sprint 6 Reveal / Occlusion Continuity, and the existing ROI renderer path. It is a contract and validation layer only: it does not render, composite, mutate the `SceneGraph`, write memory, create masks, create observed perception evidence, or generate pixels.

## Inputs

The contract builder consumes:

- `GraphDeltaContract.routing_candidates` for ordinary dynamics changes.
- `RevealContract.decisions` for reveal, block, and diagnostic states.
- `RevealContract.routing_candidates` for reveal-safe routing candidates.
- `SceneGraph` and optional `VideoMemory` / memory-manager references as read-only context.

The router reuses existing canonical region ids and the Sprint 2 body ontology rather than defining another scene graph or anatomy list.

## Decision types

Routeable decisions:

- `route_pose_update`
- `route_expression_update`
- `route_visibility_update`
- `route_interaction_update`
- `route_garment_intent`
- `route_reveal_observed_memory`
- `route_reveal_weak_memory`
- `route_preserve_visible`
- `route_memory_assisted_identity_locked`

Diagnostic / no-render decisions:

- `route_occlusion_reasoning_only`
- `route_newly_occluded_tracking_only`

Blocked decisions:

- `block_private_region`
- `block_unknown_defer`
- `block_identity_risk`
- `block_no_safe_memory`
- `block_unsupported_region`
- `block_no_route_evidence`
- `block_policy_rejected`

Blocked and diagnostic decisions are never renderer candidates and require `roi_required=False`. Runtime treats `renderable_region_ids` as the only renderer allowlist; absence from that allowlist means the ROI renderer must not execute for that region.

## Dynamics and reveal merge policy

Reveal decisions are processed before uncovered dynamics candidates. A reveal decision for a region/action/phase takes priority over the matching dynamics routing candidate. Region-level reveal blocks also prevent a dynamics candidate from becoming a normal renderer request.

Overrides include:

- `reveal_unknown_defer` becomes `block_unknown_defer`.
- `reveal_blocked_private` becomes `block_private_region`.
- `reveal_blocked_identity_risk` becomes `block_identity_risk`.
- `occlusion_reasoning_required` becomes diagnostic `route_occlusion_reasoning_only`.
- `newly_occluded` becomes tracking-only `route_newly_occluded_tracking_only`.
- `reveal_from_observed_memory` becomes `route_reveal_observed_memory`.
- `reveal_from_weak_memory` becomes `route_reveal_weak_memory` only for non-identity regions.
- If no reveal decision covers a dynamics candidate, its delta type maps to an ordinary route decision.

## Identity locks

Identity-sensitive regions (`face`, `head`, `hair`, `scalp`, and face subregions) are always identity-locked when routed. Identity-locked decisions set:

- `identity_locked=True`
- `allowed_to_modify_identity=False`
- `renderer_must_preserve_identity=True`
- `renderer_must_not_create_identity_memory=True`

Observed-memory identity reveals require `memory_authority="authoritative"`. Weak-memory identity reveal is blocked as identity risk, and generated / inferred / fallback identity material is never upgraded to authoritative identity memory.

## Private / optional regions

Private and optional sex-specific/private regions are addressable representation regions only. If they appear in dynamics or reveal inputs, Region Routing v2 blocks them loudly:

- `route_allowed=False`
- `render_candidate_allowed=False`
- `blocked=True`
- `block_reason="private_region"`
- `roi_required=False`
- no memory fallback, body/skin fallback, reveal route, or private render strategy

## Unknown / deferred regions

`reveal_unknown_defer` maps to `block_unknown_defer` with `block_reason="unknown_defer_no_safe_memory"`. It never receives an ROI request and never falls back to generated content.

Unsupported regions map to `block_unsupported_region`. Regions with no safe reveal memory map to `block_no_safe_memory`.

## Weak memory routing constraints

`route_reveal_weak_memory` is allowed only for non-identity regions after reveal validation. It carries strict downstream constraints:

- `weak_memory_candidate=True`
- `output_reference_authority="weak"`
- `memory_write_allowed=False`
- `identity_memory_write_allowed=False`
- `observed_evidence_claim_allowed=False`
- `renderer_must_not_create_observed_evidence=True`
- `renderer_must_not_create_identity_memory=True`

Renderer output from such a route remains non-authoritative until future verification.

## Reveal observed memory routing

`route_reveal_observed_memory` preserves `memory_reference_kind`, `memory_authority`, and `memory_family` from reveal evidence. For identity regions, the memory authority must be authoritative. The router itself does not create observed evidence.

## Diagnostic occlusion routing

`occlusion_reasoning_required` and `newly_occluded` are diagnostic/tracking states. They preserve provenance but do not produce render candidates, ROI requests, mask creation, or reveal generation.

## Relationship to the existing router

The existing `CanonicalRegionRouter` and runtime `RegionRoutingPlan` remain as compatibility machinery for the current ROI renderer. Region Routing v2 wraps that path as a stricter diagnostics/enforcement layer: runtime debug exposes `debug["region_routing_contract"]`, and renderer execution is allowed only for ids present in `renderable_region_ids`. Blocked, private, unknown-deferred, and diagnostic-only decisions are absent from that allowlist; diagnostic ids are not treated as hard region-level blocks when a separate routeable decision exists for the same region.

The v2 contract does not select learned renderer implementations. `selected_render_strategy` is absent unless a future compatibility path requires a non-learned placeholder.
