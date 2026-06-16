# Sprint 8 ROI Renderer Contract

The ROI renderer is the safe rendering boundary after Region Routing v2. It is an adapter contract, not a diffusion/video model. The renderer may create generated/refined pixels for an already-approved ROI, but it must never create observed perception evidence, masks-as-observed-evidence, memory, identity memory, private anatomy, hidden anatomy reconstruction, or clothing-removal output.

## Inputs

`ROIRenderRequest` is built from a renderable `RegionRoutingDecision` selected via `RegionRoutingContract.renderable_decision_for_region_id()` and the `RegionRoutingContract.renderable_region_ids` allowlist. It carries:

- region/entity/canonical region identifiers
- route decision type and GraphDelta/Reveal/Planner provenance
- action and phase ordering
- ROI bbox and current frame reference/shape
- identity preservation policy
- memory/reference authority and family
- inherited route validation reasons
- forbidden-operation constraints (`memory_write_allowed=False`, `observed_evidence_claim_allowed=False`)

The Sprint 7 `RegionRoutingDecision` remains authoritative. The renderer contract does not define a second routing policy.

## Renderer modes

Allowed contract modes are:

- `preserve`
- `warp`
- `deform`
- `refine`
- `reveal_memory`
- `weak_memory_refine`
- `expression_local_update`
- `pose_local_update`
- `garment_intent_update`

There are no private-region, hidden-anatomy, clothing-removal, or diffusion/video modes in this sprint.

## Blocked requests

The request builder raises `ROIRenderValidationError` when a route is blocked, diagnostic-only, not render-candidate-allowed, not route-allowed, not ROI-required, absent from `renderable_region_ids`, private/optional, identity-risky, weak identity reveal, occlusion-reasoning-only, newly-occluded-tracking-only, or otherwise unsupported.

Private or optional sex-specific regions such as `external_genital_region`, `male_external_genital_region`, `male_pelvic_region`, and `female_pelvic_region` cannot build requests. No fallback patch or mask should be created for them.

## Identity preservation

Identity-sensitive regions (`face`, `head`, `hair`, `scalp`, and future upstream identity-locked subregions) are rendered with:

- `identity_locked=True`
- `renderer_must_preserve_identity=True`
- `allowed_to_modify_identity=False`
- `identity_memory_write_allowed=False`

Weak identity reveal is rejected even if it somehow reaches the renderer. Observed identity reveal requires authoritative identity memory.

## Weak memory

`route_reveal_weak_memory` maps to `weak_memory_refine`, is allowed only for non-identity regions, and keeps `output_authority="weak"`. Weak-memory render output cannot be promoted to authoritative memory without future verification. For all other routes, source memory authority is preserved separately from renderer output authority; even authoritative memory references produce generated/generated-from-memory renderer output, never authoritative/reusable/observed output.

## Output

`ROIRenderOutput` records patch references, patch shape, alpha/mask runtime info, source route summary, reference usage, identity policy, safety policy, trace, and output authority. Validation rejects output that claims observed evidence, identity memory creation, memory writes, identity modification, SceneGraph mutation, private rendering, hidden anatomy generation, clothing removal, or authoritative/reusable/observed output authority.

Generated/refined renderer output remains generated/rendered output; it is not observed perception evidence.

## Runtime integration

The orchestrator builds a `ROIRenderRequest` before constructing `PatchSynthesisRequest`, passes the request metadata through `transition_context`, wraps the patch synthesizer output as `ROIRenderOutput`, and stores diagnostics in patch execution traces. Runtime debug includes `roi_renderer_contract_summary` counts.

## Training metadata hooks

Requests expose `to_training_metadata()` with route type, render mode, canonical region, memory family/authority, output authority, identity lock, weak memory flag, source delta/reveal/action/phase provenance, safety flags, and forbidden-operation flags. Generated runtime output must not be marked as an observed training target unless a future real observed-pair manifest explicitly provides that evidence.
