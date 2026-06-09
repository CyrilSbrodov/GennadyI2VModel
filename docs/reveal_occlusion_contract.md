# Sprint 6 Reveal / Occlusion Continuity Contract

The reveal layer is a policy handoff between **Dynamics / GraphDelta** and **Region Routing**. It consumes the Sprint-5 `GraphDeltaContract`, the current `SceneGraph`, and read-only memory summaries or `VideoMemory`. It decides whether a region that may become visible is safe to restore from prior observed memory, only weakly supported, unknown/deferred, or blocked.

## Non-goals

The reveal contract does **not** generate pixels, reconstruct hidden anatomy, remove clothing, run cloth or physics simulation, create masks, write memory, create observed perception evidence, call region routing, choose render modes, or call renderers. Routing candidates emitted by reveal are handoff metadata only.

## Decision types

`RevealDecisionType` values are:

- `preserve_visible`
- `newly_occluded`
- `occlusion_reasoning_required`
- `reveal_from_observed_memory`
- `reveal_from_weak_memory`
- `reveal_unknown_defer`
- `reveal_blocked_private`
- `reveal_blocked_no_evidence`
- `reveal_blocked_identity_risk`
- `reveal_blocked_unsupported_region`

## Occlusion lifecycle states

`OcclusionLifecycleState` values are:

- `visible_stable`
- `newly_occluded`
- `occluded_known`
- `occluded_unknown`
- `newly_revealed_known`
- `newly_revealed_weak`
- `newly_revealed_unknown`
- `reveal_blocked`
- `private_blocked`
- `identity_risk_blocked`

These states distinguish stable visible content, newly occluded content, known/unknown hidden content, weak reveal candidates, private blocks, and identity-risk blocks.

## Memory evidence requirements

Each `RevealDecision` carries `RevealMemoryEvidence` with memory family, reference kind, authority, material provenance, support level, score, confidence, observed/generated/inferred flags, mask evidence type, and policy reasons. The reveal layer uses the Sprint-3 memory policy (`assess_memory_candidate`) and never treats generated, inferred, fallback, synthetic, hidden, unknown, private, diagnostic-only, rejected, or no-reference memory as observed reveal evidence.

Authoritative identity memory can support identity reveal. Reusable body, skin, garment, and accessory memory can support appearance reveal. Weak memory can only produce `reveal_from_weak_memory`, which requires later routing/render validation and cannot become authoritative identity evidence. Missing memory produces `reveal_unknown_defer` or an identity-risk block.

## Identity lock behavior

Identity-sensitive regions (`face`, `head`, `hair`, `scalp`, and targeted face subregions) are always `identity_locked=True` and `allowed_to_modify_identity=False`. `reveal_from_observed_memory` for identity regions requires authoritative observed identity memory. The reveal layer never creates identity memory or embeddings.

## Private and optional region block

Private and optional sex-specific/private body regions remain no-reveal and no-render by default. They cannot become reveal targets, routing candidates, rendering candidates, memory requirements, fallback body/skin targets, or reconstructed/inferred regions under clothing. If such a region reaches reveal, it is blocked as `reveal_blocked_private` (or rejected by strict validation for invalid claims).

## Routing handoff

`RevealRoutingCandidate` contains region/action/phase provenance, reveal decision type, source delta type, identity lock status, reveal allowance, rendering-candidate requirement, and memory reference metadata. It explicitly records that no routing decision or render strategy was selected. Region Routing remains responsible for actual routing decisions later in the pipeline.
