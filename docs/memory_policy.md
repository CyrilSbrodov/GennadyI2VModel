# Identity / Appearance Memory Policy

Sprint 3 makes memory a typed policy layer between canonical scene regions and the planner/renderer. Memory records are descriptors and policy decisions, not learned identity recognition. The engine must not synthesize face-recognition embeddings or make generated pixels authoritative.

## Memory families

Every canonical memory candidate is classified into exactly one family. Entries also retain applicability, observation status, mask evidence type, parser support level, and source frame kind so policy can distinguish not-applicable/unsupported/unknown regions, parser masks, bbox projections, generated/fallback observations, and missing masks.

- `identity`: face/head/hair/scalp and future face subregion addresses (`forehead`, eyes, nose, mouth/lips, jaw/chin). These are identity-sensitive only when observed; no subregion masks are fabricated.
- `skin`: visible non-identity skin such as neck and hands.
- `body_shape`: torso, chest, abdomen, pelvis, limbs, and structural composites.
- `soft_tissue`: soft-tissue/body-motion-relevant appearance such as breasts, buttocks, and hips. This is appearance memory, not identity memory.
- `garment`: upper/lower/outer/inner garment and garment composites.
- `accessory`: external accessories.
- `private`: private/sex-specific addresses such as external genital or pelvic private regions.
- `unknown`: unrecognized region ids.

Private memory uses `reference_kind="none"` by default. It is addressable for representation safety but is not an identity, skin, body-shape, garment, or reveal reference.

## Authority levels

Policy assigns one authority value to each candidate:

- `authoritative`: observed, direct, high-confidence identity anchors only.
- `reusable`: observed appearance material that may be reused by the correct family but is not identity-authoritative.
- `weak`: low-confidence observed material retained only as weak appearance support.
- `diagnostic_only`: unsafe material that may remain traceable for debugging but cannot seed references.
- `rejected`: material that must not seed memory or retrieval.

Generated, inferred, fallback, synthetic/training-synthetic, hidden, unknown, unsupported, or private material cannot become authoritative identity memory.

## Material provenance

Memory stores material provenance separately from region family: `observed_input`, `observed_parser`, `observed_detector`, `observed_face`, `memory_reuse`, `generated`, `inferred`, `fallback`, `synthetic`, `hidden`, `unknown`, or `unsupported`. Source-frame and mask/provenance signals decide this value; generated and fallback frame material is never normalized into observed input.

## Update rules

Memory writes go through `memory.memory_policy.assess_memory_candidate` before the record is made reusable or authoritative.

- Observed high-confidence `face`, `head`, `hair`, `scalp`, ears, or supported face subaddresses can seed authoritative identity memory only from observed face, parser, or input material with direct non-bbox mask evidence.
- Lower-confidence observed identity material can be weak/reusable appearance metadata, but not an authoritative identity reference.
- Generated/inferred/fallback/synthetic identity material is downgraded to diagnostic or rejected and uses no identity reference kind. Generic detector identity material is weak/reusable unless explicitly face-specific.
- Observed garment material can seed garment memory only.
- Observed skin, body-shape, and soft-tissue material can seed non-identity appearance memory only.
- Hidden, unknown, not-applicable, or unsupported regions without masks do not create authoritative memory.
- Private regions remain `reference_kind="none"` even if parser evidence exists.

Existing observed authoritative identity anchors are protected: generated, inferred, fallback, non-direct, low-reliability, or lower-confidence candidates cannot overwrite them.

## Retrieval rules

Region memory bundles expose family-specific references only:

- identity references only for identity-family entries,
- skin references only for skin entries,
- body-shape references for body-shape and soft-tissue entries,
- garment references only for garment entries,
- accessory references only for accessory entries,
- no authoritative/private references for private entries.

Bundles include policy trace fields (`memory_family`, `reference_kind`, `authority`, `material_provenance`, `policy_decision`, `policy_reasons`, and seed/overwrite flags) so missing or rejected memory is explainable without exposing large payloads.

## Summaries and bridge behavior

Memory summaries count authoritative identity entries, diagnostic/rejected entries, generated/inferred/fallback/synthetic entries, private no-reference entries, and per-family records. Summaries expose compact policy metadata, not raw patches.

The learned bridge does not fabricate missing identity embeddings. If no real descriptor exists, the identity encoder returns an empty embedding contract.
