# Adult Human Body Ontology Contract

Sprint 2 adds one canonical adult human body ontology in `src/core/body_ontology.py`. The ontology is an anatomical representation contract for scene-graph addressing, motion planning, occlusion reasoning, routing, and memory policy. It is not a physics system, reveal system, sex classifier, clothing-removal system, or explicit anatomy renderer.

## State separation

Each canonical body region exists in the global ontology independently from a particular frame. Per-person region state keeps these concepts separate:

- `exists_in_ontology`: the region is part of the canonical global ontology.
- `applicability`: `applicable`, `not_applicable`, `unknown_applicability`, or `unsupported_by_current_parser`.
- `visibility_state`: visible/partially visible/hidden/hidden by garment/object/self/out of frame/unknown.
- `observation_status`: observed/inferred/fallback/generated/unknown/missing.
- `mask_evidence_type` and `mask_ref`: parser mask provenance; missing masks never become observed evidence.
- ontology topology and policy: parent/children, symmetry partner, motion role, memory family, parser support, and routing enablement.

Important invariants:

- Not visible does not mean not applicable.
- Not parsed does not mean not applicable or nonexistent.
- Unknown applicability is distinct from not applicable.
- Fallback, generated, inferred, hidden, unsupported, or missing regions are not observed parser evidence.

## Optional sex-specific/private regions

The ontology includes female-applicable optional regions (`left_breast`, `right_breast`, `breast_region`, `female_pelvic_region`), male-applicable optional regions (`male_chest`, `male_pelvic_region`, `male_external_genital_region`), and the general private `external_genital_region` address. These are anatomical addresses for future occlusion, garment, motion, and memory policies only.

Sprint 2 does not infer sex, infer hidden/private anatomy from clothing, generate private anatomy, or render explicit anatomical detail. For unknown persons, optional sex-specific/private regions default to `unknown_applicability` and `observation_status=unknown` unless explicit parser evidence for that exact region exists.

## Parser mapping

Human-parser labels map to canonical ontology regions without collapsing significant anatomy:

- `face` maps to `face`, not generic `head`.
- `hair` maps to `hair`; scalp remains separately addressable.
- Left/right limb labels preserve side and upper/lower limb specificity where parser labels provide it.
- `chest` maps to `chest`, not `torso`.
- Garment labels such as `upper_clothes` remain garments and never become body anatomy.
- Unsupported or absent parser classes do not create fake observed masks.

## Scene graph and routing

`SceneGraphBuilder` derives body-part node order from the ontology. Canonical region payloads preserve ontology metadata so downstream routing and renderer metadata can distinguish observed regions from unknown, hidden, unsupported, or not-applicable regions.

Region IDs can address every canonical ontology region as `entity_id:canonical_region` (for example `person_1:left_breast`, `person_1:left_knee`, or `person_1:male_external_genital_region`). Addressability does not authorize rendering. Sprint 0 routing remains required, and unknown/not-applicable/unsupported ontology states block automatic observed render updates.

## Memory policy

Identity memory remains limited to identity-sensitive regions such as face/head/hair/scalp. Body-shape and soft-tissue regions map to body/appearance reference families. Private/sex-specific private regions map to private/no-reference policy by default and cannot become authoritative identity memory from generated, inferred, fallback, hidden, or unsupported material.
