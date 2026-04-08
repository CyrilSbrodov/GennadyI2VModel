import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schema import BBox, GarmentNode, SceneObjectNode, PersonNode, SceneGraph, BodyPartNode
from text.intent_parser import IntentParser
from text.text_encoder import BaselineStructuredTextEncoder



scene = SceneGraph(
    frame_index=0,
    persons=[
        PersonNode(
            person_id="p1",
            track_id="t1",
            bbox=BBox(0.1, 0.1, 0.5, 0.8),
            mask_ref=None,
            body_parts=[
                BodyPartNode(part_id="bp_face", part_type="face", visibility="visible"),
                BodyPartNode(part_id="bp_arm", part_type="arm", visibility="visible"),
            ],
            garments=[GarmentNode(garment_id="g1", garment_type="coat")],
        )
    ],
    objects=[
        SceneObjectNode(
            object_id="chair_1",
            object_type="chair",
            bbox=BBox(0.0, 0.0, 0.2, 0.2),
        )
    ],
)

parser = IntentParser()
parsed = parser.parse_to_structured_intent(
    "Садится на стул и улыбается",
    scene_graph=scene,
)

encoder = BaselineStructuredTextEncoder()
encoded = encoder.encode(
    raw_text=parsed.normalized_text,
    parsed_intent=parsed,
    scene_graph=scene,
)

print("CLAUSES:")
for clause in parsed.clauses:
    print("-", clause.text)
    print("  actions:", [(a.semantic_family, a.semantic_action) for a in clause.action_candidates])
    print("  targets:", [(t.target_entity_class, t.target_entity_id, t.target_region, t.unresolved) for t in clause.resolved_targets])
    print("  constraints:", [c.requirement for c in clause.constraints])
    print("  ambiguities:", clause.ambiguities)

print("\nTEMPORAL:", [(r.relation, r.source_clause, r.target_clause) for r in parsed.temporal_relations])
print("\nTOKENS:", encoded.structured_action_tokens)
print("SCENE ALIGNMENT:", encoded.scene_alignment_score)
print("ENCODER CONFIDENCE:", encoded.encoder_confidence)
print("CONSTRAINTS:", encoded.constraints)
print("PLANNER HINTS:", encoded.conditioning_hints["planner"])
print("RENDERER HINTS:", encoded.conditioning_hints["renderer"])
print("MEMORY HINTS:", encoded.conditioning_hints["memory"])