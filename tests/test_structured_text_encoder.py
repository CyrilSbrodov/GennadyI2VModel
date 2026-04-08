from core.schema import BBox, GarmentNode, SceneObjectNode, PersonNode, SceneGraph
from text.intent_parser import IntentParser
from text.text_encoder import BaselineStructuredTextEncoder


def test_structured_text_encoder_builds_rich_conditioning_output() -> None:
    parser = IntentParser()
    encoder = BaselineStructuredTextEncoder()

    person = PersonNode(person_id="p1", track_id="t1", bbox=BBox(0.1, 0.1, 0.5, 0.8), mask_ref=None)
    person.garments = [GarmentNode(garment_id="g1", garment_type="coat")]
    scene = SceneGraph(frame_index=0, persons=[person], objects=[SceneObjectNode(object_id="chair_1", object_type="chair", bbox=BBox(0.0, 0.0, 0.2, 0.2))])

    parsed = parser.parse_to_structured_intent("Сначала медленно снимает пальто, затем садится на стул и улыбается", scene_graph=scene)
    out = encoder.encode(raw_text=parsed.normalized_text, parsed_intent=parsed, scene_graph=scene)

    assert out.global_text_embedding
    assert out.action_embedding
    assert out.target_embedding
    assert out.modifier_embedding
    assert out.temporal_embedding
    assert out.constraint_embedding
    assert out.grounding_embedding
    assert out.structured_action_tokens
    assert out.grounded_targets
    assert out.diagnostics.action_count > 0
    assert "planner" in out.conditioning_hints
    assert out.scene_alignment_score > 0.0
    assert out.encoder_confidence > 0.0


def test_text_encoder_adapter_is_backward_compatible() -> None:
    from text.learned_bridge import BaselineTextEncoderAdapter

    adapter = BaselineTextEncoderAdapter()
    output = adapter.encode("Поднимает руку и аккуратно поворачивает голову")

    assert output.structured_action_tokens
    assert "entities" in output.target_hints
    assert "relations" in output.temporal_hints
    assert isinstance(output.decomposition_hints, list)
    assert isinstance(output.constraints, list)
    assert output.confidence >= 0.0
