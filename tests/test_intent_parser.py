from text.intent_parser import IntentParser


def test_parser_extracts_core_actions() -> None:
    parser = IntentParser()
    plan = parser.parse("Снимает пальто и садится на стул. Улыбается.")
    actions = [a.type for a in plan.actions]
    assert "remove_garment" in actions
    assert "sit_down" in actions
    assert "smile" in actions
