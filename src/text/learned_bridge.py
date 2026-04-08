from __future__ import annotations

from core.schema import ActionPlan, SceneGraph
from learned.interfaces import TextEncoder
from text.encoder_contracts import TextEncodingOutput
from text.intent_parser import IntentParser
from text.text_encoder import BaselineStructuredTextEncoder


class BaselineTextEncoderAdapter(TextEncoder):
    """Адаптер parser + structured encoder к learned text-encoder интерфейсу."""

    def __init__(self) -> None:
        self.parser = IntentParser()
        self.encoder = BaselineStructuredTextEncoder()

    def encode(self, text: str, scene_graph: SceneGraph | None = None, action_plan: ActionPlan | None = None) -> TextEncodingOutput:
        # ActionPlan оставлен в интерфейсе для обратной совместимости, но источник conditioning — ParsedIntent.
        parsed_intent = self.parser.parse_to_structured_intent(text, scene_graph=scene_graph)
        return self.encoder.encode(raw_text=text, parsed_intent=parsed_intent, scene_graph=scene_graph)
