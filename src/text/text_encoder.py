from __future__ import annotations

import math
from typing import Protocol

from core.schema import SceneGraph
from text.contracts import ParsedIntent
from text.encoder_contracts import TextEncodingDiagnostics, TextEncodingOutput
from text.feature_encoders import (
    ActionSemanticsEncoder,
    AmbiguityConfidenceEncoder,
    ConstraintEncoder,
    ModifierEncoder,
    TargetGroundingEncoder,
    TemporalEncoder,
)


class TextConditioningEncoder(Protocol):
    """Интерфейс learned-ready text conditioning encoder."""

    def encode(self, raw_text: str, parsed_intent: ParsedIntent, scene_graph: SceneGraph | None = None) -> TextEncodingOutput:
        ...


class BaselineStructuredTextEncoder(TextConditioningEncoder):
    """Детерминированный composition-based encoder для structured text conditioning."""

    def __init__(self) -> None:
        self.action_encoder = ActionSemanticsEncoder()
        self.target_encoder = TargetGroundingEncoder()
        self.modifier_encoder = ModifierEncoder()
        self.temporal_encoder = TemporalEncoder()
        self.constraint_encoder = ConstraintEncoder()
        self.ambiguity_encoder = AmbiguityConfidenceEncoder()

    def encode(self, raw_text: str, parsed_intent: ParsedIntent, scene_graph: SceneGraph | None = None) -> TextEncodingOutput:
        action_block = self.action_encoder.encode(parsed_intent)
        target_block = self.target_encoder.encode(parsed_intent, scene_graph=scene_graph)
        modifier_embedding = self.modifier_encoder.encode(parsed_intent)
        temporal_embedding = self.temporal_encoder.encode(parsed_intent)
        constraint_embedding, constraints = self.constraint_encoder.encode(parsed_intent)
        confidence_embedding, ambiguity_score, encoder_confidence = self.ambiguity_encoder.encode(
            parsed_intent,
            weak_grounding_count=target_block.weak_grounding_count,
            unresolved_target_count=target_block.unresolved_target_count,
        )

        grounding_embedding = [
            target_block.scene_alignment_score,
            1.0 - min(1.0, target_block.unresolved_target_count / max(1, len(target_block.grounded_targets))),
            1.0 - min(1.0, target_block.weak_grounding_count / max(1, len(target_block.grounded_targets))),
            1.0 if scene_graph is not None else 0.0,
        ]
        global_text_embedding = self._build_global_embedding(
            raw_text=raw_text,
            parsed_intent=parsed_intent,
            action_embedding=action_block.embedding,
            target_embedding=target_block.embedding,
            modifier_embedding=modifier_embedding,
            temporal_embedding=temporal_embedding,
            constraint_embedding=constraint_embedding,
            grounding_embedding=grounding_embedding,
            confidence_embedding=confidence_embedding,
        )

        diagnostics = self._build_diagnostics(
            parsed_intent=parsed_intent,
            action_count=len(action_block.structured_action_tokens),
            family_distribution=action_block.family_distribution,
            grounded_target_count=len(target_block.grounded_targets),
            unresolved_target_count=target_block.unresolved_target_count,
            weak_grounding_count=target_block.weak_grounding_count,
            constraint_count=len(constraints),
            encoder_confidence=encoder_confidence,
        )

        conditioning_hints = self._build_conditioning_hints(
            action_tokens=action_block.structured_action_tokens,
            family_presence=action_block.family_presence_vector,
            modifier_embedding=modifier_embedding,
            temporal_embedding=temporal_embedding,
            constraints=constraints,
            grounded_targets=target_block.grounded_targets,
        )
        decomposition_hints = [
            {
                "clause_index": clause.index,
                "actions": [f"{a.semantic_family}:{a.semantic_action}" for a in clause.action_candidates],
                "modifiers": clause.modifiers.as_dict(),
                "ambiguities": clause.ambiguities[:],
            }
            for clause in parsed_intent.clauses
        ]

        return TextEncodingOutput(
            global_text_embedding=global_text_embedding,
            action_embedding=action_block.embedding,
            target_embedding=target_block.embedding,
            modifier_embedding=modifier_embedding,
            temporal_embedding=temporal_embedding,
            constraint_embedding=constraint_embedding,
            grounding_embedding=grounding_embedding,
            structured_action_tokens=action_block.structured_action_tokens,
            grounded_targets=target_block.grounded_targets,
            parser_confidence=float(parsed_intent.parser_confidence),
            encoder_confidence=encoder_confidence,
            diagnostics=diagnostics,
            scene_alignment_score=target_block.scene_alignment_score,
            ambiguity_score=ambiguity_score,
            conditioning_hints=conditioning_hints,
            family_presence_vector=action_block.family_presence_vector,
            target_hints={
                "entities": sorted({t.target_entity_id for t in target_block.grounded_targets if t.target_entity_id}),
                "objects": sorted({t.target_object for t in target_block.grounded_targets if t.target_object}),
                "regions": sorted({t.target_region for t in target_block.grounded_targets if t.target_region}),
            },
            temporal_hints={
                "relations": [rel.relation for rel in parsed_intent.temporal_relations],
                "edge_count": len(parsed_intent.temporal_relations),
                "clause_count": len(parsed_intent.clauses),
            },
            decomposition_hints=decomposition_hints,
            constraints=constraints,
            confidence=encoder_confidence,
            alignment={
                "source": "baseline_structured_text_encoder",
                "scene_alignment_score": target_block.scene_alignment_score,
                "parser_confidence": parsed_intent.parser_confidence,
            },
            trace=[
                "encoder=baseline_structured_text",
                f"actions={len(action_block.structured_action_tokens)}",
                f"grounded_targets={len(target_block.grounded_targets)}",
            ],
        )

    def _build_global_embedding(
        self,
        *,
        raw_text: str,
        parsed_intent: ParsedIntent,
        action_embedding: list[float],
        target_embedding: list[float],
        modifier_embedding: list[float],
        temporal_embedding: list[float],
        constraint_embedding: list[float],
        grounding_embedding: list[float],
        confidence_embedding: list[float],
    ) -> list[float]:
        """Собирает компактный глобальный embedding из block-level признаков."""

        lexical = self._lexical_embedding(raw_text, parsed_intent)
        merged = (
            lexical
            + action_embedding
            + target_embedding
            + modifier_embedding
            + temporal_embedding
            + constraint_embedding
            + grounding_embedding
            + confidence_embedding
        )
        norm = math.sqrt(sum(v * v for v in merged)) or 1.0
        return [float(v / norm) for v in merged]

    def _lexical_embedding(self, raw_text: str, parsed_intent: ParsedIntent) -> list[float]:
        """Кодирует базовые лексические статистики как устойчивые признаки."""

        text = parsed_intent.normalized_text or raw_text.lower().strip()
        token_count = len([t for t in text.split(" ") if t])
        char_count = len(text)
        comma_count = text.count(",")
        digit_count = sum(1 for c in text if c.isdigit())
        unique_ratio = (len(set(text.split())) / token_count) if token_count else 0.0
        return [
            min(1.0, token_count / 32.0),
            min(1.0, char_count / 256.0),
            min(1.0, len(parsed_intent.clauses) / 8.0),
            min(1.0, comma_count / 8.0),
            min(1.0, digit_count / 8.0),
            float(unique_ratio),
        ]

    def _build_diagnostics(
        self,
        *,
        parsed_intent: ParsedIntent,
        action_count: int,
        family_distribution: dict[str, int],
        grounded_target_count: int,
        unresolved_target_count: int,
        weak_grounding_count: int,
        constraint_count: int,
        encoder_confidence: float,
    ) -> TextEncodingDiagnostics:
        """Собирает structured diagnostics блок для explainability."""

        ambiguity_count = sum(len(clause.ambiguities) for clause in parsed_intent.clauses)
        return TextEncodingDiagnostics(
            action_count=action_count,
            family_distribution=family_distribution,
            grounded_target_count=grounded_target_count,
            unresolved_target_count=unresolved_target_count,
            weak_grounding_count=weak_grounding_count,
            temporal_relation_count=len(parsed_intent.temporal_relations),
            constraint_count=constraint_count,
            ambiguity_count=ambiguity_count,
            parser_confidence=float(parsed_intent.parser_confidence),
            encoder_confidence=encoder_confidence,
            explainability_summary={
                "clauses": [clause.text for clause in parsed_intent.clauses],
                "temporal_relations": [rel.relation for rel in parsed_intent.temporal_relations],
                "global_constraints": [c.requirement for c in parsed_intent.global_constraints],
                "ambiguity_reasons": [amb for clause in parsed_intent.clauses for amb in clause.ambiguities],
            },
        )

    def _build_conditioning_hints(
        self,
        *,
        action_tokens: list[str],
        family_presence: dict[str, float],
        modifier_embedding: list[float],
        temporal_embedding: list[float],
        constraints: list[str],
        grounded_targets: list[object],
    ) -> dict[str, object]:
        """Формирует прямые hints для planner/dynamics/renderer/memory."""

        return {
            "planner": {
                "action_families": [fam for fam, flag in family_presence.items() if flag > 0.0],
                "temporal_structure": {
                    "after": temporal_embedding[0],
                    "parallel": temporal_embedding[1],
                    "sequence": temporal_embedding[2],
                },
                "constraints": constraints,
                "target_priorities": [getattr(t, "target_region", None) for t in grounded_targets if getattr(t, "target_region", None)],
            },
            "dynamics": {
                "action_conditioning": action_tokens,
                "phase_hints": {
                    "is_parallel": temporal_embedding[1] > 0.2,
                    "is_sequential": temporal_embedding[2] > 0.2,
                },
                "modifier_intensities": {
                    "intensity": modifier_embedding[0],
                    "speed_slow": modifier_embedding[1],
                    "speed_fast": modifier_embedding[3],
                },
            },
            "renderer": {
                "reveal_hints": any(tok in {"reveal", "show", "uncover", "reveal_inner"} for tok in action_tokens),
                "face_refinement_hints": family_presence.get("expression", 0.0) > 0,
                "garment_transition_hints": family_presence.get("garment", 0.0) > 0,
                "visibility_hints": family_presence.get("visibility", 0.0) > 0,
            },
            "memory": {
                "target_family_hints": [getattr(t, "target_entity_class", "") for t in grounded_targets],
                "scene_grounding_hints": [getattr(t, "resolution_reason", "") for t in grounded_targets],
                "ambiguity_safety_hints": {
                    "has_unresolved": any(getattr(t, "unresolved", False) for t in grounded_targets),
                    "requires_conservative_routing": any(c.startswith("requires:") for c in constraints),
                },
            },
        }
