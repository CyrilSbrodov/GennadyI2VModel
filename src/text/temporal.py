from __future__ import annotations

import re

from text.contracts import TemporalRelation


def split_clauses(normalized_text: str) -> list[str]:
    """Сегментирует текст на клаузы и временные блоки."""

    return [c.strip() for c in re.split(r"[\.,;]", normalized_text) if c.strip()]


def extract_temporal_relations(clauses: list[str]) -> list[TemporalRelation]:
    """Строит temporal edges между клауза-индексами."""

    rels: list[TemporalRelation] = []
    for idx in range(1, len(clauses)):
        curr = clauses[idx]
        if any(t in curr for t in ("потом", "затем", "после")):
            rels.append(TemporalRelation(relation="after", source_clause=idx, target_clause=idx - 1, marker="then"))
        elif any(t in curr for t in ("одновременно", "вместе", "пока")):
            rels.append(TemporalRelation(relation="parallel", source_clause=idx, target_clause=idx - 1, marker="simultaneous"))
        else:
            rels.append(TemporalRelation(relation="sequence", source_clause=idx, target_clause=idx - 1, marker="default"))

    if clauses and "сначала" in clauses[0]:
        rels.append(TemporalRelation(relation="anchor_first", source_clause=0, target_clause=0, marker="first"))

    return rels
