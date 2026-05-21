"""Deterministic legal authority ranker applied after Qdrant dedup, before cross-encoder.

Scores each chunk on four legal signals and blends with the vector score:
  - court authority: binding/persuasive hierarchy in NZ courts
  - statute affinity: boost legislation chunks on statute queries
  - early chunk: chunk_index 0-2 typically contains headnote or leading principle
  - citation density: more outbound citations = richer legal authority source
  - recency: boost recent decisions when query signals recency preference

Does not replace the cross-encoder; it pre-orders chunks so that the cross-encoder
sees the most legally authoritative candidates first when the candidate pool is capped.
"""

import re
from dataclasses import dataclass

_COURT_AUTHORITY: dict[str, float] = {
    "NZSC":    1.00,
    "NZLEG":   0.95,
    "NZCA":    0.85,
    "NZHC":    0.70,
    "NZEmpC":  0.65,
    "NZEnvC":  0.60,
    "NZFC":    0.55,
    "NZERA":   0.50,
    "NZACC":   0.50,
    "NZHRRT":  0.50,
    "NZTT":    0.45,
    "NZCorC":  0.45,
    "NZLCDT":  0.40,
    "NZREADT": 0.40,
}
_DEFAULT_AUTHORITY = 0.45

_AUTHORITY_WEIGHT    = 0.12
_EARLY_CHUNK_WEIGHT  = 0.015
_CITATION_RICH_WEIGHT = 0.02
_STATUTE_BOOST       = 0.08
_RECENCY_BOOST       = 0.025
_RECENCY_THRESHOLD_YEAR = 2022

_STATUTE_RE = re.compile(
    r"\b(section|s\s*\d+[A-Z]?|ERA|RTA|HSW|FTA|CCA|Act)\b", re.IGNORECASE
)
_RECENCY_RE = re.compile(
    r"\b(latest|recent|current|2023|2024|2025|2026)\b", re.IGNORECASE
)


@dataclass
class QueryContext:
    is_statute: bool = False
    wants_recent: bool = False

    @classmethod
    def from_query(cls, question: str) -> "QueryContext":
        return cls(
            is_statute=bool(_STATUTE_RE.search(question)),
            wants_recent=bool(_RECENCY_RE.search(question)),
        )


def rerank(hits: list, ctx: QueryContext, top_k: int | None = None) -> list:
    """Re-score and sort hits using legal authority signals.

    Args:
        hits: list of SearchResult objects (must have .score and .payload)
        ctx: QueryContext derived from the user question
        top_k: if set, return only the top_k results

    Returns:
        New list sorted descending by legal-blended score. Original .score is not mutated.
    """
    scored: list[tuple[float, object]] = []
    for h in hits:
        p = h.payload

        court = p.get("court", "")
        authority = _COURT_AUTHORITY.get(court, _DEFAULT_AUTHORITY)

        chunk_index = p.get("chunk_index", 99)
        early_boost = max(0.0, _EARLY_CHUNK_WEIGHT - chunk_index * 0.005)

        citations = p.get("citations") or []
        citation_boost = min(len(citations), 5) / 5.0 * _CITATION_RICH_WEIGHT

        statute_boost = 0.0
        if ctx.is_statute and p.get("document_type", "") == "legislation":
            statute_boost = _STATUTE_BOOST

        recency_boost = 0.0
        if ctx.wants_recent:
            year = p.get("year") or 0
            if year >= _RECENCY_THRESHOLD_YEAR:
                recency_boost = _RECENCY_BOOST

        legal_score = (
            h.score * (1.0 - _AUTHORITY_WEIGHT)
            + authority * _AUTHORITY_WEIGHT
            + early_boost
            + citation_boost
            + statute_boost
            + recency_boost
        )
        scored.append((legal_score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [h for _, h in scored]
    return result[:top_k] if top_k is not None else result
