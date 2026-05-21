"""Retrieval trace and citation verification for Phase 3 of the kick-ass roadmap.

RetrievalTrace records how an answer was produced:
  - which retrieval strategy was used
  - SQL filters applied and how many point IDs they returned
  - Qdrant candidate count before and after dedup/rerank
  - latency breakdown at each pipeline stage
  - model info

CitationVerification checks whether the generated answer is grounded in
the retrieved sources without requiring an extra LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config


@dataclass
class RetrievalTrace:
    # Strategy
    strategy: str = "pure_qdrant"   # 'pure_qdrant', 'sql_first_hybrid', 'bm25'
    sql_filters: dict | None = None  # FilterParams fields that were set
    sql_point_ids_count: int = 0     # how many IDs the SQL pre-filter returned

    # Candidate counts through the pipeline
    qdrant_candidates: int = 0
    after_dedup: int = 0
    after_rerank: int = 0

    # Latency (milliseconds)
    latency_embed_ms: float = 0.0
    latency_sql_ms: float = 0.0
    latency_qdrant_ms: float = 0.0
    latency_rerank_ms: float = 0.0
    latency_generate_ms: float = 0.0
    latency_total_ms: float = 0.0

    # Scores of final context chunks
    top_scores: list[float] = field(default_factory=list)

    # Model info
    model_name: str = ""
    embedding_model: str = ""
    reranker_enabled: bool = False

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "sql_filters": self.sql_filters,
            "sql_point_ids_count": self.sql_point_ids_count,
            "counts": {
                "qdrant_candidates": self.qdrant_candidates,
                "after_dedup": self.after_dedup,
                "after_rerank": self.after_rerank,
            },
            "latency_ms": {
                "embed": round(self.latency_embed_ms, 1),
                "sql": round(self.latency_sql_ms, 1),
                "qdrant": round(self.latency_qdrant_ms, 1),
                "rerank": round(self.latency_rerank_ms, 1),
                "generate": round(self.latency_generate_ms, 1),
                "total": round(self.latency_total_ms, 1),
            },
            "top_scores": [round(s, 4) for s in self.top_scores],
            "models": {
                "llm": self.model_name,
                "embedding": self.embedding_model,
                "reranker_enabled": self.reranker_enabled,
            },
        }


@dataclass
class CitationVerification:
    has_citations: bool = False           # answer contains at least one [N] reference
    cited_count: int = 0                  # valid [N] references found in answer
    orphan_citations: list[int] = field(default_factory=list)  # [N] cited but N > source count
    uncited_sources: list[int] = field(default_factory=list)   # retrieved but never cited
    evidence_confidence: str = "low"      # 'high', 'medium', 'low'
    has_warning: bool = False             # true if orphan citations or no citations at all

    def to_dict(self) -> dict:
        return {
            "has_citations": self.has_citations,
            "cited_count": self.cited_count,
            "orphan_citations": self.orphan_citations,
            "uncited_sources": self.uncited_sources,
            "evidence_confidence": self.evidence_confidence,
            "has_warning": self.has_warning,
        }


def verify_citations(
    answer: str,
    sources: list[dict],
) -> CitationVerification:
    """Lightweight citation verifier - no extra LLM call.

    Checks:
    - Does the answer contain [N] citations?
    - Are all cited [N] within the retrieved source range?
    - Which retrieved sources were never cited?
    - What evidence confidence level does this imply?
    """
    # Only match small citation numbers [1]-[50], not years like [1994]
    cited_nums = {int(m) for m in re.findall(r'\[(\d{1,2})\]', answer)}
    available_nums = set(range(1, len(sources) + 1))

    orphan = sorted(cited_nums - available_nums)
    uncited = sorted(available_nums - cited_nums)
    valid_cited = cited_nums & available_nums

    has_citations = len(cited_nums) > 0
    has_warning = bool(orphan) or not has_citations

    # Evidence confidence
    n_sources = len(sources)
    if n_sources >= 5 and not orphan and has_citations:
        confidence = "high"
    elif n_sources >= 3 and not orphan and has_citations:
        confidence = "medium"
    else:
        confidence = "low"

    return CitationVerification(
        has_citations=has_citations,
        cited_count=len(valid_cited),
        orphan_citations=orphan,
        uncited_sources=uncited,
        evidence_confidence=confidence,
        has_warning=has_warning,
    )
