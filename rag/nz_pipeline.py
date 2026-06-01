"""NZ-enhanced RAG pipeline: Astraea base + legal authority ranker + cross-encoder reranker.

Extends core.pipeline.RAGPipeline with two post-retrieval steps that are
specific to NZ legal corpora:

  1. Legal authority ranker  - re-orders by court hierarchy (SC > CA > HC > tribunals)
                               and legal signals (citation density, statute boost, recency).
  2. Cross-encoder reranker  - optional BAAI/bge-reranker-v2-m3 for precision boost.
                               Off by default (RERANK_MODE=off). Enable with RERANK_MODE=rerank_5.
"""

from __future__ import annotations

import os

from core.pipeline import RAGPipeline, _deduplicate, _mmr_select

_TOP_K = int(os.getenv("TOP_K", "5"))
_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_RERANK_MODE = os.getenv("RERANK_MODE", "off")


def _parse_rerank_candidates(mode: str) -> int | None:
    m = mode.strip().lower()
    if m == "off":
        return None
    if m.startswith("rerank_"):
        try:
            return int(m[len("rerank_"):])
        except ValueError:
            pass
    return None


class NZLegalPipeline(RAGPipeline):
    """RAGPipeline with NZ legal authority ranker and optional cross-encoder reranker."""

    def __init__(
        self,
        collection: str,
        system_prompt: str,
        courts: list[str] | None = None,
        embedder=None,
    ) -> None:
        super().__init__(
            collection=collection,
            system_prompt=system_prompt,
            courts=courts,
            embedder=embedder,
        )
        rerank_candidates = _parse_rerank_candidates(_RERANK_MODE)
        if rerank_candidates is not None:
            from rag.reranker import Reranker
            self._reranker = Reranker()
            self._rerank_candidates = rerank_candidates
        else:
            self._reranker = None
            self._rerank_candidates = None

    async def retrieve(
        self,
        question: str,
        top_k: int = _TOP_K,
        min_score: float = 0.0,
        min_chunks: int = 1,
        strategy: str = "vector",
    ) -> tuple[list[str], list[dict]]:
        query_vector = await self._embedder.embed(question)
        raw_hits = self._store.search(query_vector, top_k=top_k * 3, courts=self._courts)

        if not raw_hits:
            return [], []

        hits = _deduplicate(raw_hits, top_k * 2)

        if strategy != "mmr":
            from rag.legal_ranker import QueryContext, rerank as legal_rerank
            ctx = QueryContext.from_query(question)
            hits = legal_rerank(hits, ctx)

        if self._reranker is not None:
            candidates = hits[:self._rerank_candidates]
            hits = self._reranker.rerank(question, candidates, top_k)
        elif strategy == "mmr":
            hits = _mmr_select(hits, top_k)
        else:
            hits = hits[:top_k]

        if min_score > 0.0:
            hits = [h for h in hits if h.score >= min_score]
        if len(hits) < min_chunks:
            return [], []

        context_texts = [h.text for h in hits]
        sources = [
            {
                "case_id": h.case_id,
                "title": h.title,
                "court_name": h.court_name,
                "date": h.date,
                "url": h.url,
                "_score": round(h.score, 4),
            }
            for h in hits
        ]
        return context_texts, sources
