"""Full RAG pipeline: embed query -> retrieve -> deduplicate -> rerank -> generate."""

from dataclasses import dataclass, field

import config
from rag.embedder import Embedder
from rag.generator import Generator
from rag.reranker import Reranker
from rag.retriever import SearchResult, VectorStore


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict]
    scores: list[float]
    context_texts: list[str] = field(default_factory=list)


def _deduplicate(hits: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Keep the highest-scoring chunk per case_id to avoid context monopolisation."""
    seen: dict[str, SearchResult] = {}
    for h in hits:
        cid = h.case_id
        if cid not in seen or h.score > seen[cid].score:
            seen[cid] = h
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]


class RAGPipeline:
    def __init__(self) -> None:
        self._embedder = Embedder()
        self._store = VectorStore()
        self._generator = Generator()
        self._reranker = Reranker() if config.RERANKER_ENABLED else None

    async def ask(
        self,
        question: str,
        top_k: int = config.TOP_K,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        flags: list[str] | None = None,
    ) -> RAGResponse:
        query_vector = await self._embedder.embed(question)

        # Fetch more candidates than needed so dedup + rerank have room to work
        raw_hits = self._store.search(
            query_vector,
            top_k=top_k * 3,
            courts=courts,
            year_from=year_from,
            year_to=year_to,
            flags=flags,
        )

        if not raw_hits:
            return RAGResponse(
                question=question,
                answer="No relevant NZ legal documents found for this query. "
                       "The database may not contain decisions on this topic yet.",
                sources=[],
                scores=[],
                context_texts=[],
            )

        # Deduplicate: one chunk per case_id (best score wins)
        hits = _deduplicate(raw_hits, top_k * 2)

        # Rerank with cross-encoder if enabled
        if self._reranker is not None:
            hits = self._reranker.rerank(question, hits, top_k)
        else:
            hits = hits[:top_k]

        context_texts = [h.text for h in hits]
        sources = [
            {
                "case_id": h.case_id,
                "title": h.title,
                "court_name": h.court_name,
                "date": h.date,
                "url": h.url,
            }
            for h in hits
        ]

        answer = await self._generator.generate(question, context_texts, sources)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            scores=[h.score for h in hits],
            context_texts=context_texts,
        )

    async def search_only(
        self,
        query: str,
        top_k: int = config.TOP_K,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        flags: list[str] | None = None,
    ) -> list[SearchResult]:
        query_vector = await self._embedder.embed(query)
        hits = self._store.search(
            query_vector, top_k=top_k * 3,
            courts=courts, year_from=year_from, year_to=year_to, flags=flags,
        )
        hits = _deduplicate(hits, top_k)
        if self._reranker is not None:
            hits = self._reranker.rerank(query, hits, top_k)
        return hits

    def search_notable(
        self,
        flags: list[str] | None = None,
        min_outcome_osi: float | None = None,
        max_outcome_osi: float | None = None,
        min_recovery_rate: float | None = None,
        max_recovery_rate: float | None = None,
        min_awarded: float | None = None,
        max_awarded: float | None = None,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        limit: int = 50,
    ) -> list[SearchResult]:
        return self._store.scroll_notable(
            flags=flags,
            min_outcome_osi=min_outcome_osi,
            max_outcome_osi=max_outcome_osi,
            min_recovery_rate=min_recovery_rate,
            max_recovery_rate=max_recovery_rate,
            min_awarded=min_awarded,
            max_awarded=max_awarded,
            courts=courts,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
        )

    async def close(self) -> None:
        await self._embedder.close()
        await self._generator.close()
