"""Full RAG pipeline: embed query -> retrieve -> deduplicate -> rerank -> generate.

Retrieval strategies:
  - Pure Qdrant (default): semantic search with optional court/year/flags filters.
  - SQL-first hybrid: PostgreSQL narrows candidates by structured fields
    (offence, sentencing ranges, employment outcomes, flags), then Qdrant
    ranks semantically within that set. Pass sql_filter=FilterParams(...) to ask().
  - BM25 (keyword): full-text search via PostgreSQL tsvector. Use db.filter.bm25_search().
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import config
from rag.embedder import Embedder
from rag.generator import Generator
from rag.reranker import Reranker
from rag.retriever import SearchResult, VectorStore
from rag.trace import CitationVerification, RetrievalTrace, verify_citations

if TYPE_CHECKING:
    from db.filter import FilterParams


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict]
    scores: list[float]
    context_texts: list[str] = field(default_factory=list)
    trace: RetrievalTrace | None = None
    citation_verification: CitationVerification | None = None


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
        sql_filter: "FilterParams | None" = None,
        trace: bool = False,
    ) -> RAGResponse:
        t_total = time.monotonic()
        tr = RetrievalTrace(
            model_name=config.LLM_MODEL,
            embedding_model=getattr(config, "EMBED_MODEL", ""),
            reranker_enabled=self._reranker is not None,
        ) if trace else None

        # Embed
        t0 = time.monotonic()
        query_vector = await self._embedder.embed(question)
        if tr:
            tr.latency_embed_ms = (time.monotonic() - t0) * 1000

        if sql_filter is not None:
            # SQL-first hybrid: pre-filter with PostgreSQL, rank with Qdrant
            from db.filter import get_point_ids
            if tr:
                tr.strategy = "sql_first_hybrid"
                tr.sql_filters = {
                    k: v for k, v in sql_filter.__dict__.items()
                    if v is not None and k != "max_ids"
                }
            t0 = time.monotonic()
            point_ids = get_point_ids(sql_filter)
            if tr:
                tr.latency_sql_ms = (time.monotonic() - t0) * 1000
                tr.sql_point_ids_count = len(point_ids)
            t0 = time.monotonic()
            raw_hits = self._store.search_within(
                query_vector, point_ids, top_k=top_k * 3
            )
            if tr:
                tr.latency_qdrant_ms = (time.monotonic() - t0) * 1000
        else:
            # Pure Qdrant search
            t0 = time.monotonic()
            raw_hits = self._store.search(
                query_vector,
                top_k=top_k * 3,
                courts=courts,
                year_from=year_from,
                year_to=year_to,
                flags=flags,
            )
            if tr:
                tr.latency_qdrant_ms = (time.monotonic() - t0) * 1000

        if tr:
            tr.qdrant_candidates = len(raw_hits)

        if not raw_hits:
            return RAGResponse(
                question=question,
                answer="No relevant NZ legal documents found for this query. "
                       "The database may not contain decisions on this topic yet.",
                sources=[],
                scores=[],
                context_texts=[],
                trace=tr,
            )

        # Deduplicate: one chunk per case_id (best score wins)
        hits = _deduplicate(raw_hits, top_k * 2)
        if tr:
            tr.after_dedup = len(hits)

        # Rerank with cross-encoder if enabled
        t0 = time.monotonic()
        if self._reranker is not None:
            hits = self._reranker.rerank(question, hits, top_k)
        else:
            hits = hits[:top_k]
        if tr:
            tr.latency_rerank_ms = (time.monotonic() - t0) * 1000
            tr.after_rerank = len(hits)
            tr.top_scores = [h.score for h in hits]

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

        t0 = time.monotonic()
        answer = await self._generator.generate(question, context_texts, sources)
        if tr:
            tr.latency_generate_ms = (time.monotonic() - t0) * 1000
            tr.latency_total_ms = (time.monotonic() - t_total) * 1000

        citation_check = verify_citations(answer, sources) if trace else None

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            scores=[h.score for h in hits],
            context_texts=context_texts,
            trace=tr,
            citation_verification=citation_check,
        )

    async def search_only(
        self,
        query: str,
        top_k: int = config.TOP_K,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        flags: list[str] | None = None,
        sql_filter: "FilterParams | None" = None,
    ) -> list[SearchResult]:
        query_vector = await self._embedder.embed(query)

        if sql_filter is not None:
            from db.filter import get_point_ids
            point_ids = get_point_ids(sql_filter)
            hits = self._store.search_within(query_vector, point_ids, top_k=top_k * 3)
        else:
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
        counsel_surname: str | None = None,
        crown_counsel: str | None = None,
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
            counsel_surname=counsel_surname,
            crown_counsel=crown_counsel,
            courts=courts,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
        )

    def search_sentencing(
        self,
        flags: list[str] | None = None,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        sentence_type: str | None = None,
        min_starting_point: float | None = None,
        max_starting_point: float | None = None,
        min_final_sentence: float | None = None,
        max_final_sentence: float | None = None,
        has_guilty_plea: bool | None = None,
        limit: int = 50,
    ) -> list[SearchResult]:
        return self._store.scroll_sentencing(
            flags=flags,
            courts=courts,
            year_from=year_from,
            year_to=year_to,
            sentence_type=sentence_type,
            min_starting_point=min_starting_point,
            max_starting_point=max_starting_point,
            min_final_sentence=min_final_sentence,
            max_final_sentence=max_final_sentence,
            has_guilty_plea=has_guilty_plea,
            limit=limit,
        )

    def search_pg(
        self,
        grievance_types: list[str] | None = None,
        reinstatement: bool | None = None,
        min_contributory: float | None = None,
        max_contributory: float | None = None,
        min_compensation: float | None = None,
        max_compensation: float | None = None,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        limit: int = 50,
    ) -> list[SearchResult]:
        return self._store.scroll_pg(
            grievance_types=grievance_types,
            reinstatement=reinstatement,
            min_contributory=min_contributory,
            max_contributory=max_contributory,
            min_compensation=min_compensation,
            max_compensation=max_compensation,
            courts=courts,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
        )

    async def contrasting_cases(
        self,
        query: str,
        domain: str,
        split_by: str | None = None,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        top_k: int = 5,
    ):
        from rag.contrasting import find_contrasting_cases
        query_vector = await self._embedder.embed(query)
        return find_contrasting_cases(
            query=query,
            domain=domain,
            query_vector=query_vector,
            store=self._store,
            split_by=split_by,
            courts=courts,
            year_from=year_from,
            year_to=year_to,
            top_k=top_k,
        )

    async def close(self) -> None:
        await self._embedder.close()
        await self._generator.close()
