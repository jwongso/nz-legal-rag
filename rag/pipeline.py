"""Full RAG pipeline: embed query -> retrieve -> generate answer with citations."""

from dataclasses import dataclass

import config
from rag.embedder import Embedder
from rag.generator import Generator
from rag.retriever import SearchResult, VectorStore


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict]
    scores: list[float]


class RAGPipeline:
    def __init__(self) -> None:
        self._embedder = Embedder()
        self._store = VectorStore()
        self._generator = Generator()

    async def ask(
        self,
        question: str,
        top_k: int = config.TOP_K,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> RAGResponse:
        query_vector = await self._embedder.embed(question)

        hits: list[SearchResult] = self._store.search(
            query_vector,
            top_k=top_k,
            courts=courts,
            year_from=year_from,
            year_to=year_to,
        )

        if not hits:
            return RAGResponse(
                question=question,
                answer="No relevant NZ legal documents found for this query. "
                       "The database may not contain decisions on this topic yet.",
                sources=[],
                scores=[],
            )

        context_chunks = [h.text for h in hits]
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

        answer = await self._generator.generate(question, context_chunks, sources)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            scores=[h.score for h in hits],
        )

    async def search_only(
        self,
        query: str,
        top_k: int = config.TOP_K,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[SearchResult]:
        query_vector = await self._embedder.embed(query)
        return self._store.search(query_vector, top_k=top_k, courts=courts, year_from=year_from, year_to=year_to)

    async def close(self) -> None:
        await self._embedder.close()
        await self._generator.close()
