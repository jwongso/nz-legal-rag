"""
Cross-encoder reranker using bge-reranker-v2-m3.

Runs on CPU - no GPU needed. At 5 candidates and 512 token max_length,
reranking adds ~50-100ms per query but significantly improves context precision
for legal text where boilerplate confuses cosine similarity.

# TODO: Once AI Max+ 395 is available, increase max_length to 1024 and run on
#        NPU/GPU for sub-10ms reranking even with 20 candidates.
"""

from sentence_transformers import CrossEncoder

import config
from rag.device import select_device
from rag.retriever import SearchResult


class Reranker:
    def __init__(self) -> None:
        self._model = CrossEncoder(config.RERANKER_MODEL, max_length=512, device=select_device())

    def rerank(self, query: str, hits: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not hits:
            return hits
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:top_k]]
