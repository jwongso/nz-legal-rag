"""Qdrant vector store: upsert and search with optional metadata filters.

# TODO: Once AI Max+ 395 Node 3 is available, move Qdrant to persistent storage there.
#       Also add a reranker step after retrieval: cross-encoder (e.g. bge-reranker-v2)
#       running on Node 1 CPU would push context precision above 0.85.
#
# TODO: Once Blackwell is affordable, explore building a case citation graph in
#       Neo4j alongside Qdrant. Graph traversal for "cases that cite X" gives
#       structured retrieval that pure vector search cannot do.
"""

import uuid
from typing import Any

# Deterministic UUID namespace for point IDs - ensures re-ingest is idempotent
_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _point_id(case_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NS, f"{case_id}:{chunk_index}"))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

import config


class SearchResult:
    def __init__(self, payload: dict[str, Any], score: float) -> None:
        self.payload = payload
        self.score = score

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def case_id(self) -> str:
        return self.payload.get("case_id", "")

    @property
    def title(self) -> str:
        return self.payload.get("title", "")

    @property
    def court_name(self) -> str:
        return self.payload.get("court_name", "")

    @property
    def url(self) -> str:
        return self.payload.get("url", "")

    @property
    def date(self) -> str:
        return self.payload.get("date", "")


class VectorStore:
    def __init__(self) -> None:
        self._client = QdrantClient(url=config.QDRANT_URL)

    def ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if config.QDRANT_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=config.EMBED_DIM,
                    distance=Distance.COSINE,
                ),
            )
            # Payload indexes for fast filtered search
            for field in ("court", "year"):
                self._client.create_payload_index(
                    collection_name=config.QDRANT_COLLECTION,
                    field_name=field,
                    field_schema="keyword" if field == "court" else "integer",
                )

    def upsert(self, vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        points = [
            PointStruct(
                id=_point_id(payload["case_id"], payload["chunk_index"]),
                vector=vec,
                payload=payload,
            )
            for vec, payload in zip(vectors, payloads)
        ]
        self._client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = config.TOP_K,
        courts: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[SearchResult]:
        must = []

        if courts:
            must.append(FieldCondition(key="court", match=MatchAny(any=courts)))

        if year_from is not None or year_to is not None:
            must.append(
                FieldCondition(
                    key="year",
                    range=Range(
                        gte=year_from if year_from is not None else 1900,
                        lte=year_to if year_to is not None else 2100,
                    ),
                )
            )

        query_filter = Filter(must=must) if must else None

        hits = self._client.search(
            collection_name=config.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [SearchResult(h.payload, h.score) for h in hits]

    def get_by_case_id(self, case_id: str) -> list[SearchResult]:
        results, _ = self._client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
            ),
            with_payload=True,
            limit=50,
        )
        return [SearchResult(r.payload, 1.0) for r in results]

    def collection_stats(self) -> dict[str, Any]:
        info = self._client.get_collection(config.QDRANT_COLLECTION)
        return {
            "points_count": info.points_count or 0,
            "indexed_vectors_count": info.indexed_vectors_count or 0,
            "status": str(info.status),
        }
