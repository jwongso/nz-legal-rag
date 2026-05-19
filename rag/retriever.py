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


def _penalty_weight(r: "SearchResult") -> float:
    """Sort key for notable case ranking: higher OSI/recovery_rate = more notable."""
    p = r.payload.get("penalty", {})
    osi = p.get("outcome_osi") or 0.0
    rr  = min(p.get("recovery_rate") or 0.0, 1.5)
    return max(osi, rr)

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
        flags: list[str] | None = None,
    ) -> list[SearchResult]:
        must: list = []
        should: list = []

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

        # Flag filter: OR logic - chunks that contain any of the requested flags
        if flags:
            for f in flags:
                should.append(FieldCondition(key="flags", match=MatchValue(value=f)))

        query_filter = (
            Filter(must=must or None, should=should or None)
            if (must or should)
            else None
        )

        hits = self._client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [SearchResult(h.payload, h.score) for h in hits]

    def scroll_notable(
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
        """Scroll Qdrant with flag/penalty filters. Deduplicates to 1 chunk per case_id."""
        must: list = []
        should: list = []

        if courts:
            must.append(FieldCondition(key="court", match=MatchAny(any=courts)))

        if year_from is not None or year_to is not None:
            must.append(FieldCondition(
                key="year",
                range=Range(
                    gte=year_from if year_from is not None else 1900,
                    lte=year_to if year_to is not None else 2100,
                ),
            ))

        if min_outcome_osi is not None or max_outcome_osi is not None:
            must.append(FieldCondition(
                key="penalty.outcome_osi",
                range=Range(
                    gte=min_outcome_osi if min_outcome_osi is not None else 0.0,
                    lte=max_outcome_osi if max_outcome_osi is not None else 1.0,
                ),
            ))

        if min_recovery_rate is not None or max_recovery_rate is not None:
            must.append(FieldCondition(
                key="penalty.recovery_rate",
                range=Range(
                    gte=min_recovery_rate if min_recovery_rate is not None else 0.0,
                    lte=max_recovery_rate if max_recovery_rate is not None else 99999.0,
                ),
            ))

        if min_awarded is not None or max_awarded is not None:
            must.append(FieldCondition(
                key="penalty.awarded_amount",
                range=Range(
                    gte=min_awarded if min_awarded is not None else 0.0,
                    lte=max_awarded if max_awarded is not None else 999_999_999.0,
                ),
            ))

        if flags:
            for f in flags:
                should.append(FieldCondition(key="flags", match=MatchValue(value=f)))

        query_filter = (
            Filter(must=must or None, should=should or None)
            if (must or should)
            else None
        )

        # Fetch limit*4 candidates to have room after dedup
        raw, _ = self._client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=query_filter,
            limit=limit * 4,
            with_payload=True,
            with_vectors=False,
        )

        results = [SearchResult(r.payload, 1.0) for r in raw]

        # Dedup: 1 chunk per case_id - keep the one with the highest penalty weight
        seen: dict[str, SearchResult] = {}
        for r in results:
            cid = r.case_id
            if cid not in seen or _penalty_weight(r) > _penalty_weight(seen[cid]):
                seen[cid] = r

        # Sort by penalty weight descending
        deduped = sorted(seen.values(), key=_penalty_weight, reverse=True)
        return deduped[:limit]

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
