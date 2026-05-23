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
    HasIdCondition,
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
    def __init__(self, collection: str | None = None) -> None:
        self._client = QdrantClient(url=config.QDRANT_URL)
        self._collection = collection or config.QDRANT_COLLECTION

    def ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if config.QDRANT_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=config.EMBED_DIM,
                    distance=Distance.COSINE,
                ),
            )
            # Payload indexes for fast filtered search
            for field in ("court", "year"):
                self._client.create_payload_index(
                    collection_name=self._collection,
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
        self._client.upsert(collection_name=self._collection, points=points)

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
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [SearchResult(h.payload, h.score) for h in hits]

    def search_filtered(
        self,
        query_vector: list[float],
        query_filter: Filter,
        top_k: int = config.TOP_K,
    ) -> list[SearchResult]:
        """Semantic search with an arbitrary pre-built Qdrant filter."""
        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [SearchResult(h.payload, h.score) for h in hits]

    def search_within(
        self,
        query_vector: list[float],
        point_ids: list[str],
        top_k: int = config.TOP_K,
    ) -> list[SearchResult]:
        """Semantic search restricted to a specific set of qdrant_point_ids.

        Used for SQL-first hybrid retrieval: PostgreSQL narrows the candidate set,
        Qdrant ranks semantically within it.
        """
        if not point_ids:
            return []

        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=Filter(must=[HasIdCondition(has_id=point_ids)]),
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
        counsel_surname: str | None = None,
        crown_counsel: str | None = None,
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

        # Counsel filters - exact match on indexed keyword arrays
        if counsel_surname:
            must.append(FieldCondition(
                key="counsel.all_surnames",
                match=MatchValue(value=counsel_surname),
            ))
        if crown_counsel:
            must.append(FieldCondition(
                key="counsel.crown",
                match=MatchValue(value=crown_counsel),
            ))

        query_filter = (
            Filter(must=must or None, should=should or None)
            if (must or should)
            else None
        )

        # Fetch limit*4 candidates to have room after dedup
        raw, _ = self._client.scroll(
            collection_name=self._collection,
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

    def scroll_sentencing(
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
        """Scroll criminal chunks with sentencing data. Merges per case_id."""
        must: list = [FieldCondition(key="sentencing.has_data", match=MatchValue(value=True))]
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
        if sentence_type:
            must.append(FieldCondition(
                key="sentencing.sentence_type",
                match=MatchValue(value=sentence_type),
            ))
        if min_starting_point is not None or max_starting_point is not None:
            must.append(FieldCondition(
                key="sentencing.starting_point_months",
                range=Range(
                    gte=min_starting_point if min_starting_point is not None else 0.0,
                    lte=max_starting_point if max_starting_point is not None else 9999.0,
                ),
            ))
        if min_final_sentence is not None or max_final_sentence is not None:
            must.append(FieldCondition(
                key="sentencing.final_sentence_months",
                range=Range(
                    gte=min_final_sentence if min_final_sentence is not None else 0.0,
                    lte=max_final_sentence if max_final_sentence is not None else 9999.0,
                ),
            ))
        if has_guilty_plea is not None:
            must.append(FieldCondition(
                key="sentencing.has_guilty_plea",
                match=MatchValue(value=has_guilty_plea),
            ))
        if flags:
            for f in flags:
                should.append(FieldCondition(key="flags", match=MatchValue(value=f)))

        query_filter = Filter(must=must, should=should or None)

        raw, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=query_filter,
            limit=limit * 6,
            with_payload=True,
            with_vectors=False,
        )

        results = [SearchResult(r.payload, 1.0) for r in raw]

        # Merge per case_id: accumulate sentencing fields across all chunks, keep
        # the chunk with the most populated sentencing fields as the representative.
        _key_fields = [
            "starting_point_months", "final_sentence_months",
            "home_detention_months", "community_work_hours",
            "guilty_plea_discount_pct",
        ]

        def _completeness(r: SearchResult) -> int:
            s = r.payload.get("sentencing", {})
            return sum(1 for k in _key_fields if s.get(k) is not None)

        case_best: dict[str, SearchResult] = {}
        case_merged: dict[str, dict] = {}

        for r in results:
            cid = r.case_id
            s = r.payload.get("sentencing", {})
            if cid not in case_best:
                case_best[cid] = r
                case_merged[cid] = {k: v for k, v in s.items() if k != "has_data"}
            else:
                if _completeness(r) > _completeness(case_best[cid]):
                    case_best[cid] = r
                for k, v in s.items():
                    if k != "has_data" and case_merged[cid].get(k) is None and v is not None:
                        case_merged[cid][k] = v

        merged_results: list[SearchResult] = []
        for cid, best in case_best.items():
            merged_payload = {
                **best.payload,
                "sentencing": {**case_merged[cid], "has_data": True},
            }
            merged_results.append(SearchResult(merged_payload, 1.0))

        # Sort by starting_point_months desc (most serious first), fall back to final sentence
        def _sort_key(r: SearchResult) -> float:
            s = r.payload.get("sentencing", {})
            return s.get("starting_point_months") or s.get("final_sentence_months") or 0.0

        return sorted(merged_results, key=_sort_key, reverse=True)[:limit]

    def scroll_pg(
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
        """Scroll ERA/NZEmpC chunks with PG outcome data. Deduplicates per case_id."""
        must: list = [FieldCondition(key="pg.has_data", match=MatchValue(value=True))]
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
        if reinstatement is not None:
            must.append(FieldCondition(
                key="pg.reinstatement_ordered",
                match=MatchValue(value=reinstatement),
            ))
        if min_contributory is not None or max_contributory is not None:
            must.append(FieldCondition(
                key="pg.contributory_conduct_pct",
                range=Range(
                    gte=min_contributory if min_contributory is not None else 0.0,
                    lte=max_contributory if max_contributory is not None else 100.0,
                ),
            ))
        if min_compensation is not None or max_compensation is not None:
            must.append(FieldCondition(
                key="penalty.awarded_amount",
                range=Range(
                    gte=min_compensation if min_compensation is not None else 0.0,
                    lte=max_compensation if max_compensation is not None else 999_999_999.0,
                ),
            ))
        if grievance_types:
            for gt in grievance_types:
                should.append(FieldCondition(key="pg.grievance_types", match=MatchValue(value=gt)))

        query_filter = Filter(must=must, should=should or None)

        raw, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=query_filter,
            limit=limit * 4,
            with_payload=True,
            with_vectors=False,
        )

        results = [SearchResult(r.payload, 1.0) for r in raw]

        # Merge per case_id: accumulate pg fields
        case_best: dict[str, SearchResult] = {}
        case_merged_pg: dict[str, dict] = {}

        def _pg_completeness(r: SearchResult) -> int:
            pg = r.payload.get("pg", {})
            score = len(pg.get("grievance_types") or [])
            if pg.get("reinstatement_ordered") is not None:
                score += 2
            if pg.get("contributory_conduct_pct") is not None:
                score += 1
            return score

        for r in results:
            cid = r.case_id
            pg = r.payload.get("pg", {})
            if cid not in case_best:
                case_best[cid] = r
                case_merged_pg[cid] = {k: v for k, v in pg.items() if k != "has_data"}
            else:
                if _pg_completeness(r) > _pg_completeness(case_best[cid]):
                    case_best[cid] = r
                for k, v in pg.items():
                    if k == "grievance_types":
                        existing = case_merged_pg[cid].get("grievance_types") or []
                        for gt in (v or []):
                            if gt not in existing:
                                existing.append(gt)
                        case_merged_pg[cid]["grievance_types"] = existing
                    elif k != "has_data" and case_merged_pg[cid].get(k) is None and v is not None:
                        case_merged_pg[cid][k] = v

        merged: list[SearchResult] = []
        for cid, best in case_best.items():
            merged_payload = {
                **best.payload,
                "pg": {**case_merged_pg[cid], "has_data": True},
            }
            merged.append(SearchResult(merged_payload, 1.0))

        # Sort by compensation desc
        def _comp_key(r: SearchResult) -> float:
            return r.payload.get("penalty", {}).get("awarded_amount") or 0.0

        return sorted(merged, key=_comp_key, reverse=True)[:limit]

    def get_by_case_id(self, case_id: str) -> list[SearchResult]:
        results, _ = self._client.scroll(
            collection_name=self._collection,
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
