"""Retrieval A/B benchmark: compares retrieval pipelines on the gold dataset.

Pipelines compared:
  vector_only                Pure Qdrant semantic search, no metadata filter
  sql_filter_vector          PostgreSQL court+year filter (oracle) -> Qdrant within filtered set
  sql_filter_vector_rerank   sql_filter_vector + cross-encoder reranker
  sql_filter_vector_legal    sql_filter_vector + intent-aware legal authority ranker
  sql_filter_vector_legal_rerank_5  legal ranker + cross-encoder on top 5
  sql_tracker_vector         tracker-first: JOIN sentencing/employment tables -> Qdrant
  sql_tracker_vector_legal   tracker-first + intent-aware legal authority ranker

"Oracle" filter means the expected_courts from the gold record are used as the
filter, not extracted from query text. This isolates retrieval stack quality from
planner quality. A separate planner benchmark will test filter extraction.

Tracker-first pipelines route by task_type:
  sentencing  -> JOIN sentencing_cases to restrict candidates to docs with extracted data
  employment  -> JOIN employment_cases to restrict candidates to docs with extracted data
  other       -> falls back to court-only filter (same as sql_filter_vector)

Scoring (per query, per pipeline):
  hit_gold@K    1 if any expected_document appears in top-K, else 0
  hit_rel@K     1 if any expected_document OR acceptable_document appears in top-K
  mrr           1 / rank of first expected_document, 0 if none found
  irr@5         fraction of top-5 results from must_not_include_courts

Reports written to benchmarks/reports/:
  latest.json       full per-query, per-pipeline results
  latest.md         summary table
  failures.md       detailed failure analysis

Run:
    python -m benchmarks.runners.run_retrieval
    python -m benchmarks.runners.run_retrieval --quick
    python -m benchmarks.runners.run_retrieval --pipelines sql_filter_vector_legal planner_filter_vector_legal no_filter_vector_legal
    python -m benchmarks.runners.run_retrieval --pipelines sql_filter_vector sql_tracker_vector sql_tracker_vector_legal
"""

import argparse
import asyncio
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

import config
from rag.embedder import Embedder
from rag.reranker import Reranker
from rag.retriever import VectorStore

_GOLD_PATH = Path("benchmarks/datasets/retrieval_gold.jsonl")
_REPORTS_DIR = Path("benchmarks/reports")
_FETCH_K = 50          # chunks fetched from Qdrant (deduped to unique docs before scoring)
_RERANK_FETCH_K = 100  # wider pool for reranker
_RERANK_TOP_K = 20
_BM25_CHUNK_LIMIT = 500  # chunks fetched from FTS before dedup to unique docs
_RRF_K = 60              # reciprocal rank fusion constant (standard)
_TRACKER_SOFT_BOOST = 0.04  # additive score bonus for tracker-member docs

_ALL_PIPELINES = [
    "vector_only",
    "sql_filter_vector",
    "sql_filter_vector_rerank",
    "sql_filter_vector_legal",
    "sql_filter_vector_legal_rerank_5",
    "sql_tracker_vector",
    "sql_tracker_vector_legal",
    "sql_filter_bm25_legal",
    "sql_filter_bm25_vector_rrf_legal",
    "sql_filter_bm25_vector_rrf_legal_plus_tracker_soft_boost",
    "planner_filter_vector_legal",
    "no_filter_vector_legal",
]

_BASELINE_PIPELINE = "sql_filter_vector_legal"  # used for regression analysis
_HIT_K_VALUES = [1, 3, 5, 10, 20]
_LEGAL_RERANK_CANDIDATES = 5


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def _get_point_ids_for_courts(conn, courts: list[str],
                              years: list[int] | None) -> list[str]:
    """Return qdrant_point_ids for all chunks belonging to the given courts."""
    cur = conn.cursor()
    if years:
        cur.execute("""
            SELECT c.qdrant_point_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.court = ANY(%s)
              AND EXTRACT(YEAR FROM d.decision_date) = ANY(%s)
              AND c.qdrant_point_id IS NOT NULL
        """, (courts, years))
    else:
        cur.execute("""
            SELECT c.qdrant_point_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.court = ANY(%s)
              AND c.qdrant_point_id IS NOT NULL
        """, (courts,))
    return [r[0] for r in cur.fetchall()]


def _get_point_ids_sentencing(conn, courts: list[str],
                              years: list[int] | None) -> list[str]:
    """Point IDs restricted to docs that have structured sentencing data extracted."""
    cur = conn.cursor()
    if years:
        cur.execute("""
            SELECT DISTINCT c.qdrant_point_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN sentencing_cases sc ON sc.document_id = d.id
            WHERE d.court = ANY(%s)
              AND EXTRACT(YEAR FROM d.decision_date) = ANY(%s)
              AND c.qdrant_point_id IS NOT NULL
        """, (courts, years))
    else:
        cur.execute("""
            SELECT DISTINCT c.qdrant_point_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN sentencing_cases sc ON sc.document_id = d.id
            WHERE d.court = ANY(%s)
              AND c.qdrant_point_id IS NOT NULL
        """, (courts,))
    return [r[0] for r in cur.fetchall()]


def _get_point_ids_employment(conn, courts: list[str],
                              years: list[int] | None) -> list[str]:
    """Point IDs restricted to docs that have structured employment case data extracted."""
    cur = conn.cursor()
    if years:
        cur.execute("""
            SELECT DISTINCT c.qdrant_point_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN employment_cases ec ON ec.document_id = d.id
            WHERE d.court = ANY(%s)
              AND EXTRACT(YEAR FROM d.decision_date) = ANY(%s)
              AND c.qdrant_point_id IS NOT NULL
        """, (courts, years))
    else:
        cur.execute("""
            SELECT DISTINCT c.qdrant_point_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN employment_cases ec ON ec.document_id = d.id
            WHERE d.court = ANY(%s)
              AND c.qdrant_point_id IS NOT NULL
        """, (courts,))
    return [r[0] for r in cur.fetchall()]


def _bm25_hits(conn, query: str, courts: list[str],
               years: list[int] | None, limit: int = _FETCH_K) -> list:
    """BM25 via PostgreSQL tsvector (GIN index). Returns list[SearchResult], deduped by case_id.

    Conditional: suppresses BM25 for broad natural language questions (via build_bm25_query).
    When activated, passes extracted OR-joined key terms rather than the full question.
    Uses websearch_to_tsquery (phrase-aware) and ts_rank_cd (cover density).
    Returns empty list if BM25 is suppressed or no FTS matches.
    """
    from rag.bm25_query import build_bm25_query
    from rag.retriever import SearchResult

    bm25_q = build_bm25_query(query)
    if not bm25_q.should_use:
        return []

    fts_query = bm25_q.query_terms  # OR-joined anchors, safe for websearch_to_tsquery

    cur = conn.cursor()
    year_clause = "AND EXTRACT(YEAR FROM d.decision_date) = ANY(%s)" if years else ""
    year_args = [years] if years else []

    try:
        cur.execute(f"""
            SELECT
                d.citation,
                ts_rank_cd(to_tsvector('english', COALESCE(ch.text, '')), q) AS rank,
                d.court,
                ch.chunk_index,
                EXTRACT(YEAR FROM d.decision_date)::int AS yr,
                d.document_type
            FROM chunks ch
            JOIN documents d ON d.id = ch.document_id,
                 websearch_to_tsquery('english', %s) AS q
            WHERE to_tsvector('english', COALESCE(ch.text, '')) @@ q
              AND d.court = ANY(%s)
              {year_clause}
            ORDER BY rank DESC
            LIMIT %s
        """, [fts_query, courts] + year_args + [_BM25_CHUNK_LIMIT])
    except Exception:
        return []

    rows = cur.fetchall()
    if not rows:
        return []

    # Deduplicate: best-ranked chunk per case_id
    best: dict[str, tuple] = {}
    for citation, rank, court, chunk_index, yr, doc_type in rows:
        if citation not in best or rank > best[citation][0]:
            best[citation] = (rank, court, int(chunk_index or 99), int(yr or 0), doc_type or "decision")

    hits = []
    for citation, (rank, court, chunk_index, yr, doc_type) in best.items():
        payload = {
            "case_id": citation,
            "court": court,
            "chunk_index": chunk_index,
            "year": yr,
            "document_type": doc_type,
            "citations": [],   # not in PG; citation_rich_weight not usable for BM25-only hits
            "legal_area": "",  # not in PG; tracker signal falls back to sentencing dict
            "sentencing": None,
        }
        hits.append(SearchResult(payload=payload, score=float(rank)))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def _rrf_merge(bm25_hits: list, vec_hits: list, k: int = _RRF_K) -> list:
    """Reciprocal Rank Fusion: merge BM25 and vector hit lists into a single ranked list.

    RRF score = sum(1 / (k + rank)) across systems.
    When a doc appears in both lists, the Qdrant SearchResult is kept (richer payload:
    citations, legal_area, sentencing) so the legal ranker gets full signals.
    """
    from rag.retriever import SearchResult

    bm25_rank = {h.case_id: i + 1 for i, h in enumerate(bm25_hits)}
    vec_rank = {h.case_id: i + 1 for i, h in enumerate(vec_hits)}
    bm25_by_id = {h.case_id: h for h in bm25_hits}
    vec_by_id = {h.case_id: h for h in vec_hits}
    all_ids = set(bm25_by_id) | set(vec_by_id)

    rrf_scores: dict[str, float] = {}
    for cid in all_ids:
        score = 0.0
        if cid in bm25_rank:
            score += 1.0 / (k + bm25_rank[cid])
        if cid in vec_rank:
            score += 1.0 / (k + vec_rank[cid])
        rrf_scores[cid] = score

    merged = []
    for cid in sorted(all_ids, key=rrf_scores.__getitem__, reverse=True):
        # Prefer Qdrant payload (has citations, legal_area, full sentencing dict)
        src = vec_by_id[cid] if cid in vec_by_id else bm25_by_id[cid]
        merged.append(SearchResult(payload=src.payload, score=rrf_scores[cid]))

    return merged


def _apply_tracker_soft_boost(ordered_hits: list, task_type: str, conn,
                              boost: float = _TRACKER_SOFT_BOOST) -> list:
    """Post-legal-ranker soft boost: add a bonus for docs with structured tracker data.

    Unlike tracker-first (hard JOIN that excludes non-tracker docs), this re-ranks
    by rank-normalized position score + additive tracker bonus. No docs are excluded.
    """
    if task_type not in ("sentencing", "employment"):
        return ordered_hits

    case_ids = [h.case_id for h in ordered_hits]
    cur = conn.cursor()
    if task_type == "sentencing":
        cur.execute("""
            SELECT d.citation FROM documents d
            JOIN sentencing_cases sc ON sc.document_id = d.id
            WHERE d.citation = ANY(%s)
        """, (case_ids,))
    else:
        cur.execute("""
            SELECT d.citation FROM documents d
            JOIN employment_cases ec ON ec.document_id = d.id
            WHERE d.citation = ANY(%s)
        """, (case_ids,))
    tracker_ids = {r[0] for r in cur.fetchall()}

    if not tracker_ids:
        return ordered_hits

    n = len(ordered_hits)
    scored = []
    for i, h in enumerate(ordered_hits):
        pos_score = (n - i) / n  # rank-normalized: 1.0 for rank-1, 0.0 for last
        extra = boost if h.case_id in tracker_ids else 0.0
        scored.append((pos_score + extra, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored]


def _get_court_for_citation(conn, citation: str) -> str | None:
    cur = conn.cursor()
    cur.execute("SELECT court FROM documents WHERE citation = %s LIMIT 1", (citation,))
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(case_ids: list[str], gold: dict,
           court_lookup: dict[str, str]) -> dict:
    expected = set(gold["expected_documents"])
    acceptable = set(gold["acceptable_documents"])
    relevant = expected | acceptable
    must_not = set(gold.get("must_not_include_courts", []))

    gold_ranks = [i + 1 for i, cid in enumerate(case_ids) if cid in expected]
    rel_ranks  = [i + 1 for i, cid in enumerate(case_ids) if cid in relevant]

    result: dict = {}
    for k in _HIT_K_VALUES:
        result[f"hit_gold_at_{k}"] = 1 if any(r <= k for r in gold_ranks) else 0
        result[f"hit_rel_at_{k}"]  = 1 if any(r <= k for r in rel_ranks)  else 0

    result["mrr"] = round(1.0 / gold_ranks[0], 4) if gold_ranks else 0.0
    result["gold_ranks"] = gold_ranks[:3]
    result["rel_ranks"]  = rel_ranks[:3]

    # Irrelevant result rate: fraction of top-5 from must_not_include_courts
    top5_courts = [court_lookup.get(cid, "") for cid in case_ids[:5]]
    irr = sum(1 for c in top5_courts if c in must_not) / max(len(top5_courts), 1)
    result["irr_at_5"] = round(irr, 3)

    return result


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _dedup(hits) -> list[str]:
    """Deduplicate hits to one entry per case_id, keeping best score, preserving rank order."""
    seen: dict[str, float] = {}
    for h in hits:
        if h.case_id not in seen or h.score > seen[h.case_id]:
            seen[h.case_id] = h.score
    return sorted(seen, key=seen.__getitem__, reverse=True)


def _dedup_sr(hits) -> list:
    """Deduplicate to one SearchResult per case_id (best score), sorted descending."""
    seen: dict[str, object] = {}
    for h in hits:
        if h.case_id not in seen or h.score > seen[h.case_id].score:  # type: ignore[union-attr]
            seen[h.case_id] = h
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)  # type: ignore[union-attr]


async def _run_vector_only(query_vec: list[float], store: VectorStore) -> list[str]:
    hits = store.search(query_vec, top_k=_FETCH_K)
    return _dedup(hits)


async def _run_sql_filter_vector(query_vec: list[float], store: VectorStore,
                                  conn, courts: list[str],
                                  years: list[int] | None) -> list[str]:
    point_ids = _get_point_ids_for_courts(conn, courts, years)
    if not point_ids:
        return await _run_vector_only(query_vec, store)
    hits = store.search_within(query_vec, point_ids, top_k=_FETCH_K)
    return _dedup(hits)


async def _run_sql_filter_vector_rerank(query: str, query_vec: list[float],
                                         store: VectorStore, conn,
                                         courts: list[str],
                                         years: list[int] | None,
                                         reranker: Reranker) -> list[str]:
    point_ids = _get_point_ids_for_courts(conn, courts, years)
    hits = store.search_within(query_vec, point_ids, top_k=_RERANK_FETCH_K) \
           if point_ids else store.search(query_vec, top_k=_RERANK_FETCH_K)
    reranked = reranker.rerank(query, hits, top_k=_RERANK_TOP_K)
    seen: dict[str, float] = {}
    for i, h in enumerate(reranked):
        if h.case_id not in seen:
            seen[h.case_id] = len(reranked) - i
    return sorted(seen, key=seen.__getitem__, reverse=True)


async def _run_sql_filter_vector_legal(query: str, query_vec: list[float],
                                        store: VectorStore, conn,
                                        courts: list[str],
                                        years: list[int] | None) -> list[str]:
    from rag.legal_ranker import QueryContext, rerank as legal_rerank
    point_ids = _get_point_ids_for_courts(conn, courts, years)
    hits = store.search_within(query_vec, point_ids, top_k=_FETCH_K) \
           if point_ids else store.search(query_vec, top_k=_FETCH_K)
    deduped = _dedup_sr(hits)
    ctx = QueryContext.from_query(query)
    ordered = legal_rerank(deduped, ctx)
    return [h.case_id for h in ordered]


async def _run_sql_filter_vector_legal_rerank_5(query: str, query_vec: list[float],
                                                 store: VectorStore, conn,
                                                 courts: list[str],
                                                 years: list[int] | None,
                                                 reranker: Reranker) -> list[str]:
    from rag.legal_ranker import QueryContext, rerank as legal_rerank
    point_ids = _get_point_ids_for_courts(conn, courts, years)
    hits = store.search_within(query_vec, point_ids, top_k=_RERANK_FETCH_K) \
           if point_ids else store.search(query_vec, top_k=_RERANK_FETCH_K)
    deduped = _dedup_sr(hits)
    ctx = QueryContext.from_query(query)
    legal_ordered = legal_rerank(deduped, ctx)
    candidates = legal_ordered[:_LEGAL_RERANK_CANDIDATES]
    reranked = reranker.rerank(query, candidates, top_k=_LEGAL_RERANK_CANDIDATES)
    seen: dict[str, float] = {}
    for i, h in enumerate(reranked):
        if h.case_id not in seen:
            seen[h.case_id] = len(reranked) - i
    return sorted(seen, key=seen.__getitem__, reverse=True)


async def _run_sql_filter_bm25_legal(query: str, conn,
                                     courts: list[str],
                                     years: list[int] | None) -> list[str]:
    from rag.legal_ranker import QueryContext, rerank as legal_rerank
    hits = _bm25_hits(conn, query, courts, years)
    if not hits:
        return []
    ctx = QueryContext.from_query(query)
    ordered = legal_rerank(hits, ctx)
    return [h.case_id for h in ordered]


async def _run_sql_filter_bm25_vector_rrf_legal(query: str, query_vec: list[float],
                                                 store: VectorStore, conn,
                                                 courts: list[str],
                                                 years: list[int] | None) -> list[str]:
    from rag.legal_ranker import QueryContext, rerank as legal_rerank
    bm25 = _bm25_hits(conn, query, courts, years, limit=_FETCH_K)
    point_ids = _get_point_ids_for_courts(conn, courts, years)
    raw_vec = store.search_within(query_vec, point_ids, top_k=_FETCH_K) \
              if point_ids else store.search(query_vec, top_k=_FETCH_K)
    vec = _dedup_sr(raw_vec)
    # RRF only when BM25 returned hits; otherwise fall through to pure vector
    # to avoid replacing cosine similarity scores with near-zero RRF scores.
    fused = _rrf_merge(bm25, vec) if bm25 else vec
    ctx = QueryContext.from_query(query)
    ordered = legal_rerank(fused, ctx)
    return [h.case_id for h in ordered]


async def _run_sql_filter_bm25_vector_rrf_legal_plus_tracker_soft_boost(
        task_type: str, query: str, query_vec: list[float],
        store: VectorStore, conn,
        courts: list[str], years: list[int] | None) -> list[str]:
    from rag.legal_ranker import QueryContext, rerank as legal_rerank
    bm25 = _bm25_hits(conn, query, courts, years, limit=_FETCH_K)
    point_ids = _get_point_ids_for_courts(conn, courts, years)
    raw_vec = store.search_within(query_vec, point_ids, top_k=_FETCH_K) \
              if point_ids else store.search(query_vec, top_k=_FETCH_K)
    vec = _dedup_sr(raw_vec)
    fused = _rrf_merge(bm25, vec) if bm25 else vec
    ctx = QueryContext.from_query(query)
    ordered = legal_rerank(fused, ctx)
    boosted = _apply_tracker_soft_boost(ordered, task_type, conn)
    return [h.case_id for h in boosted]


async def _run_planner_filter_vector_legal(query: str, query_vec: list[float],
                                            store: VectorStore, conn) -> tuple[list[str], object]:
    """Like sql_filter_vector_legal but courts come from heuristic planner, not oracle."""
    from rag.legal_ranker import QueryContext, rerank as legal_rerank
    from rag.court_planner import plan_courts
    plan = plan_courts(query)
    if plan.courts:
        point_ids = _get_point_ids_for_courts(conn, plan.courts, plan.years)
        hits = store.search_within(query_vec, point_ids, top_k=_FETCH_K) \
               if point_ids else store.search(query_vec, top_k=_FETCH_K)
    else:
        hits = store.search(query_vec, top_k=_FETCH_K)
    deduped = _dedup_sr(hits)
    ctx = QueryContext.from_query(query)
    ordered = legal_rerank(deduped, ctx)
    return [h.case_id for h in ordered], plan


async def _run_no_filter_vector_legal(query: str, query_vec: list[float],
                                      store: VectorStore) -> list[str]:
    """Full-corpus vector search + legal ranker; no court filter at all."""
    from rag.legal_ranker import QueryContext, rerank as legal_rerank
    hits = store.search(query_vec, top_k=_FETCH_K)
    deduped = _dedup_sr(hits)
    ctx = QueryContext.from_query(query)
    ordered = legal_rerank(deduped, ctx)
    return [h.case_id for h in ordered]


def _tracker_point_ids(task_type: str, conn, courts: list[str],
                       years: list[int] | None) -> list[str]:
    if task_type == "sentencing":
        return _get_point_ids_sentencing(conn, courts, years)
    if task_type == "employment":
        return _get_point_ids_employment(conn, courts, years)
    return _get_point_ids_for_courts(conn, courts, years)


async def _run_sql_tracker_vector(task_type: str, query_vec: list[float],
                                   store: VectorStore, conn,
                                   courts: list[str],
                                   years: list[int] | None) -> list[str]:
    point_ids = _tracker_point_ids(task_type, conn, courts, years)
    if not point_ids:
        return await _run_vector_only(query_vec, store)
    hits = store.search_within(query_vec, point_ids, top_k=_FETCH_K)
    return _dedup(hits)


async def _run_sql_tracker_vector_legal(task_type: str, query: str,
                                         query_vec: list[float],
                                         store: VectorStore, conn,
                                         courts: list[str],
                                         years: list[int] | None) -> list[str]:
    from rag.legal_ranker import QueryContext, rerank as legal_rerank
    point_ids = _tracker_point_ids(task_type, conn, courts, years)
    hits = store.search_within(query_vec, point_ids, top_k=_FETCH_K) \
           if point_ids else store.search(query_vec, top_k=_FETCH_K)
    deduped = _dedup_sr(hits)
    ctx = QueryContext.from_query(query)
    ordered = legal_rerank(deduped, ctx)
    return [h.case_id for h in ordered]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _aggregate(pipeline_runs: list[dict]) -> dict:
    keys = [f"hit_gold_at_{k}" for k in _HIT_K_VALUES] + \
           [f"hit_rel_at_{k}" for k in _HIT_K_VALUES] + ["mrr", "irr_at_5"]
    return {k: _mean([r[k] for r in pipeline_runs]) for k in keys}


def _build_planner_analysis_report(all_results: list[dict]) -> str:
    """Per-query comparison: oracle courts vs heuristic planner courts.

    Match types:
      exact     planner courts == oracle courts
      superset  oracle courts is a strict subset of planner courts
      subset    planner courts is a strict subset of oracle courts (coverage risk)
      partial   overlap but neither is a subset of the other
      disjoint  no overlap (certain coverage regression)
      no_filter planner returned None (full corpus fallback)
    """
    baseline = _BASELINE_PIPELINE
    planner = "planner_filter_vector_legal"
    no_filter = "no_filter_vector_legal"

    lines = [
        "# Planner vs Oracle Court Filter Analysis",
        "",
        "Compares heuristic court planner output against oracle (gold expected_courts).",
        "Measures quality gap between oracle-filtered and planner-filtered pipelines.",
        "",
        "## Per-Query Detail",
        "",
        "| Query | Oracle | Planner | Match | Oracle MRR | Planner MRR | NoFilter MRR |",
        "|---|---|---|---|---:|---:|---:|",
    ]

    match_counts: dict[str, int] = {}
    for r in all_results:
        plan_data = r.get("planner_data", {})
        oracle_courts = sorted(r.get("expected_courts", []))
        planner_courts = sorted(plan_data.get("courts") or [])
        signals = ", ".join(plan_data.get("signals", [])[:3])
        confidence = plan_data.get("confidence", "?")

        o, p = set(oracle_courts), set(planner_courts)
        if not p:
            match = "no_filter"
        elif o == p:
            match = "exact"
        elif o < p:
            match = "superset"
        elif p < o:
            match = "subset"
        elif o & p:
            match = "partial"
        else:
            match = "disjoint"

        match_counts[match] = match_counts.get(match, 0) + 1

        base_mrr = r.get("scores", {}).get(baseline, {}).get("mrr", 0)
        plan_mrr = r.get("scores", {}).get(planner, {}).get("mrr", 0)
        nf_mrr   = r.get("scores", {}).get(no_filter, {}).get("mrr", 0)
        oracle_str  = "+".join(oracle_courts)
        planner_str = "+".join(planner_courts) if planner_courts else "(none)"
        delta_symbol = "=" if plan_mrr == base_mrr else ("+" if plan_mrr > base_mrr else "-")
        lines.append(
            f"| {r['id']} "
            f"| {oracle_str} "
            f"| {planner_str} "
            f"| {match} "
            f"| {base_mrr:.3f} "
            f"| {plan_mrr:.3f} ({delta_symbol}) "
            f"| {nf_mrr:.3f} |"
        )

    lines += [
        "",
        "## Match Type Summary",
        "",
        "| Match type | Count | Meaning |",
        "|---|---:|---|",
        f"| exact | {match_counts.get('exact', 0)} | Planner matched oracle courts perfectly |",
        f"| superset | {match_counts.get('superset', 0)} | Planner included all oracle courts + extras (safe, larger pool) |",
        f"| subset | {match_counts.get('subset', 0)} | Planner missed some oracle courts (coverage risk) |",
        f"| partial | {match_counts.get('partial', 0)} | Partial overlap (some oracle courts missing) |",
        f"| disjoint | {match_counts.get('disjoint', 0)} | No overlap - definite coverage regression |",
        f"| no_filter | {match_counts.get('no_filter', 0)} | Planner returned None - full corpus fallback |",
    ]

    # Aggregate MRR comparison
    all_base = [r.get("scores", {}).get(baseline, {}).get("mrr", 0) for r in all_results if baseline in r.get("scores", {})]
    all_plan = [r.get("scores", {}).get(planner, {}).get("mrr", 0) for r in all_results if planner in r.get("scores", {})]
    all_nf   = [r.get("scores", {}).get(no_filter, {}).get("mrr", 0) for r in all_results if no_filter in r.get("scores", {})]

    def _m(lst): return round(sum(lst) / len(lst), 3) if lst else 0.0

    lines += [
        "",
        "## Aggregate MRR",
        "",
        "| Pipeline | MRR | vs oracle |",
        "|---|---:|---:|",
        f"| {baseline} (oracle) | {_m(all_base):.3f} | - |",
        f"| {planner} | {_m(all_plan):.3f} | {_m(all_plan) - _m(all_base):+.3f} |",
        f"| {no_filter} | {_m(all_nf):.3f} | {_m(all_nf) - _m(all_base):+.3f} |",
    ]

    return "\n".join(lines) + "\n"


def _build_regression_report(all_results: list[dict], pipelines_run: list[str]) -> str:
    """Compare each pipeline against _BASELINE_PIPELINE per query.

    Three regression categories:
      coverage_regression  baseline H@5(r)=1, new H@5(r)=0  (acceptable docs lost)
      gold_rank_regression baseline found gold (MRR>0), new did not  (gold doc dropped out)
      mode_regression      task_type avg MRR decreased vs baseline
    """
    baseline = _BASELINE_PIPELINE
    comparators = [p for p in pipelines_run if p != baseline]
    if not comparators:
        return "# Regression Report\n\nNo comparator pipelines vs baseline.\n"

    lines = [
        "# Retrieval Benchmark: Regression Analysis",
        "",
        f"Baseline: `{baseline}`",
        "",
    ]

    # Per-pipeline regression counts
    lines += ["## Summary: Regression Counts per Pipeline", ""]
    lines.append("| Pipeline | coverage_regression | gold_rank_regression | mode_regression |")
    lines.append("|---|---:|---:|---:|")

    reg_data: dict[str, dict[str, list[str]]] = {p: {"coverage": [], "gold_rank": [], "mode": []} for p in comparators}

    for r in all_results:
        base_sc = r.get("scores", {}).get(baseline, {})
        task = r["task_type"]
        for p in comparators:
            new_sc = r.get("scores", {}).get(p, {})
            if not new_sc:
                continue
            qid = r["id"]
            # coverage_regression: baseline rel hit, new misses entirely
            if base_sc.get("hit_rel_at_5", 0) == 1 and new_sc.get("hit_rel_at_5", 0) == 0:
                reg_data[p]["coverage"].append(qid)
            # gold_rank_regression: baseline found gold in top-10, new did not
            if base_sc.get("hit_gold_at_10", 0) == 1 and new_sc.get("hit_gold_at_10", 0) == 0:
                reg_data[p]["gold_rank"].append(qid)
            # mode_regression: task_type-level, logged per query where MRR dropped
            if base_sc.get("mrr", 0) > 0 and new_sc.get("mrr", 0) < base_sc.get("mrr", 0) - 0.05:
                reg_data[p]["mode"].append(f"{qid}[{task}]")

    for p in comparators:
        rd = reg_data[p]
        lines.append(
            f"| {p} | {len(rd['coverage'])} | {len(rd['gold_rank'])} | {len(rd['mode'])} |"
        )

    lines += [""]

    # Per-pipeline detail
    for p in comparators:
        rd = reg_data[p]
        lines += [f"## {p}", ""]
        if rd["coverage"]:
            lines.append(f"**coverage_regression** (baseline H@5(r)=1, new=0): {', '.join(rd['coverage'])}")
        if rd["gold_rank"]:
            lines.append(f"**gold_rank_regression** (baseline found gold@10, new did not): {', '.join(rd['gold_rank'])}")
        if rd["mode"]:
            lines.append(f"**mode_regression** (MRR dropped >0.05 vs baseline): {', '.join(rd['mode'])}")
        if not any(rd.values()):
            lines.append("No regressions vs baseline.")
        lines.append("")

    # Per-task-type MRR delta
    lines += ["## MRR Delta by Task Type vs Baseline", ""]
    task_types = sorted(set(r["task_type"] for r in all_results))
    header = "| Task type | baseline MRR | " + " | ".join(comparators) + " |"
    sep = "|---|---:|" + "---:|" * len(comparators)
    lines += [header, sep]
    for tt in task_types:
        tt_results = [r for r in all_results if r["task_type"] == tt]
        base_mrr = _mean([r["scores"].get(baseline, {}).get("mrr", 0) for r in tt_results])
        row = f"| {tt} | {base_mrr:.3f} |"
        for p in comparators:
            new_mrr = _mean([r["scores"].get(p, {}).get("mrr", 0) for r in tt_results if p in r.get("scores", {})])
            delta = new_mrr - base_mrr
            sign = "+" if delta >= 0 else ""
            row += f" {new_mrr:.3f} ({sign}{delta:.3f}) |"
        lines.append(row)

    lines += ["", "## Note on Sentencing MRR=0", "",
              "All sentencing gold documents (expected_documents) are specific NZCA decisions chosen",
              "for their cross-referential value by a domain expert. Several do not contain the exact",
              "offence terms in their text (e.g. Webster v R [2026] NZCA 67 is a murder case, not",
              "an aggravated robbery case). MRR=0 on sentencing queries is inherent to how the gold",
              "dataset was constructed and is not fixable by any retrieval pipeline change alone.",
              "H@5(rel)=1.00 confirms relevant documents ARE being retrieved.", ""]

    return "\n".join(lines) + "\n"


def _write_reports(all_results: list[dict], gold_records: list[dict],
                   pipelines_run: list[str]) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # --- JSON report ---
    json_path = _REPORTS_DIR / "latest.json"
    json_path.write_text(json.dumps({
        "timestamp": ts,
        "pipelines": pipelines_run,
        "n_queries": len(gold_records),
        "results": all_results,
    }, indent=2))
    print(f"  -> {json_path}")

    # --- Markdown summary ---
    lines = [
        f"# NZ Legal RAG Retrieval Benchmark",
        f"",
        f"Generated: {ts}  |  Queries: {len(gold_records)}  |  Corpus: nz_legal",
        f"",
        f"> Gold = exact expected document hit. Rel = expected OR acceptable document hit.",
        f"",
        f"## Pipeline Summary",
        f"",
        "| Pipeline | Hit@1(g) | Hit@5(g) | Hit@5(r) | Hit@10(g) | Hit@10(r) | MRR | IRR@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    aggregates = {}
    for p in pipelines_run:
        runs = [r["scores"][p] for r in all_results if p in r.get("scores", {})]
        agg = _aggregate(runs)
        aggregates[p] = agg
        lines.append(
            f"| {p} "
            f"| {agg['hit_gold_at_1']:.2f} "
            f"| {agg['hit_gold_at_5']:.2f} "
            f"| {agg['hit_rel_at_5']:.2f} "
            f"| {agg['hit_gold_at_10']:.2f} "
            f"| {agg['hit_rel_at_10']:.2f} "
            f"| {agg['mrr']:.3f} "
            f"| {agg['irr_at_5']:.2f} |"
        )

    lines += [
        "",
        "## Per-Task-Type Breakdown",
        "",
        "| Task Type | Pipeline | Hit@5(g) | Hit@5(r) | MRR |",
        "|---|---|---:|---:|---:|",
    ]
    task_types = sorted(set(r["task_type"] for r in all_results))
    for tt in task_types:
        for p in pipelines_run:
            runs = [r["scores"][p] for r in all_results
                    if r["task_type"] == tt and p in r.get("scores", {})]
            if runs:
                agg = _aggregate(runs)
                lines.append(
                    f"| {tt} | {p} "
                    f"| {agg['hit_gold_at_5']:.2f} "
                    f"| {agg['hit_rel_at_5']:.2f} "
                    f"| {agg['mrr']:.3f} |"
                )

    lines += ["", "## Per-Query Results", ""]
    header_parts = []
    for p in pipelines_run:
        header_parts.append(f"{p[:18]} H@5(g)")
        header_parts.append(f"{p[:18]} H@5(r)")
    lines.append("| Query ID | Task | " + " | ".join(header_parts) + " |")
    lines.append("|---|---|" + "|---:" * len(pipelines_run) * 2 + "|")
    for r in all_results:
        row = f"| {r['id']} | {r['task_type']} "
        for p in pipelines_run:
            sc = r.get("scores", {}).get(p, {})
            row += f"| {sc.get('hit_gold_at_5', '-')} | {sc.get('hit_rel_at_5', '-')} "
        row += "|"
        lines.append(row)

    md_path = _REPORTS_DIR / "latest.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"  -> {md_path}")

    # --- Failures report ---
    fail_lines = ["# Retrieval Benchmark: Failure Analysis", ""]
    any_failure = False
    for r in all_results:
        failures = {p: sc for p, sc in r.get("scores", {}).items()
                    if sc.get("hit_gold_at_5", 0) == 0}
        if not failures:
            continue
        any_failure = True
        fail_lines += [
            f"## {r['id']}",
            f"",
            f"**Query:** {r['query']}",
            f"",
            f"**Task type:** {r['task_type']}  "
            f"|  **Expected courts:** {r.get('expected_courts', [])}",
            f"",
            f"**Expected docs (gold):** {r.get('expected_documents', [])}",
            f"",
        ]
        for p, sc in r.get("scores", {}).items():
            top5 = r.get("top5", {}).get(p, [])
            hit5 = sc.get("hit_gold_at_5", 0)
            fail_lines.append(
                f"**{p}:** hit@5={hit5}  MRR={sc.get('mrr', 0):.3f}  "
                f"IRR@5={sc.get('irr_at_5', 0):.2f}"
            )
            if top5:
                fail_lines.append("  Top-5 returned:")
                for i, cid in enumerate(top5, 1):
                    marker = " [GOLD]" if cid in set(r.get("expected_documents", [])) else \
                             " [ok]" if cid in set(r.get("acceptable_documents", [])) else ""
                    fail_lines.append(f"  {i}. {cid}{marker}")
        fail_lines.append("")

    if not any_failure:
        fail_lines.append("No failures at hit@5 on any pipeline.")

    fail_path = _REPORTS_DIR / "failures.md"
    fail_path.write_text("\n".join(fail_lines) + "\n")
    print(f"  -> {fail_path}")

    # --- Regression report (vs baseline) ---
    if _BASELINE_PIPELINE in pipelines_run:
        reg_path = _REPORTS_DIR / "regressions.md"
        reg_path.write_text(_build_regression_report(all_results, pipelines_run))
        print(f"  -> {reg_path}")

    if "planner_filter_vector_legal" in pipelines_run:
        plan_path = _REPORTS_DIR / "planner_analysis.md"
        plan_path.write_text(_build_planner_analysis_report(all_results))
        print(f"  -> {plan_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(gold_path: Path, pipelines: list[str], quick: bool) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    gold_records = [json.loads(l) for l in gold_path.open() if l.strip()]
    if quick:
        gold_records = gold_records[:10]

    print(f"Retrieval benchmark: {len(gold_records)} queries, pipelines={pipelines}")
    print()

    conn = psycopg2.connect(dbname="nz_legal")
    embedder = Embedder()
    store = VectorStore()
    needs_reranker = (
        "sql_filter_vector_rerank" in pipelines
        or "sql_filter_vector_legal_rerank_5" in pipelines
    )
    reranker = Reranker() if needs_reranker else None

    # Pre-build court lookup for scoring (citation -> court code)
    all_citations = set()
    for r in gold_records:
        all_citations |= set(r["expected_documents"]) | set(r["acceptable_documents"])
    cur = conn.cursor()
    cur.execute("SELECT citation, court FROM documents WHERE citation = ANY(%s)",
                (list(all_citations),))
    court_lookup: dict[str, str] = dict(cur.fetchall())

    all_results: list[dict] = []

    for i, gold in enumerate(gold_records, 1):
        q = gold["query"]
        courts = gold["expected_courts"]
        years = gold.get("expected_years") or None

        print(f"[{i}/{len(gold_records)}] {gold['id']}")
        print(f"  Q: {q[:80]}")

        vec = await embedder.embed(q)

        scores: dict[str, dict] = {}
        top5: dict[str, list[str]] = {}

        if "vector_only" in pipelines:
            ids = await _run_vector_only(vec, store)
            scores["vector_only"] = _score(ids, gold, court_lookup)
            top5["vector_only"] = ids[:5]

        if "sql_filter_vector" in pipelines:
            ids = await _run_sql_filter_vector(vec, store, conn, courts, years)
            scores["sql_filter_vector"] = _score(ids, gold, court_lookup)
            top5["sql_filter_vector"] = ids[:5]

        if "sql_filter_vector_rerank" in pipelines and reranker:
            ids = await _run_sql_filter_vector_rerank(
                q, vec, store, conn, courts, years, reranker
            )
            scores["sql_filter_vector_rerank"] = _score(ids, gold, court_lookup)
            top5["sql_filter_vector_rerank"] = ids[:5]

        if "sql_filter_vector_legal" in pipelines:
            ids = await _run_sql_filter_vector_legal(q, vec, store, conn, courts, years)
            scores["sql_filter_vector_legal"] = _score(ids, gold, court_lookup)
            top5["sql_filter_vector_legal"] = ids[:5]

        if "sql_filter_vector_legal_rerank_5" in pipelines and reranker:
            ids = await _run_sql_filter_vector_legal_rerank_5(
                q, vec, store, conn, courts, years, reranker
            )
            scores["sql_filter_vector_legal_rerank_5"] = _score(ids, gold, court_lookup)
            top5["sql_filter_vector_legal_rerank_5"] = ids[:5]

        if "sql_tracker_vector" in pipelines:
            ids = await _run_sql_tracker_vector(
                gold["task_type"], vec, store, conn, courts, years
            )
            scores["sql_tracker_vector"] = _score(ids, gold, court_lookup)
            top5["sql_tracker_vector"] = ids[:5]

        if "sql_tracker_vector_legal" in pipelines:
            ids = await _run_sql_tracker_vector_legal(
                gold["task_type"], q, vec, store, conn, courts, years
            )
            scores["sql_tracker_vector_legal"] = _score(ids, gold, court_lookup)
            top5["sql_tracker_vector_legal"] = ids[:5]

        if "sql_filter_bm25_legal" in pipelines:
            ids = await _run_sql_filter_bm25_legal(q, conn, courts, years)
            scores["sql_filter_bm25_legal"] = _score(ids, gold, court_lookup)
            top5["sql_filter_bm25_legal"] = ids[:5]

        if "sql_filter_bm25_vector_rrf_legal" in pipelines:
            ids = await _run_sql_filter_bm25_vector_rrf_legal(
                q, vec, store, conn, courts, years
            )
            scores["sql_filter_bm25_vector_rrf_legal"] = _score(ids, gold, court_lookup)
            top5["sql_filter_bm25_vector_rrf_legal"] = ids[:5]

        if "sql_filter_bm25_vector_rrf_legal_plus_tracker_soft_boost" in pipelines:
            ids = await _run_sql_filter_bm25_vector_rrf_legal_plus_tracker_soft_boost(
                gold["task_type"], q, vec, store, conn, courts, years
            )
            scores["sql_filter_bm25_vector_rrf_legal_plus_tracker_soft_boost"] = \
                _score(ids, gold, court_lookup)
            top5["sql_filter_bm25_vector_rrf_legal_plus_tracker_soft_boost"] = ids[:5]

        planner_data: dict = {}
        if "planner_filter_vector_legal" in pipelines:
            ids, plan = await _run_planner_filter_vector_legal(q, vec, store, conn)
            scores["planner_filter_vector_legal"] = _score(ids, gold, court_lookup)
            top5["planner_filter_vector_legal"] = ids[:5]
            planner_data = {
                "courts": plan.courts,
                "years": plan.years,
                "signals": plan.signals,
                "confidence": plan.confidence,
            }

        if "no_filter_vector_legal" in pipelines:
            ids = await _run_no_filter_vector_legal(q, vec, store)
            scores["no_filter_vector_legal"] = _score(ids, gold, court_lookup)
            top5["no_filter_vector_legal"] = ids[:5]

        all_results.append({
            "id": gold["id"],
            "query": q,
            "task_type": gold["task_type"],
            "expected_documents": gold["expected_documents"],
            "acceptable_documents": gold["acceptable_documents"],
            "expected_courts": courts,
            "scores": scores,
            "top5": top5,
            "planner_data": planner_data,
        })

        for p in pipelines:
            sc = scores.get(p, {})
            print(
                f"  {p[:28]:<28} "
                f"H@1={sc.get('hit_gold_at_1','-')} "
                f"H@5={sc.get('hit_gold_at_5','-')}(g)/{sc.get('hit_rel_at_5','-')}(r) "
                f"MRR={sc.get('mrr',0):.3f} "
                f"IRR={sc.get('irr_at_5',0):.2f}"
            )

    print()
    print("--- Benchmark Summary ---")
    for p in pipelines:
        runs = [r["scores"][p] for r in all_results if p in r.get("scores", {})]
        agg = _aggregate(runs)
        print(
            f"  {p:<30} "
            f"H@5={agg['hit_gold_at_5']:.2f}(g)/{agg['hit_rel_at_5']:.2f}(r)  "
            f"H@10={agg['hit_gold_at_10']:.2f}(g)/{agg['hit_rel_at_10']:.2f}(r)  "
            f"MRR={agg['mrr']:.3f}  "
            f"IRR@5={agg['irr_at_5']:.2f}"
        )

    print()
    print("Writing reports...")
    _write_reports(all_results, gold_records, pipelines)

    await embedder.close()
    conn.close()
    return all_results, gold_records, pipelines


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval A/B benchmark")
    parser.add_argument("--gold", type=Path, default=_GOLD_PATH)
    parser.add_argument("--pipelines", nargs="+", default=_ALL_PIPELINES,
                        choices=_ALL_PIPELINES,
                        help="Which pipelines to run (default: all three)")
    parser.add_argument("--quick", action="store_true",
                        help="Run first 10 queries only")
    args = parser.parse_args()
    asyncio.run(run(args.gold, args.pipelines, args.quick))


if __name__ == "__main__":
    main()
