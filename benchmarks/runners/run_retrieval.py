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

_ALL_PIPELINES = [
    "vector_only",
    "sql_filter_vector",
    "sql_filter_vector_rerank",
    "sql_filter_vector_legal",
    "sql_filter_vector_legal_rerank_5",
    "sql_tracker_vector",
    "sql_tracker_vector_legal",
]
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

        all_results.append({
            "id": gold["id"],
            "query": q,
            "task_type": gold["task_type"],
            "expected_documents": gold["expected_documents"],
            "acceptable_documents": gold["acceptable_documents"],
            "expected_courts": courts,
            "scores": scores,
            "top5": top5,
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
