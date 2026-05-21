"""Retrieval A/B benchmark: compares three retrieval pipelines on the gold dataset.

Pipelines compared:
  vector_only              Pure Qdrant semantic search, no metadata filter
  sql_filter_vector        PostgreSQL court+year filter (oracle) -> Qdrant within filtered set
  sql_filter_vector_rerank sql_filter_vector + cross-encoder reranker

"Oracle" filter means the expected_courts from the gold record are used as the
filter, not extracted from query text. This isolates retrieval stack quality from
planner quality. A separate planner benchmark will test filter extraction.

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
    python -m benchmarks.runners.run_retrieval --pipelines vector_only sql_filter_vector
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
_TOP_K = 10
_RERANK_TOP_K = 5

_ALL_PIPELINES = ["vector_only", "sql_filter_vector", "sql_filter_vector_rerank"]
_HIT_K_VALUES = [1, 3, 5, 10]


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

async def _run_vector_only(query_vec: list[float], store: VectorStore) -> list[str]:
    hits = store.search(query_vec, top_k=_TOP_K)
    seen: dict[str, float] = {}
    for h in hits:
        if h.case_id not in seen or h.score > seen[h.case_id]:
            seen[h.case_id] = h.score
    return sorted(seen, key=seen.__getitem__, reverse=True)


async def _run_sql_filter_vector(query_vec: list[float], store: VectorStore,
                                  conn, courts: list[str],
                                  years: list[int] | None) -> list[str]:
    point_ids = _get_point_ids_for_courts(conn, courts, years)
    if not point_ids:
        return await _run_vector_only(query_vec, store)
    hits = store.search_within(query_vec, point_ids, top_k=_TOP_K)
    seen: dict[str, float] = {}
    for h in hits:
        if h.case_id not in seen or h.score > seen[h.case_id]:
            seen[h.case_id] = h.score
    return sorted(seen, key=seen.__getitem__, reverse=True)


async def _run_sql_filter_vector_rerank(query: str, query_vec: list[float],
                                         store: VectorStore, conn,
                                         courts: list[str],
                                         years: list[int] | None,
                                         reranker: Reranker) -> list[str]:
    point_ids = _get_point_ids_for_courts(conn, courts, years)
    if not point_ids:
        hits = store.search(query_vec, top_k=_TOP_K * 2)
    else:
        hits = store.search_within(query_vec, point_ids, top_k=_TOP_K * 2)

    reranked = reranker.rerank(query, hits, top_k=_RERANK_TOP_K)
    seen: dict[str, float] = {}
    for i, h in enumerate(reranked):
        if h.case_id not in seen:
            seen[h.case_id] = len(reranked) - i
    return sorted(seen, key=seen.__getitem__, reverse=True)


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
        f"## Pipeline Summary",
        f"",
        "| Pipeline | Hit@1 | Hit@5 | Hit@10 | MRR | IRR@5 |",
        "|---|---:|---:|---:|---:|---:|",
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
            f"| {agg['hit_gold_at_10']:.2f} "
            f"| {agg['mrr']:.3f} "
            f"| {agg['irr_at_5']:.2f} |"
        )

    lines += [
        "",
        "## Per-Task-Type Breakdown",
        "",
        "| Task Type | Pipeline | Hit@5 | MRR |",
        "|---|---|---:|---:|",
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
                    f"| {agg['mrr']:.3f} |"
                )

    lines += ["", "## Per-Query Results", ""]
    lines.append("| Query ID | Task | " +
                 " | ".join(f"{p[:20]} H@5" for p in pipelines_run) + " |")
    lines.append("|---|---|" + "|---:" * len(pipelines_run) + "|")
    for r in all_results:
        row = f"| {r['id']} | {r['task_type']} "
        for p in pipelines_run:
            sc = r.get("scores", {}).get(p, {})
            row += f"| {sc.get('hit_gold_at_5', '-')} "
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
    reranker = Reranker() if "sql_filter_vector_rerank" in pipelines else None

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
                f"H@5={sc.get('hit_gold_at_5','-')} "
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
            f"H@1={agg['hit_gold_at_1']:.2f}  "
            f"H@5={agg['hit_gold_at_5']:.2f}  "
            f"H@10={agg['hit_gold_at_10']:.2f}  "
            f"MRR={agg['mrr']:.3f}  "
            f"IRR@5={agg['irr_at_5']:.2f}"
        )

    print()
    print("Writing reports...")
    _write_reports(all_results, gold_records, pipelines)

    await embedder.close()
    conn.close()


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
