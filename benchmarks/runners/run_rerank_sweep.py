"""Reranker candidate-size sweep: quality and latency at N=5/10/20/50.

Answers: "How many candidates should we pass to the reranker?"

Pipelines:
  sql_filter_vector              Baseline - no reranking
  sql_filter_vector_rerank_5     Cross-encoder over top 5 chunks
  sql_filter_vector_rerank_10    Cross-encoder over top 10 chunks
  sql_filter_vector_rerank_20    Cross-encoder over top 20 chunks
  sql_filter_vector_rerank_50    Cross-encoder over top 50 chunks

Pool: always _POOL_FETCH_K chunks from Qdrant; top N of those go to
the cross-encoder. Only the reranker call is timed (embed + Qdrant
time is shared across all pipelines and excluded).

Metrics per pipeline:
  Hit@5(g)    1 if any expected_document in top-5 unique docs
  Hit@5(r)    1 if any expected OR acceptable document in top-5
  MRR         1 / rank of first expected_document
  p50 lat     median reranker latency across queries (ms)
  p95 lat     95th-pct reranker latency (ms)
  regressions queries where this pipeline lost rel@5 vs. baseline

Run:
    python -m benchmarks.runners.run_rerank_sweep
    python -m benchmarks.runners.run_rerank_sweep --quick
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

import config
from rag.embedder import Embedder
from rag.reranker import Reranker
from rag.retriever import VectorStore
from benchmarks.runners.run_retrieval import (
    _get_point_ids_for_courts,
    _score,
    _dedup,
    _aggregate,
    _GOLD_PATH,
    _HIT_K_VALUES,
)

_REPORTS_DIR = Path("benchmarks/reports")
_POOL_FETCH_K = 100   # chunks fetched from Qdrant (shared pool for all N)
_RERANK_SIZES = [5, 10, 20, 50]
_BASELINE = "sql_filter_vector"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(int(len(s) * pct / 100), len(s) - 1))
    return round(s[k], 1)


def _mean_f(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _rerank_name(n: int) -> str:
    return f"sql_filter_vector_rerank_{n}"


def _dedup_reranked(hits) -> list[str]:
    """Dedup reranked hits preserving the reranker's rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h.case_id not in seen:
            seen.add(h.case_id)
            out.append(h.case_id)
    return out


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_reports(
    all_results: list[dict],
    gold_records: list[dict],
    pipelines: list[str],
    latencies: dict[str, list[float]],
) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # --- JSON ---
    json_path = _REPORTS_DIR / "rerank_sweep.json"
    json_path.write_text(json.dumps({
        "timestamp": ts,
        "pipelines": pipelines,
        "n_queries": len(gold_records),
        "pool_fetch_k": _POOL_FETCH_K,
        "results": all_results,
        "latencies_ms": latencies,
    }, indent=2))
    print(f"  -> {json_path}")

    # --- Markdown ---
    lines = [
        "# Reranker Candidate-Size Sweep",
        "",
        f"Generated: {ts}  |  Queries: {len(gold_records)}  |  "
        f"Pool: {_POOL_FETCH_K} chunks fetched, top N to cross-encoder",
        "",
        "> Gold (g) = hit on expected_documents. "
        "Rel (r) = hit on expected OR acceptable. "
        "Latency = reranker inference only (embed+Qdrant excluded).",
        "",
        "## Summary",
        "",
        "| Pipeline | H@5(g) | H@5(r) | MRR | p50 lat (ms) | p95 lat (ms) | Regressions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    baseline_rel = {r["id"]: r["scores"][_BASELINE].get("hit_rel_at_5", 0)
                    for r in all_results if _BASELINE in r["scores"]}
    baseline_gold = {r["id"]: r["scores"][_BASELINE].get("hit_gold_at_5", 0)
                     for r in all_results if _BASELINE in r["scores"]}

    pipeline_regressions: dict[str, list[str]] = {p: [] for p in pipelines}
    pipeline_improvements: dict[str, list[str]] = {p: [] for p in pipelines}

    for p in pipelines:
        runs = [r["scores"][p] for r in all_results if p in r.get("scores", {})]
        agg = _aggregate(runs)
        lat = latencies.get(p, [])
        p50 = _percentile(lat, 50) if lat else "-"
        p95 = _percentile(lat, 95) if lat else "-"

        reg_count = 0
        for r in all_results:
            qid = r["id"]
            sc = r["scores"].get(p, {})
            rel_now = sc.get("hit_rel_at_5", 0)
            rel_base = baseline_rel.get(qid, 0)
            gold_base = baseline_gold.get(qid, 0)
            if rel_now < rel_base:
                pipeline_regressions[p].append(qid)
                reg_count += 1
            elif rel_now > rel_base or (sc.get("hit_gold_at_5", 0) > gold_base):
                pipeline_improvements[p].append(qid)

        lat_str = f"{p50} / {p95}" if lat else "-"
        lines.append(
            f"| {p} "
            f"| {agg['hit_gold_at_5']:.2f} "
            f"| {agg['hit_rel_at_5']:.2f} "
            f"| {agg['mrr']:.3f} "
            f"| {p50} "
            f"| {p95} "
            f"| {reg_count} |"
        )

    # Per-task-type
    lines += [
        "",
        "## Per-Task-Type",
        "",
        "| Task | Pipeline | H@5(g) | H@5(r) | MRR |",
        "|---|---|---:|---:|---:|",
    ]
    task_types = sorted(set(r["task_type"] for r in all_results))
    for tt in task_types:
        for p in pipelines:
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

    # Latency detail
    lines += [
        "",
        "## Reranker Latency Detail",
        "",
        "| Pipeline | Mean (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in pipelines:
        lat = latencies.get(p, [])
        if not lat:
            continue
        lines.append(
            f"| {p} "
            f"| {_mean_f(lat):.1f} "
            f"| {_percentile(lat, 50)} "
            f"| {_percentile(lat, 95)} "
            f"| {min(lat):.1f} "
            f"| {max(lat):.1f} |"
        )

    # Regression analysis
    lines += [
        "",
        "## Regression Analysis",
        "",
        "A regression is a query where hit_rel@5 dropped vs. sql_filter_vector baseline.",
        "",
    ]
    for p in pipelines:
        if p == _BASELINE:
            continue
        regs = pipeline_regressions[p]
        imps = pipeline_improvements[p]
        lines += [
            f"### {p}",
            "",
            f"Regressions: {len(regs)}  |  Improvements: {len(imps)}",
            "",
        ]
        if regs:
            lines.append("**Regressions (reranker hurt rel@5):**")
            lines.append("")
            for qid in regs:
                r = next((x for x in all_results if x["id"] == qid), None)
                if not r:
                    continue
                base_sc = r["scores"].get(_BASELINE, {})
                p_sc = r["scores"].get(p, {})
                top5_base = r.get("top5", {}).get(_BASELINE, [])
                top5_p = r.get("top5", {}).get(p, [])
                expected = r.get("expected_documents", [])
                acceptable = r.get("acceptable_documents", [])
                lines += [
                    f"**{qid}** ({r['task_type']})",
                    f"- Baseline: H@5(g)={base_sc.get('hit_gold_at_5',0)} "
                    f"H@5(r)={base_sc.get('hit_rel_at_5',0)} "
                    f"MRR={base_sc.get('mrr',0):.3f}",
                    f"- {p}: H@5(g)={p_sc.get('hit_gold_at_5',0)} "
                    f"H@5(r)={p_sc.get('hit_rel_at_5',0)} "
                    f"MRR={p_sc.get('mrr',0):.3f}",
                ]
                if top5_base:
                    base_labeled = [
                        f"{c} [GOLD]" if c in expected else
                        f"{c} [ok]" if c in acceptable else c
                        for c in top5_base
                    ]
                    lines.append(f"- Baseline top-5: {', '.join(base_labeled)}")
                if top5_p:
                    p_labeled = [
                        f"{c} [GOLD]" if c in expected else
                        f"{c} [ok]" if c in acceptable else c
                        for c in top5_p
                    ]
                    lines.append(f"- {p} top-5: {', '.join(p_labeled)}")
                lines.append("")
        if imps:
            lines.append("**Improvements (reranker helped):**")
            lines.append("")
            for qid in imps:
                r = next((x for x in all_results if x["id"] == qid), None)
                if not r:
                    continue
                base_sc = r["scores"].get(_BASELINE, {})
                p_sc = r["scores"].get(p, {})
                lines.append(
                    f"- **{qid}**: baseline H@5(r)={base_sc.get('hit_rel_at_5',0)} "
                    f"-> {p} H@5(r)={p_sc.get('hit_rel_at_5',0)}  "
                    f"MRR {base_sc.get('mrr',0):.3f} -> {p_sc.get('mrr',0):.3f}"
                )
            lines.append("")

    # Per-query grid
    lines += [
        "## Per-Query Results",
        "",
    ]
    abbrev = {_BASELINE: "base"}
    for n in _RERANK_SIZES:
        abbrev[_rerank_name(n)] = f"rr{n}"
    h_parts = [f"{abbrev.get(p, p)} H@5(g)" for p in pipelines]
    h_parts += [f"{abbrev.get(p, p)} H@5(r)" for p in pipelines]
    sep = ["---", "---"] + ["---:"] * len(pipelines) * 2
    lines.append("| Query ID | Task | " + " | ".join(h_parts) + " |")
    lines.append("|" + "|".join(sep) + "|")
    for r in all_results:
        gold_cells = " ".join(
            f"| {r['scores'].get(p, {}).get('hit_gold_at_5', '-')} "
            for p in pipelines
        )
        rel_cells = " ".join(
            f"| {r['scores'].get(p, {}).get('hit_rel_at_5', '-')} "
            for p in pipelines
        )
        lines.append(f"| {r['id']} | {r['task_type']} {gold_cells}{rel_cells}|")

    md_path = _REPORTS_DIR / "rerank_sweep.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"  -> {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(gold_path: Path, quick: bool) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    gold_records = [json.loads(l) for l in gold_path.open() if l.strip()]
    if quick:
        gold_records = gold_records[:10]

    pipelines = [_BASELINE] + [_rerank_name(n) for n in _RERANK_SIZES]
    print(f"Reranker sweep: {len(gold_records)} queries")
    print(f"Pool: {_POOL_FETCH_K} chunks  |  N tested: {_RERANK_SIZES}")
    print()

    conn = psycopg2.connect(dbname="nz_legal")
    embedder = Embedder()
    store = VectorStore()
    reranker = Reranker()

    # Pre-build court lookup for scoring
    all_citations: set[str] = set()
    for r in gold_records:
        all_citations |= set(r["expected_documents"]) | set(r["acceptable_documents"])
    cur = conn.cursor()
    cur.execute("SELECT citation, court FROM documents WHERE citation = ANY(%s)",
                (list(all_citations),))
    court_lookup: dict[str, str] = dict(cur.fetchall())

    all_results: list[dict] = []
    latencies: dict[str, list[float]] = {p: [] for p in pipelines if p != _BASELINE}

    for i, gold in enumerate(gold_records, 1):
        q = gold["query"]
        courts = gold["expected_courts"]
        years = gold.get("expected_years") or None

        print(f"[{i}/{len(gold_records)}] {gold['id']}")

        vec = await embedder.embed(q)
        point_ids = _get_point_ids_for_courts(conn, courts, years)

        # Shared pool fetch
        if point_ids:
            pool = store.search_within(vec, point_ids, top_k=_POOL_FETCH_K)
        else:
            pool = store.search(vec, top_k=_POOL_FETCH_K)

        scores: dict[str, dict] = {}
        top5: dict[str, list[str]] = {}

        # Baseline: sql_filter_vector (dedup pool, no reranking)
        base_ids = _dedup(pool)
        scores[_BASELINE] = _score(base_ids, gold, court_lookup)
        top5[_BASELINE] = base_ids[:5]

        # Reranker sweep
        for n in _RERANK_SIZES:
            pname = _rerank_name(n)
            candidates = pool[:n]
            if not candidates:
                scores[pname] = _score([], gold, court_lookup)
                top5[pname] = []
                continue
            t0 = time.monotonic()
            reranked = reranker.rerank(q, candidates, top_k=min(n, 20))
            elapsed_ms = (time.monotonic() - t0) * 1000
            latencies[pname].append(round(elapsed_ms, 1))
            ids = _dedup_reranked(reranked)
            scores[pname] = _score(ids, gold, court_lookup)
            top5[pname] = ids[:5]

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

        # Print row
        base_sc = scores[_BASELINE]
        print(
            f"  baseline  H@5(g)={base_sc['hit_gold_at_5']} "
            f"H@5(r)={base_sc['hit_rel_at_5']} "
            f"MRR={base_sc['mrr']:.3f}"
        )
        for n in _RERANK_SIZES:
            pname = _rerank_name(n)
            sc = scores[pname]
            lat_ms = latencies[pname][-1] if latencies[pname] else 0
            print(
                f"  rerank_{n:<3} H@5(g)={sc['hit_gold_at_5']} "
                f"H@5(r)={sc['hit_rel_at_5']} "
                f"MRR={sc['mrr']:.3f}  "
                f"lat={lat_ms:.0f}ms"
            )

    # Summary
    print()
    print("--- Reranker Sweep Summary ---")
    print(f"  {'Pipeline':<35} {'H@5(g)':>8} {'H@5(r)':>8} {'MRR':>8} "
          f"{'p50(ms)':>10} {'p95(ms)':>10}")
    for p in pipelines:
        runs = [r["scores"][p] for r in all_results if p in r["scores"]]
        agg = _aggregate(runs)
        lat = latencies.get(p, [])
        p50 = f"{_percentile(lat, 50):.0f}" if lat else "-"
        p95 = f"{_percentile(lat, 95):.0f}" if lat else "-"
        print(
            f"  {p:<35} {agg['hit_gold_at_5']:>8.2f} {agg['hit_rel_at_5']:>8.2f} "
            f"{agg['mrr']:>8.3f} {p50:>10} {p95:>10}"
        )

    print()
    print("Writing reports...")
    _write_reports(all_results, gold_records, pipelines, latencies)

    await embedder.close()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reranker candidate-size sweep")
    parser.add_argument("--gold", type=Path, default=_GOLD_PATH)
    parser.add_argument("--quick", action="store_true",
                        help="Run first 10 queries only")
    args = parser.parse_args()
    asyncio.run(run(args.gold, args.quick))


if __name__ == "__main__":
    main()
