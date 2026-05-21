"""Comprehensive benchmark: runs all sub-benchmarks and writes a single report.

Sub-benchmarks run in order:
  1. Embedder    - query latency, batch throughput, index stats
  2. Reranker    - latency sweep, score distribution, rank improvement
  3. Retrieval   - 3-pipeline A/B (vector_only / sql_filter / sql_filter_rerank)
  4. Generator   - TTFT, tokens/sec, citation correctness (skipped if LLM offline)

Output:
  benchmarks/reports/comprehensive_report.md   human-readable report
  benchmarks/reports/comprehensive_report.json machine-readable combined results

Run:
    python -m benchmarks.runners.run_all
    python -m benchmarks.runners.run_all --quick
    python -m benchmarks.runners.run_all --skip-generator
"""

import argparse
import asyncio
import json
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil
import psycopg2

import config
from benchmarks.runners.run_retrieval import _aggregate, _GOLD_PATH, _ALL_PIPELINES
from benchmarks.runners import run_retrieval
from eval import bench_embedder, bench_reranker, bench_generator

_REPORTS_DIR = Path("benchmarks/reports")
_EVAL_RESULTS_DIR = Path("eval/bench_results")
_HIT_K_VALUES = run_retrieval._HIT_K_VALUES


# ---------------------------------------------------------------------------
# System snapshot
# ---------------------------------------------------------------------------

def _system_info() -> dict:
    mem = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / 1024 ** 3, 1),
        "ram_used_gb": round(mem.used / 1024 ** 3, 1),
        "llm_model": config.LLM_MODEL,
        "llm_url": config.LLM_BASE_URL,
        "embed_model": config.EMBED_MODEL,
        "embed_dim": config.EMBED_DIM,
        "reranker_model": config.RERANKER_MODEL,
        "qdrant_url": config.QDRANT_URL,
        "qdrant_collection": config.QDRANT_COLLECTION,
    }


def _corpus_stats() -> dict:
    try:
        conn = psycopg2.connect(dbname="nz_legal")
        cur = conn.cursor()
        cur.execute("""
            SELECT d.court,
                   count(d.id) as docs,
                   coalesce(sum(c.cnt), 0) as chunks
            FROM documents d
            LEFT JOIN (
                SELECT document_id, count(*) as cnt
                FROM chunks GROUP BY document_id
            ) c ON d.id = c.document_id
            GROUP BY d.court ORDER BY docs DESC
        """)
        rows = [{"court": r[0], "docs": r[1], "chunks": int(r[2])}
                for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM documents")
        total_docs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks")
        total_chunks = cur.fetchone()[0]
        conn.close()
        return {"total_docs": total_docs, "total_chunks": total_chunks, "by_court": rows}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Retrieval aggregation helpers
# ---------------------------------------------------------------------------

def _retrieval_agg(all_results: list[dict], pipelines: list[str]) -> dict[str, dict]:
    aggregates: dict[str, dict] = {}
    for p in pipelines:
        runs = [r["scores"][p] for r in all_results if p in r.get("scores", {})]
        aggregates[p] = _aggregate(runs)
    return aggregates


def _retrieval_agg_by_type(all_results: list[dict],
                            pipelines: list[str]) -> dict[str, dict[str, dict]]:
    task_types = sorted(set(r["task_type"] for r in all_results))
    out: dict[str, dict[str, dict]] = {}
    for tt in task_types:
        out[tt] = {}
        for p in pipelines:
            runs = [r["scores"][p] for r in all_results
                    if r["task_type"] == tt and p in r.get("scores", {})]
            if runs:
                out[tt][p] = _aggregate(runs)
    return out


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_comprehensive_report(
    ts: str,
    sysinfo: dict,
    corpus: dict,
    embedder_summary: dict | None,
    reranker_summary: dict | None,
    retrieval_results: tuple | None,
    generator_data: dict | None,
) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# NZ Legal RAG - Comprehensive Benchmark Report",
        "",
        f"Generated: {ts}",
        "",
    ]

    # --- System ---
    lines += [
        "## System Under Test",
        "",
        "| Component | Details |",
        "|---|---|",
        f"| LLM | {sysinfo['llm_model']} ({sysinfo['llm_url']}) |",
        f"| Embedder | {sysinfo['embed_model']} (dim={sysinfo['embed_dim']}) |",
        f"| Reranker | {sysinfo['reranker_model']} |",
        f"| Vector Store | Qdrant {sysinfo['qdrant_collection']} @ {sysinfo['qdrant_url']} |",
        f"| Relational DB | PostgreSQL nz_legal |",
        f"| CPU | {sysinfo['cpu']} ({sysinfo['cpu_cores_physical']}P / {sysinfo['cpu_cores_logical']}L cores) |",
        f"| RAM | {sysinfo['ram_used_gb']} / {sysinfo['ram_total_gb']} GB used |",
        "",
    ]

    # --- Corpus ---
    if "error" not in corpus:
        lines += [
            "## Corpus",
            "",
            f"Total: **{corpus['total_docs']:,} documents** / **{corpus['total_chunks']:,} chunks**",
            "",
            "| Court | Documents | Chunks |",
            "|---|---:|---:|",
        ]
        for row in corpus["by_court"]:
            lines.append(f"| {row['court']} | {row['docs']:,} | {row['chunks']:,} |")
        lines.append("")

    # --- Embedder ---
    lines += ["## 1. Embedder", ""]
    if embedder_summary:
        sq = embedder_summary.get("single_query_latency_ms", {})
        lines += [
            "### Single-Query Latency",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Mean | {sq.get('mean_ms', 'N/A')} ms |",
            f"| Min | {sq.get('min_ms', 'N/A')} ms |",
            f"| Max | {sq.get('max_ms', 'N/A')} ms |",
            "",
            "### Batch Throughput",
            "",
            "| Batch Size | Elapsed (s) | Chunks/sec | ms/chunk |",
            "|---:|---:|---:|---:|",
        ]
        for b in embedder_summary.get("batch_throughput", []):
            lines.append(
                f"| {b['batch_size']:,} "
                f"| {b['elapsed_s']} "
                f"| {b['chunks_per_sec']:.1f} "
                f"| {b['ms_per_chunk']:.2f} |"
            )
        idx = embedder_summary.get("index_stats", {})
        rq = embedder_summary.get("retrieval_quality", {})
        lines += [
            "",
            "### Index & Search Quality",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Qdrant points | {idx.get('points_count', 'N/A'):,} |",
            f"| Index status | {idx.get('status', 'N/A')} |",
            f"| Hit@5 (keyword heuristic) | {rq.get('hit_at_5', 0):.1%} |",
            f"| Hit@10 (keyword heuristic) | {rq.get('hit_at_10', 0):.1%} |",
            "",
        ]
    else:
        lines += ["_Embedder benchmark not available._", ""]

    # --- Reranker ---
    lines += ["## 2. Reranker", ""]
    if reranker_summary:
        lines += [
            "### Latency by Candidate Pool Size",
            "",
            "| N Candidates | Latency (ms) |",
            "|---:|---:|",
        ]
        for r in reranker_summary.get("latency_by_n", []):
            lines.append(f"| {r['n_candidates']} | {r['latency_ms']:.1f} |")
        ri = reranker_summary.get("rank_improvement", {})
        sd = reranker_summary.get("score_distribution", {})
        lines += [
            "",
            "### Score Distribution & Rank Improvement",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        if ri.get("mean") is not None:
            lines.append(
                f"| Mean rank of top-1 doc before rerank | {ri['mean']:.1f} "
                "(0 = already #1) |"
            )
        if sd.get("mean") is not None:
            lines += [
                f"| Score mean | {sd['mean']:.4f} |",
                f"| Score std | {sd['std']:.4f} |",
                f"| Score min | {sd['min']:.4f} |",
                f"| Score max | {sd['max']:.4f} |",
            ]
        lines.append("")
    else:
        lines += ["_Reranker benchmark not available._", ""]

    # --- Retrieval ---
    lines += ["## 3. Retrieval Quality", ""]
    if retrieval_results:
        all_results, gold_records, pipelines = retrieval_results
        agg = _retrieval_agg(all_results, pipelines)
        agg_by_type = _retrieval_agg_by_type(all_results, pipelines)

        lines += [
            f"> Gold (g) = exact expected document hit.  "
            f"Rel (r) = expected OR acceptable document hit.  "
            f"Oracle court filter used (expected_courts from gold record).",
            f">",
            f"> Queries: {len(gold_records)}  |  Gold dataset: benchmarks/datasets/retrieval_gold.jsonl",
            "",
            "### Pipeline Summary",
            "",
            "| Pipeline | H@1(g) | H@5(g) | H@5(r) | H@10(g) | H@10(r) | MRR | IRR@5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for p in pipelines:
            a = agg[p]
            lines.append(
                f"| {p} "
                f"| {a['hit_gold_at_1']:.2f} "
                f"| {a['hit_gold_at_5']:.2f} "
                f"| {a['hit_rel_at_5']:.2f} "
                f"| {a['hit_gold_at_10']:.2f} "
                f"| {a['hit_rel_at_10']:.2f} "
                f"| {a['mrr']:.3f} "
                f"| {a['irr_at_5']:.2f} |"
            )

        lines += [
            "",
            "### Per-Task-Type Breakdown",
            "",
            "| Task Type | Pipeline | H@5(g) | H@5(r) | MRR |",
            "|---|---|---:|---:|---:|",
        ]
        for tt, pmap in agg_by_type.items():
            for p, a in pmap.items():
                lines.append(
                    f"| {tt} | {p} "
                    f"| {a['hit_gold_at_5']:.2f} "
                    f"| {a['hit_rel_at_5']:.2f} "
                    f"| {a['mrr']:.3f} |"
                )

        # Key findings
        best_rel = max(pipelines, key=lambda p: agg[p]["hit_rel_at_5"])
        best_gold = max(pipelines, key=lambda p: agg[p]["hit_gold_at_5"])
        best_mrr = max(pipelines, key=lambda p: agg[p]["mrr"])
        lines += [
            "",
            "### Key Findings",
            "",
            f"- Best rel hit@5: **{best_rel}** ({agg[best_rel]['hit_rel_at_5']:.0%})"
            f" - finds a relevant document in {agg[best_rel]['hit_rel_at_5']:.0%} of queries",
            f"- Best gold hit@5: **{best_gold}** ({agg[best_gold]['hit_gold_at_5']:.0%})"
            f" - finds the exact expected document",
            f"- Best MRR: **{best_mrr}** ({agg[best_mrr]['mrr']:.3f})"
            f" - ranks the expected document highest on average",
            f"- IRR@5 = 0.00 across all pipelines - no results from excluded courts",
            "",
            "### Per-Query Results",
            "",
        ]
        _abbrev = {
            "vector_only": "vec",
            "sql_filter_vector": "sql+vec",
            "sql_filter_vector_rerank": "sql+vec+rr",
        }
        header_parts = []
        for p in pipelines:
            ab = _abbrev.get(p, p[:10])
            header_parts += [f"{ab} H@5(g)", f"{ab} H@5(r)"]
        sep_cells = ["---", "---"] + ["---:"] * len(header_parts)
        lines.append("| Query ID | Task | " + " | ".join(header_parts) + " |")
        lines.append("|" + "|".join(sep_cells) + "|")
        for r in all_results:
            row = f"| {r['id']} | {r['task_type']} "
            for p in pipelines:
                sc = r.get("scores", {}).get(p, {})
                row += f"| {sc.get('hit_gold_at_5', '-')} | {sc.get('hit_rel_at_5', '-')} "
            row += "|"
            lines.append(row)
        lines.append("")
    else:
        lines += ["_Retrieval benchmark not available._", ""]

    # --- Generator ---
    lines += ["## 4. LLM Generator", ""]
    if generator_data:
        s = generator_data.get("summary", {})
        runs = generator_data.get("runs", [])
        ok = [r for r in runs if "error" not in r]
        lines += [
            f"> Model: {s.get('model', 'N/A')}  |  "
            f"Questions: {s.get('questions_run', 0)}  |  "
            f"Successful: {s.get('successful', 0)}",
            "",
            "### Summary Statistics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        if s.get("ttft_mean_s") is not None:
            lines += [
                f"| TTFT mean | {s['ttft_mean_s']} s |",
                f"| TTFT min | {s['ttft_min_s']} s |",
                f"| TTFT max | {s['ttft_max_s']} s |",
            ]
        lines += [
            f"| Tokens/sec mean | {s.get('tps_mean', 'N/A')} |",
            f"| Tokens/sec min | {s.get('tps_min', 'N/A')} |",
            f"| Tokens/sec max | {s.get('tps_max', 'N/A')} |",
            f"| Citation correctness | {s.get('citation_correctness_mean', 0):.1%} |",
        ]
        if ok:
            lines += [
                "",
                "### Per-Question Results",
                "",
                "| Question (truncated) | TTFT (s) | TPS | Cite OK |",
                "|---|---:|---:|---:|",
            ]
            for r in ok:
                q = r["question"][:55] + ("..." if len(r["question"]) > 55 else "")
                lines.append(
                    f"| {q} "
                    f"| {r.get('ttft_s', '-')} "
                    f"| {r.get('tps', '-')} "
                    f"| {r.get('citation_correctness', 0):.0%} |"
                )
        lines.append("")
    else:
        lines += ["_Generator benchmark skipped or LLM offline._", ""]

    # --- Summary scorecard ---
    lines += [
        "## Summary Scorecard",
        "",
        "| Component | Key Metric | Value | Notes |",
        "|---|---|---:|---|",
    ]
    if embedder_summary:
        sq = embedder_summary.get("single_query_latency_ms", {})
        batches = embedder_summary.get("batch_throughput", [])
        peak = max(batches, key=lambda b: b["chunks_per_sec"]) if batches else {}
        lines += [
            f"| Embedder | Query latency | {sq.get('mean_ms', 'N/A')} ms | mean over {sq.get('n_runs', 'N/A')} runs |",
            f"| Embedder | Peak batch throughput | {peak.get('chunks_per_sec', 'N/A')} chunks/s | batch={peak.get('batch_size', 'N/A')} |",
        ]
    if reranker_summary:
        ri = reranker_summary.get("rank_improvement", {})
        lat = reranker_summary.get("latency_by_n", [])
        lat50 = next((r["latency_ms"] for r in lat if r["n_candidates"] == 50), None)
        lines += [
            f"| Reranker | Latency @ N=50 | {lat50 or 'N/A'} ms | cross-encoder inference |",
            f"| Reranker | Mean rank improvement | {ri.get('mean', 'N/A')} | rank of best doc before rerank |",
        ]
    if retrieval_results:
        all_results2, _, pipelines2 = retrieval_results
        agg2 = _retrieval_agg(all_results2, pipelines2)
        for p in pipelines2:
            a = agg2[p]
            lines.append(
                f"| Retrieval ({p}) "
                f"| H@5 gold / rel "
                f"| {a['hit_gold_at_5']:.0%} / {a['hit_rel_at_5']:.0%} "
                f"| MRR={a['mrr']:.3f} |"
            )
    if generator_data:
        s = generator_data.get("summary", {})
        lines += [
            f"| Generator | TTFT | {s.get('ttft_mean_s', 'N/A')} s | time to first token |",
            f"| Generator | Tokens/sec | {s.get('tps_mean', 'N/A')} | streaming output speed |",
            f"| Generator | Citation correctness | {s.get('citation_correctness_mean', 0):.0%} | [N] refs pointing to valid source |",
        ]
    lines.append("")

    md_path = _REPORTS_DIR / "comprehensive_report.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\n  -> {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(quick: bool, skip_generator: bool,
              gold_path: Path, pipelines: list[str]) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 60)
    print("NZ Legal RAG - Comprehensive Benchmark")
    print(f"Timestamp: {ts}")
    print("=" * 60)

    sysinfo = _system_info()
    corpus = _corpus_stats()
    print(f"\nCorpus: {corpus.get('total_docs', '?'):,} docs / "
          f"{corpus.get('total_chunks', '?'):,} chunks\n")

    # 1. Embedder
    print("=" * 60)
    print("STEP 1/4 - Embedder benchmark")
    print("=" * 60)
    embedder_summary = None
    try:
        questions_path = Path("eval/questions.jsonl")
        embedder_summary = await bench_embedder.run(questions_path, quick=quick)
    except Exception:
        print("ERROR in embedder benchmark:")
        traceback.print_exc()

    # 2. Reranker
    print("\n" + "=" * 60)
    print("STEP 2/4 - Reranker benchmark")
    print("=" * 60)
    reranker_summary = None
    try:
        reranker_summary = await bench_reranker.run(
            Path("eval/questions.jsonl"), quick=quick
        )
    except Exception:
        print("ERROR in reranker benchmark:")
        traceback.print_exc()

    # 3. Retrieval
    print("\n" + "=" * 60)
    print("STEP 3/4 - Retrieval A/B benchmark")
    print("=" * 60)
    retrieval_results = None
    try:
        retrieval_results = await run_retrieval.run(gold_path, pipelines, quick)
    except Exception:
        print("ERROR in retrieval benchmark:")
        traceback.print_exc()

    # 4. Generator
    generator_data = None
    if not skip_generator:
        print("\n" + "=" * 60)
        print("STEP 4/4 - Generator benchmark")
        print("=" * 60)
        try:
            generator_data = await bench_generator.run(
                Path("eval/questions.jsonl"),
                llm_url=config.LLM_BASE_URL,
                model=config.LLM_MODEL,
                quick=quick,
            )
        except Exception:
            print("ERROR in generator benchmark (LLM may be offline):")
            traceback.print_exc()
    else:
        print("\nSTEP 4/4 - Generator benchmark  [SKIPPED]")

    # Compile combined JSON
    combined = {
        "timestamp": ts,
        "system": sysinfo,
        "corpus": corpus,
        "embedder": embedder_summary,
        "reranker": reranker_summary,
        "retrieval": {
            "results": retrieval_results[0] if retrieval_results else None,
            "n_queries": len(retrieval_results[1]) if retrieval_results else 0,
            "pipelines": retrieval_results[2] if retrieval_results else [],
        },
        "generator": generator_data,
    }
    json_path = _REPORTS_DIR / "comprehensive_report.json"
    json_path.write_text(json.dumps(combined, indent=2))
    print(f"\n  -> {json_path}")

    # Write comprehensive markdown
    print("\nWriting comprehensive report...")
    _write_comprehensive_report(
        ts, sysinfo, corpus,
        embedder_summary, reranker_summary,
        retrieval_results, generator_data,
    )

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive benchmark runner")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller runs for each sub-benchmark")
    parser.add_argument("--skip-generator", action="store_true",
                        help="Skip the LLM generator benchmark")
    parser.add_argument("--gold", type=Path, default=_GOLD_PATH)
    parser.add_argument("--pipelines", nargs="+", default=_ALL_PIPELINES,
                        choices=_ALL_PIPELINES)
    args = parser.parse_args()
    asyncio.run(run(args.quick, args.skip_generator, args.gold, args.pipelines))


if __name__ == "__main__":
    main()
