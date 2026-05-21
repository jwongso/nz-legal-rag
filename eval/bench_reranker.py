"""Reranker benchmark: latency at N candidates, score distribution, rank improvement.

For each question, embeds the query, retrieves up to 50 candidates from Qdrant,
then measures how long the cross-encoder takes to rerank pools of N=5/10/20/50.
Also reports where the final top-1 case sat in the original vector-ranked list -
a proxy for how much value the reranker adds.

Metrics:
  latency_ms        Reranker wall time at each candidate pool size
  rank_improvement  Mean rank position of top-1 before reranking (0 = already first)
  score_dist        Cross-encoder score distribution (mean/min/max/std)

Run:
    python -m eval.bench_reranker
    python -m eval.bench_reranker --quick
"""

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from rag.embedder import Embedder
from rag.reranker import Reranker
from rag.retriever import VectorStore

_RESULTS_DIR = Path("eval/bench_results")


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "min": None, "max": None, "std": None}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "mean": round(mean, 4),
        "min":  round(min(values), 4),
        "max":  round(max(values), 4),
        "std":  round(math.sqrt(variance), 4),
        "n":    n,
    }


def _latency_sweep(reranker: Reranker, query: str,
                   pool: list, sizes: list[int], top_k: int = 5) -> list[dict]:
    results = []
    for n in sizes:
        candidates = pool[:n]
        if len(candidates) < n:
            print(f"  [skip N={n}: only {len(candidates)} candidates available]")
            continue
        t0 = time.monotonic()
        reranker.rerank(query, candidates, top_k=top_k)
        elapsed = time.monotonic() - t0
        results.append({
            "n_candidates": n,
            "top_k": top_k,
            "latency_ms": round(elapsed * 1000, 1),
        })
        print(f"  N={n:>3} -> {elapsed * 1000:.1f}ms")
    return results


async def run(questions_path: Path, quick: bool) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(questions_path) as f:
        questions = [json.loads(l) for l in f if l.strip()]
    if quick:
        questions = questions[:3]

    candidate_sizes = [5, 10, 20] if quick else [5, 10, 20, 50]
    max_candidates = max(candidate_sizes)

    print(f"Reranker benchmark  model={config.RERANKER_MODEL}")
    print(f"Questions: {len(questions)}  candidate sizes: {candidate_sizes}")
    print()

    embedder = Embedder()
    store = VectorStore()
    reranker = Reranker()

    latency_runs: list[dict] = []
    rank_improvements: list[int] = []
    all_scores: list[float] = []
    per_question: list[dict] = []

    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"[{i}/{len(questions)}] {q[:65]}")

        vec = await embedder.embed(q)
        pool = store.search(vec, top_k=max_candidates)

        if not pool:
            print("  no candidates, skipping")
            continue

        # Latency sweep on first question only (avoids loading the model repeatedly)
        if i == 1:
            print("  Latency sweep:")
            latency_runs = _latency_sweep(reranker, q, pool, candidate_sizes)

        # Score distribution and rank improvement
        pairs = [(q, h.text) for h in pool]
        raw_scores: list[float] = reranker._model.predict(pairs).tolist()
        all_scores.extend(raw_scores)

        # Best case_id by cross-encoder score
        best_idx = max(range(len(raw_scores)), key=lambda j: raw_scores[j])
        best_case_id = pool[best_idx].case_id

        # Where did it sit in the original vector-ranked list?
        original_rank = next(
            (j for j, h in enumerate(pool) if h.case_id == best_case_id), -1
        )
        rank_improvements.append(original_rank)

        t0 = time.monotonic()
        reranker.rerank(q, pool, top_k=5)
        elapsed = time.monotonic() - t0

        per_question.append({
            "question": q,
            "n_candidates": len(pool),
            "top1_was_rank": original_rank,
            "rerank_latency_ms": round(elapsed * 1000, 1),
            "score_stats": _stats(raw_scores),
        })
        print(
            f"  top-1 was at rank {original_rank} before rerank  "
            f"({elapsed * 1000:.1f}ms  "
            f"score range [{min(raw_scores):.3f}, {max(raw_scores):.3f}])"
        )

    summary = {
        "model": config.RERANKER_MODEL,
        "latency_by_n": latency_runs,
        "rank_improvement": {
            "description": "position of the best-scoring chunk in the pre-reranked list",
            "mean": round(sum(rank_improvements) / len(rank_improvements), 1)
            if rank_improvements else None,
            "values": rank_improvements,
        },
        "score_distribution": _stats(all_scores),
        "per_question": per_question,
    }

    print()
    print("--- Reranker Benchmark Summary ---")
    for r in latency_runs:
        print(f"  N={r['n_candidates']:>3} -> {r['latency_ms']}ms")
    if rank_improvements:
        print(
            f"  Mean rank improvement: {summary['rank_improvement']['mean']:.1f} positions "
            f"(higher = reranker lifted best doc further up)"
        )
    sd = summary["score_distribution"]
    if sd["mean"] is not None:
        print(
            f"  Score dist: mean={sd['mean']}  min={sd['min']}  "
            f"max={sd['max']}  std={sd['std']}"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _RESULTS_DIR / f"reranker_{ts}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults -> {out}")

    await embedder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reranker benchmark")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--quick", action="store_true", help="Fewer questions, smaller N")
    args = parser.parse_args()
    asyncio.run(run(args.questions, args.quick))


if __name__ == "__main__":
    main()
