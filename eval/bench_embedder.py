"""Embedder benchmark: query latency, batch throughput, Qdrant index stats, hit rate.

Metrics:
  single_query_latency   Time to embed one query (mean/min/max over N runs, ms)
  batch_throughput       Chunks/sec at batch sizes 10, 100, 500, 1000
  index_stats            Qdrant collection size and status
  hit@5 / hit@10         Fraction of questions where a relevant chunk appears
                         in the top-K results (keyword-overlap heuristic)

Run:
    python -m eval.bench_embedder
    python -m eval.bench_embedder --quick
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from rag.embedder import Embedder
from rag.retriever import VectorStore

_RESULTS_DIR = Path("eval/bench_results")

_SAMPLE_TEXT = (
    "The Employment Relations Authority held that the employer failed to follow "
    "a fair and reasonable process before dismissing the employee. The Authority "
    "found that the employer did not give the employee an adequate opportunity to "
    "respond to the concerns raised. The dismissal was therefore unjustified within "
    "the meaning of section 103A of the Employment Relations Act 2000. The Authority "
    "ordered the employer to pay compensation for lost wages and distress."
)

_STOP_WORDS = {
    "under", "must", "that", "this", "their", "which", "before", "after",
    "least", "with", "from", "have", "been", "they", "were", "also", "than",
}


def _keywords(text: str) -> set[str]:
    return {w.lower() for w in text.split() if len(w) > 4 and w.lower() not in _STOP_WORDS}


async def _bench_single(embedder: Embedder, n_runs: int) -> dict:
    query = "What is the standard for unjustified dismissal in New Zealand?"
    times = []
    for _ in range(n_runs):
        t0 = time.monotonic()
        await embedder.embed(query)
        times.append(time.monotonic() - t0)
    times = times[1:]  # drop first (model warmup)
    return {
        "n_runs": n_runs,
        "mean_ms": round(sum(times) / len(times) * 1000, 1),
        "min_ms":  round(min(times) * 1000, 1),
        "max_ms":  round(max(times) * 1000, 1),
    }


async def _bench_batch(embedder: Embedder, sizes: list[int]) -> list[dict]:
    results = []
    for n in sizes:
        texts = [f"Case {i}: {_SAMPLE_TEXT}" for i in range(n)]
        t0 = time.monotonic()
        await embedder.embed_batch(texts)
        elapsed = time.monotonic() - t0
        tps = n / elapsed
        results.append({
            "batch_size": n,
            "elapsed_s": round(elapsed, 2),
            "chunks_per_sec": round(tps, 1),
            "ms_per_chunk": round(elapsed / n * 1000, 2),
        })
        print(f"  batch={n:>5}: {elapsed:.2f}s  ({tps:.1f} chunks/s)")
    return results


async def _retrieval_quality(embedder: Embedder, store: VectorStore,
                             questions: list[dict], top_k_values: list[int]) -> dict:
    hits: dict[int, int] = {k: 0 for k in top_k_values}
    max_k = max(top_k_values)

    for item in questions:
        q = item["question"]
        gt_terms = _keywords(item.get("ground_truth", ""))
        vec = await embedder.embed(q)
        search_hits = store.search(vec, top_k=max_k)
        for k in top_k_values:
            matched = any(
                any(term in h.text.lower() for term in gt_terms)
                for h in search_hits[:k]
            )
            if matched:
                hits[k] += 1

    n = len(questions)
    return {f"hit_at_{k}": round(hits[k] / n, 3) for k in top_k_values}


async def run(questions_path: Path, quick: bool) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(questions_path) as f:
        questions = [json.loads(l) for l in f if l.strip()]
    if quick:
        questions = questions[:4]

    batch_sizes = [10, 100] if quick else [10, 100, 500, 1000]
    n_single = 5 if quick else 10

    print(f"Embedder benchmark  model=nomic-ai/nomic-embed-text-v1.5  dim={config.EMBED_DIM}")
    print(f"Questions: {len(questions)}")
    print()

    embedder = Embedder()

    print("Single-query latency...")
    single = await _bench_single(embedder, n_single)
    print(f"  mean={single['mean_ms']}ms  min={single['min_ms']}ms  max={single['max_ms']}ms")
    print()

    print("Batch throughput...")
    batches = await _bench_batch(embedder, batch_sizes)
    print()

    print("Qdrant index stats...")
    store = VectorStore()
    stats = store.collection_stats()
    print(f"  points:  {stats['points_count']:,}")
    print(f"  indexed: {stats.get('indexed_vectors_count', 'N/A')}")
    print(f"  status:  {stats['status']}")
    print()

    print(f"Retrieval quality ({len(questions)} questions, keyword-overlap heuristic)...")
    top_k_values = [5, 10]
    quality = await _retrieval_quality(embedder, store, questions, top_k_values)
    for k in top_k_values:
        print(f"  hit@{k}: {quality[f'hit_at_{k}']:.1%}")
    print()

    summary = {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "embed_dim": config.EMBED_DIM,
        "single_query_latency_ms": single,
        "batch_throughput": batches,
        "index_stats": stats,
        "retrieval_quality": quality,
        "n_questions": len(questions),
    }

    print("--- Embedder Benchmark Summary ---")
    print(f"  Single query:    mean={single['mean_ms']}ms")
    if batches:
        peak = max(batches, key=lambda b: b["chunks_per_sec"])
        print(f"  Peak throughput: {peak['chunks_per_sec']} chunks/s (batch={peak['batch_size']})")
    print(f"  Index points:    {stats['points_count']:,}")
    for k in top_k_values:
        print(f"  hit@{k}:          {quality[f'hit_at_{k}']:.1%}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _RESULTS_DIR / f"embedder_{ts}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults -> {out}")

    await embedder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedder benchmark")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--quick", action="store_true", help="Smaller runs for quick check")
    args = parser.parse_args()
    asyncio.run(run(args.questions, args.quick))


if __name__ == "__main__":
    main()
