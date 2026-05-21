# NZ Legal RAG - Benchmark Results

Benchmarks run on 2026-05-21. Source data: `benchmarks/reports/comprehensive_report.md`
and `benchmarks/reports/rerank_sweep.md`.

---

## Setup

### Hardware

| Component | Value |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 (12P / 24L cores) |
| RAM | 17.3 / 30.5 GB used |
| GPU | Radeon 890M (integrated) |

### Software Stack

| Component | Model / Version |
|---|---|
| LLM (generator) | qwen3 via llama-server (http://localhost:8080/v1) |
| Embedder | nomic-embed-text (dim=768, via Ollama) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder, CPU) |
| Vector store | Qdrant collection `nz_legal` (http://localhost:6333) |
| Relational DB | PostgreSQL database `nz_legal` |

### Corpus

| Metric | Value |
|---|---|
| Total documents | 23,561 |
| Total chunks | 982,361 |
| Qdrant index status | green |

| Court | Documents | Chunks |
|---|---:|---:|
| NZCA | 17,074 | 813,668 |
| NZLEG | 2,333 | 4,708 |
| NZEnvC | 882 | 17,877 |
| NZACC | 663 | 41,305 |
| NZCorC | 586 | 6,129 |
| NZEmpC | 300 | 11,691 |
| NZTT | 300 | 5,920 |
| NZHC | 299 | 18,081 |
| NZERA | 298 | 16,704 |
| NZSC | 294 | 8,759 |
| NZFC | 231 | 19,408 |
| NZLCDT | 107 | 5,406 |
| NZHRRT | 105 | 5,989 |
| NZREADT | 89 | 6,716 |

---

## 1. Embedder

### Single-query latency (10 runs)

| Metric | Value |
|---|---:|
| Mean | 35.7 ms |
| Min | 30.7 ms |
| Max | 45.4 ms |

### Batch throughput

| Batch size | Elapsed (s) | Chunks/sec | ms/chunk |
|---:|---:|---:|---:|
| 10 | 0.74 | 13.5 | 74.3 |
| 100 | 7.09 | 14.1 | 70.9 |
| 500 | 29.61 | 16.9 | 59.2 |
| 1,000 | 51.11 | 19.6 | 51.1 |

Embedding a single query adds ~36 ms to every request. Batch throughput peaks at
19.6 chunks/sec (batch=1000), which determines ingestion speed for new corpus additions.

---

## 2. Reranker - Candidate Size Sweep

Cross-encoder (BAAI/bge-reranker-v2-m3) inference time by candidate pool size.

### Latency

| N candidates | Latency (ms) |
|---:|---:|
| 5 | 929.6 |
| 10 | 1,124.1 |
| 20 | 3,908.0 |
| 50 | 7,209.5 |

Note: N=20 is 4x slower than N=5, and N=50 is 8x slower. The jump from N=10 to N=20
is non-linear due to cross-encoder batch overhead on CPU.

### Quality vs candidate count

Baseline: `sql_filter_vector` (no reranker). Queries: 30. Oracle court filter used.

| Pipeline | H@5(g) | H@5(r) | MRR | p50 latency | Regressions |
|---|---:|---:|---:|---:|---:|
| sql_filter_vector (baseline) | 0.23 | 1.00 | 0.115 | - | - |
| sql_filter_vector_rerank_5 | 0.23 | 1.00 | 0.113 | 1,321 ms | 0 |
| sql_filter_vector_rerank_10 | 0.17 | 0.93 | 0.127 | 2,059 ms | 2 |
| sql_filter_vector_rerank_20 | 0.17 | 0.80 | 0.133 | 4,660 ms | 6 |
| sql_filter_vector_rerank_50 | 0.13 | 0.77 | 0.132 | 9,546 ms | 7 |

Regression = query where H@5(rel) dropped vs. the no-reranker baseline.

Key finding: more candidates fed to the cross-encoder causes more regressions and
slower inference, with no quality gain above N=5. The BGE cross-encoder rewards
semantically dense explanatory chunks over short citation-dense authority chunks.
Capping at N=5 preserves 100% relevant hit rate at 929 ms vs. 7,210 ms for N=50.

---

## 3. Retrieval Pipeline Comparison

Gold = exact expected document hit. Rel = expected OR acceptable document hit.
Oracle court filter (expected_courts from gold record) used for SQL-filtered pipelines.
Queries: 30. Gold dataset: `benchmarks/datasets/retrieval_gold.jsonl`.

### Summary

| Pipeline | H@1(g) | H@5(g) | H@5(r) | H@10(g) | H@10(r) | MRR | IRR@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector_only | 0.00 | 0.13 | 0.70 | 0.13 | 0.87 | 0.045 | 0.00 |
| sql_filter_vector | 0.03 | 0.23 | 1.00 | 0.27 | 1.00 | 0.112 | 0.00 |
| sql_filter_vector_rerank | 0.07 | 0.20 | 0.80 | 0.30 | 0.87 | 0.142 | 0.00 |

### Per task type

| Task type | Pipeline | H@5(g) | H@5(r) | MRR |
|---|---|---:|---:|---:|
| employment | vector_only | 0.00 | 0.25 | 0.004 |
| employment | sql_filter_vector | 0.25 | 1.00 | 0.079 |
| employment | sql_filter_vector_rerank | 0.25 | 1.00 | 0.108 |
| general | vector_only | 0.20 | 0.70 | 0.086 |
| general | sql_filter_vector | 0.30 | 1.00 | 0.192 |
| general | sql_filter_vector_rerank | 0.20 | 0.80 | 0.178 |
| sentencing | vector_only | 0.00 | 1.00 | 0.000 |
| sentencing | sql_filter_vector | 0.00 | 1.00 | 0.000 |
| sentencing | sql_filter_vector_rerank | 0.00 | 0.75 | 0.000 |
| statute | vector_only | 0.50 | 1.00 | 0.113 |
| statute | sql_filter_vector | 0.50 | 1.00 | 0.201 |
| statute | sql_filter_vector_rerank | 0.50 | 0.50 | 0.404 |

Key finding: SQL court filtering is the single largest quality improvement.
`sql_filter_vector` finds a relevant document in 100% of queries, compared to
70% for vector search alone. The generic reranker hurts statute and sentencing
queries - it demotes exact statute sections below broader explanatory case law.

---

## 4. LLM Generator

Model: qwen3. Questions: 8. All successful.

| Metric | Value |
|---|---:|
| TTFT mean | 6.28 s |
| TTFT min | 3.97 s |
| TTFT max | 16.45 s |
| Tokens/sec mean | 4.6 |
| Tokens/sec min | 1.4 |
| Tokens/sec max | 6.6 |
| Citation correctness | 100% |

The generator reliably cites sources using [N] reference markers. All citations
pointed to documents present in the retrieved context. Generation speed is limited
by CPU-only inference on this hardware.

---

## 5. Reranker Mean Rank Improvement

| Metric | Value |
|---|---:|
| Mean rank of top-1 doc before rerank | 11.1 |
| Score mean | 0.26 |
| Score std | 0.34 |
| Score min | 0.00 |
| Score max | 0.99 |

The cross-encoder moves the best document from rank 11 to rank 1 on average,
which explains the MRR improvement even when H@5(rel) drops. The relevant
document is being found and promoted, but sometimes at the cost of displacing
other relevant documents from the top-5 window.

---

## Decisions Made

Based on the benchmark results, the following production changes were applied:

### 1. RERANK_MODE enum (off by default)

Replaces the old RERANKER_ENABLED + RERANKER_CANDIDATES pair.

```
RERANK_MODE=off          # default - skip cross-encoder, use legal ranker only
RERANK_MODE=rerank_5     # cross-encoder on top 5 legal-ranked candidates (~930 ms)
RERANK_MODE=rerank_N     # cross-encoder on top N candidates
```

At N=5: 929 ms, 0 regressions, H@5(rel)=1.00 - identical to the no-reranker baseline.
At N=20 (old implicit behavior): 3,908 ms, 6 regressions, H@5(rel)=0.80.
Defaulting to off saves ~930 ms per query with no quality loss at N=5.

### 2. Legal authority ranker added (rag/legal_ranker.py)

Deterministic re-scoring applied after Qdrant dedup, before the cross-encoder.
Blends vector score with court authority hierarchy, early chunk boost, citation
density, and statute/recency signals from the query. Pre-orders candidates so
the cross-encoder's capped-5 window contains the most authoritative sources.

Court authority weights:
NZSC=1.00, NZLEG=0.95, NZCA=0.85, NZHC=0.70, NZEmpC=0.65 ...

### 3. New benchmark pipelines added

`sql_filter_vector_legal` and `sql_filter_vector_legal_rerank_5` added to
`run_retrieval.py` to quantify the legal ranker's impact. Results pending
the next benchmark run.

---

## Next Benchmarks

In priority order based on analysis:

1. Legal ranker impact: compare `sql_filter_vector_legal` vs. `sql_filter_vector`
2. Gated rerank strategy: skip reranker when top-1 vector score exceeds threshold
3. Hybrid BM25 + vector with RRF fusion
4. Oracle filters vs. auto-planner filters (measures planner readiness)
5. Statute-specific exact lookup route
6. Tracker-first route for sentencing and PG queries
7. Context packing benchmark (chunk count, metadata headers, grouping)
