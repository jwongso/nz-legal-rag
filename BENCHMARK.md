# NZ Legal RAG - Benchmark Results

Benchmarks run on 2026-05-21 (initial), 2026-05-22 (legal ranker), and 2026-05-22
(tracker-first). Source data: `benchmarks/reports/comprehensive_report.md`,
`benchmarks/reports/rerank_sweep.md`, and `benchmarks/reports/latest.json`.

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
| sql_filter_vector_rerank (N=20) | 0.07 | 0.20 | 0.80 | 0.30 | 0.87 | 0.142 | 0.00 |
| sql_filter_vector_legal | 0.10 | 0.23 | 1.00 | 0.27 | 1.00 | **0.154** | 0.00 |
| sql_filter_vector_legal_rerank_5 | 0.10 | 0.23 | 0.93 | 0.23 | 0.93 | 0.118 | 0.00 |
| sql_tracker_vector | 0.03 | 0.23 | 0.97 | 0.27 | 0.97 | 0.115 | 0.00 |
| sql_tracker_vector_legal | 0.10 | 0.23 | 0.97 | 0.27 | 0.97 | 0.158 | 0.00 |

**Current production pipeline: `sql_filter_vector_legal` (RERANK_MODE=off)**

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

### 2. Intent-aware legal ranker (rag/legal_ranker.py)

Deterministic re-scoring applied after Qdrant dedup. Four profiles, each tuned
for a different query intent detected from the query text:

| Mode | authority_weight | Key signal | Triggered by |
|---|---:|---|---|
| authority | 0.18 | court hierarchy, citation density | principle/doctrine queries (default) |
| example | 0.04 | recency +0.05 | "find", "examples of", year in query |
| tracker | 0.05 | structured payload +0.04 | starting points, guilty plea, sentencing range |
| statute | 0.22 | legislation chunk +0.15 | s103A, section 127, specific section refs |

The planner can set the mode explicitly; heuristic detection is the fallback.

Result: MRR improved from 0.112 to 0.154 (+37%) with zero regressions at H@5(r).
H@10(g) also maintained at 0.27 - profile-aware weighting stopped example/tracker
queries from over-promoting high-authority NZCA docs over lower-court gold docs.

Court authority weights (used at varying strength per profile):
NZSC=1.00, NZLEG=0.95, NZCA=0.85, NZHC=0.70, NZEmpC=0.65 ...

### 3. Cross-encoder confirmed harmful even at N=5 after legal ranker

`sql_filter_vector_legal_rerank_5` regresses on statute queries (H@5(r) drops
from 1.00 to 0.93). The statute_mode ranker surfaces legislation chunks first,
but the cross-encoder at N=5 then picks the wrong one from those candidates.

RERANK_MODE=off is the confirmed production default. The profile-aware legal
ranker alone outperforms legal ranker + cross-encoder.

### 4. Tracker-first retrieval ruled out as general pipeline

`sql_tracker_vector` and `sql_tracker_vector_legal` JOIN sentencing_cases or
employment_cases before vector search, restricting candidates to documents that
have structured extraction data.

Result: H@5(r) regression from 1.00 to 0.97. One query (`pg_employment_court_challenge_era`)
dropped from H@5(r)=1 to 0 because its acceptable documents are procedural challenge
decisions that are not in employment_cases (they carry no PG outcome data).

Root cause of sentencing MRR=0 is NOT pool size. All 8 sentencing gold documents
are confirmed in sentencing_cases. The problem is specificity: gold docs are
recent specific decisions (2024-2026 NZCA) among 5,280 equally-semantically-similar
sentencing docs. Vector search cannot distinguish which specific decision is the
gold one when hundreds of others discuss the same sentencing methodology.

Tracker-first is not adopted as production pipeline. Remains as an explicit
structured-data query mode for the future planner (e.g., "find decisions where
home detention was imposed" - field filter, not vector).

---

## Next Benchmarks

In priority order based on analysis:

1. ~~Legal ranker impact~~ - done. MRR +37%, zero regressions. Locked as baseline.
2. ~~Reranker candidate sweep and gated strategy~~ - done. RERANK_MODE=off confirmed.
3. ~~Tracker-first route for sentencing and PG queries~~ - done. H@5(r) regression.
   Ruled out as general pipeline. Useful only for explicit structured-field queries.
4. Hybrid BM25 + vector with RRF fusion - likely win for statute and exact-phrase queries.
   Sentencing MRR=0 root cause: offence-name matching (aggravated robbery, manslaughter)
   is a keyword problem, not a semantic problem. BM25 should surface exact offence matches.
5. Oracle filters vs. auto-planner filters - measures planner readiness for production
6. Context packing benchmark (chunk count, metadata headers, grouping)
7. Citation support benchmark (citation_exists / in_context / supports_claim)
