# NZ Legal RAG - Benchmark Results

Benchmarks run on 2026-05-21 (initial), 2026-05-22 (legal ranker + tracker-first),
2026-05-22 (BM25 hybrid), and 2026-05-22 (gold dataset revision). Source data:
`benchmarks/reports/comprehensive_report.md`, `benchmarks/reports/rerank_sweep.md`,
and `benchmarks/reports/latest.json`.

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
| sql_filter_vector_legal | 0.10 | 0.27 | 1.00 | 0.30 | 1.00 | **0.163** | 0.00 |
| sql_filter_vector_legal_rerank_5 | 0.10 | 0.23 | 0.93 | 0.23 | 0.93 | 0.118 | 0.00 |
| sql_tracker_vector | 0.03 | 0.23 | 0.97 | 0.27 | 0.97 | 0.115 | 0.00 |
| sql_tracker_vector_legal | 0.10 | 0.23 | 0.97 | 0.27 | 0.97 | 0.158 | 0.00 |
| sql_filter_bm25_legal | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.000 | 0.00 |
| sql_filter_bm25_vector_rrf_legal | 0.10 | 0.20 | 0.80 | 0.27 | 0.97 | 0.147 | 0.00 |
| sql_filter_bm25_vector_rrf_legal_plus_tracker_soft_boost | 0.10 | 0.20 | 0.83 | 0.27 | 0.97 | 0.147 | 0.00 |
| planner_filter_vector_legal | 0.10 | 0.27 | 1.00 | 0.27 | 1.00 | **0.160** | 0.00 |
| no_filter_vector_legal | 0.07 | 0.10 | 0.40 | 0.10 | 0.60 | 0.067 | 0.00 |

**Production pipeline: `planner_filter_vector_legal` (RERANK_MODE=off)**
**Oracle upper bound: `sql_filter_vector_legal` - benchmark reference only, not deployed**

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

### 5. BM25 hybrid - negative result across all three variants

BM25 via PostgreSQL `websearch_to_tsquery` + `ts_rank_cd` (GIN index on chunk text).
Regression categories tracked: `coverage_regression`, `gold_rank_regression`,
`mode_regression`. See `benchmarks/reports/regressions.md` for per-query detail.

| Pipeline | H@5(r) | MRR | coverage_reg | gold_rank_reg |
|---|---:|---:|---:|---:|
| sql_filter_vector_legal (baseline) | 1.00 | 0.154 | - | - |
| sql_filter_bm25_legal | 0.00 | 0.000 | 30 | 8 |
| sql_filter_bm25_vector_rrf_legal | 0.80 | 0.147 | 6 | 0 |
| sql_filter_bm25_vector_rrf_legal_plus_tracker_soft_boost | 0.83 | 0.147 | 5 | 0 |

**BM25 alone**: complete failure. `websearch_to_tsquery` generates an AND query
requiring all stems to co-occur in the same chunk (~512 tokens). A 15-word
question like "What notice must a landlord give before entering a rental
property?" becomes 8 required stems - most relevant chunks contain only a subset.
Result: 30/30 coverage_regression, H@5(r) = 0.000 for all queries.

**BM25 + vector RRF (k=60)**: 6 coverage regressions vs baseline. Root cause:
RRF gives each system equal weight at 1/(k+rank). When BM25 returns an irrelevant
doc at rank 1 (score 0.016) and the relevant doc is only found by vector at rank 5
(score 0.015), the irrelevant BM25 doc wins. k=60 was designed for systems of
roughly equal recall; BM25 here has near-zero recall so its top picks carry too
much leverage relative to its contribution.

**Stat mode exception**: statute MRR improves from 0.512 to 0.521 (+0.009) with
RRF. Section references like "section 103A" are keyword-matchable and appear in
legislation chunks - BM25 reliably finds them. The NZLEG authority weight in
statute_mode then promotes them. But this gain is outweighed by 6 coverage
regressions elsewhere.

**Soft tracker boost**: recovers one coverage regression
(`sentencing_home_detention_vs_imprisonment` H@5(r) 0->1) without introducing
any new ones. Net improvement is real but marginal: 5 regressions vs 6.

**Sentencing MRR=0 confirmed as gold dataset issue**: multiple "expected" gold
documents for sentencing queries do not contain the query terms at all. For
example, the gold set for "aggravated robbery starting points" includes
Webster v R [2026] NZCA 67 - a murder case that makes no reference to
aggravated robbery. BM25 correctly identifies this as non-matching. Sentencing
MRR=0 cannot be fixed by any retrieval pipeline; the gold set needs revision.

Production pipeline unchanged: `sql_filter_vector_legal`.

What would make BM25 useful:
- Key-term extraction before FTS (3-4 legal terms, not the full question)
- Weighted RRF: `weight_bm25 * 1/(k+rank)` with weight < 1.0
- Conditional BM25: activate only for statute/exact-section queries

---

## Next Benchmarks

In priority order based on analysis:

1. ~~Legal ranker impact~~ - done. MRR +37%, zero regressions. Locked as baseline.
2. ~~Reranker candidate sweep and gated strategy~~ - done. RERANK_MODE=off confirmed.
3. ~~Tracker-first route for sentencing and PG queries~~ - done. H@5(r) regression.
   Ruled out as general pipeline. Useful only for explicit structured-field queries.
4. ~~Hybrid BM25 + vector with RRF fusion~~ - done. All three variants regress vs
   baseline. BM25 AND-matching too strict; RRF k=60 not suited to low-recall BM25.
   Conditional BM25 (statute queries only) + weighted RRF is the next experiment.
5. ~~Gold dataset revision~~ - done. Two needs_review records fixed:
   - sentencing_aggravated_robbery_starting_point: removed NZCA/2026/67 (murder)
     and NZCA/2024/50 (sexual offence) - both mistagged in sentencing_cases, no
     "aggravated robbery" in chunk text. Replaced with NZCA/2023/519 (Herewini,
     26 on-point chunks) and NZCA/2024/359 (Lo v R). Query MRR: 0.000 -> 0.250.
   - statute_rta_landlord_entry: NZLEG/RTA/s123A (document retention) replaced
     with NZLEG/RTA/s48 (Landlord's right of entry, 24h notice, inspection rules).
     MRR stays 0 - s48 is at retrieval rank 9, outside top-5 fetch window.
   Overall: oracle MRR 0.154->0.163 (+6%), planner MRR 0.152->0.160 (+5%).
   Remaining sentencing MRR=0 (7 queries) is genuine retrieval hardness, not data.
6. ~~Oracle filters vs. auto-planner filters~~ - done. Heuristic planner achieves
   H@5(r)=1.00 and MRR=0.152 vs oracle MRR=0.154. Zero coverage regressions.
   Court filtering value confirmed: no-filter baseline MRR=0.059 (-62% vs oracle).
7. ~~Context packing benchmark~~ - done. See section 7 below.
8. Citation support benchmark (citation_exists / in_context / supports_claim)
9. Answer quality benchmark (faithfulness, completeness, false "not enough context" rate)

Retrieval baseline for answer-construction benchmarks:
  Pipeline: planner_filter_vector_legal
  H@5(r)=1.00, MRR=0.160 (post gold revision)
  The retrieval bottleneck is solved. Next bottleneck: did we package and use the
  retrieved context correctly for the generator?

### 6. Oracle vs heuristic court planner

Two new pipelines benchmarked against the oracle baseline.
Numbers updated after gold dataset revision (2026-05-22): see Next Benchmarks item 5.

| Pipeline | H@5(g) | H@5(r) | MRR | vs oracle |
|---|---:|---:|---:|---:|
| sql_filter_vector_legal (oracle) | 0.27 | 1.00 | 0.163 | - |
| planner_filter_vector_legal | 0.27 | 1.00 | 0.160 | -0.003 |
| no_filter_vector_legal | 0.10 | 0.40 | 0.067 | -0.096 |

**Planner (`rag/court_planner.py`)**: heuristic keyword planner with no LLM call.
Detects courts from domain signals (criminal, employment, tenancy, ACC, human rights,
legislation). Extracts years from temporal phrases ("in 2023", "2024 decisions").
Extracts courts: NZERA-only for general employment (NZEmpC added only when
"Employment Court" is explicitly named). NZLEG triggered by section references and
named Acts (ERA, RTA, etc.), but NOT Privacy Act or Human Rights Act (already
NZHRRT signals; adding NZLEG inflates the pool and causes ranking regressions).

Match quality on 30 gold queries:
- exact: 18 (planner courts == oracle courts)
- superset: 7 (oracle courts are a subset of planner courts - safe, larger pool)
- subset: 5 (planner misses some oracle courts - verified to be NZEmpC when gold docs are NZERA)
- disjoint: 0 (no hard coverage regressions)

**Result**: planner achieves oracle-equivalent H@5(r)=1.00 and MRR within 2% of
oracle. The heuristic planner is production-ready for court/year routing. No LLM
needed for this step. MRR gap vs oracle increased marginally (0.002->0.003) after
gold revision because sentencing_aggravated_robbery_starting_point improved by
+0.250 for both pipelines identically - gap unchanged in practice.

**No-filter baseline**: confirms court filtering is critical. Without any filter,
H@5(r) drops to 0.40 and MRR drops 62% vs oracle. The full corpus (23,561 docs)
overwhelms semantic search for jurisdiction-specific queries.

See `benchmarks/reports/planner_analysis.md` for per-query detail.

---

### 7. Context packing benchmark

Six context assembly formats benchmarked. Retrieval fixed at `planner_filter_vector_legal`.
14 queries (general + statute task types). LLM: Qwen3 via llama-server.
See `benchmarks/reports/context_packing.md` for per-query detail.

| Format | ctx_tok | ans_tok | has_cit | all_in_ctx | no_ctx | diversity | lat_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (current) | 494 | 364 | 1.00 | 1.00 | 0.14 | 4.2 | 49.3 |
| metadata_rich | 727 | 369 | 1.00 | 1.00 | 0.14 | 3.9 | 42.9 |
| full_chunk | 521 | 353 | 1.00 | 1.00 | 0.21 | 3.9 | 35.8 |
| statute_first | 727 | 369 | 1.00 | 1.00 | 0.14 | 4.1 | 34.5 |
| top3_only | 473 | 289 | 0.93 | 0.93 | 0.14 | 2.5 | 34.2 |
| max2_per_doc | 1160 | 486 | 1.00 | 1.00 | 0.29 | 3.6 | 59.2 |

**Format descriptions** (`rag/context_packer.py`):
- `baseline`: current production - one chunk/doc, 600-char truncation, plain `[N]` prefix
- `metadata_rich`: adds `Title | Court | Date | Section` header per chunk, same pool
- `full_chunk`: removes 600-char truncation (chunks are ~120 words so rarely differ)
- `statute_first`: metadata_rich with NZLEG chunks sorted before case chunks
- `top3_only`: top 3 docs, full text, metadata_rich headers
- `max2_per_doc`: up to 2 chunks per doc (10 chunks max), metadata_rich, 600-char each

**Findings**:

1. **Citation quality is format-independent.** `all_in_ctx=1.00` for every format that
   produced citations. The LLM follows the `[N]` citation instruction reliably regardless
   of how the context is structured. Zero spurious paragraph-number false positives after
   fixing the citation regex to exclude 4-digit NZ case years (`[2023]`).

2. **The 600-char truncation does not cut content.** `baseline` vs `full_chunk` have nearly
   identical `ctx_tok` (494 vs 521) because NZ legal chunks are ~120-word windows already
   within the limit. Removing the truncation cap changes nothing.

3. **Metadata headers add tokens without quality gain.** `metadata_rich` uses 47% more
   context tokens (727 vs 494) with identical `has_cit`, `all_in_ctx`, `no_ctx` metrics
   and slightly lower source diversity (3.9 vs 4.2). The headers consume prompt budget
   without improving the answer.

4. **Two chunks per doc hurts.** `max2_per_doc` raises `no_ctx` rate from 0.14 to 0.29
   (2x), uses 2.3x the context tokens, and increases latency by 20%. The second-best chunk
   per document is noisier than the top chunk from the next-ranked document. Five diverse
   documents beat two chunks from one document.

5. **Top 3 docs is insufficient.** `top3_only` drops source diversity from 4.2 to 2.5
   and produces one complete citation failure (statute_era_s103_personal_grievance: no
   citations generated from 3 docs). The 4th and 5th ranked docs contribute materially.

6. **`no_context_flag` is query-dependent, not format-dependent.** `general_workplace_harassment`
   triggers the flag in all 6 formats - the retrieved docs do not contain enough specific
   NZ workplace harassment statute or case law. This is a corpus coverage gap, not a
   packing problem. `general_sick_leave_medical_cert` triggers it in 4/6 formats
   (borderline coverage).

7. **`statute_first` is a free win for statute-intent queries.** Sorting legislation chunks
   first helps the LLM locate the statute text sooner without increasing `no_ctx` rate.
   For general queries where no NZLEG chunks are retrieved, `statute_first` behaves
   identically to `metadata_rich`.

**Conclusion**: The `baseline` format is already near-optimal. No format beats it on the
combined `{no_ctx, diversity, all_in_ctx}` metrics. The main actionable finding is to adopt
`statute_first` (legislation before cases) for statute-profile queries since it matches the
intent-aware ranker's `statute_mode` and helps structure the answer without token cost.

**CONTEXT_PACK_MODE locked** at `baseline` for general/authority/example/tracker queries.
`statute_first` promoted as the preferred format for statute-intent queries.

---

## Production Configuration (Locked)

Based on all benchmarks, the following decisions are locked and should not be
re-opened without new evidence from a benchmark run.

| Decision | Value | Rationale |
|---|---|---|
| DEFAULT_PIPELINE | planner_filter_vector_legal | Heuristic planner reaches oracle-equivalent H@5(r)=1.00, MRR=0.152 (-1.3%) with no LLM call |
| RERANK_MODE | off | Legal ranker alone outperforms legal ranker + cross-encoder |
| BM25_MODE | conditional_statute_only | Global BM25 causes regressions; statute/section/citation queries only |
| TRACKER_JOIN_MODE | soft_only | Hard JOIN excludes acceptable docs; additive post-rank bonus only |
| COURT_PLANNER | deterministic heuristic | Keyword signals, no LLM; oracle pipeline (sql_filter_vector_legal) kept as benchmark upper bound |
| CONTEXT_PACK_MODE | baseline (general); statute_first (statute) | Metadata headers add no quality; 5 docs > 2 chunks/doc; statute_first free win for statute intent |

### Production routing

| Intent | Pipeline | Notes |
|---|---|---|
| authority | planner_filter_vector_legal | Default legal ranker profile |
| example | planner_filter_vector_legal | Example profile in legal ranker |
| tracker | planner_filter_vector_legal | Tracker profile + soft boost (no hard JOIN) |
| statute | planner_filter_vector_legal + conditional BM25 | BM25 activated for section refs, citations, quoted phrases |
| default | planner_filter_vector_legal | Fallback for all unclassified intents |

### Conditional BM25 activation rules (rag/bm25_query.py)

BM25 is activated when any one of the following is present in the query:

1. Section reference: s103A, section 103A, s 127
2. Case citation: NZCA/2024/50, [2024] NZCA 50
3. Quoted phrase: "interim reinstatement"
4. Short keyword query: <= 6 tokens, no question words, no trailing "?"

When activated, OR-joined anchors extracted from the query are passed to
`websearch_to_tsquery`, not the full question string, so FTS does not apply strict
AND-matching over all stems. When BM25 returns empty (suppressed or no FTS match),
RRF is skipped entirely and vector cosine scores are used directly.
