# nz-legal-rag

On-premise RAG pipeline for New Zealand legal research - designed to scale toward
500,000+ public legal decisions, with the current benchmark corpus covering 23,561
documents and 982,361 chunks. Structured trackers comparable to Westlaw NZ, running
entirely on local hardware.

This project explores an on-premise RAG architecture for privacy-sensitive legal workflows,
where data residency, legal professional privilege, and operational control are important
design constraints. All inference, retrieval, and storage run on client hardware - no data
leaves the machine.

---

## Why on-premise matters for NZ legal

| Constraint | Cloud AI risk | This project |
|---|---|---|
| Legal professional privilege | Client queries sent to third-party servers | All inference runs locally |
| Privacy Act 2020 | Processing personal data requires disclosure and consent | No data leaves the machine |
| Health Information Privacy Code | Clinical data cannot go to US cloud servers | Air-gapped deployment possible |
| Data residency | NZ firms may require NZ-hosted data | Runs on client hardware in NZ |

---

## Benchmark Results

Full retrieval, generation, and answer quality results: **[BENCHMARK.md](BENCHMARK.md)**

The benchmark covers the complete pipeline end-to-end: 12 retrieval pipelines across
30 gold queries, 6 context packing formats, citation faithfulness judging, and answer
quality scoring. Key numbers from the locked 8B GPU baseline:

| Dimension | Result |
|---|---|
| Retrieval H@5(r) | 1.00 - heuristic planner matches oracle filter, no LLM needed for routing |
| Retrieval MRR | 0.160 - legal ranker gives +46% MRR over raw vector search |
| Generator TTFT | 62 ms mean (101x faster than the 35B CPU comparison) |
| Generator throughput | 59.5 tok/s, 100% citation format compliance |
| Citation faithfulness | 0.86 (LLM-as-judge, confirmed by both 35B and 8B judges) |
| Answer faithfulness | 4.00 / 5 (8B judge) |
| Answer completeness | 3.64 / 5 - gaps trace to corpus coverage, not pipeline failures |

The report also documents what was tried and ruled out: global BM25 (collapsed H@5(r) to 0),
cross-encoder reranker (caused regressions), tracker hard JOIN (dropped valid cases), and
no court filter (H@5(r) dropped to 0.40).

---

## Architecture

```
                        +------------------+
                        |   User / Client  |
                        +--------+---------+
                                 |
              +------------------+------------------+
              |                                     |
    +---------v---------+               +-----------v----------+
    |   MCP Server      |               |   REST API (FastAPI)  |
    |  search_nz_law()  |               |   POST /ask          |
    |  get_case()       |               |   GET  /search       |
    +--------+----------+               |   POST /notable      |
             |                          |   POST /sentencing-  |
             |                          |         tracker      |
             |                          |   POST /pg-tracker   |
             +------------------+       +-----------+----------+
                                |                   |
                    +-----------v-------------------v-----------+
                    |    RAG Pipeline (rag/pipeline.py)         |
                    |  1. embed query                           |
                    |  2. court planner: domain signals -> SQL |
                    |     filter (no LLM call)                |
                    |  3. SQL pre-filter (court + year)        |
                    |  4. Qdrant semantic search within filter |
                    |  5. deduplicate (one chunk per case)     |
                    |  6. legal authority ranker (intent-aware)|
                    |  7. conditional BM25 + RRF [statute only]|
                    |  8. generate answer via LLM             |
                    |  9. verify citations (no LLM call)      |
                    +-----------+-----------+-----------+-------+
                                |           |           |
          +---------------------+     +-----v------+   |
          |                           |  PostgreSQL |   |
+---------v----------+               |  nz_legal   |   +----------------------+
|  Qdrant            |               |  documents  |   |  llama.cpp server    |
|  Vector DB         |               |  chunks     |   |  (local inference)   |
|  :6333             |               |  sentencing |   |  :8080               |
|  Chunks + metadata |               |  pg_cases   |   +----------------------+
|  (payload filters) |               |  citations  |
+--------------------+               |  :5432      |
                                     +-------------+

  Embeddings (nomic-embed-text-v1.5) and reranker (bge-reranker-v2-m3)
  run in-process via sentence-transformers on CPU. No Ollama required.


Data ingestion:

  NZLII (nzlii.org) - public legal information repository
     |
     +-- HTML decisions: NZHC, NZCA, NZERA, NZEmpC, NZSC
     +-- PDF decisions:  NZTT (Tenancy Tribunal, PDF-only on NZLII)
     |
    +---------v---------+
    |  scraper.py       |
    |  subprocess curl  |  (bypasses Cloudflare TLS fingerprinting)
    +--------+----------+
             |
    +--------v----------+
    |  chunker.py       |
    |  section-aware    |
    |  120-word windows |
    +--------+----------+
             |
    +--------v----------+
    |  pipeline.py      |   --> Qdrant (vectors + metadata)
    |  embed + upsert   |   --> PostgreSQL via migrate_from_qdrant.py
    +-------------------+
```

---

## Data sources

| Source | Content | Courts indexed |
|---|---|---|
| NZLII (nzlii.org) | NZ legal information institute - free public access | NZSC, NZCA, NZHC, NZERA, NZEmpC, NZTT |

NZLII hosts HTML decisions for most courts. Tenancy Tribunal (NZTT) decisions
are PDF-only on NZLII - the scraper fetches and extracts these via pypdf.

Current coverage: **2020-2026** across all courts (NZCA fully indexed from 2020).
982,000+ chunks indexed across 23,500+ documents. All sources are publicly available.
No proprietary data required.

---

## Live demo

https://nz-legal-rag.localrun.ai

On-premise instance running on local hardware, exposed via Cloudflare Tunnel.
No data leaves the machine - the tunnel only carries HTTP traffic to the UI.

---

## Quick start

### 1. Start infrastructure

```bash
docker compose up -d
```

Starts Qdrant (port 6333). Ollama is not required - embeddings run in-process
via sentence-transformers.

PostgreSQL must be running locally. Create the database:

```bash
createdb nz_legal
psql nz_legal -f db/schema.sql
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
pip install einops  # required by nomic-embed-text-v1.5
```

### 3. Start the LLM inference server

```bash
llama-server --model /path/to/model.gguf --n-gpu-layers 20 --port 8080
```

Any OpenAI-compatible endpoint works (Ollama, vLLM, LM Studio).

### 4. Ingest NZ legal data

```bash
# Ingest NZCA decisions (2020-2026)
python -m ingest.pipeline --court NZCA --years 2020 2021 2022 2023 2024 2025 2026 --threads 16

# Ingest High Court decisions
python -m ingest.pipeline --court NZHC --years 2022 2023 2024 --threads 16

# Ingest Employment Relations Authority decisions
python -m ingest.pipeline --court NZERA --years 2022 2023 2024 --threads 16

# Ingest Tenancy Tribunal (PDF extraction handled automatically)
python -m ingest.pipeline --court NZTT --years 2022 2023 2024 --threads 16
```

`--threads` caps CPU usage from sentence-transformers (default: 16).
`--max-per-year` limits decisions per year (default: 200).

### 5. Populate PostgreSQL

After Qdrant is populated, migrate metadata to PostgreSQL for structured filtering:

```bash
python -m db.migrate_from_qdrant
```

Then run the structured data backfill pipelines:

```bash
python -u -m ingest.sentencing_pipeline  # criminal sentencing factors
python -u -m ingest.pg_pipeline          # employment personal grievance outcomes
python -u -m ingest.counsel_pipeline     # counsel/appearances extraction
python -u -m ingest.flag_pipeline        # legal category flags
python -u -m ingest.recovery_agg         # civil recovery rate aggregation
```

### 6. Ask a question

```bash
# REST API
python -m api.server

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can a landlord enter a property without 24 hours notice?"}'
```

```bash
# MCP server (for Claude Code / Claude Desktop integration)
python -m mcp.server
```

---

## Example output

```
Q: Can a landlord increase rent more than once in 12 months?

A: No. Under section 24 of the Residential Tenancies Act 1986 (as amended by the
Residential Tenancies Amendment Act 2020), a landlord may not increase rent more
than once in any 12-month period.

The Tenancy Tribunal has consistently applied this restriction. In Hooper v Zhang
[2023] NZTT Wellington 4521/23, the Tribunal found that a second rent increase
within 11 months breached s24 and ordered the overpaid rent refunded.

Sources:
  [1] Residential Tenancies Act 1986, s24 (as at 12 Aug 2021)
  [2] Hooper v Zhang [2023] NZTT Wellington 4521/23
  [3] Chen v Patel [2022] NZTT Auckland 3187/22
```

---

## Retrieval strategies

Three retrieval strategies are supported, selectable per-request:

### 1. Pure Qdrant (default)

Semantic search over the full corpus with optional Qdrant payload filters (court,
year, flags). Best for open-ended questions with no structured constraints.

```python
pipeline.ask("What is unjustified dismissal?", courts=["NZERA"])
```

### 2. SQL-first hybrid

PostgreSQL pre-filter narrows candidates to a structured subset, then Qdrant ranks
semantically within that set via `HasIdCondition`. Accurate when you need to combine
structured constraints (sentencing range, offence type, employment outcome) with
a semantic question.

```python
from db.filter import FilterParams
params = FilterParams(courts=["NZHC"], min_final_sentence=24, max_final_sentence=60)
pipeline.ask("robbery aggravated factors", sql_filter=params)
```

The pre-filter is capped at 5,000 point IDs to keep Qdrant latency bounded.

### 3. Conditional BM25 (statute and exact-reference queries only)

Full-text search via PostgreSQL `tsvector`/GIN index. Activated automatically only
when the query contains a section reference (`s103A`, `section 127`), a case citation
(`[2024] NZCA 50`), a quoted phrase, or is a short keyword query (<= 6 words, no
question words). For all other queries BM25 is suppressed.

When active, OR-joined anchors extracted from the query are passed to
`websearch_to_tsquery` - not the full question. This avoids the AND-strictness
problem where a 15-word question requires all stems to co-occur in one 512-token chunk.
BM25 results are fused with vector results via RRF (k=60). If BM25 returns empty,
RRF is skipped and vector scores are used directly.

```python
from rag.bm25_query import build_bm25_query
q = build_bm25_query("section 103A Employment Relations Act")
# q.should_use = True, q.query_terms = '"section 103A" OR "Employment Relations Act"'
```

Benchmarked result: global BM25 hybrid caused 6 coverage regressions vs vector-only
baseline. Conditional BM25 (this design) shows zero regressions and adds BM25 signal
only where exact matching is reliable. See `BENCHMARK.md` for full results.

---

### Retrieval design summary

The benchmark showed that general BM25 hybrid retrieval was harmful for broad legal
questions. PostgreSQL-filtered vector search with a profile-aware legal ranker remained
Pareto-optimal, achieving 100% relevant Hit@5 and the best production-safe MRR. BM25
is therefore reserved for targeted statute, section, citation, and rare-phrase queries
rather than used as a global hybrid retrieval signal.

The production pipeline is `planner_filter_vector_legal`. The oracle pipeline
(`sql_filter_vector_legal`) is retained as a benchmark upper bound only.

---

### Court planner design decisions

Four non-obvious lessons emerged from benchmarking the heuristic court planner against
oracle filters on 30 gold queries (`benchmarks/datasets/retrieval_gold.jsonl`):

**1. Employment queries route to NZERA by default; NZEmpC is added only when
the query explicitly mentions "Employment Court".**
General employment signals ("dismissal", "redundancy", "employer") almost always
target ERA decisions. Adding NZEmpC to every employment query dilutes the NZERA
result pool and causes ranking regressions on NZERA-specific gold documents.
NZEmpC is appended to the filter only when the query contains "Employment Court"
or "NZEmpC" as an explicit court reference.

**2. Privacy Act and Human Rights Act queries route to NZHRRT, not legislation.**
"Privacy Act" and "Human Rights Act" are NZHRRT signals, not triggers for generic
legislation retrieval. Routing them to NZLEG inflated the candidate pool with statute
chunks and caused ranking regressions on queries seeking court decisions rather than
statute text. The two Acts remain excluded from the `_LEGISLATION` signal group.

**3. Year extraction applies only to temporal phrases, not Act names.**
"in 2023" and "2024 decisions" correctly extract year=2023/2024. "Privacy Act 2020"
and "Employment Relations Act 2000" do not - Act-name years are not decision-date
filters. A naive year regex would restrict "Privacy Act 2020" queries to 2020
decisions, excluding all recent case law. The regex requires a temporal preposition
("in", "from", "during", "of") or a decision-noun suffix ("decisions", "cases").

**4. Planner quality is evaluated using four court-match categories.**
- exact: planner courts == oracle courts (18/30 gold queries)
- superset: planner includes all oracle courts plus extras - safe, larger pool (7/30)
- subset: planner misses some oracle courts - coverage risk, but zero MRR loss (5/30)
- disjoint: no overlap with oracle - hard regression (0/30)

Result: H@5(r)=1.00, MRR=0.152 vs oracle MRR=0.154 (-1.3%). No LLM required
for court routing. See `rag/court_planner.py` and `BENCHMARK.md` section 6.

---

## Developer trace and citation verification

Enable the retrieval trace by passing `"trace": true` in the `/ask` request body,
or click the **DEV** button in the web UI.

```json
POST /ask
{
  "question": "What is unjustified dismissal?",
  "top_k": 5,
  "trace": true
}
```

Response includes a `trace` object:

```json
{
  "answer": "...",
  "trace": {
    "strategy": "pure_qdrant",
    "sql_point_ids_count": 0,
    "counts": {
      "qdrant_candidates": 30,
      "after_dedup": 15,
      "after_rerank": 5
    },
    "latency_ms": {
      "embed": 42.1,
      "sql": 0.0,
      "qdrant": 87.3,
      "rerank": 0.0,
      "generate": 31204.8,
      "total": 31334.2
    },
    "top_scores": [0.8821, 0.8754, 0.8612, 0.8489, 0.8311],
    "models": {
      "llm": "qwen3:latest",
      "embedding": "nomic-embed-text-v1.5",
      "reranker_enabled": false
    }
  },
  "citation_verification": {
    "has_citations": true,
    "cited_count": 4,
    "orphan_citations": [],
    "uncited_sources": [5],
    "evidence_confidence": "high",
    "has_warning": false
  }
}
```

Citation verification is lightweight - no extra LLM call. It checks whether
the generated answer contains `[N]` references matching the retrieved sources,
flags orphan citations (cited but not retrieved), and assigns a confidence level:

- **high**: 5+ sources, all cited, no orphans
- **medium**: 3+ sources, all cited, no orphans
- **low**: no citations, or orphan citations present

---

## Tracker endpoints (Westlaw-comparable)

These structured-data endpoints compete directly with Westlaw NZ's premium tracker products,
delivered on-premise with no per-user SaaS fee.

| This project | Westlaw NZ equivalent |
|---|---|
| `POST /sentencing-tracker` | Sentencing Tracker |
| `POST /pg-tracker` | Personal Grievance Tracker |
| `POST /notable` (OSI + flags) | OSH Tracker / Resource Management Tracker |
| `POST /contrasting-cases` | (no direct equivalent - contrastive retrieval) |

### Sentencing Tracker (`POST /sentencing-tracker`)

Extracts structured criminal sentencing factors from NZHC, NZCA and NZSC decisions.

Payload fields extracted per chunk (stored under `sentencing.*` in Qdrant and
`sentencing_cases` in PostgreSQL):

| Field | Type | Description |
|---|---|---|
| `starting_point_months` | float | Judicial starting point before discounts |
| `final_sentence_months` | float | Imprisonment term actually imposed |
| `home_detention_months` | float | Home detention term |
| `community_work_hours` | int | Community work hours ordered |
| `reparation_amount` | float | Reparation ordered ($) |
| `fine_amount` | float | Fine imposed ($) |
| `guilty_plea_discount_pct` | float | Discount % applied for guilty plea (5-50) |
| `sentence_type` | keyword | imprisonment / home_detention / community_work / fine / supervision |
| `has_guilty_plea` | bool | Guilty plea present in the text |
| `has_remorse` | bool | Remorse expressed |
| `has_previous_convictions` | bool | Prior convictions positively stated |

Filter by: offence flags, courts, year range, sentence type, starting point range,
final sentence range, guilty plea presence.

UI: Sentencing Tracker tab computes and displays median starting point, median sentence,
and median guilty plea discount across results.

Backfill (regex): `python -u -m ingest.sentencing_pipeline`
Backfill (LLM): `python -u -m ingest.llm_extract_pipeline --domain criminal --limit 200`

### Personal Grievance Tracker (`POST /pg-tracker`)

Extracts ERA / NZEmpC personal grievance outcome data.

Payload fields extracted per chunk (stored under `pg.*` in Qdrant and
`pg_cases` in PostgreSQL):

| Field | Type | Description |
|---|---|---|
| `grievance_types` | keyword[] | unjustified_dismissal, constructive_dismissal, disadvantage, harassment, discrimination, unjustified_action |
| `reinstatement_ordered` | bool | True = ordered, False = declined |
| `contributory_conduct_pct` | float | Reduction % for employee's own conduct |
| `has_contributory_conduct` | bool | Discussed but % not parsed |

Filter by: grievance type, reinstatement outcome, contributory conduct range,
compensation range (uses `penalty.awarded_amount`), court, year.

UI: PG Tracker tab shows reinstatement rate, median compensation, median contributory conduct.

Backfill (regex): `python -u -m ingest.pg_pipeline`
Backfill (LLM): `python -u -m ingest.llm_extract_pipeline --domain employment`

### Similar Cases With Opposite Outcomes (`POST /contrasting-cases`)

Finds semantically similar cases where the court reached a different outcome. Two Qdrant
semantic searches run in parallel - one per outcome group - each restricted to chunks
carrying the target structured outcome field.

```json
POST /contrasting-cases
{
  "query": "aggravated robbery weapon group offending youth",
  "domain": "criminal",
  "split_by": "sentence_type",
  "courts": ["NZHC"],
  "top_k": 5
}
```

Supported domains and splits:

| Domain | split_by | Group A | Group B |
|---|---|---|---|
| `criminal` | `sentence_type` (default) | Imprisonment | Home detention |
| `criminal` | `guilty_plea` | Guilty plea | No guilty plea |
| `employment` | `reinstatement` (default) | Reinstatement ordered | Reinstatement declined |

Each case in the response includes `structured` fields (sentencing factors or PG outcome)
so the caller can see what drove the different result.

Contrasting outcomes are drawn from public court decisions and reflect what courts decided
in those specific cases. They do not predict what a court would decide in any other
situation. Not legal advice.

---

## Counsel extraction

Appearances block extracted from all decisions and stored under `counsel.*` in Qdrant.

| Field | Type | Description |
|---|---|---|
| `has_data` | bool | Appearances block found |
| `all_names` | keyword[] | All counsel full names |
| `all_surnames` | keyword[] | Surnames only (for fuzzy search) |
| `crown` | keyword[] | Crown / prosecution counsel |
| `defence` | keyword[] | Defence / appellant counsel |
| `entries` | list | Structured [{names, role}] entries |

Searchable via the Notable Cases tab "Counsel" filter or the `/notable` endpoint.

Backfill: `python -u -m ingest.counsel_pipeline`

---

## Flag and penalty system

Every chunk carries `flags` (list of matched categories) and `penalty` (structured outcome data).

### Flags (19 categories)

Detected by regex in `ingest/flags.py`. Used across all tracker endpoints as an OR filter.

Criminal-relevant: self_defence, provocation, diminished_responsibility, necessity, duress,
mental_health, intoxication, youth, tikanga_maori, cultural_factors, lack_of_motive,
suppressed_identity, jurisdictional_challenge, novel_argument, procedural_irregularity.

Civil/general: exemplary_damages, contempt, whistleblower, self_represented.

### Penalty fields (`penalty.*`)

| Field | Description |
|---|---|
| `court_type` | criminal / civil_financial / civil_nonfinancial / civil_mixed / coronal / civil_disciplinary |
| `outcome_osi` | Criminal Outcome Severity Index (0.0 discharge to 1.0 life without parole) |
| `awarded_amount` | Civil monetary award ($) |
| `recovery_rate` | awarded / claimed (>1.0 = exemplary damages) |
| `recovery_class` | full_recovery / partial / exemplary |

Backfill: `python -u -m ingest.flag_pipeline`
Civil aggregation: `python -u -m ingest.recovery_agg`

---

## Analytics SQL

Pre-written SQL queries in `db/queries/` for corpus analysis without writing raw SQL:

| File | Purpose |
|---|---|
| `coverage.sql` | Corpus coverage per court/year, gap detection |
| `sentencing_analysis.sql` | Median sentencing, GPD factor prevalence, year-over-year trends |
| `employment_analysis.sql` | Compensation stats, reinstatement rates, contributory conduct |
| `citations.sql` | Most-cited cases, citation graph traversal, unresolved citations |
| `hybrid_search.sql` | SQL-first hybrid, BM25, and pure Qdrant query patterns |
| `bm25_index.sql` | GIN tsvector index creation and example BM25 query |

---

## Test suite

```bash
pytest tests/ -v
```

145 tests across 7 files. All tests run against live Qdrant and PostgreSQL
(no mocking). LLM-dependent tests are automatically skipped when the inference
server is not running.

| File | Tests | Coverage |
|---|---|---|
| `test_api.py` | 21 | All REST endpoints: health, search, ask (LLM-gated), notable, sentencing-tracker, pg-tracker |
| `test_filter.py` | 20 | SQL pre-filter (`get_point_ids`), BM25 search, `get_document_metadata` |
| `test_pipeline.py` | 15 | Embedder output, VectorStore search, deduplication, SQL-first hybrid roundtrip |
| `test_quality.py` | 12 | Domain relevance (employment vs criminal courts), BM25 snippet quality, score calibration |
| `test_trace.py` | 12 | `RetrievalTrace` serialisation, `CitationVerification` logic edge cases |
| `test_contrasting.py` | 22 | `POST /contrasting-cases` endpoint, split configs, data quality, filter combinations |
| `test_llm_extract.py` | 31 | JSON parsing from LLM output, field validation and clamping, text selection strategy |
| `test_perf.py` | 13 | Latency budgets: SQL (<0.5s), BM25 (<1s), embed (<3s), Qdrant (<5s), hybrid (<8s) |

---

## Evaluation

Uses RAGAS metrics against a curated NZ legal Q&A benchmark:

```bash
python -m eval.ragas_eval --questions eval/questions.jsonl --output eval/results.json
```

| Metric | Description | Target |
|---|---|---|
| Faithfulness | Answer grounded in retrieved context | > 0.85 |
| Answer relevance | Answer addresses the question | > 0.80 |
| Context precision | Retrieved chunks are relevant | > 0.75 |
| Context recall | Relevant chunks are retrieved | > 0.70 |

---

## MCP integration

Add to your Claude Desktop or Claude Code MCP config:

```json
{
  "mcpServers": {
    "nz-legal": {
      "command": "python",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/nz-legal-rag"
    }
  }
}
```

Available tools:
- `search_nz_law(query, court?, date_from?, date_to?, top_k?)` - semantic search
- `get_case(case_id)` - retrieve a specific decision
- `list_courts()` - list indexed courts and decision counts

---

## Stack

| Component | Technology |
|---|---|
| Vector database | Qdrant |
| Relational database | PostgreSQL (structured filters, BM25, analytics) |
| Embeddings | nomic-embed-text-v1.5 via sentence-transformers (in-process, no server) |
| LLM inference | llama.cpp (OpenAI-compatible) |
| Reranker | bge-reranker-v2-m3 via sentence-transformers (CPU, ~50ms per query) |
| SQL pre-filter | `db/filter.py` - FilterParams + `HasIdCondition` for SQL-first hybrid |
| BM25 search | PostgreSQL `tsvector` / GIN index |
| Retrieval trace | `rag/trace.py` - per-stage latency + citation verification |
| MCP server | Python MCP SDK |
| REST API | FastAPI |
| Evaluation | RAGAS |
| Ingestion | subprocess curl + BeautifulSoup4 |

---

## Hardware

This project runs on a single machine with a consumer GPU. No Blackwell required.

Current inference: llama-server with Qwen3.6-35B-A3B-UD-Q5_K_M (Unsloth Dynamic 5-bit,
MTP-capable). 10 GPU layers, 4096 context.

Embeddings: nomic-embed-text-v1.5 via sentence-transformers, running in-process on CPU.
No Ollama server required. Thread count is capped at 16 during ingest to leave headroom
for the inference server. Not a bottleneck at the scales the NZ legal corpus requires
(NZLII has ~50k decisions total).

Upcoming: GMKtec EVO-T1 mini PC (Intel Core Ultra 9 285H, 64GB DDR5) + WD Blue SN5000
2TB NVMe. The Arc 140T iGPU shares system RAM as VRAM, enabling larger model context
via SYCL/oneAPI. Planned as 24/7 always-on inference node running Gentoo Linux.

---

## Roadmap

### Done

- [x] NZLII decision scraper (NZSC, NZCA, NZHC, NZERA, NZEmpC, NZTT) via subprocess curl (Cloudflare bypass)
- [x] PDF extraction for Tenancy Tribunal decisions
- [x] Section-aware legal document chunker (120-word windows, 20-word minimum)
- [x] Qdrant ingestion pipeline with metadata filtering and deterministic UUID5 IDs
- [x] 182,000+ chunks indexed across 13 courts
- [x] RAG pipeline with citation grounding
- [x] MCP server for Claude Code / Claude Desktop
- [x] FastAPI REST interface with web chat UI
- [x] RAGAS evaluation harness with NZ legal Q&A benchmark
- [x] Cross-encoder reranker (bge-reranker-v2-m3, CPU) - benchmarked and ruled out (RERANK_MODE=off)
- [x] Case deduplication in retrieval
- [x] Flag system (19 categories, regex-based)
- [x] Penalty extraction: criminal OSI scale + civil awarded amount + recovery rate
- [x] Notable Cases endpoint (`POST /notable`)
- [x] Counsel / appearances extraction and Qdrant indexing
- [x] Sentencing Tracker (`POST /sentencing-tracker`)
- [x] Personal Grievance Tracker (`POST /pg-tracker`)
- [x] PostgreSQL dual-backend: structured metadata store alongside Qdrant
- [x] SQL-first hybrid retrieval: `FilterParams` + Qdrant `HasIdCondition` (Phase 2)
- [x] BM25 full-text search via PostgreSQL tsvector/GIN index
- [x] Developer retrieval trace: per-stage latency breakdown (Phase 3)
- [x] Citation verification: grounding check without extra LLM call (Phase 3)
- [x] DEV mode in UI: collapsible trace panel per answer
- [x] Automated test suite: 114 tests covering API, retrieval, quality, and latency
- [x] Similar Cases With Opposite Outcomes (`POST /contrasting-cases`) - contrastive semantic retrieval split by structured outcome (Phase 4)
- [x] LLM extraction backfill pipeline (`ingest/llm_extract_pipeline.py`) - fills structured fields the regex pipelines miss (compensation, reinstatement, aggravating/mitigating factors)
- [x] Intent-aware legal ranker (`rag/legal_ranker.py`) - four profiles (authority, example, tracker, statute), MRR +37% vs unranked baseline
- [x] Heuristic court planner (`rag/court_planner.py`) - domain signals to court/year filter with no LLM call, oracle-equivalent H@5(r)=1.00
- [x] Conditional BM25 (`rag/bm25_query.py`) - statute/section/citation queries only, OR-joined anchors, RRF-fused
- [x] 30-query retrieval benchmark suite with gold dataset, planner analysis, and locked production config

### Near-term (current hardware)

- [ ] legislation.govt.nz statute ingestion (NZLEG court)
- [ ] Synthetic Q&A generation from ingested corpus (knowledge distillation)
- [ ] Expand eval benchmark to 50+ questions
- [ ] Citation graph: resolve `to_document_id` for cross-linked decisions
- [ ] Legislation-to-Case Bridge: reverse lookup from Act + section to citing cases
- [ ] Always-Fresh Ingestion: cron job for NZLII new decisions

### Once AMD Ryzen AI Max+ 395 cluster is available (early 2027)

- [ ] LoRA fine-tuning on NZ legal Q&A dataset (128GB unified memory)
- [ ] Serve full Qwen3-72B with all layers on GPU
- [ ] Case citation graph with Neo4j or PostgreSQL recursive CTEs
- [ ] Similar Cases With Opposite Outcomes: contrastive retrieval for legal argument
- [ ] LLM-based penalty extraction for complex sentences

### Once Blackwell is affordable

- [ ] vLLM with speculative decoding for sub-second TTFT
- [ ] GPU-accelerated embedding at 512+ batch size for large corpus reindex

---

## License

MIT. Not legal advice. Always consult a qualified NZ lawyer.
