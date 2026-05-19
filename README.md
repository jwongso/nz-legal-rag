# nz-legal-rag

On-premise retrieval-augmented generation pipeline for New Zealand legal research.
Indexes NZ court decisions and legislation into a local vector database, then answers
legal questions with citations - entirely on your own hardware.

Built for law firms, barristers, and legal practitioners who cannot send client
queries to a cloud AI due to legal professional privilege and the NZ Privacy Act 2020.

---

## Personal note

I am not a big fan of cloud-first architecture. I prefer on-premise, or at minimum
a hybrid approach where sensitive data never leaves your own hardware.

This is not just a technical preference. It is a risk management position.

Most cloud AI providers operate under US law and store data on US soil. The CLOUD Act
(2018) allows US law enforcement to compel American companies to hand over data stored
anywhere in the world, without notifying the data owner. The legal and political
environment around that is not stable. If the current US administration - or any future
one - decides to act aggressively on data access, there is very little a NZ law firm
or a NZ government agency can do to protect client data stored on AWS, Azure, or GCP.

This is not theoretical. In Microsoft Corp. v. United States, the Second Circuit
actually ruled in Microsoft's favour - finding that US warrants could not compel
production of data stored in Dublin, Ireland. Rather than accept that limit, Congress
passed the CLOUD Act in March 2018, which legislatively overrode the ruling and
explicitly extended US warrant authority to data stored anywhere in the world by
US-based companies. The Supreme Court then dismissed the case as moot on April 17,
2018. A NZ company has essentially no standing to challenge a US government data
demand in a US court. Your NZ lawyers cannot help you there.

The phrase "national security" is the catch-all that makes normal legal protections
disappear. We have watched the current US administration use that label - and similar
vague justifications - to impose trade tariffs, sanction companies, and rewrite
agreements overnight with no warning. The same logic applied to data access is not a
stretch. It is the same playbook.

If your sensitive business data sits on AWS, Azure, or GCP - regardless of which
region - you are operating under US jurisdiction whether you agreed to that or not.
"Australia region" is not a legal shield. It is a marketing label on US-company
infrastructure subject to US law.

You may think that is unlikely to affect you. Maybe. But "unlikely" is not the
standard a lawyer applies to their client's privileged communications. And it is not
the standard I apply when designing systems that handle sensitive data.

On-premise is not about being anti-cloud or anti-American. It is about knowing where
your data is, who controls it, and what jurisdiction it lives under. For NZ legal,
health, and financial data, the answer should be: here, us, and New Zealand.

---

## Why on-premise matters for NZ legal

| Constraint | Cloud AI risk | This project |
|---|---|---|
| Legal professional privilege | Client queries sent to third-party servers | All inference runs locally |
| Privacy Act 2020 | Processing personal data requires disclosure and consent | No data leaves the machine |
| Health Information Privacy Code | Clinical data cannot go to US cloud servers | Air-gapped deployment possible |
| Data residency | NZ firms may require NZ-hosted data | Runs on client hardware in NZ |

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
    +--------+----------+               +-----------+----------+
             |                                      |
             +------------------+-------------------+
                                |
                    +-----------v-----------+
                    |    RAG Pipeline       |
                    |  1. embed query       |
                    |  2. retrieve top-k   |
                    |  3. deduplicate      |
                    |  4. rerank           |
                    |  5. generate answer  |
                    +-----------+-----------+
                                |
          +---------------------+---------------------+
          |                                           |
+---------v----------+                    +-----------v----------+
|  Qdrant            |                    |  llama.cpp server    |
|  Vector DB         |                    |  (local inference)   |
|  :6333             |                    |  :8080               |
+--------------------+                    +----------------------+

  Embeddings (nomic-embed-text-v1.5) and reranker (bge-reranker-v2-m3)
  run in-process via sentence-transformers on CPU. No Ollama required.


Data ingestion:

  NZLII (nzlii.org) - public legal information repository
     |
     +-- HTML decisions: NZHC, NZERA
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
    |  pipeline.py      |
    |  embed + upsert   |
    +-------------------+
```

---

## Data sources

| Source | Content | Courts indexed |
|---|---|---|
| NZLII (nzlii.org) | NZ legal information institute - free public access | NZSC, NZCA, NZHC, NZERA, NZEmpC, NZTT |

NZLII hosts HTML decisions for most courts. Tenancy Tribunal (NZTT) decisions
are PDF-only on NZLII - the scraper fetches and extracts these via pypdf.

Current coverage: **2022-2024**, 100 decisions per court per year (~75k chunks across 6 courts).
All sources are publicly available. No proprietary data is required.

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

This starts Qdrant (port 6333). Ollama is no longer required - embeddings run
in-process via sentence-transformers.

### 2. Install dependencies

```bash
pip install -e ".[dev]"
pip install einops  # required by nomic-embed-text-v1.5
```

### 3. Start your llama.cpp inference server

```bash
llama-server --model /path/to/model.gguf --n-gpu-layers 20 --port 8080
```

Any OpenAI-compatible endpoint works (Ollama, vLLM, LM Studio).

### 4. Ingest NZ legal data

```bash
# Ingest Tenancy Tribunal decisions (PDF extraction handled automatically)
python -m ingest.pipeline --court NZTT --years 2022 2023 2024 --threads 16

# Ingest High Court decisions
python -m ingest.pipeline --court NZHC --years 2022 2023 2024 --threads 16

# Ingest Employment Relations Authority decisions
python -m ingest.pipeline --court NZERA --years 2022 2023 2024 --threads 16
```

`--threads` caps CPU usage from sentence-transformers embedding (default: 16).
`--max-per-year` limits decisions per year (default: 200).

### 5. Ask a question

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
| Embeddings | nomic-embed-text-v1.5 via sentence-transformers (in-process, no server) |
| LLM inference | llama.cpp (OpenAI-compatible) |
| Reranker | bge-reranker-v2-m3 via sentence-transformers (CPU, ~50ms per query) |
| MCP server | Python MCP SDK |
| REST API | FastAPI |
| Evaluation | RAGAS |
| Ingestion | subprocess curl + BeautifulSoup4 |

---

## Hardware

This project runs on a single machine with a consumer GPU. No Blackwell required.

Current inference: llama-server with Qwen3.6-35B-A3B-UD-Q5_K_M (Unsloth Dynamic 5-bit,
MTP-capable). 10 GPU layers, 4096 context. MTP disabled due to 8GB VRAM headroom.

Embeddings: nomic-embed-text-v1.5 via sentence-transformers, running in-process on CPU.
No Ollama server required. Thread count is capped at 16 during ingest to leave headroom
for the inference server. Not a bottleneck at the scales NZ legal corpus requires
(NZLII has ~50k decisions total).

---

## Tracker endpoints (Westlaw-comparable)

These structured-data endpoints compete directly with Westlaw NZ's premium tracker products,
delivered on-premise with no per-user SaaS fee.

| This project | Westlaw NZ equivalent |
|---|---|
| `POST /sentencing-tracker` | Sentencing Tracker |
| `POST /pg-tracker` | Personal Grievance Tracker |
| `POST /notable` (OSI + flags) | OSH Tracker / Resource Management Tracker |

### Sentencing Tracker (`POST /sentencing-tracker`)

Extracts structured criminal sentencing factors from NZHC, NZCA and NZSC decisions.

Payload fields extracted per chunk (stored under `sentencing.*` in Qdrant):

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

Backfill: `python -m ingest.sentencing_pipeline`

### Personal Grievance Tracker (`POST /pg-tracker`)

Extracts ERA / NZEmpC personal grievance outcome data.

Payload fields extracted per chunk (stored under `pg.*` in Qdrant):

| Field | Type | Description |
|---|---|---|
| `grievance_types` | keyword[] | unjustified_dismissal, constructive_dismissal, disadvantage, harassment, discrimination, unjustified_action |
| `reinstatement_ordered` | bool | True = ordered, False = declined |
| `contributory_conduct_pct` | float | Reduction % for employee's own conduct |
| `has_contributory_conduct` | bool | Discussed but % not parsed |

Filter by: grievance type, reinstatement outcome, contributory conduct range,
compensation range (uses `penalty.awarded_amount`), court, year.

UI: PG Tracker tab shows reinstatement rate, median compensation, median contributory conduct.

Backfill: `python -m ingest.pg_pipeline`

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

Backfill: `python -m ingest.counsel_pipeline`

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

Backfill: `python -m ingest.flag_pipeline`
Civil aggregation: `python -m ingest.recovery_agg`

---

## Roadmap

### Done
- [x] NZLII decision scraper (NZSC, NZCA, NZHC, NZERA, NZEmpC, NZTT) via subprocess curl (Cloudflare bypass)
- [x] PDF extraction for Tenancy Tribunal decisions (HTML pages are metadata wrappers)
- [x] Section-aware legal document chunker (120-word windows, 20-word minimum)
- [x] Qdrant ingestion pipeline with metadata filtering and deterministic UUID5 IDs
- [x] 182,000+ chunks indexed across 13 courts, 2022-2024
- [x] RAG pipeline with citation grounding
- [x] MCP server for Claude Code / Claude Desktop
- [x] FastAPI REST interface with web chat UI
- [x] RAGAS evaluation harness with NZ legal Q&A benchmark
- [x] Cross-encoder reranker (bge-reranker-v2-m3, CPU) - significant precision improvement
- [x] Case deduplication in retrieval - one chunk per case, diverse context window
- [x] Flag system (19 categories, regex-based, OR filter across all endpoints)
- [x] Penalty extraction: criminal OSI scale + civil awarded amount + recovery rate
- [x] Notable Cases endpoint (`POST /notable`) - filter by flags, OSI, compensation, counsel
- [x] Counsel / appearances extraction and Qdrant indexing
- [x] Sentencing Tracker (`POST /sentencing-tracker`) - starting point, final sentence, GP discount
- [x] Personal Grievance Tracker (`POST /pg-tracker`) - reinstatement, contributory conduct, compensation

### Near-term (current hardware)
- [ ] legislation.govt.nz statute ingestion
- [ ] Synthetic Q&A generation from ingested corpus (Claude API, knowledge distillation)
- [ ] Ingest progress tracking (crash-resume for long runs)
- [ ] Expand eval benchmark to 50+ questions
- [ ] Counterfactual "What if?" endpoint using extended thinking mode

### Once AMD Ryzen AI Max+ 395 cluster is available (early 2027)
- [ ] LoRA fine-tuning on NZ legal Q&A dataset (Node 2, 128GB unified memory)
- [ ] Serve full Qwen3-72B with all layers on GPU (Node 1)
- [ ] Dedicated Qdrant + Ollama on Node 3 (persistent storage node)
- [ ] Case citation graph: cross-link related decisions automatically
- [ ] LLM-based penalty extraction for complex sentences (improves recovery rate coverage)

### Once Blackwell is affordable
- [ ] vLLM with speculative decoding for sub-second TTFT
- [ ] GPU-accelerated embedding at 512+ batch size for large corpus reindex
- [ ] Neo4j citation graph alongside Qdrant for structured "cases citing X" queries

---

## License

MIT. Not legal advice. Always consult a qualified NZ lawyer.
