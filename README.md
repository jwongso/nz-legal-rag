# nz-legal-rag

On-premise RAG pipeline for New Zealand legal research - covering 2.8 million+
decision chunks across 13 NZ courts and tribunals, with structured trackers comparable
to Westlaw NZ, running entirely on local hardware.

**Live public demo - NZ Tenancy Tribunal assistant:** https://tenancy.localrun.ai

Ask questions about your rights as a tenant or landlord. Answers are grounded in
31,000+ Tenancy Tribunal decisions and the Residential Tenancies Act 1986, with
cited sources for every response. No login required.

**Full legal research tool (private, all courts):** https://nz-legal-rag.localrun.ai

---

## Architecture

The API and runtime are now powered by [Astraea](https://github.com/jwongso/astraea) -
an open-source framework extracted from this project. This repo contains the NZ-specific
pipeline extension, ingestion scripts, and data.

```
tenancy.localrun.ai                nz-legal-rag.localrun.ai
        |                                    |
tenancy/app.py                  jurisdictions/nz_legal/app.py
        |                                    |
 NZLegalPipeline                      (Astraea core)
 (rag/nz_pipeline.py)            /search, /notable, /sentencing-tracker
  + legal_ranker                  /pg-tracker, /contrasting-cases
  + optional reranker                         |
        |                                     |
        +------------------+-----------------+
                           |
                 Astraea core (create_app)
                 SSE streaming, queue, statute routing,
                 legislation anchors, feedback, debug
                           |
               +-----------+-----------+
               |                       |
           Qdrant                  llama.cpp
         (vectors)              (local inference)
```

---

## Why on-premise matters for NZ legal

| Constraint | Cloud AI risk | This project |
|---|---|---|
| Legal professional privilege | Client queries sent to third-party servers | All inference runs locally |
| Privacy Act 2020 | Processing personal data requires disclosure and consent | No data leaves the machine |
| Data residency | NZ firms may require NZ-hosted data | Runs on client hardware in NZ |

---

## Data sources

| Source | Content | Courts indexed | Collection |
|---|---|---|---|
| NZLII (nzlii.org) | NZ legal information institute - free public access | NZSC, NZCA, NZHC, NZERA, NZEmpC, NZEnvC, NZACC, NZCorC, NZFC, NZLCDT, NZHRRT, NZREADT, NZTT | `nz_legal` |
| Ministry of Justice (forms.justice.govt.nz) | Tenancy Tribunal decisions via public Solr index - Crown copyright, non-commercial reuse with attribution permitted under NZGOAL | NZTT (2023-2026) | `nztt_moj` |

**Current coverage:**

| Court | Coverage |
|---|---|
| NZCA | 1985-2026 (full history) |
| NZHC | 2000-2021 |
| NZSC | 2000-2021 |
| NZERA | 2000-2021 |
| NZEmpC | 2000-2021 |
| NZEnvC, NZACC, NZCorC, NZFC, NZLCDT, NZHRRT, NZREADT | 1996-2021 |
| NZTT | 2022-2024 (NZLII) + 2023-2026 (MoJ, 31,000+ decisions) |

2.8M+ chunks indexed. All sources are publicly available. No proprietary data required.

---

## Tracker endpoints

These structured-data endpoints are specific to the `nz_legal` jurisdiction and are
registered via Astraea's `register_routes()` hook. They compete directly with Westlaw NZ's
premium tracker products, delivered on-premise with no per-user SaaS fee.

| Endpoint | Westlaw NZ equivalent |
|---|---|
| `POST /sentencing-tracker` | Sentencing Tracker |
| `POST /pg-tracker` | Personal Grievance Tracker |
| `POST /notable` | OSH Tracker / Resource Management Tracker |
| `POST /contrasting-cases` | (no direct equivalent - contrastive retrieval) |
| `GET /search` | Semantic search without generation |

### Sentencing Tracker (`POST /sentencing-tracker`)

Structured criminal sentencing factors from NZHC, NZCA, NZSC decisions.
Filter by: courts, year range, sentence type, starting point, final sentence, guilty plea.

### Personal Grievance Tracker (`POST /pg-tracker`)

ERA / NZEmpC personal grievance outcome data.
Filter by: grievance type, reinstatement, contributory conduct, compensation range, court, year.

### Similar Cases With Opposite Outcomes (`POST /contrasting-cases`)

Finds semantically similar cases where the court reached a different outcome.

```json
POST /contrasting-cases
{
  "query": "aggravated robbery weapon group offending youth",
  "domain": "criminal",
  "split_by": "sentence_type",
  "top_k": 5
}
```

Supported domains: `criminal` (split by `sentence_type` or `guilty_plea`),
`employment` (split by `reinstatement`).

---

## NZ-specific pipeline (`rag/nz_pipeline.py`)

`NZLegalPipeline` extends Astraea's `RAGPipeline` with two post-retrieval steps
specific to NZ legal corpora:

1. **Legal authority ranker** (`rag/legal_ranker.py`) - re-orders by court hierarchy
   (SC > CA > HC > tribunals) and legal signals (citation density, statute boost, recency).
   MRR +37% vs raw vector search baseline.

2. **Cross-encoder reranker** (`rag/reranker.py`) - optional BAAI/bge-reranker-v2-m3.
   Off by default (`RERANK_MODE=off`). Enable with `RERANK_MODE=rerank_5`.

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

## Ingestion

```bash
# Ingest NZCA decisions (2020-2026)
python -m ingest.pipeline --court NZCA --years 2020 2021 2022 2023 2024 2025 2026

# Ingest Tenancy Tribunal (MoJ Solr, 2023-2026)
python -m ingest.run_nztt_moj --years 2023 2024 2025 2026

# Structured data backfill
python -u -m ingest.sentencing_pipeline  # criminal sentencing factors
python -u -m ingest.pg_pipeline          # employment PG outcomes
python -u -m ingest.counsel_pipeline     # counsel/appearances
python -u -m ingest.flag_pipeline        # legal category flags
python -u -m ingest.recovery_agg         # civil recovery rate aggregation
```

---

## Stack

| Component | Technology |
|---|---|
| Framework | [Astraea](https://github.com/jwongso/astraea) |
| Vector database | Qdrant |
| Embeddings | nomic-embed-text-v1.5 via sentence-transformers (in-process) |
| LLM inference | llama.cpp (OpenAI-compatible, local) |
| Legal ranker | `rag/legal_ranker.py` - court hierarchy + intent-aware scoring |
| Reranker | bge-reranker-v2-m3 (optional, off by default) |
| MCP server | Python MCP SDK |

---

## Hardware

Current inference: llama.cpp with Qwen3-30B-A3B (MoE, 3B active params), Mac Mini M4 Pro 48GB.
Embeddings: nomic-embed-text-v1.5 via sentence-transformers.
Services managed via systemd --user, exposed via Cloudflare Tunnel.

---

## License

MIT. Not legal advice. Always consult a qualified NZ lawyer.
