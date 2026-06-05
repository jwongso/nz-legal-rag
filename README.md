# nz-legal-rag

> **ARCHIVED** - This repository is no longer the source of the live services.
> Both `tenancy.localrun.ai` and `nz-legal-rag.localrun.ai` now run from
> [jwongso/astraea](https://github.com/jwongso/astraea). All future development
> happens there. This repo is kept for historical reference and ingestion scripts.

---

On-premise RAG pipeline for New Zealand legal research - covering 2.8 million+
decision chunks across 13 NZ courts and tribunals, with structured trackers comparable
to Westlaw NZ, running entirely on local hardware.

**Live public demo - NZ Tenancy Tribunal assistant:** https://tenancy.localrun.ai

Ask questions about your rights as a tenant or landlord. Answers are grounded in
31,000+ Tenancy Tribunal decisions and the Residential Tenancies Act 1986, with
cited sources for every response. No login required.

**Full legal research tool (private, all courts):** https://nz-legal-rag.localrun.ai

Both services now run the [Astraea framework](https://github.com/jwongso/astraea).

---

## Migration history

This project was the original monolithic NZ legal RAG implementation. The runtime
(API, routing, streaming, queue, legislation anchors) has been extracted and
generalised into the [Astraea framework](https://github.com/jwongso/astraea).

What moved to Astraea:
- `tenancy/app.py` -> `jurisdictions/nz_tenancy/` in Astraea
- `jurisdictions/nz_legal/` -> `jurisdictions/nz_legal/` in Astraea
- `rag/` core pipeline -> `core/` in Astraea
- `tenancy/static/` frontend -> `jurisdictions/nz_tenancy/static/` in Astraea

What stays here (still useful):
- `ingest/` - ingestion scripts for re-indexing court decisions into Qdrant
- `data/` - local data and configuration specific to this deployment
- Historical reference for the original architecture

---

## Architecture (current - Astraea)

```
tenancy.localrun.ai              nz-legal-rag.localrun.ai
        |                                    |
  astraea repo                         astraea repo
  jurisdictions/nz_tenancy/       jurisdictions/nz_legal/
        |                                    |
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

## Ingestion (still maintained here)

The ingestion scripts that populate the Qdrant collections live in `ingest/` and
are still the authoritative way to re-index or extend the corpus.

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
| Legal ranker | court hierarchy + intent-aware scoring |
| Reranker | bge-reranker-v2-m3 (optional, off by default) |

---

## Hardware

Current inference: llama.cpp with Qwen3-30B-A3B (MoE, 3B active params), Mac Mini M4 Pro 48GB.
Embeddings: nomic-embed-text-v1.5 via sentence-transformers.
Services managed via systemd --user, exposed via Cloudflare Tunnel.

---

## License

MIT. Not legal advice. Always consult a qualified NZ lawyer.
