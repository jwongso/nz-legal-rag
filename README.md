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
                    |  3. build context    |
                    |  4. generate answer  |
                    +-----------+-----------+
                                |
          +---------------------+---------------------+
          |                                           |
+---------v----------+                    +-----------v----------+
|  Qdrant            |                    |  llama.cpp server    |
|  Vector DB         |                    |  (local inference)   |
|  :6333             |                    |  :8080               |
+---------+----------+                    +----------------------+
          |
+---------v----------+
|  Ollama            |
|  nomic-embed-text  |
|  :11434            |
+--------------------+


Data ingestion:

  NZLII         Tenancy Tribunal    legislation.govt.nz
     |                 |                    |
     +--------+--------+                    |
              |                             |
    +---------v---------+       +-----------v----------+
    |  scraper.py       |       |  legislation.py      |
    |  fetch decisions  |       |  fetch statutes      |
    +--------+----------+       +-----------+----------+
             |                              |
    +--------v------------------------------v----------+
    |              chunker.py                          |
    |  section-aware split, preserve metadata          |
    +--------+-----------------------------------------+
             |
    +--------v----------+
    |  pipeline.py      |
    |  embed + upsert   |
    +-------------------+
```

---

## Data sources

| Source | Content | URL |
|---|---|---|
| NZLII | Court decisions (HC, CA, SC, ERA, NZTT) | nzlii.org |
| Tenancy Tribunal | Residential tenancy decisions | tenancy.govt.nz |
| legislation.govt.nz | Acts, regulations, amendments | legislation.govt.nz |
| Employment NZ | ERA and Employment Court decisions | employment.govt.nz |

All sources are publicly available. No proprietary data is required.

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
# Ingest Tenancy Tribunal decisions (2020-2024)
python -m ingest.pipeline --court NZTT --years 2020 2021 2022 2023 2024

# Ingest High Court decisions
python -m ingest.pipeline --court NZHC --years 2023 2024

# Ingest specific legislation
python -m ingest.pipeline --legislation "Privacy Act 2020" "Residential Tenancies Act 1986"
```

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

Current inference: llama-server with Qwen3.6-35B-A3B (MoE, only 3B active params per
token). 14 GPU layers, 4096 context. Handles legal Q&A well within these constraints.

Embeddings: nomic-embed-text via Ollama on CPU. Adequate for ingest; not a bottleneck
at the scales NZ legal corpus requires (NZLII has ~50k decisions total).

---

## Roadmap

### Done
- [x] NZLII decision scraper (HC, CA, SC, NZTT, ERA)
- [x] Section-aware legal document chunker
- [x] Qdrant ingestion pipeline with metadata filtering
- [x] RAG pipeline with citation grounding
- [x] MCP server for Claude Code / Claude Desktop
- [x] FastAPI REST interface
- [x] RAGAS evaluation harness with NZ legal Q&A benchmark

### Near-term (current hardware)
- [x] Cross-encoder reranker (bge-reranker-v2-m3, CPU) - significant precision improvement
- [x] Case deduplication in retrieval - one chunk per case, diverse context window
- [ ] legislation.govt.nz statute ingestion
- [ ] Synthetic Q&A generation from ingested corpus (Claude API, knowledge distillation)
- [ ] Ingest progress tracking (crash-resume for long runs)
- [ ] Expand eval benchmark to 50+ questions

### Once AMD Ryzen AI Max+ 395 cluster is available (early 2027)
- [ ] LoRA fine-tuning on NZ legal Q&A dataset (Node 2, 128GB unified memory)
- [ ] Serve full Qwen3-72B with all layers on GPU (Node 1)
- [ ] Dedicated Qdrant + Ollama on Node 3 (persistent storage node)
- [ ] Case citation graph: cross-link related decisions automatically

### Once Blackwell is affordable
- [ ] vLLM with speculative decoding for sub-second TTFT
- [ ] GPU-accelerated embedding at 512+ batch size for large corpus reindex
- [ ] Neo4j citation graph alongside Qdrant for structured "cases citing X" queries

---

## License

MIT. Not legal advice. Always consult a qualified NZ lawyer.
