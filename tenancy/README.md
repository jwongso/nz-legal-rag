# Tenancy Tool - tenancy.localrun.ai

Free public RAG tool for New Zealand residential tenancy law. Answers questions
grounded in 31,240+ Tenancy Tribunal decisions (Ministry of Justice data) and
the live Residential Tenancies Act 1986 text from legislation.govt.nz.

This document covers the tenancy-specific layer only. For the underlying RAG
pipeline, corpus ingestion, and evaluation harness see the root `README.md`.

---

## What makes it different from the general RAG API

| Feature | General API | Tenancy tool |
|---|---|---|
| Corpus | All 13 courts, 2.8M+ chunks | NZTT only (MoJ 31,240 decisions + NZLII) |
| BM25 | Conditional (statute/citation queries) | Two-pass AND/OR with legal stopwords |
| Live legislation | Not included | RTA 1986 fetched via headless browser, 1h cache |
| Web verify | Not included | DDG search grounded by law section, Redis 7d cache |
| Compare mode | Not included | 4 retrieval strategies side-by-side |
| Public access | Token-gated (private) | Public API token |
| Request queue | Not included | Per-IP concurrency limit with fairness queue |

---

## Quality gates

Every production change must pass these checks before deployment:

| Gate | Target |
|---|---:|
| Retrieval H@5(r) | >= 0.95 |
| Correct RTA section retrieval | required for all statute/property-change tests |
| Forbidden anchor text | 0 occurrences of Schedule 1A / penalty-table rows |
| Citation source validity | 100% of citations point to retrieved context |
| Spurious citation rate | 0 |
| No-context refusal accuracy | 100% justified (no hallucinated answer when sources are absent) |
| p95 retrieval latency | tracked per release |
| p95 full answer latency | tracked per release |

Regression tests live in `benchmarks/datasets/retrieval_gold.jsonl`. Each entry
specifies expected legislation sections, acceptable case documents, and forbidden
context strings. New regression cases should be added whenever a retrieval bug
is diagnosed and fixed.

---

## Architecture

```
Browser client (SSE)
        |
        | POST /ask/stream  (single strategy)
        | POST /ask/stream/compare  (up to 4 strategies)
        |
+-------v-----------+
|  FastAPI / uvicorn |  tenancy/app.py
|  (tenancy-api.service) |
+-------+-----------+
        |
        +--- _rewrite_query()       [llama-server, query clarification]
        |
        +--- asyncio.gather():
        |       _pipeline.retrieve()     [Qdrant + BM25, NZTT filter]
        |       _retrieve_rta_anchor()   [leg_store vector search, RTA sections]
        |
        +--- _web_verify()              [DDG search via Playwright Firefox]
        |       |
        |       +-- Redis cache check   [key: sorted section IDs, TTL 7d]
        |       +-- BrowserSession.search_ddg()  [if cache miss]
        |       +-- Redis write
        |
        +--- _pipeline.generate_stream()  [llama-server, Qwen3-8B]
        |
        +--- _verify_sections()         [live RTA text, no LLM call]
        |
        SSE events -> browser
```

### Services

| Service | Description |
|---|---|
| `tenancy-api.service` | FastAPI app, port 8081, public via Cloudflare Tunnel |
| `llama-server.service` | Qwen3-8B-Q5_K_M, port 8080, 8GB GPU, ctx 5120, parallel 1 |
| `redis.service` | Web search cache, default port 6379 |

---

## Retrieval strategies

Four strategies are available, selectable in debug mode:

| Strategy | Description |
|---|---|
| `vector` | Qdrant semantic search with legal authority reranker |
| `vector_no_rerank` | Qdrant semantic search, raw score order |
| `mmr` | Maximal Marginal Relevance - diversified results |
| `bm25` | PostgreSQL full-text (two-pass AND/OR, legal stopwords) |

In single-strategy mode (default for public users), `vector` is always used.
In debug mode, one or more strategies can be selected. Selecting more than one
activates compare mode.

---

## Compare mode

When more than one strategy is selected, the request goes to `POST /ask/stream/compare`.
Each strategy runs as an independent `asyncio` task, all in parallel.

```
Question submitted
    |
    +-- _retrieve_rta_anchor()          [shared: same question, same RTA sections]
    |
    +-- _web_verify() task              [background: one search for all strategies]
    |
    +-- strategy task: vector           \
    +-- strategy task: vector_no_rerank  |  all parallel
    +-- strategy task: mmr_diverse       |
    +-- strategy task: bm25_keyword     /
    |
    await all strategy tasks
    await web_verify task
    emit web_results -> all_done
```

Key design decisions:
- Web search runs once per question, not once per strategy. The law sections
  relevant to a question do not change based on retrieval strategy.
- `_retrieve_rta_anchor()` is called once and the result shared across all strategy
  tasks (saves 4x repeated vector searches for RTA sections).
- The web verify task runs concurrently with all strategy tasks. Since LLM
  generation dominates latency (~5-30 seconds), the web search (~2-5 seconds)
  adds zero wall-clock time in practice.

---

## Web verify

Web verify adds live online sources to the LLM context and shows them to the user
with a LIVE or CACHED badge. It runs for all users on every request.

### Cache design

Cache key: sorted law section IDs from `leg_sources`, joined with `|`.

```
nz_tenancy:web_verify:NZLEG/RTA/s51|NZLEG/RTA/s60A
```

If no section IDs are found, the cache key falls back to the first 80 characters
of the question (lowercased). TTL is 7 days - appropriate for NZ legislation which
changes via Parliament (months) or regulations (weeks), not daily.

When `alwaysonline=True` (not exposed in UI, API-only), the Redis check is skipped
and a fresh search is always performed.

### Search implementation

Uses the shared `BrowserSession` (Playwright Firefox, headless) to fetch
`html.duckduckgo.com/html/`. BeautifulSoup parses `.result` elements and extracts
title, URL, and snippet. Ad redirects (`duckduckgo.com/y.js`) are filtered out.
DDG redirect URLs (`uddg=` parameter) are unwrapped to the real destination.

The query sent to DDG is:

```
NZ residential tenancy law {question[:120]}
```

Up to 3 results are returned. All degrade silently - if the browser is unavailable,
the search times out (20s limit), or DDG returns empty, the LLM context is unchanged
and no `web_results` event is emitted.

### Event flow

```
SSE event: web_results
  {
    "type": "web_results",
    "results": [{"title": "...", "url": "...", "body": "..."}],
    "cached": true | false
  }
```

In single mode: emitted before token events, rendered in the UI at the `done` event
(after the full answer, to ensure correct DOM order).

In compare mode: emitted before `all_done`, rendered spanning all strategy columns.

---

## SSE event reference

### Single mode (`POST /ask/stream`)

| Event type | When | Payload |
|---|---|---|
| `sources` | After retrieval | `sources`, `legislation` arrays |
| `confidence` | After retrieval | `level`, `message` |
| `web_results` | After web verify | `results`, `cached` |
| `debug` | Debug mode only | `strategy`, `retrieve_ms`, `scores`, `chunks` |
| `token` | During generation | `text` (streamed token) |
| `debug_done` | Debug mode only, after tokens | `generate_ms`, `total_ms` |
| `done` | After all tokens | (empty) |
| `verification` | After done | `sections` (live RTA section excerpts) |
| `error` | On failure | `message` |

### Compare mode (`POST /ask/stream/compare`)

| Event type | When | Payload |
|---|---|---|
| `col_start` | Strategy task begins | `strategy`, `col` index |
| `col_sources` | After retrieval | `strategy`, `sources`, `legislation` |
| `col_debug` | After retrieval | `strategy`, retrieval timing and scores |
| `col_think` | After `</think>` tag | `strategy`, `text` (thinking block) |
| `col_token` | During generation | `strategy`, `text` |
| `col_done` | After generation | `strategy`, `generate_ms`, `total_ms` |
| `col_error` | On strategy failure | `strategy`, `message` |
| `web_results` | After all strategies | `results`, `cached` |
| `all_done` | End of stream | (empty) |

---

## API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Liveness check |
| `GET` | `/token` | None | Returns public API token for browser clients |
| `POST` | `/ask/stream` | Token | SSE stream, single strategy |
| `POST` | `/ask/stream/compare` | Token | SSE stream, multi-strategy |
| `POST` | `/retrieve` | Token | Non-streaming context retrieval (benchmark runner) |
| `POST` | `/search` | Token | Web search via Playwright DDG (benchmark runner) |
| `POST` | `/feedback` | Token | Thumbs up/down feedback |
| `GET` | `/legislation/cases` | Token | Cases citing a given RTA section |
| `GET` | `/debug/ping` | Debug key | Debug mode activation check |

### AskRequest fields

```python
class AskRequest(BaseModel):
    question: str
    strategy: str = "vector"
    debug_key: str = ""
    irac: bool = False         # structure answer as IRAC
    verify: bool = True        # run web verify (default on)
    alwaysonline: bool = False # bypass Redis cache, always fetch fresh
```

---

## Configuration

Environment variables (set in `tenancy-api.service`):

| Variable | Purpose |
|---|---|
| `TENANCY_API_TOKEN` | Public API token served to browser clients at `/token` |
| `TENANCY_DEBUG_KEY` | Key to activate debug mode (Ctrl+Shift+D in browser) |
| `TENANCY_REDIS_URL` | Redis URL (default: `redis://localhost:6379`) |

The app degrades gracefully if Redis is unavailable - web verify still works, it
just skips the cache and always performs a live search.

### llama-server settings (Qwen3-8B-Q5_K_M, 8GB GPU)

```
--ctx-size 5120
--parallel 1        (sequential benchmark; no concurrency needed)
--cache-type-k q8_0
--cache-type-v q8_0
```

The q8_0 KV cache quantization halves VRAM usage (~400MB vs ~800MB at fp16 for
Qwen3's 36-layer model). This leaves enough headroom alongside the 5.5GB model
weights and compute buffers on an 8GB card.

---

## Frontend

Static files are in `tenancy/static/`. Both the JS and CSS use a version query
parameter for cache busting:

```html
<link rel="stylesheet" href="/static/style.css?v=12">
<script src="/static/app.js?v=30"></script>
```

Both values must be bumped together whenever the files change. The service must
be restarted for changes to take effect, and the browser needs a hard refresh
(Ctrl+Shift+R) to pick up the new versions.

### CSP compliance

The app serves a strict `style-src 'self'` Content Security Policy. Inline
`style="..."` attributes are blocked. Dynamic styles (score bar widths) are
applied via `element.style.width = ...` in JavaScript after innerHTML is set,
which is allowed by CSP.

### Debug mode

Activated in the browser with Ctrl+Shift+D and the debug key. Enables:
- Strategy selector (single or multi-strategy compare mode)
- Retrieval debug panel (scores, timing, strategy label)
- IRAC toggle
- Thinking mode toggle (compare mode only)

The web verify panel (LIVE/CACHED badge + source list) is always visible to all
users - it is not gated behind debug mode.

---

## Pipeline correctness notes

### RTA section extraction - heading-aware matching

`_extract_rta_section(full_text, num)` is the single function used by both
`_build_live_anchor()` and `_verify_sections()` to pull substantive text for a
given RTA section from the live legislation.govt.nz page.

The RTA page contains Schedule 1A, a penalty table with rows like:

```
42A(7)   Landlord failing to respond to written request seeking consent for fixtures   1,500
19(2)    Breaching duties on receipt of bond   1,500
```

A naive regex (`\n42A\b`) matches these table rows and feeds the model penalty
schedule data instead of legal text. The correct pattern requires a real section
heading - the section number at line start followed by whitespace and a capital
letter title, not by `(`:

```python
heading_re = re.compile(rf"(?m)^\s*{re.escape(num)}\s+[A-Z][^\n]{{3,}}")
```

Matches near "schedule", "infringement fee", "penalty", or "maximum amount" are
also rejected. The result must contain subsection markers `(1)`, `(2)` etc to
confirm it is substantive text.

### Property-change query routing

Queries that mention `plant`, `tree`, `garden`, `backyard`, `fixture`,
`alteration`, `install`, `renovate`, or similar terms trigger two adjustments:

**Legislation:** A synthetic embedding query targets s40/s42A/s42B directly,
because raw question embeddings ("planted trees backyard") have low cosine
similarity to section titles ("Consent for tenant's fixtures"). The desired
sections are prepended to leg_store results so they are always selected first.

**Cases:** A topical relevance gate keeps only chunks that mention
garden/land/premises AND alteration/consent/fixture. This filters irrelevant
cases (unlawful termination, business use of property) that match broadly on
"property" and "breach" but contribute no useful reasoning.

Both filters fall back to the original results if the gate removes everything.

### "Already done" prompt rule

The system prompt instructs the model to answer in two parts when the user
describes something already completed (past tense verbs): (1) the likely legal
position, and (2) concrete practical next steps to reduce risk. This prevents
answers that only say "you should have asked permission first."

### Regression test

`benchmarks/datasets/retrieval_gold.jsonl` entry `tenancy_planted_trees_backyard`
asserts:

- Legislation retrieved: `NZLEG/RTA/s42A`, `NZLEG/RTA/s42B` (not s19 or s16A)
- Cases acceptable: `NZTT-MOJ-5408751` (backyard structures), `NZTT-MOJ-4717450` (s40 garden)
- Forbidden in anchor: "Schedule 1A", "infringement fee", "19(2)", "42A(7)"
