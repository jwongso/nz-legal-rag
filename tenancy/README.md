# Tenancy Tool - tenancy.localrun.ai

Free public RAG tool for New Zealand residential tenancy law. Answers questions
grounded in 31,240+ Tenancy Tribunal decisions (Ministry of Justice data) and
the live Residential Tenancies Act 1986 text from legislation.govt.nz.

This document covers the tenancy-specific layer only. For the underlying RAG
pipeline, corpus ingestion, and evaluation harness see the root `README.md`.

---

## Known limitations

- This tool is for information and research support only, not legal advice.
- Tribunal decisions are fact-specific; the outcome in a cited decision may not
  apply directly to a different set of facts.
- The tool only covers residential tenancy law under the RTA 1986. Issues
  involving employment, immigration, consumer law, or criminal matters are out
  of scope.
- Web verify is supplementary. It can fail silently or return generic guidance
  rather than the specific answer needed.
- AI-generated answers may be inaccurate, incomplete, or outdated. Always verify
  with Tenancy Services (0800 836 262) or a Community Law centre before acting.
- Do not enter private or confidential information into the public demo. Queries
  are logged for quality improvement.

---

## Current production configuration

| Component | Value |
|---|---|
| Generator | Qwen3-8B-Q5_K_M |
| Context size | 5120 tokens |
| Default retrieval | `vector` (semantic + legal authority reranker) |
| Statute routing | `rta_routes.py` - 8 intents, forced section injection |
| RTA extraction | Live heading-aware extractor (legislation.govt.nz, 1h cache) |
| Context packing | Statute-first (RTA anchor prepended before case chunks) |
| Web verify | Enabled for all users, DDG via Playwright, Redis 7-day cache |
| Debug mode | Key-gated, not exposed to public users |

---

## What makes it different from the general RAG API

| Feature | General API | Tenancy tool |
|---|---|---|
| Corpus | All 13 courts, 2.8M+ chunks | Dedicated NZTT corpus: MoJ 31,240 decisions + NZLII (separate from the multi-court benchmark corpus in the root README) |
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

## Tenancy-specific benchmark

The tenancy layer has behavior that the root pipeline evaluation does not cover:
live RTA anchors, property-change routing, web verify, compare mode, and public
access constraints. The table below defines what each test dimension measures and
what a failure indicates.

| Test dimension | What it measures | Failure mode caught |
|---|---|---|
| RTA section retrieval | Correct legislation sections surface for the query (e.g. s42A/s42B for fixture queries, not s16A/s19) | Lexicon injection not firing; embedding mismatch |
| Anchor correctness | Live anchor contains substantive section text, not Schedule 1A penalty-table rows | Heading-aware extraction regex regressed |
| NZTT case relevance | Retrieved tribunal chunks are on-point for the query topic | Prop-change gate not filtering; score threshold too low |
| Web verify relevance | DDG results address the same law section or issue as the question | Query too generic; stale Redis entry with wrong content |
| Citation correctness | Every [SN] citation in the answer maps to a retrieved source ID | Model hallucinating section numbers or case references |
| Practical-step quality | "Already done" questions include concrete next steps, not only hindsight | System prompt rule absent; model ignoring past-tense signal |
| Compare mode parity | All 4 strategies receive the same RTA anchor and web verify result | Shared anchor or web task not propagating correctly |

Each dimension maps to at least one entry in `retrieval_gold.jsonl`. The
planted-trees regression (`tenancy_planted_trees_backyard`) covers the first
four dimensions simultaneously.

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

### Statute routing layer

`tenancy/rta_routes.py` is a structured routing table that sits between query
rewrite and `leg_store` vector search. It solves a recurring problem: casual
question language has low cosine similarity to formal RTA section titles, so
the embedding search returns wrong or irrelevant sections.

#### Design

```
original query + rewritten query
  -> match_routes()       -- check against ROUTES table
  -> for each match:
       embed route.synthetic_query
       leg_store.search()
       filter to route.forced_sections
       prepend to vector results (deduplicated)
  -> allow_section()      -- suppress known false-positive sections
  -> deduplicate, take top 2
```

Both original and rewritten query are combined for matching. The original
often contains colloquial signals ("work and income", "broken oven"); the
rewrite carries formal legal terms ("bond lodgement", "repair obligations").

#### Routes

| Intent | Trigger terms (sample) | Forced sections |
|---|---|---|
| `wear_and_tear` | "fair wear and tear", "damage claim", "landlord charge" | s49A, s49B, s40 |
| `property_change` | "plant", "tree", "garden", "fixture", "alteration" | s40, s42A, s42B |
| `repairs_maintenance` | "not working", "broken", "mould", "hot water", "maintenance" | s45 |
| `agreement_form` | "tenancy agreement", "before signing", "copy of agreement" | s13A, s13B |
| `bond` | "bond lodgement", "work and income", "bond receipt", "proof of bond" | s18 |
| `landlord_entry` | "landlord entry", "inspection notice", "24 hours notice" | s48 |
| `termination_notice` | "evict", "90 days notice", "end the tenancy", "notice to leave" | s51 |
| `rent_payment` | "rent increase", "raise the rent", "weeks rent in advance" | s28, s28A |

**Important:** Section IDs must match what is actually indexed in Qdrant. The corpus
contains sections from the RTA itself plus its associated regulations, all under the
`NZLEG/RTA/` namespace. Verify any new `forced_sections` value exists with the correct
content before adding it to a route - some IDs resolve to regulation sections rather
than the main Act (e.g. `s13` is the Smoke Alarms Regulations, not "Form of tenancy
agreement"; the correct ID is `s13A`). Use the Qdrant scroll API to verify.

Each route has a `synthetic_query` string tuned to retrieve the forced
sections from the leg_store embedding index. This bridges the gap between
casual question language and formal section titles.

#### Section suppression guard

`LOW_PRIORITY_SECTIONS` lists sections that are false positives for many
common queries. `allow_section()` suppresses them unless the query
explicitly contains terms that make them relevant.

| Suppressed section | Suppressed unless query mentions |
|---|---|
| s16A (landlord must have NZ agent if overseas) | "landlord overseas", "out of new zealand", "21 consecutive days" |

This directly fixed a live failure: s16A was surfacing for a bond/agreement
formation question because "bond" embeddings have some overlap with agent
notification obligations.

#### Case relevance gate (property-change only)

For property-change queries, a separate case chunk filter (`_filter_prop_change_chunks`)
keeps only chunks that mention garden/land/premises AND alteration/consent/fixture.
This filters irrelevant cases that match broadly on "property" and "breach" but
contribute no useful reasoning. Falls back to the original results if the gate
removes everything.

#### Debug visibility

Debug mode is key-gated because it exposes internal prompts, retrieved context,
route decisions, and raw source snippets. It must not be enabled for public users.

The `context_debug` SSE event (always emitted, panel rendered in debug mode only)
includes a `statute_routing` block:

```json
{
  "statute_routing": {
    "triggered": true,
    "matched_routes": ["agreement_form", "bond"],
    "trigger_terms": ["tenancy agreement", "bond", "work and income"],
    "forced_sections": ["NZLEG/RTA/s13A", "NZLEG/RTA/s13B", "NZLEG/RTA/s18"],
    "suppressed_sections": [
      {
        "section": "NZLEG/RTA/s16A",
        "reason": "low_priority_section - query does not mention relevant terms"
      }
    ]
  }
}
```

This block is also stored in every thumbs-down `feedback_full.jsonl` entry
(for all users, not just debug mode), making routing failures immediately
diagnosable from the feedback log.

#### Adding a new route

Add one `StatuteRoute(...)` block to `ROUTES` in `tenancy/rta_routes.py`:

```python
StatuteRoute(
    intent=RouteIntent.MY_NEW_INTENT,
    include_any=("term1", "term2", ...),
    forced_sections=("NZLEG/RTA/sXX",),
    synthetic_query="formal legal language matching the target section...",
    exclude_any=(),   # optional: prevent false matches with other intents
    notes="What this route covers.",
),
```

Then add a regression entry to `benchmarks/datasets/retrieval_gold.jsonl`
with `expected_documents` and `must_not_include` for the old wrong sections.

### "Already done" prompt rule

The system prompt instructs the model to answer in two parts when the user
describes something already completed (past tense verbs): (1) the likely legal
position, and (2) concrete practical next steps to reduce risk. This prevents
answers that only say "you should have asked permission first."

### Live RTA extraction tests

`tests/test_rta_extractor.py` contains 22 deterministic unit tests for
`_extract_rta_section()`. No network calls, no Qdrant, no llama-server required.

Run with: `pytest tests/test_rta_extractor.py -v`

| Section | Must contain | Must not contain |
|---|---|---|
| s40 | "tenant", "clean" or "responsibilities" | "40(3A)(a)", "40(3A)(b)", penalty amounts |
| s40 | "land" or "garden" | "1,800", "4,000" |
| s42A | "consent", "fixture" or "improvement" | "42A(7)   Landlord", "1,500" |
| s42A | "unreasonably" | Penalty table amounts |
| s42B | "minor", "change" | "42B(3)", "42B(6)", "1,500" |
| s48 | "entry" or "enter", "landlord" | "48(5)", "infringement fee" |
| s48 | "24" or "notice" | Penalty content |
| All | subsection markers "(1)" | Anything from adjacent sections |

Edge cases covered: nonexistent section returns None, empty text returns None,
penalty-only text returns None, text without subsection markers returns None,
result length between 100 and 1800 chars.

### Regression tests

`benchmarks/datasets/retrieval_gold.jsonl` contains routing regression entries.
Each was added after a confirmed failure was observed and fixed.

| Entry | Route triggered | Must include | Must not include | Failure it catches |
|---|---|---|---|---|
| `tenancy_planted_trees_backyard` | `property_change` | s42A, s42B | s19, s16A | Fixture-consent sections not surfacing for garden queries |
| `tenancy_fair_wear_and_tear` | `wear_and_tear` | s49A, s49B | s66N | s49A missing; s66N (mitigation) surfacing instead |
| `tenancy_bond_proof_before_agreement` | `agreement_form`, `bond` | s13A, s18 | s16A, s13 | s16A (overseas agent) surfacing; s13 resolves to Smoke Alarms Regulations |
| `tenancy_landlord_repairs_maintenance` | `repairs_maintenance` | s45 | s42A, s42B | Fixture-consent sections surfacing for repair-obligation query |

---

## Feedback to regression workflow

When a user gives thumbs-down, a full artifact is silently posted to
`POST /feedback/full` and appended to `data/feedback_full.jsonl`. Each entry
contains the query, rewritten query, statute routing block, retrieved sources,
RTA anchor sections, and answer metadata. This makes routing and retrieval
failures diagnosable without reproducing the query.

Workflow:

1. User gives thumbs-down.
2. `feedback_full.jsonl` records `context_debug.statute_routing` (matched routes,
   trigger terms, forced sections, suppressed sections), chunk cards, and the
   full generated answer.
3. Developer inspects the entry - check `statute_routing.matched_routes`,
   `anchor.sections`, and `chunks` to identify the failure mode.
4. If the failure is real (wrong section surfaced, correct section missing),
   add a regression entry to `benchmarks/datasets/retrieval_gold.jsonl` with
   `expected_documents` and `must_not_include` set to the confirmed good/bad IDs.
5. Add or update the route in `tenancy/rta_routes.py`, or update
   `LOW_PRIORITY_SECTIONS` if a false-positive section needs suppression.
6. Run the benchmark runner against `retrieval_gold.jsonl` and confirm the new
   entry passes before deploying.
