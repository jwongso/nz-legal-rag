# nz-legal-rag - Critical Review

An honest assessment of the architecture, code quality, and roadmap. Written to be useful,
not polite.

---

## Overall Impression

The project is well-scoped and the thesis is sound: NZ lawyers cannot ethically send
privileged queries to US cloud providers, so on-prem RAG is the right call. The code is
clean, the README is excellent, and the stack choices (Qdrant, Ollama, llama.cpp, FastAPI)
are pragmatic for a single-machine deployment.

That said, the project is at "solid prototype" stage. It works end-to-end but has gaps that
would hurt real-world accuracy and reliability. Here's what I'd change, ordered by impact.

---

## 1. Bugs that need fixing now

### 1.1 eval/ragas_eval.py contexts extraction is broken

Line 33:
```python
"contexts": [s["text"] for s in response.sources] if hasattr(response.sources[0], "text") else [],
```

`response.sources` is a `list[dict]`. Dicts don't have attributes, they have keys.
`hasattr(response.sources[0], "text")` is always `False`, so contexts is always `[]`.
This means your RAGAS faithfulness and context_precision scores are meaningless - they're
evaluating against zero context.

Additionally, `response.sources[0]` will raise `IndexError` when sources is empty.

Fix:
```python
"contexts": [s.get("text", "") for s in response.sources] if response.sources else [],
```

But wait - `response.sources` doesn't contain `"text"`. The sources dicts in `RAGPipeline.ask()`
are: `{case_id, title, court_name, date, url}`. The actual text is in `context_chunks`, but
that's not returned in `RAGResponse`. You need to either add `context_texts: list[str]` to
`RAGResponse` or reconstruct contexts from the hits. Without this, you literally cannot
evaluate faithfulness.

### 1.2 scraper.py double-fetches on cache miss

Lines 161-167:
```python
doc = await fetch_case(client, url, court, year)   # first GET
if doc:
    cache_file.write_text(
        (await client.get(url, headers=_HEADERS)).text,  # second GET
    )
```

Every uncached case hits NZLII twice. The second fetch could return different content
(or 429/503). Save the response from the first fetch instead.

### 1.3 scraper.py creates unused soup on cache hit

Line 155:
```python
soup = BeautifulSoup(html, "html.parser")  # created, never used
```

Dead code. `_parse_cached()` creates its own soup. Delete line 155.

### 1.4 config.py DATA_DIR.mkdir

Line 21:
```python
DATA_DIR.mkdir(exist_ok=True)
```

Should be `DATA_DIR.mkdir(parents=True, exist_ok=True)`. If someone sets
`DATA_DIR=/opt/nzlegal/data`, this fails because `/opt/nzlegal/` doesn't exist.

---

## 2. Architecture concerns

### 2.1 Embedding is embarrassingly slow

`embed_batch()` calls Ollama one text at a time, even within a batch:
```python
for text in batch:
    results.append(await self.embed(text))
```

With nomic-embed-text on CPU and 50k decisions at ~5 chunks each, that's ~250k sequential
HTTP round trips. Even at 50ms each, that's 3.5 hours just for embedding.

Fix: Use `asyncio.gather()` with a semaphore for concurrent requests within each batch.
Or better yet, Ollama now supports batch embedding via `/api/embed` (plural). Switch to
that - one request per batch of 16 texts, ~15x faster.

```python
async def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = await self._client.post(
            "/api/embed",
            json={"model": config.EMBED_MODEL, "input": batch},
        )
        resp.raise_for_status()
        results.extend(resp.json()["embeddings"])
    return results
```

### 2.2 No reranker - context precision will suffer

Vector search with nomic-embed-text at 768 dimensions is decent for recall but weak for
precision. Legal texts are full of similar boilerplate ("pursuant to section...",
"the Tribunal finds that...") that confuse cosine similarity.

The README roadmap says "cross-encoder reranker (CPU, bge-reranker-v2)". This should be
near-term priority #1, not deferred. bge-reranker-v2-m3 runs on CPU at ~50ms per query
with 5 candidates. The accuracy improvement is dramatic for legal text.

You can use sentence-transformers CrossEncoder with no GPU:
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
scores = reranker.predict([(query, chunk.text) for chunk in hits])
```

This fits current hardware. No AI Max+ needed.

### 2.3 No deduplication in retrieval

If a case is chunked into 5 pieces and 3 of them are in the top-5 results, you get 3
chunks from the same case and miss other relevant cases entirely. This is a known problem
with chunk-level vector search.

Fix: After retrieval, group by `case_id`, take the best-scoring chunk per case, then
re-rank. This gives diversity without losing relevance.

### 2.4 MCP search_nz_law returns only the answer text

```python
return response.answer  # line 67
```

Claude Code/Desktop cannot see the sources. This defeats half the value of RAG for a
legal tool - lawyers need to verify citations. Return the sources and scores as structured
text so the MCP client can present them.

### 2.5 MCP get_case creates a fresh VectorStore per call

```python
store = VectorStore()  # new Qdrant connection every call
```

Should reuse the pipeline's store instance. Same for `list_courts()`.

---

## 3. Chunking strategy

The section-aware chunker is good in principle but has issues:

### 3.1 Legal document structure is more complex than headings

NZ court decisions follow a pattern:
```
TITLE
PARTIES
REPRESENTATION
JUDGMENT OF Judge Name
[1] First paragraph (numbered)
[2] Second paragraph
...
FINDINGS
[24] The Tribunal finds that...
ORDER
[30] The respondent is ordered to...
```

The current heading detection (`r"^\d+\.?\s+[A-Z]"`) would match numbered paragraphs
like `1. The applicant seeks...` as headings when they're actually body text. This
over-segments.

Consider: numbered paragraphs `[N]` in NZ decisions are NOT section breaks. They're
just paragraph numbers. Only ALL-CAPS standalone lines (FINDINGS, ORDER, DISCUSSION)
are real section breaks.

### 3.2 Chunk overlap is in words, not tokens

With `CHUNK_SIZE=600` words and `CHUNK_OVERLAP=80` words, the actual token count varies
wildly. Legal text with citations and statute references can be 1.5-2x the word count in
tokens. Some chunks may exceed nomic-embed-text's 8192 token limit on long sections.

Not critical now, but worth adding a token count sanity check.

---

## 4. Generation quality

### 4.1 The system prompt is good but lacks guardrails for hallucinated citations

The prompt says "do not invent cases" but models still hallucinate case names. There's no
post-generation verification that cited cases actually exist in the retrieved context.

Cheap fix: after generation, extract citations from the answer (`[YYYY] XXXX NNN` pattern)
and verify each against the source list. Flag or strip any citation not grounded in context.

### 4.2 Source list is built but not in the prompt

`source_list` in generator.py is constructed (lines 45-48) but never injected into
`user_message`. It's only appended post-hoc if the model didn't include sources. The
model can't cite sources it can't see.

The context blocks do have `[1]`, `[2]` numbered prefixes, which is good, but the model
doesn't see the full citation metadata (case name, date, court). Add the source list to
the user prompt so the model can cite properly.

### 4.3 `enable_thinking: False` may hurt legal reasoning

Qwen3 models benefit from thinking mode for multi-step reasoning. Legal analysis often
requires: find relevant statute -> apply to facts -> consider exceptions -> conclude.
Disabling thinking trades accuracy for speed. Consider making this configurable, or
enabling it for the generation step (where accuracy matters most) while keeping it off
for parsing tasks.

---

## 5. Missing features that matter for credibility

### 5.1 legislation.govt.nz scraper doesn't exist

The README says it's a data source. The roadmap says "near-term". There's no code. For a
legal RAG tool, having Acts and Regulations is arguably more important than case law -
you can't answer "Can a landlord do X?" without the Residential Tenancies Act itself.

The site is well-structured XML/HTML. A scraper for the top 20 statutes would take a day
and dramatically improve answer quality for the most common questions.

### 5.2 No citation linking

Decisions cite other decisions. The scraper extracts citations (`_extract_citations`) but
they're stored as a flat list in the chunk payload and never used for retrieval. Building
even a simple "this case cites those cases" forward/reverse index would let you:
- Retrieve the authoritative case when a user asks about a principle
- Show "cited by N later cases" as a relevance signal
- Chain citations: "X was applied in Y, which was followed in Z"

You don't need Neo4j for this. A SQLite table `(citing_case_id, cited_case_id)` with a
simple BFS traversal would work. This is achievable on current hardware.

### 5.3 Evaluation benchmark is too small

8 questions is not a benchmark. It's a smoke test. For meaningful RAGAS scores, you need
at least 50 questions spanning:
- Tenancy law (most of your corpus)
- Employment law
- Privacy law
- Multi-hop questions ("What happened when Smith v Jones was appealed?")
- Questions with no answer in the corpus (to test refusal)

Consider using Claude API to generate 50-100 Q&A pairs from your ingested corpus, then
manually verify and curate them. This is the "synthetic Q&A generation" item in the
roadmap - it should be higher priority than it appears.

---

## 6. Robustness / operational concerns

### 6.1 No retry/backoff on any external call

- Embedding calls to Ollama: no retry
- LLM generation calls: no retry
- NZLII scraping: no retry (except implicit httpx timeout)
- Qdrant upsert: no retry

All of these can transiently fail. Use tenacity or a simple retry decorator with
exponential backoff. Especially important for the ingest pipeline, which can take hours.

### 6.2 No ingest progress tracking

If the pipeline crashes at decision 847/1200, you start from scratch. The cache_dir helps
(cached HTML won't be re-fetched), but re-embedding and re-upserting all cached documents
is wasteful.

Simple fix: write a `progress.json` to `data/raw/{court}/{year}/` tracking which case_ids
have been successfully upserted. On restart, skip those.

### 6.3 VectorStore.upsert uses random UUIDs

```python
id=str(uuid.uuid4()),
```

If you re-ingest the same case, you get duplicate chunks in Qdrant. Use a deterministic
ID based on case_id + chunk_index:
```python
id = f"{case_id}:{chunk_index}"
```

This makes upsert truly idempotent.

---

## 7. Hardware notes

### Current hardware

The stack fits well:
- Qwen3.6-35B-A3B (3B active) is a smart choice for the MoE efficiency
- nomic-embed-text on CPU is fine for NZ legal scale
- Qdrant is lightweight; 250k vectors with payload fits in <2GB RAM

### What actually needs more hardware

| Bottleneck | Current impact | What solves it |
|---|---|---|
| Embedding throughput | Ingest takes hours | Ollama batch API (free, today) |
| Reranker | Precision ~0.70 | bge-reranker-v2 on CPU (free, today) |
| Generation quality | Occasional hallucinations | 70B model needs >48GB VRAM |
| Fine-tuning | Not possible | Needs 128GB unified memory |
| Fast TTFT | 2-4s currently | Speculative decoding needs powerful GPU |

The first two don't need new hardware. They need code changes. I'd focus there before
waiting for AI Max+ or Blackwell.

### AI Max+ 395 cluster plan

The 3-node plan is sound:
- Node 1: inference (128GB unified = full 72B Q4_K_M on-GPU)
- Node 2: fine-tuning (LoRA with unsloth)
- Node 3: storage + embedding

But consider: do you need 3 nodes? The AI Max+ 395 has 128GB unified memory. You could
run Qdrant, Ollama embedding, and 70B inference on a single node and use Node 2 only
for fine-tuning jobs. Two nodes might be enough.

### Blackwell wishlist

vLLM with speculative decoding on Blackwell is the right long-term target. But realistically,
a single RTX 5090 (32GB) would already be transformative:
- Full 35B model with all layers on GPU
- GPU-accelerated embedding at 500+ docs/sec
- CrossEncoder reranker at <10ms
- Total cost: ~$3k NZD

That might arrive before an AI Max+ cluster and solve 80% of the same problems.

---

## 8. What I'd do next (priority order)

If I had a weekend:

1. **Fix eval contexts bug** - your metrics are currently meaningless
2. **Fix scraper double-fetch** - trivial, prevents NZLII rate limit issues
3. **Switch to Ollama batch embed API** - 10x ingest speedup, 15 min of work
4. **Add deterministic point IDs** - makes re-ingest safe
5. **Add bge-reranker-v2** - biggest accuracy win on current hardware

If I had a week:

6. **Build legislation.govt.nz scraper** - top 20 NZ statutes
7. **Expand eval to 50+ questions** - use Claude API to seed, then curate
8. **Add citation dedup in retrieval** - group by case_id before generation
9. **Add source list to generation prompt** - model needs citation metadata
10. **Add progress tracking to ingest** - crash-resume for long runs

---

## Summary

The project has good bones. The README is better than most production systems. The privacy
thesis is correct and well-argued. The code is clean and straightforward.

But the evaluation is broken, the embedding is needlessly slow, and the two highest-impact
accuracy improvements (reranker + legislation) don't need new hardware. Ship those before
chasing the AI Max+ dream.

---

*Written 2026-05-18*
