"""
tenancy.localrun.ai - Free NZ tenancy law research tool.
Wraps the existing RAG pipeline with NZTT-only filtering,
a tenancy-focused system prompt, and a fair queue.
"""

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from cachetools import TTLCache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

import config
from rag.generator import Generator
from rag.live_verify.browser import BrowserSession
from rag.pipeline import RAGPipeline
from rag.retriever import VectorStore
from tenancy.queue import acquire, get_client_ip, queue_status, release

_TENANCY_SYSTEM_PROMPT = """You are a free legal research assistant helping New Zealand tenants understand \
their rights based on real Tenancy Tribunal decisions.

Rules:
- The governing legislation is the Residential Tenancies Act 1986 (RTA 1986). Never name any other Act. If the sources cite a section, use that section number; if not, do not invent one.
- Answer only from the provided Tenancy Tribunal decisions. Do not invent cases, laws, section numbers, or dates.
- Cite every claim with [SN] notation (e.g. [S1], [S2]) matching the source index. Never use other citation formats.
- Use plain, simple English that any tenant can understand. Explain legal terms when you use them.
- Be empathetic - users may be stressed about their housing situation.
- If the context does not contain enough information to answer confidently, say so clearly.
- Focus only on NZ residential tenancy matters: bonds, damage, rent arrears, notice periods, repairs, entry rights.
- If the user describes something they have already done (past tense: "I planted", "I built", "I installed", "I painted"), answer in two parts: (1) the likely legal position based only on the retrieved sources, and (2) concrete practical next steps to reduce risk now that it is done. Do not only say what should have been done beforehand.
- End every answer with: "For advice on your specific situation, contact Community Law (free) at \
communitylaw.org.nz or Tenancy Services on 0800 836 262."
"""

_pipeline: RAGPipeline | None = None
_leg_store: VectorStore | None = None  # nz_legal collection for RTA anchor chunks
_browser: BrowserSession | None = None
_PUBLIC_TOKEN = os.getenv("TENANCY_API_TOKEN", "")
_DEBUG_KEY = os.getenv("TENANCY_DEBUG_KEY", "")

_RTA_URL = "https://www.legislation.govt.nz/act/public/1986/120/en/latest/"
_rta_page_cache: tuple[str, float] | None = None
_RTA_CACHE_TTL = 3600  # seconds

_redis: aioredis.Redis | None = None
_WEB_CACHE_TTL = 604800  # 7 days - NZ law changes via Parliament (months) or regulations (weeks)
_WEB_CACHE_PREFIX = "nz_tenancy:web_verify:"

_ALLOWED_ORIGIN = "https://tenancy.localrun.ai"
_MAX_BODY_BYTES = 20_480  # 20 KB

# Common prompt injection patterns
_INJECTION_RE = re.compile(
    r"ignore\s+(previous|all|prior|above)\s+(instructions?|rules?|prompts?)"
    r"|forget\s+(previous|all|prior|above)\s+(instructions?|rules?|prompts?)"
    r"|you\s+are\s+now\s+(a\s+|an\s+)?"
    r"|act\s+as\s+(if\s+)?(you\s+are\s+)?"
    r"|pretend\s+(you|to\s+be)"
    r"|system\s*prompt\s*:"
    r"|<\s*system\s*>",
    re.IGNORECASE,
)


def _sanitize_question(text: str) -> str:
    """Strip control characters and detect obvious prompt injection attempts."""
    # Remove control chars except newline and tab
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in ("Cc", "Cf") or c in "\n\t"
    )
    if _INJECTION_RE.search(text):
        raise HTTPException(
            status_code=400,
            detail={"error": "Question contains content that cannot be processed."},
        )
    return text

_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://static.cloudflareinsights.com; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self' https://cloudflareinsights.com; "
    "frame-ancestors 'none';"
)


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = _CSP
        return response


class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_BYTES:
            return Response(
                content='{"detail": "Request body too large."}',
                status_code=413,
                media_type="application/json",
            )
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _leg_store, _browser, _redis
    _pipeline = RAGPipeline()
    _pipeline._generator = Generator(system_prompt=_TENANCY_SYSTEM_PROMPT)
    _pipeline._store = VectorStore(collection=config.QDRANT_TENANCY_COLLECTION)
    _leg_store = VectorStore(collection=config.QDRANT_COLLECTION)
    _browser = BrowserSession()
    await _browser.open()
    try:
        _redis = aioredis.from_url("redis://127.0.0.1:6379/0", decode_responses=True)
        await _redis.ping()
    except Exception:
        _redis = None  # degrade gracefully if Redis unavailable
    asyncio.create_task(_fetch_rta_cached())  # warm cache before first request
    yield
    if _pipeline:
        await _pipeline.close()
    if _browser:
        await _browser.close()
    if _redis:
        await _redis.aclose()


_REWRITE_SYSTEM = (
    "Rewrite the following as a concise formal legal question using standard "
    "NZ residential tenancy law terminology. Output only the rewritten question, "
    "no explanation, no preamble."
)


async def _rewrite_query(question: str) -> str:
    """Rephrase informal/colloquial questions into formal tenancy law language.

    Bridges the gap between casual user language (abbreviations, run-on sentences,
    informal phrasing) and formal Tribunal decision language used in the vector index.
    Falls back to the original question on any error.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{config.LLM_BASE_URL}/chat/completions",
                json={
                    "model": config.LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": _REWRITE_SYSTEM},
                        {"role": "user", "content": question},
                    ],
                    "max_tokens": 100,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            r.raise_for_status()
            rewritten = r.json()["choices"][0]["message"]["content"].strip()
            return rewritten if rewritten else question
    except Exception:
        return question


# Property-change query detection: terms that suggest a tenant altered the property.
_PROP_CHANGE_TERMS = frozenset({
    "plant", "planted", "planting", "tree", "trees", "shrub", "hedge",
    "garden", "backyard", "back yard", "lawn", "landscap", "outdoor",
    "built", "build", "building", "install", "installed", "installing",
    "fixture", "improve", "improvement", "alteration", "alter", "altered",
    "renovate", "renovation", "structure", "fence", "dug", "dig",
    "added", "modify", "modification", "paint", "painted",
})

# Synthetic query tuned to surface s40/s42A/s42B from the leg_store embedding index.
# Raw questions like "planted trees backyard" have low cosine similarity to those
# section titles; this query bridges the gap.
_PROP_CHANGE_SYNTHETIC_QUERY = (
    "tenant obligations alter improve add fixtures to land garden written consent landlord "
    "section 40 42A 42B residential tenancies act"
)

_PROP_CHANGE_DESIRED_SECTIONS = frozenset({"NZLEG/RTA/s40", "NZLEG/RTA/s42A", "NZLEG/RTA/s42B"})


def _detect_prop_change(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _PROP_CHANGE_TERMS)


# Fair wear and tear / tenant damage query detection.
# "wear and tear" has low cosine similarity to "s49A Tenant not liable" section titles,
# so the embedding index often surfaces s66N (mitigation) or s49B instead of the core
# rule. Force s49A and s49B to the front for these queries.
_WEAR_TEAR_TERMS = frozenset({
    "fair wear and tear", "wear and tear",
    "tenant damage", "damage claim",
    "landlord charge", "repair cost",
    "liable for damage", "damage to the",
    "s49a", "s49b",
})

_WEAR_TEAR_SYNTHETIC_QUERY = (
    "tenant not liable fair wear tear exception section 49A damage landlord cannot charge "
    "deterioration reasonable use natural forces residential tenancies act"
)

_WEAR_TEAR_DESIRED_SECTIONS = frozenset({"NZLEG/RTA/s49A", "NZLEG/RTA/s49B", "NZLEG/RTA/s40"})


def _detect_wear_tear(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _WEAR_TEAR_TERMS)


async def _retrieve_rta_anchor(question: str) -> tuple[str, list[dict]]:
    """Return (anchor_text, leg_sources) for the top 2 RTA sections most relevant to the question.

    anchor_text is injected before the numbered [S1]-[SN] block so the LLM reads the
    actual Act text and cites correct section numbers without hallucinating them.
    leg_sources is sent to the frontend for display alongside Tribunal decisions.
    Lexicon injections prepend desired sections for query types where raw embeddings
    produce poor section matches (property-change, wear-and-tear).
    """
    if _leg_store is None or _pipeline is None:
        return "", []
    try:
        vector = await _pipeline._embedder.embed(question)
        raw = _leg_store.search(vector, top_k=12, courts=["NZLEG"])

        if _detect_prop_change(question):
            synth_vector = await _pipeline._embedder.embed(_PROP_CHANGE_SYNTHETIC_QUERY)
            synth_raw = _leg_store.search(synth_vector, top_k=8, courts=["NZLEG"])
            existing_ids = {h.case_id for h in raw}
            desired_hits = [
                h for h in synth_raw
                if h.case_id in _PROP_CHANGE_DESIRED_SECTIONS and h.case_id not in existing_ids
            ]
            raw = desired_hits + raw  # prepend desired sections

        if _detect_wear_tear(question):
            synth_vector = await _pipeline._embedder.embed(_WEAR_TEAR_SYNTHETIC_QUERY)
            synth_raw = _leg_store.search(synth_vector, top_k=8, courts=["NZLEG"])
            existing_ids = {h.case_id for h in raw}
            desired_hits = [
                h for h in synth_raw
                if h.case_id in _WEAR_TEAR_DESIRED_SECTIONS and h.case_id not in existing_ids
            ]
            raw = desired_hits + raw  # prepend desired sections

        rta = [h for h in raw if h.case_id.startswith("NZLEG/RTA/")]
        seen: set[str] = set()
        hits = []
        for h in rta:
            if h.case_id not in seen:
                seen.add(h.case_id)
                hits.append(h)
            if len(hits) >= 2:
                break
        if not hits:
            return "", []
        lines = [
            "Relevant sections of the Residential Tenancies Act 1986 "
            "(legislative context - use for grounding section numbers only, "
            "do not cite with [SN] notation):"
        ]
        for h in hits:
            lines.append(f"\n{h.title}\n{h.text[:600]}")
        leg_sources = [
            {"case_id": h.case_id, "title": h.title, "url": h.url}
            for h in hits
        ]
        return "\n".join(lines), leg_sources
    except Exception:
        return "", []


_PROP_CHANGE_CHUNK_LOCATION = frozenset({
    "garden", "backyard", "back yard", "yard", "land", "premises", "section 40", "s40",
    "outdoor", "lawn", "shrub", "hedge", "outside",
})
_PROP_CHANGE_CHUNK_ACTION = frozenset({
    "fixture", "improvement", "alteration", "alter", "addition", "structure",
    "consent", "section 42", "s42", "renovate", "renovation", "install",
    "permission", "authoris", "authoriz", "permitted", "written consent",
})


def _prop_change_chunk_relevant(text: str) -> bool:
    t = text.lower()
    return (
        any(term in t for term in _PROP_CHANGE_CHUNK_LOCATION)
        and any(term in t for term in _PROP_CHANGE_CHUNK_ACTION)
    )


def _filter_prop_change_chunks(
    context_texts: list[str], sources: list[dict]
) -> tuple[list[str], list[dict], dict]:
    """Keep only chunks that mention garden/land AND alteration/consent.

    Returns (filtered_texts, filtered_sources, gate_stats). Falls back to the
    original lists if the filter would remove everything (gate_stats.fallback_used=True).
    """
    before = len(context_texts)
    pairs = [(t, s) for t, s in zip(context_texts, sources) if _prop_change_chunk_relevant(t)]
    if not pairs:
        return context_texts, sources, {
            "candidates_before": before, "survived": before,
            "fallback_used": True, "rejected": [],
        }
    rejected = [s.get("case_id", "") for t, s in zip(context_texts, sources) if not _prop_change_chunk_relevant(t)]
    return [p[0] for p in pairs], [p[1] for p in pairs], {
        "candidates_before": before, "survived": len(pairs),
        "fallback_used": False, "rejected": rejected[:6],
    }


def _web_cache_key(leg_sources: list[dict], fallback: str) -> str:
    """Build Redis cache key from legislation section IDs, falling back to query text."""
    ids = sorted({s.get("case_id", "") for s in leg_sources if s.get("case_id")})
    slug = "|".join(ids) if ids else fallback[:80].lower().strip()
    return f"{_WEB_CACHE_PREFIX}{slug}"


async def _web_verify(
    question: str,
    leg_sources: list[dict],
    alwaysonline: bool = False,
) -> tuple[str, list[dict], bool]:
    """Return (anchor_text, raw_results, from_cache).

    Checks Redis first (keyed by law section IDs). Falls back to live search
    on cache miss or when alwaysonline=True. Degrades silently on any error.
    """
    if _browser is None:
        return "", [], False

    cache_key = _web_cache_key(leg_sources, question)

    if not alwaysonline and _redis is not None:
        try:
            cached = await _redis.get(cache_key)
            if cached:
                payload = json.loads(cached)
                return payload["text"], payload["results"], True
        except Exception:
            pass

    query = f"NZ residential tenancy law {question[:120]}"
    try:
        results = await asyncio.wait_for(
            _browser.search_ddg(query, max_results=3),
            timeout=20,
        )
    except Exception as exc:
        logging.warning("_web_verify search failed: %s", exc)
        return "", [], False

    logging.info("_web_verify got %d results for: %s", len(results), query[:60])
    if not results:
        return "", [], False

    lines = ["Current online sources (use to verify recent law changes):"]
    for r in results:
        lines.append(f"- {r['title']} | {r['url']}\n  {r['body']}")
    text = "\n".join(lines)

    if _redis is not None:
        try:
            payload = json.dumps({"text": text, "results": results, "query": query})
            await _redis.setex(cache_key, _WEB_CACHE_TTL, payload)
        except Exception:
            pass

    return text, results, False


async def _fetch_rta_cached() -> str | None:
    """Return full RTA page text, fetching via headless browser and caching for 1 hour."""
    global _rta_page_cache
    if _rta_page_cache:
        cached_text, ts = _rta_page_cache
        if time.monotonic() - ts < _RTA_CACHE_TTL:
            return cached_text
    if _browser is None:
        return None
    try:
        text = await asyncio.wait_for(
            _browser.fetch_text(_RTA_URL, wait="networkidle"),
            timeout=20,
        )
        _rta_page_cache = (text, time.monotonic())
        return text
    except Exception:
        return None


# Detects the start of the next section heading or schedule boundary.
# Used to truncate extraction before bleeding into adjacent sections.
_NEXT_SECTION_RE = re.compile(r"(?m)^\s*(?:\d+[A-Z]*\s+[A-Z]|Schedule\b|---)")

# llama-server --ctx-size value (tokens). Shown in context budget meter.
_LLM_CTX_TOKENS = 5120

# Terms that must never appear in a live RTA anchor excerpt (penalty-table leakage).
_FORBIDDEN_ANCHOR_TERMS = [
    "Schedule 1A", "infringement fee", "42A(7)", "19(2)",
    "penalty notice", "unlawful acts and penalties",
]


def _anchor_debug_cards(
    leg_sources: list[dict], anchor_method: str, live_text: str | None
) -> list[dict]:
    """Build debug metadata cards for each legislation anchor section."""
    cards = []
    for s in leg_sources:
        doc_id = s.get("case_id", "")
        m = re.search(r'/s?(\d+[A-Z]?)$', doc_id, re.IGNORECASE)
        full_excerpt = (_extract_rta_section(live_text, m.group(1).upper()) or "") if (m and live_text) else ""
        forbidden = {t: (t.lower() in full_excerpt.lower()) for t in _FORBIDDEN_ANCHOR_TERMS}
        cards.append({
            "document_id": doc_id,
            "title": s.get("title", ""),
            "url": s.get("url", ""),
            "anchor_method": anchor_method,
            "tokens": len(full_excerpt) // 4,
            "preview": full_excerpt[:400],
            "forbidden_terms": forbidden,
        })
    return cards


def _chunk_debug_cards(
    context_texts: list[str], sources: list[dict], prop_change_triggered: bool
) -> list[dict]:
    """Build debug metadata cards for case chunk context.

    Includes full_text (exact text sent to model, up to 1500 chars) so the
    frontend can offer progressive disclosure without a round-trip.
    """
    cards = []
    for i, (text, src) in enumerate(zip(context_texts, sources)):
        sent = text[:1500]  # matches generator truncation
        card: dict = {
            "source_index": i + 1,
            "document_id": src.get("case_id", ""),
            "title": src.get("title", ""),
            "court": src.get("court_name", ""),
            "date": src.get("date", ""),
            "score": src.get("_score"),
            "tokens": len(sent) // 4,
            "preview": text[:300],
            "full_text": sent,
        }
        if prop_change_triggered:
            card["passed_gate"] = True
        cards.append(card)
    return cards


def _extract_rta_section(full_text: str, num: str) -> str | None:
    """Return the substantive text of RTA section `num` from the full page text.

    Discriminates real headings from Schedule 1A penalty table rows via the
    heading regex alone: a real heading has whitespace + capital letter title
    after the number ("42A  Consent for..."), while penalty rows have a
    parenthesised sub-number ("42A(7)   ...") which does not match.
    Extraction stops at the next section heading or schedule boundary so
    the result never bleeds into adjacent sections or the penalty table.
    Sections may legitimately cross-reference Schedule 1A inline.
    """
    heading_re = re.compile(rf"(?m)^\s*{re.escape(num)}\s+[A-Z][^\n]{{3,}}")
    for m in heading_re.finditer(full_text):
        candidate = full_text[m.start(): m.start() + 2500]
        # Skip past the matched heading itself before searching for the next one.
        nxt = _NEXT_SECTION_RE.search(candidate, 10)
        if nxt:
            candidate = candidate[:nxt.start()]
        if re.search(r"\(\d+\)", candidate):
            return candidate[:1800].strip()
    return None


def _build_live_anchor(full_text: str, leg_sources: list[dict]) -> str:
    """Build a legislation anchor from live RTA page text using section refs in leg_sources."""
    section_refs: list[str] = []
    seen: set[str] = set()
    for s in leg_sources:
        m = re.search(r'/s?(\d+[A-Z]?)$', s.get('case_id', ''), re.IGNORECASE)
        if m:
            key = m.group(1).upper()
            if key not in seen:
                seen.add(key)
                section_refs.append(m.group(1))
    if not section_refs:
        return ""
    lines = [
        "Relevant sections of the Residential Tenancies Act 1986 "
        "(current live text from legislation.govt.nz - use for grounding section numbers only, "
        "do not cite with [SN] notation):"
    ]
    for ref in section_refs[:3]:
        num = re.sub(r'^[sS]', '', ref)
        excerpt = _extract_rta_section(full_text, num)
        if excerpt:
            lines.append(f"\ns{num} {excerpt}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _extract_section_refs(text: str) -> list[str]:
    """Return unique s[0-9]+ refs from text, preserving first-seen order."""
    found = re.findall(r'\bs(\d+[A-Z]?)\b', text)
    seen: set[str] = set()
    result = []
    for ref in found:
        key = ref.upper()
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result


async def _verify_sections(answer: str, leg_sources: list[dict]) -> list[dict]:
    """Return live RTA excerpts for sections referenced in the answer and leg_sources."""
    leg_refs: list[str] = []
    seen: set[str] = set()
    for s in leg_sources:
        m = re.search(r'/s?(\d+[A-Z]?)$', s.get("case_id", ""), re.IGNORECASE)
        if m:
            key = m.group(1).upper()
            if key not in seen:
                seen.add(key)
                leg_refs.append(m.group(1))
    for ref in _extract_section_refs(answer):
        if ref.upper() not in seen:
            seen.add(ref.upper())
            leg_refs.append(ref)
    all_refs = leg_refs[:4]
    if not all_refs:
        return []
    full_text = await _fetch_rta_cached()
    if not full_text:
        return []
    results = []
    for ref in all_refs:
        num = re.sub(r'^[sS]', '', ref)
        excerpt = _extract_rta_section(full_text, num)
        if excerpt:
            results.append({"reference": f"s{num}", "excerpt": excerpt, "url": _RTA_URL})
    return results


_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="NZ Tenancy Help",
    description="Free NZ tenancy law research - powered by real Tribunal decisions",
    version="1.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(_BodySizeLimitMiddleware)
app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_ALLOWED_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", **queue_status()}


@app.get("/token")
async def token() -> dict:
    """Return the public API token for browser clients."""
    return {"token": _PUBLIC_TOKEN}


@app.get("/debug/ping")
async def debug_ping(request: Request) -> dict:
    """Validate the debug key without running a full request."""
    _check_token(request)
    key = request.headers.get("X-Debug-Key", "")
    if not (_DEBUG_KEY and key == _DEBUG_KEY):
        raise HTTPException(status_code=403, detail={"error": "Invalid debug key."})
    return {"ok": True}


@app.get("/legislation/cases")
async def legislation_cases(request: Request, section: str = "", limit: int = 8) -> dict:
    """Return NZTT decisions that mention a specific RTA section in their text."""
    _check_token(request)
    section = section.strip().lstrip("sS").strip()
    if not section.isdigit() and not re.match(r"^\d+[A-Z]?$", section, re.IGNORECASE):
        raise HTTPException(status_code=400, detail={"error": "Invalid section number."})
    limit = min(max(limit, 1), 20)

    import psycopg2
    pattern = rf"(?:s\.?\s*{re.escape(section)}|section\s+{re.escape(section)})(?:\s*\(|\b)"
    sql = """
        SELECT d.citation, d.source_url,
               to_char(d.decision_date, 'DD/MM/YYYY') AS date,
               count(*) AS mentions
        FROM chunks ch
        JOIN documents d ON ch.document_id = d.id
        WHERE d.court = 'NZTT'
          AND d.citation LIKE 'NZTT-MOJ-%%'
          AND ch.text ~* %s
        GROUP BY d.id, d.citation, d.source_url, d.decision_date
        ORDER BY mentions DESC, d.decision_date DESC
        LIMIT %s
    """
    try:
        conn = psycopg2.connect(dbname="nz_legal")
        cur = conn.cursor()
        cur.execute(sql, (pattern, limit))
        rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": str(exc)})

    return {
        "section": f"s{section}",
        "cases": [
            {"citation": r[0], "url": r[1], "date": r[2], "mentions": r[3]}
            for r in rows
        ],
    }


async def _check_llm() -> None:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{config.LLM_BASE_URL}/models")
            if r.status_code != 200:
                raise Exception()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={"error": "The AI model is currently loading. Please try again in 30 seconds."},
        )


def _check_token(request: Request) -> None:
    if not _PUBLIC_TOKEN:
        return
    if request.headers.get("X-API-Key") != _PUBLIC_TOKEN:
        raise HTTPException(
            status_code=401,
            detail={"error": "Ask the maintainer for a public API token."},
        )


_FEEDBACK_LOG = Path("data/tenancy_feedback.jsonl")
_FEEDBACK_FULL_LOG = Path("data/feedback_full.jsonl")
_FEEDBACK_COOLDOWN_S = 30
_feedback_last: TTLCache = TTLCache(maxsize=2000, ttl=_FEEDBACK_COOLDOWN_S)
_feedback_full_last: TTLCache = TTLCache(maxsize=4000, ttl=1)


_VALID_STRATEGIES = {"vector", "vector_no_legal", "mmr", "bm25"}

_IRAC_SUFFIX = (
    "\n\nStructure your answer as a legal research memo using IRAC format:\n"
    "**Issue:** State the specific legal question raised.\n"
    "**Rule:** State the legal principles that appear in the retrieved decisions, with [SN] citations.\n"
    "**Application:** Explain how those principles apply to these facts, with [SN] citations.\n"
    "**Conclusion:** Summarise what comparable decisions suggest about this situation."
)


def _confidence(scores: list[float], strategy: str) -> dict:
    n = len(scores)
    if n == 0:
        return {"level": "low", "chunks": 0, "message": "No relevant decisions found."}
    top = max(scores)
    if strategy == "bm25":
        level = "high" if n >= 4 else "medium" if n >= 2 else "low"
    else:
        level = "high" if top >= 0.82 and n >= 4 else "medium" if top >= 0.77 and n >= 2 else "low"
    messages = {
        "high": f"Found {n} directly relevant decisions.",
        "medium": f"Found {n} relevant decisions - review the cited sources carefully.",
        "low": f"Found only {n} loosely related decisions - verify independently before acting.",
    }
    return {"level": level, "chunks": n, "message": messages[level]}


class AskRequest(BaseModel):
    question: str
    debug_key: str = ""
    strategy: str = "vector"
    irac: bool = False
    verify: bool = True       # include web search results in context
    alwaysonline: bool = False  # bypass Redis cache, always fetch fresh


class CompareRequest(BaseModel):
    question: str
    debug_key: str
    strategies: list[str]
    thinking: bool = False


class FeedbackRequest(BaseModel):
    question: str
    rating: int  # 1 = helpful, -1 = not helpful
    comment: str = ""


class FeedbackFullRequest(BaseModel):
    question: str
    rating: int
    comment: str = ""
    strategy: str = ""
    irac: bool = False
    think: bool = False
    debug_mode: bool = False
    ts_start: str = ""
    ts_end: str = ""
    user_agent: str = ""
    viewport: dict = {}
    answer: str = ""
    sources: list = []
    legislation: list = []
    confidence: dict | None = None
    web_results: dict | None = None
    verification: list | None = None
    debug: dict | None = None
    debug_timing: dict | None = None
    context_debug: dict | None = None


@app.post("/ask")
async def ask(req: AskRequest, request: Request) -> dict:
    _check_token(request)
    await _check_llm()
    question = _sanitize_question(req.question.strip())
    if not question:
        raise HTTPException(status_code=400, detail={"error": "Question must not be empty."})
    if len(question) > 5000:
        raise HTTPException(status_code=400, detail={"error": "Question too long (max 5000 characters)."})

    ip = await acquire(request)
    try:
        result = await _pipeline.ask(
            question=question,
            top_k=5,
        )
        # Strip the appended Sources block from answer - rendered separately on frontend
        answer = result.answer
        idx = answer.rfind("\n\nSources:")
        if idx != -1:
            answer = answer[:idx].strip()

        sources = [
            {k: v for k, v in s.items() if k != "title"}
            for s in result.sources
        ]
        return {
            "answer": answer,
            "sources": sources,
        }
    finally:
        release(ip)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
    _check_token(request)
    await _check_llm()
    question = _sanitize_question(req.question.strip())
    if not question:
        raise HTTPException(status_code=400, detail={"error": "Question must not be empty."})
    if len(question) > 5000:
        raise HTTPException(status_code=400, detail={"error": "Question too long (max 5000 characters)."})

    ip = await acquire(request)

    debug_mode = bool(_DEBUG_KEY and req.debug_key == _DEBUG_KEY)
    strategy = req.strategy if debug_mode and req.strategy in _VALID_STRATEGIES else "vector"

    async def _event_stream():
        t0 = time.monotonic()
        try:
            # BM25 uses a different score scale - skip cosine min_score filter
            retrieve_kwargs: dict = {"top_k": 5, "strategy": strategy}
            if strategy != "bm25":
                retrieve_kwargs.update({"min_score": 0.75, "min_chunks": 2})
            else:
                retrieve_kwargs["min_chunks"] = 1
            retrieval_question = await _rewrite_query(question)

            # Vector retrieve + RTA anchor run in parallel (web search needs leg_sources first)
            (context_texts, sources), (anchor_vstore, leg_sources) = await asyncio.gather(
                _pipeline.retrieve(retrieval_question, **retrieve_kwargs),
                _retrieve_rta_anchor(retrieval_question),
            )
            t_retrieve = time.monotonic() - t0

            gate_stats: dict = {}
            if _detect_prop_change(retrieval_question):
                context_texts, sources, gate_stats = _filter_prop_change_chunks(context_texts, sources)

            web_text, web_results, from_cache = "", [], False
            if req.verify:
                web_text, web_results, from_cache = await _web_verify(
                    retrieval_question, leg_sources, alwaysonline=req.alwaysonline
                )

            if not context_texts:
                yield f"data: {json.dumps({'type': 'error', 'message': 'I could not find enough relevant Tenancy Tribunal decisions to answer this question reliably. This tool covers NZ residential tenancy matters only.'})}\n\n"
                return

            scores = [s["_score"] for s in sources]
            public_sources = [{k: v for k, v in s.items() if k not in ("title", "_score")} for s in sources]

            # Use live RTA text if the cache is already warm (zero latency); else fall back
            live_text = (
                _rta_page_cache[0]
                if _rta_page_cache and time.monotonic() - _rta_page_cache[1] < _RTA_CACHE_TTL
                else None
            )
            anchor = _build_live_anchor(live_text, leg_sources) if live_text and leg_sources else anchor_vstore
            if web_text:
                anchor = (anchor + "\n\n---\n\n" if anchor else "") + web_text

            yield f"data: {json.dumps({'type': 'sources', 'sources': public_sources, 'legislation': leg_sources})}\n\n"
            if web_results:
                yield f"data: {json.dumps({'type': 'web_results', 'results': web_results, 'cached': from_cache})}\n\n"
            yield f"data: {json.dumps({'type': 'confidence', **_confidence(scores, strategy)})}\n\n"

            if debug_mode:
                yield f"data: {json.dumps({'type': 'debug', 'strategy': strategy, 'retrieve_ms': round(t_retrieve * 1000), 'scores': scores, 'top': max(scores), 'min': min(scores), 'avg': round(sum(scores) / len(scores), 4), 'chunks': len(scores)})}\n\n"

            prop_change = bool(gate_stats)
            wear_tear = _detect_wear_tear(question)
            trigger_terms = [t for t in sorted(_PROP_CHANGE_TERMS) if t in retrieval_question.lower()][:6] if prop_change else []
            wear_tear_terms = [t for t in sorted(_WEAR_TEAR_TERMS) if t in question.lower()][:6] if wear_tear else []
            anchor_method = "live_heading_aware" if (live_text and leg_sources) else ("vector_store" if leg_sources else "none")
            chunk_cards = _chunk_debug_cards(context_texts, sources, prop_change)
            anchor_cards = _anchor_debug_cards(leg_sources, anchor_method, live_text)
            anchor_tokens = sum(c["tokens"] for c in anchor_cards)
            chunk_tokens = sum(c["tokens"] for c in chunk_cards)
            truncated = sum(1 for t in context_texts if len(t) > 1500)
            ctx_debug_payload = {
                "type": "context_debug",
                "original_query": question,
                "rewritten_query": retrieval_question,
                "rewrite_used": retrieval_question.strip() != question.strip(),
                "planner": {
                    "property_change_triggered": prop_change,
                    "wear_tear_triggered": wear_tear,
                    "trigger_terms": trigger_terms,
                    "wear_tear_terms": wear_tear_terms,
                    "forced_sections": sorted(_PROP_CHANGE_DESIRED_SECTIONS) if prop_change else (sorted(_WEAR_TEAR_DESIRED_SECTIONS) if wear_tear else []),
                    "gate": gate_stats if prop_change else None,
                },
                "anchor": {"method": anchor_method, "sections": anchor_cards},
                "chunks": chunk_cards,
                "budget": {
                    "total_tokens": anchor_tokens + chunk_tokens,
                    "ctx_limit": _LLM_CTX_TOKENS,
                    "anchor_tokens": anchor_tokens,
                    "chunk_tokens": chunk_tokens,
                    "sources_sent": len(context_texts),
                    "leg_sections": len(leg_sources),
                    "truncated_chunks": truncated,
                },
            }
            yield f"data: {json.dumps(ctx_debug_payload)}\n\n"

            gen_question = question + _IRAC_SUFFIX if req.irac else question
            t_gen = time.monotonic()
            full_answer: list[str] = []
            async for token in _pipeline._generator.generate_stream(gen_question, context_texts, sources, legislation_anchor=anchor or None):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            if debug_mode:
                yield f"data: {json.dumps({'type': 'debug_done', 'generate_ms': round((time.monotonic() - t_gen) * 1000), 'total_ms': round((time.monotonic() - t0) * 1000)})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            verification = await _verify_sections("".join(full_answer), leg_sources)
            if verification:
                yield f"data: {json.dumps({'type': 'verification', 'sections': verification})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            release(ip)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ask/stream/compare")
async def ask_stream_compare(req: CompareRequest, request: Request) -> StreamingResponse:
    """Debug-only multi-strategy comparison. Runs all strategies in parallel."""
    _check_token(request)
    if not (_DEBUG_KEY and req.debug_key == _DEBUG_KEY):
        raise HTTPException(status_code=403, detail={"error": "Debug key required."})
    await _check_llm()

    question = _sanitize_question(req.question.strip())
    if not question:
        raise HTTPException(status_code=400, detail={"error": "Question must not be empty."})
    if len(question) > 5000:
        raise HTTPException(status_code=400, detail={"error": "Question too long."})

    strategies = [s for s in req.strategies if s in _VALID_STRATEGIES][:4]
    if not strategies:
        raise HTTPException(status_code=400, detail={"error": "No valid strategies selected."})

    thinking = req.thinking
    ip = await acquire(request)

    async def _compare_stream():
        import time
        queue: asyncio.Queue = asyncio.Queue()

        # Shared RTA anchor + leg_sources for web verify (same question, run once)
        shared_anchor, shared_leg_sources = await _retrieve_rta_anchor(question)

        # Emit shared context debug (query + anchor info, shared across all strategy columns)
        _live_text_cmp = (
            _rta_page_cache[0]
            if _rta_page_cache and time.monotonic() - _rta_page_cache[1] < _RTA_CACHE_TTL
            else None
        )
        _cmp_prop_change = _detect_prop_change(question)
        _cmp_trigger_terms = [t for t in sorted(_PROP_CHANGE_TERMS) if t in question.lower()][:6] if _cmp_prop_change else []
        _cmp_anchor_method = "live_heading_aware" if (_live_text_cmp and shared_leg_sources) else ("vector_store" if shared_leg_sources else "none")
        yield f"data: {json.dumps({'type': 'shared_context_debug', 'original_query': question, 'rewrite_mode': 'disabled_in_compare', 'planner': {'property_change_triggered': _cmp_prop_change, 'trigger_terms': _cmp_trigger_terms, 'forced_sections': sorted(_PROP_CHANGE_DESIRED_SECTIONS) if _cmp_prop_change else []}, 'anchor': {'method': _cmp_anchor_method, 'sections': _anchor_debug_cards(shared_leg_sources, _cmp_anchor_method, _live_text_cmp)}})}\n\n"

        async def _run_strategy(strat: str, col_idx: int) -> None:
            t0 = time.monotonic()
            await queue.put({"type": "col_start", "strategy": strat, "col": col_idx})

            retrieve_kwargs: dict = {"top_k": 5, "strategy": strat}
            if strat != "bm25":
                retrieve_kwargs.update({"min_score": 0.75, "min_chunks": 2})
            else:
                retrieve_kwargs["min_chunks"] = 1

            try:
                context_texts, sources = await _pipeline.retrieve(question, **retrieve_kwargs)
            except Exception as exc:
                await queue.put({"type": "col_error", "strategy": strat, "message": str(exc)})
                return

            t_retrieve = time.monotonic() - t0

            if _detect_prop_change(question):
                context_texts, sources, _ = _filter_prop_change_chunks(context_texts, sources)

            if not context_texts:
                await queue.put({"type": "col_error", "strategy": strat, "message": "No relevant decisions found for this strategy."})
                return

            public_sources = [{k: v for k, v in s.items() if k not in ("title", "_score")} for s in sources]
            scores = [s["_score"] for s in sources]
            await queue.put({"type": "col_sources", "strategy": strat, "sources": public_sources, "legislation": shared_leg_sources})
            await queue.put({"type": "col_debug", "strategy": strat, "retrieve_ms": round(t_retrieve * 1000), "scores": scores, "top": max(scores), "min": min(scores), "avg": round(sum(scores) / len(scores), 4), "chunks": len(scores), "chunk_cards": _chunk_debug_cards(context_texts, sources, _detect_prop_change(question))})
            in_think = False
            think_parts: list[str] = []
            t_gen = time.monotonic()

            try:
                async for token in _pipeline._generator.generate_stream(
                    question, context_texts, sources, thinking=thinking,
                    legislation_anchor=shared_anchor or None,
                ):
                    if "<think>" in token:
                        in_think = True
                        token = token.replace("<think>", "")
                    if "</think>" in token:
                        before, _, after = token.partition("</think>")
                        if before:
                            think_parts.append(before)
                        in_think = False
                        if think_parts:
                            await queue.put({"type": "col_think", "strategy": strat, "text": "".join(think_parts)})
                            think_parts = []
                        token = after
                    if in_think:
                        think_parts.append(token)
                        continue
                    if token:
                        await queue.put({"type": "col_token", "strategy": strat, "text": token})
            except Exception as exc:
                await queue.put({"type": "col_error", "strategy": strat, "message": str(exc)})
                return

            await queue.put({"type": "col_done", "strategy": strat, "generate_ms": round((time.monotonic() - t_gen) * 1000), "total_ms": round((time.monotonic() - t0) * 1000)})

        # Run web verify in parallel with strategy tasks
        web_task = asyncio.create_task(_web_verify(question, shared_leg_sources))
        tasks = [asyncio.create_task(_run_strategy(s, i)) for i, s in enumerate(strategies)]
        finished = 0
        total = len(strategies)

        try:
            while finished < total:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("col_done", "col_error"):
                    finished += 1

            web_text, web_results, from_cache = await web_task
            if web_results:
                yield f"data: {json.dumps({'type': 'web_results', 'results': web_results, 'cached': from_cache})}\n\n"
            yield f"data: {json.dumps({'type': 'all_done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'col_error', 'strategy': 'all', 'message': str(exc)})}\n\n"
        finally:
            web_task.cancel()
            for t in tasks:
                t.cancel()
            await asyncio.gather(web_task, *tasks, return_exceptions=True)
            release(ip)

    return StreamingResponse(
        _compare_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class RetrieveRequest(BaseModel):
    question: str
    strategy: str = "vector"


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest, request: Request) -> dict:
    """Return RAG context without generating. Used by the agentic benchmark runner."""
    _check_token(request)
    question = _sanitize_question(req.question.strip())
    if not question:
        raise HTTPException(status_code=400, detail={"error": "Question must not be empty."})

    strategy = req.strategy if req.strategy in _VALID_STRATEGIES else "vector"
    retrieve_kwargs: dict = {"top_k": 5, "strategy": strategy}
    if strategy != "bm25":
        retrieve_kwargs.update({"min_score": 0.75, "min_chunks": 2})
    else:
        retrieve_kwargs["min_chunks"] = 1

    retrieval_question = await _rewrite_query(question)

    (context_texts, sources), (anchor_vstore, leg_sources) = await asyncio.gather(
        _pipeline.retrieve(retrieval_question, **retrieve_kwargs),
        _retrieve_rta_anchor(retrieval_question),
    )

    live_text = (
        _rta_page_cache[0]
        if _rta_page_cache and time.monotonic() - _rta_page_cache[1] < _RTA_CACHE_TTL
        else None
    )
    anchor = _build_live_anchor(live_text, leg_sources) if live_text and leg_sources else anchor_vstore
    public_sources = [{k: v for k, v in s.items() if k not in ("title", "_score")} for s in sources]

    return {
        "context_texts": context_texts,
        "sources": public_sources,
        "legislation": leg_sources,
        "anchor": anchor,
    }


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


@app.post("/search")
async def search(req: SearchRequest, request: Request) -> dict:
    """DuckDuckGo web search via the shared Playwright browser. Used by the agentic benchmark runner."""
    _check_token(request)
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail={"error": "Query must not be empty."})
    max_results = max(1, min(req.max_results, 10))
    if _browser is None:
        raise HTTPException(status_code=503, detail={"error": "Browser session not available."})
    try:
        results = await _browser.search_ddg(query, max_results=max_results)
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"error": f"Search failed: {exc}"})
    return {"results": results}


@app.post("/feedback")
async def feedback(req: FeedbackRequest, request: Request) -> dict:
    _check_token(request)
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1.")
    ip = get_client_ip(request)
    if ip in _feedback_last:
        raise HTTPException(status_code=429, detail="Please wait before submitting more feedback.")
    _feedback_last[ip] = 1
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": req.question[:500],
        "rating": req.rating,
        "comment": req.comment[:500],
    }
    _FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True}


@app.post("/feedback/full")
async def feedback_full(req: FeedbackFullRequest, request: Request) -> dict:
    _check_token(request)
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1.")
    ip = get_client_ip(request)
    if ip in _feedback_full_last:
        raise HTTPException(status_code=429, detail="Duplicate submission.")
    _feedback_full_last[ip] = 1
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rating": req.rating,
        "comment": req.comment[:1000],
        "strategy": req.strategy,
        "irac": req.irac,
        "think": req.think,
        "debug_mode": req.debug_mode,
        "ts_start": req.ts_start,
        "ts_end": req.ts_end,
        "user_agent": req.user_agent[:300],
        "viewport": req.viewport,
        "question": req.question[:500],
        "answer": req.answer[:12000],
        "sources": req.sources,
        "legislation": req.legislation,
        "confidence": req.confidence,
        "web_results": req.web_results,
        "verification": req.verification,
        "debug": req.debug,
        "debug_timing": req.debug_timing,
        "context_debug": req.context_debug,
    }
    _FEEDBACK_FULL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_FULL_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True}
