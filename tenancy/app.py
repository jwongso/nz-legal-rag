"""
tenancy.localrun.ai - Free NZ tenancy law research tool.
Wraps the existing RAG pipeline with NZTT-only filtering,
a tenancy-focused system prompt, and a fair queue.
"""

import json
import os
import re
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from cachetools import TTLCache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

import config
from rag.generator import Generator
from rag.pipeline import RAGPipeline
from rag.retriever import VectorStore
from tenancy.queue import acquire, get_client_ip, queue_status, release

_TENANCY_SYSTEM_PROMPT = """You are a free legal research assistant helping New Zealand tenants understand \
their rights based on real Tenancy Tribunal decisions.

Rules:
- Answer only from the provided Tenancy Tribunal decisions. Do not invent cases, laws, or dates.
- Cite every claim with [SN] notation (e.g. [S1], [S2]) matching the source index. Never use other citation formats.
- Use plain, simple English that any tenant can understand. Explain legal terms when you use them.
- Be empathetic - users may be stressed about their housing situation.
- If the context does not contain enough information to answer confidently, say so clearly.
- Focus only on NZ residential tenancy matters: bonds, damage, rent arrears, notice periods, repairs, entry rights.
- End every answer with: "For advice on your specific situation, contact Community Law (free) at \
communitylaw.org.nz or Tenancy Services on 0800 836 262."
"""

_pipeline: RAGPipeline | None = None
_PUBLIC_TOKEN = os.getenv("TENANCY_API_TOKEN", "")

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
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
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
    global _pipeline
    _pipeline = RAGPipeline()
    _pipeline._generator = Generator(system_prompt=_TENANCY_SYSTEM_PROMPT)
    _pipeline._store = VectorStore(collection=config.QDRANT_TENANCY_COLLECTION)
    yield
    if _pipeline:
        await _pipeline.close()


_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="NZ Tenancy Help",
    description="Free NZ tenancy law research - powered by real Tribunal decisions",
    version="1.0.0",
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
_FEEDBACK_COOLDOWN_S = 30
_feedback_last: TTLCache = TTLCache(maxsize=2000, ttl=_FEEDBACK_COOLDOWN_S)


class AskRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    rating: int  # 1 = helpful, -1 = not helpful
    comment: str = ""


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

    async def _event_stream():
        try:
            context_texts, sources = await _pipeline.retrieve(question, top_k=5)
            if not context_texts:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant decisions found for this question.'})}\n\n"
                return

            public_sources = [{k: v for k, v in s.items() if k != "title"} for s in sources]
            yield f"data: {json.dumps({'type': 'sources', 'sources': public_sources})}\n\n"

            async for token in _pipeline._generator.generate_stream(question, context_texts, sources):
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
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
