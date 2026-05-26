"""
tenancy.localrun.ai - Free NZ tenancy law research tool.
Wraps the existing RAG pipeline with NZTT-only filtering,
a tenancy-focused system prompt, and a fair queue.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.generator import Generator
from rag.pipeline import RAGPipeline
from tenancy.queue import acquire, queue_status, release

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = RAGPipeline()
    _pipeline._generator = Generator(system_prompt=_TENANCY_SYSTEM_PROMPT)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", **queue_status()}


_FEEDBACK_LOG = Path("data/tenancy_feedback.jsonl")


class AskRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    rating: int  # 1 = helpful, -1 = not helpful
    comment: str = ""


@app.post("/ask")
async def ask(req: AskRequest, request: Request) -> dict:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail={"error": "Question must not be empty."})
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail={"error": "Question too long (max 2000 characters)."})

    ip = await acquire(request)
    try:
        result = await _pipeline.ask(
            question=question,
            courts=["NZTT"],
            top_k=5,
        )
        # Strip the appended Sources block from answer - rendered separately on frontend
        answer = result.answer
        idx = answer.rfind("\n\nSources:")
        if idx != -1:
            answer = answer[:idx].strip()

        return {
            "answer": answer,
            "sources": result.sources,
        }
    finally:
        release(ip)


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1.")
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
