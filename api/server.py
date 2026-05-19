"""FastAPI REST interface for the NZ Legal RAG pipeline."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag.pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = RAGPipeline()
    yield
    if _pipeline:
        await _pipeline.close()


_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="NZ Legal RAG",
    description="On-premise retrieval-augmented generation for NZ law",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


class AskRequest(BaseModel):
    question: str
    courts: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    top_k: int = config.TOP_K
    flags: list[str] | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    scores: list[float]


class SearchResult(BaseModel):
    case_id: str
    title: str
    court_name: str
    date: str
    url: str
    text: str
    score: float


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Ask a question. Returns answer with citations from indexed NZ decisions."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    response = await _pipeline.ask(
        question=req.question,
        top_k=min(req.top_k, 20),
        courts=req.courts,
        year_from=req.year_from,
        year_to=req.year_to,
        flags=req.flags or None,
    )
    return AskResponse(
        question=response.question,
        answer=response.answer,
        sources=response.sources,
        scores=response.scores,
    )


@app.get("/search", response_model=list[SearchResult])
async def search(
    q: Annotated[str, Query(description="Search query")],
    courts: Annotated[list[str] | None, Query()] = None,
    year_from: int | None = None,
    year_to: int | None = None,
    top_k: int = config.TOP_K,
) -> list[SearchResult]:
    """Semantic search without generation. Returns raw matching chunks."""
    hits = await _pipeline.search_only(
        query=q,
        top_k=min(top_k, 20),
        courts=courts,
        year_from=year_from,
        year_to=year_to,
    )
    return [
        SearchResult(
            case_id=h.case_id,
            title=h.title,
            court_name=h.court_name,
            date=h.date,
            url=h.url,
            text=h.text,
            score=h.score,
        )
        for h in hits
    ]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "collection": config.QDRANT_COLLECTION}


class NotableRequest(BaseModel):
    flags: list[str] | None = None
    min_osi: float | None = None
    max_osi: float | None = None
    min_recovery: float | None = None
    max_recovery: float | None = None
    min_awarded: float | None = None
    max_awarded: float | None = None
    courts: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    limit: int = 30


class NotableResult(BaseModel):
    case_id: str
    title: str
    court_name: str
    date: str
    url: str
    flags: list[str]
    penalty: dict


@app.post("/notable", response_model=list[NotableResult])
async def notable(req: NotableRequest) -> list[NotableResult]:
    """Filter notable cases by flags and penalty severity. No question needed."""
    hits = _pipeline.search_notable(
        flags=req.flags or None,
        min_outcome_osi=req.min_osi,
        max_outcome_osi=req.max_osi,
        min_recovery_rate=req.min_recovery,
        max_recovery_rate=req.max_recovery,
        min_awarded=req.min_awarded,
        max_awarded=req.max_awarded,
        courts=req.courts or None,
        year_from=req.year_from,
        year_to=req.year_to,
        limit=min(req.limit, 100),
    )
    return [
        NotableResult(
            case_id=h.case_id,
            title=h.title,
            court_name=h.court_name,
            date=h.date,
            url=h.url,
            flags=h.payload.get("flags") or [],
            penalty=h.payload.get("penalty") or {},
        )
        for h in hits
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
