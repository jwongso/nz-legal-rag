"""tenancy.localrun.ai - powered by Astraea framework.

Uses Astraea's create_app() for all routing, queue, SSE streaming, statute
injection, legislation caching, and security middleware. NZLegalPipeline
extends the base with NZ-specific legal authority ranker and optional
cross-encoder reranker.

Question logging is appended here as middleware so it survives framework
upgrades without any changes to Astraea core.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request
from core.api import create_app
from jurisdictions.nz_tenancy import jurisdiction
from rag.nz_pipeline import NZLegalPipeline

_QUESTION_LOG = Path("data/question_log.jsonl")
log = logging.getLogger(__name__)

app = create_app(
    jurisdiction,
    pipeline_factory=NZLegalPipeline,
    static_dir=Path(__file__).parent / "static",
)


@app.middleware("http")
async def _log_questions(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/ask/stream":
        body = await request.body()
        try:
            data = json.loads(body)
            q = data.get("question", "")
            if q:
                _QUESTION_LOG.parent.mkdir(parents=True, exist_ok=True)
                with _QUESTION_LOG.open("a") as _f:
                    _f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "q": q}) + "\n")
        except Exception:
            pass
        request._body = body
    return await call_next(request)
