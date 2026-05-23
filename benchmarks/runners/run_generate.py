"""Generation pass for multi-model benchmark.

Runs retrieval + context packing + LLM generation for each question and saves
results to a JSONL file. The judging pass (run_judge.py) is fully decoupled -
run generation with any model, then judge with any other model later.

TTFT is captured via streaming: first token timestamp minus request timestamp.

Output record per query (one JSON object per line):
  query_id          str
  query             str
  generator_model   str   model name as reported by /models
  retrieval_pipeline str  always "planner_filter_vector_legal"
  context_pack_mode str
  sources           list  [{case_id, title, court_name, date, url, score, text}]
  context_block     str   full formatted context sent to LLM
  answer            str
  no_context_flag   int   1 if answer says "not enough context"
  timing            dict  {ttft_ms, tokens_per_sec, completion_tokens, total_ms}

Run:
    python -m benchmarks.runners.run_generate
    python -m benchmarks.runners.run_generate --model Qwen3-8B-Q5_K_M --out gen_q5.jsonl
    python -m benchmarks.runners.run_generate --max-questions 5 --context-pack-mode baseline
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx
import psycopg2

import config
from rag.context_packer import pack
from rag.court_planner import plan_courts
from rag.embedder import Embedder
from rag.legal_ranker import QueryContext, rerank as legal_rerank
from rag.retriever import VectorStore

_QUESTIONS_PATH = Path("benchmarks/datasets/generator_questions.jsonl")
_REPORTS_DIR = Path("benchmarks/reports/generations")
_FETCH_K = 50

_NO_CONTEXT_RE = re.compile(
    r"not enough (information|context|detail)|"
    r"(do|does|did) not (have|contain|include|provide)|"
    r"cannot (answer|determine|find)|"
    r"no (relevant|specific|sufficient)|"
    r"unable to (answer|find|determine)",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "You are a legal research assistant specialising in New Zealand law.\n\n"
    "Rules:\n"
    "- Answer only from the provided context. Do not invent cases, statutes, or dates.\n"
    "- Always cite the source document for each claim using [N] notation.\n"
    "- If the context does not contain enough information to answer, say so clearly.\n"
    "- Use plain English. Avoid legal jargon unless quoting directly from a source.\n"
    "- Do not give legal advice. Remind the user to consult a qualified NZ lawyer.\n"
)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _get_point_ids(conn, courts: list[str], years: list[int] | None) -> list[str]:
    cur = conn.cursor()
    if years:
        cur.execute(
            "SELECT c.qdrant_point_id FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE d.court = ANY(%s) AND d.decision_date IS NOT NULL "
            "AND EXTRACT(YEAR FROM d.decision_date) = ANY(%s)",
            (courts, years),
        )
    else:
        cur.execute(
            "SELECT c.qdrant_point_id FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE d.court = ANY(%s)",
            (courts,),
        )
    return [r[0] for r in cur.fetchall()]


async def _retrieve(query: str, qvec: list[float], store: VectorStore, conn):
    plan = plan_courts(query)
    if plan.courts:
        ids = _get_point_ids(conn, plan.courts, plan.years)
        raw = store.search_within(qvec, ids, top_k=_FETCH_K) if ids \
              else store.search(qvec, top_k=_FETCH_K)
    else:
        raw = store.search(qvec, top_k=_FETCH_K)
    ctx = QueryContext.from_query(query)
    return legal_rerank(raw, ctx)


# ---------------------------------------------------------------------------
# LLM generation with streaming TTFT capture
# ---------------------------------------------------------------------------

async def _generate(
    query: str,
    context_block: str,
    sources: list[dict],
    model: str,
    base_url: str,
) -> tuple[str, dict]:
    """Stream the LLM response and capture TTFT + tok/s. Returns (answer, timing)."""
    source_header = "\n".join(
        f"  [{i+1}] {s.get('title','Unknown')} | {s.get('court_name','')} "
        f"| {s.get('date','')} | {s.get('url','')}"
        for i, s in enumerate(sources)
    )
    user_msg = (
        f"Source index:\n{source_header}\n\n"
        f"Context documents (numbered to match source index):\n\n{context_block}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        f"Answer using only the context above. Cite sources with [N] notation "
        f"matching the source index. After your answer, list every source you cited."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": config.LLM_MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t_start = time.monotonic()
    ttft_ms: float | None = None
    parts: list[str] = []
    completion_tokens = 0

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
                async with client.stream("POST", "/chat/completions", json=payload) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        if not raw_line.startswith("data: "):
                            continue
                        data = raw_line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if ttft_ms is None:
                                ttft_ms = (time.monotonic() - t_start) * 1000
                            parts.append(content)
                        usage = chunk.get("usage")
                        if usage and usage.get("completion_tokens"):
                            completion_tokens = usage["completion_tokens"]
            break
        except (httpx.RemoteProtocolError, httpx.TimeoutException,
                httpx.ConnectError, httpx.HTTPStatusError) as e:
            if attempt == 2:
                raise
            wait = 20 * (attempt + 1)
            print(f"  [retry {attempt + 1}/2 after {wait}s: {type(e).__name__}]")
            await asyncio.sleep(wait)
            t_start = time.monotonic()
            ttft_ms = None
            parts = []
            completion_tokens = 0

    total_ms = (time.monotonic() - t_start) * 1000
    answer = "".join(parts).strip()

    if completion_tokens == 0:
        completion_tokens = max(1, len(answer) // 4)

    tok_per_sec = completion_tokens / (total_ms / 1000) if total_ms > 0 else 0

    timing = {
        "ttft_ms": round(ttft_ms or 0, 1),
        "tokens_per_sec": round(tok_per_sec, 1),
        "completion_tokens": completion_tokens,
        "total_ms": round(total_ms, 1),
    }
    return answer, timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    questions = [
        json.loads(line)
        for line in Path(args.question_set).read_text().splitlines()
        if line.strip()
    ]
    if args.task_types:
        allowed = set(args.task_types.split(","))
        questions = [q for q in questions if q.get("task_type") in allowed]
    if args.max_questions:
        questions = questions[: args.max_questions]

    model = args.model or config.LLM_MODEL
    base_url = args.base_url or config.LLM_BASE_URL
    pack_mode = args.context_pack_mode

    # Resolve actual model ID from server
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
            resp = await client.get("/models")
            resp.raise_for_status()
        models = resp.json().get("data", [])
        server_model_id = models[0]["id"] if models else model
    except Exception:
        server_model_id = model

    out_path = Path(args.out) if args.out else (
        _REPORTS_DIR / f"gen_{model.replace('/', '_').replace(' ', '_')}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embed_model = args.embed_model or None
    collection = args.collection or None

    print(f"Generation pass")
    print(f"  model:        {model} (server reports: {server_model_id})")
    print(f"  base_url:     {base_url}")
    print(f"  pack_mode:    {pack_mode}")
    print(f"  embed_model:  {embed_model or 'default (nomic)'}")
    print(f"  collection:   {collection or 'default'}")
    print(f"  questions:    {len(questions)}")
    print(f"  out:          {out_path}")
    print()

    embedder = Embedder(model_name=embed_model)
    store = VectorStore(collection=collection)
    conn = psycopg2.connect(dbname="nz_legal")

    with out_path.open("w") as fout:
        for qi, q in enumerate(questions, 1):
            qid = q["id"]
            query = q["query"]
            print(f"[{qi}/{len(questions)}] {qid}")
            print(f"  Q: {query[:80]}")

            qvec = await embedder.embed(query)
            hits = await _retrieve(query, qvec, store, conn)
            packed = pack(hits, pack_mode)

            # Build sources list with chunk texts (needed by judge pass)
            best_by_case: dict[str, tuple[float, str]] = {}
            for h in hits:
                if h.case_id not in best_by_case or h.score > best_by_case[h.case_id][0]:
                    best_by_case[h.case_id] = (h.score, h.text)

            sources_rich = []
            for s in packed.sources:
                cid = s["case_id"]
                score, text = best_by_case.get(cid, (0.0, ""))
                sources_rich.append({**s, "score": round(score, 4), "text": text})

            answer, timing = await _generate(
                query, packed.context_block, packed.sources, model, base_url
            )

            no_ctx = 1 if _NO_CONTEXT_RE.search(answer) else 0

            print(
                f"  ttft={timing['ttft_ms']:.0f}ms  "
                f"tok/s={timing['tokens_per_sec']:.1f}  "
                f"tokens={timing['completion_tokens']}  "
                f"total={timing['total_ms']/1000:.1f}s"
                + ("  [no_ctx]" if no_ctx else "")
            )

            record = {
                "query_id": qid,
                "query": query,
                "task_type": q.get("task_type", ""),
                "difficulty": q.get("difficulty", "standard"),
                "generator_model": model,
                "server_model_id": server_model_id,
                "embed_model": embed_model or "nomic-ai/nomic-embed-text-v1.5",
                "qdrant_collection": collection or config.QDRANT_COLLECTION,
                "retrieval_pipeline": "planner_filter_vector_legal",
                "context_pack_mode": pack_mode,
                "sources": sources_rich,
                "context_block": packed.context_block,
                "answer": answer,
                "no_context_flag": no_ctx,
                "timing": timing,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    conn.close()
    print(f"\nDone. {len(questions)} records written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation pass for multi-model benchmark")
    parser.add_argument("--model", default=None,
                        help="LLM model name (default: config.LLM_MODEL)")
    parser.add_argument("--base-url", default=None,
                        help="LLM server base URL (default: config.LLM_BASE_URL)")
    parser.add_argument("--context-pack-mode", default="statute_first",
                        choices=["baseline", "metadata_rich", "full_chunk",
                                 "statute_first", "top3_only", "max2_per_doc"],
                        help="Context assembly format (default: statute_first)")
    parser.add_argument("--question-set", default=str(_QUESTIONS_PATH),
                        help="Path to JSONL question file")
    parser.add_argument("--task-types", default=None,
                        help="Comma-separated task types to include (e.g. general,statute)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--embed-model", default=None,
                        help="Embedding model name (default: nomic-ai/nomic-embed-text-v1.5)")
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection name (default: config.QDRANT_COLLECTION)")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
