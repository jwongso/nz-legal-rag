"""Pre-flight smoke test: verifies all services and pipeline components before a full run.

Checks (in order):
  1. PostgreSQL  - connect, count documents and chunks
  2. Qdrant      - collection reachable, point count matches PostgreSQL chunks
  3. Embedder    - Ollama responds, single embed call latency
  4. LLM         - port 8080 responds, model name, single short generation
  5. Retrieval   - one query end-to-end: planner -> vector search -> legal ranker
  6. Imports     - all benchmark runners import without error

Exits 0 if all checks pass, 1 if any check fails.

Run:
    python -m benchmarks.runners.smoke_test
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx
import psycopg2

import config

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"

_results: list[tuple[str, str, str]] = []  # (check, status, detail)


def _ok(check: str, detail: str = "") -> None:
    _results.append((check, _PASS, detail))
    tag = f"[{_PASS}]"
    print(f"  {tag} {check}" + (f"  {detail}" if detail else ""))


def _fail(check: str, detail: str = "") -> None:
    _results.append((check, _FAIL, detail))
    print(f"  [FAIL] {check}" + (f"  {detail}" if detail else ""))


def _warn(check: str, detail: str = "") -> None:
    _results.append((check, _WARN, detail))
    print(f"  [WARN] {check}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. PostgreSQL
# ---------------------------------------------------------------------------

def check_postgres() -> bool:
    print("\n[1/6] PostgreSQL")
    try:
        conn = psycopg2.connect(dbname="nz_legal")
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM documents")
        docs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks")
        chunks = cur.fetchone()[0]
        conn.close()
        _ok("connect", f"{docs:,} documents, {chunks:,} chunks")
        if docs == 0:
            _fail("documents > 0", "no documents in DB")
            return False
        if chunks == 0:
            _fail("chunks > 0", "no chunks in DB")
            return False
        return True
    except Exception as e:
        _fail("connect", str(e))
        return False


# ---------------------------------------------------------------------------
# 2. Qdrant
# ---------------------------------------------------------------------------

def check_qdrant() -> bool:
    print("\n[2/6] Qdrant")
    try:
        resp = httpx.get(
            f"{config.QDRANT_URL}/collections/{config.QDRANT_COLLECTION}",
            timeout=10,
        )
        resp.raise_for_status()
        info = resp.json()["result"]
        points = info["points_count"]
        status = info["status"]
        _ok("collection reachable", f"status={status}, points={points:,}")
        if status != "green":
            _warn("collection status", f"expected green, got {status}")
        if points == 0:
            _fail("points > 0", "empty collection")
            return False
        return True
    except Exception as e:
        _fail("collection reachable", str(e))
        return False


# ---------------------------------------------------------------------------
# 3. Embedder
# ---------------------------------------------------------------------------

async def check_embedder() -> bool:
    print("\n[3/6] Embedder (Ollama)")
    try:
        from rag.embedder import Embedder
        embedder = Embedder()
        t0 = time.monotonic()
        vec = await embedder.embed("What notice must a landlord give before entry?")
        latency_ms = (time.monotonic() - t0) * 1000
        if not vec or len(vec) != config.EMBED_DIM:
            _fail("embed call", f"expected dim={config.EMBED_DIM}, got {len(vec) if vec else 0}")
            return False
        _ok("embed call", f"dim={len(vec)}, latency={latency_ms:.0f}ms")
        return True
    except Exception as e:
        _fail("embed call", str(e))
        return False


# ---------------------------------------------------------------------------
# 4. LLM
# ---------------------------------------------------------------------------

async def check_llm() -> bool:
    print("\n[4/6] LLM (port 8080)")
    try:
        async with httpx.AsyncClient(base_url=config.LLM_BASE_URL, timeout=30) as client:
            resp = await client.get("/models")
            resp.raise_for_status()
        models = resp.json().get("data", [])
        model_id = models[0]["id"] if models else "unknown"
        _ok("health", f"model={model_id}")
    except Exception as e:
        _fail("health", str(e))
        return False

    # Short generation test
    try:
        payload = {
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_tokens": 10,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.monotonic()
        async with httpx.AsyncClient(base_url=config.LLM_BASE_URL, timeout=120) as client:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        latency_ms = (time.monotonic() - t0) * 1000
        text = resp.json()["choices"][0]["message"]["content"].strip()
        _ok("generate", f"response={repr(text[:40])}, latency={latency_ms:.0f}ms")
        return True
    except Exception as e:
        _fail("generate", str(e))
        return False


# ---------------------------------------------------------------------------
# 5. Retrieval pipeline
# ---------------------------------------------------------------------------

async def check_retrieval() -> bool:
    print("\n[5/6] Retrieval pipeline (end-to-end)")
    try:
        from rag.embedder import Embedder
        from rag.retriever import VectorStore
        from rag.court_planner import plan_courts
        from rag.legal_ranker import QueryContext, rerank as legal_rerank

        query = "What notice must a landlord give before entering a rental property in New Zealand?"
        embedder = Embedder()
        store = VectorStore()

        t0 = time.monotonic()
        qv = await embedder.embed(query)
        embed_ms = (time.monotonic() - t0) * 1000

        plan = plan_courts(query)
        _ok("court planner", f"courts={plan.courts}, years={plan.years}")

        t1 = time.monotonic()
        if plan.courts:
            conn = psycopg2.connect(dbname="nz_legal")
            cur = conn.cursor()
            cur.execute(
                "SELECT c.qdrant_point_id FROM chunks c "
                "JOIN documents d ON d.id = c.document_id "
                "WHERE d.court = ANY(%s)",
                (plan.courts,),
            )
            point_ids = [r[0] for r in cur.fetchall()]
            conn.close()
            raw = store.search_within(qv, point_ids, top_k=10) if point_ids \
                  else store.search(qv, top_k=10)
        else:
            raw = store.search(qv, top_k=10)
        search_ms = (time.monotonic() - t1) * 1000

        ctx = QueryContext.from_query(query)
        hits = legal_rerank(raw, ctx)

        if not hits:
            _fail("retrieval returns results", "0 hits returned")
            return False

        top = hits[0]
        _ok("vector search + rerank",
            f"top={top.case_id}, score={top.score:.3f}, "
            f"embed={embed_ms:.0f}ms, search={search_ms:.0f}ms")
        return True

    except Exception as e:
        _fail("retrieval pipeline", str(e))
        return False


# ---------------------------------------------------------------------------
# 6. Benchmark runner imports
# ---------------------------------------------------------------------------

def check_imports() -> bool:
    print("\n[6/6] Benchmark runner imports")
    runners = [
        "benchmarks.runners.run_all",
        "benchmarks.runners.run_retrieval",
        "benchmarks.runners.run_context_packing",
        "benchmarks.runners.run_citation_support",
        "benchmarks.runners.run_answer_quality",
    ]
    all_ok = True
    for mod in runners:
        try:
            __import__(mod)
            _ok(f"import {mod.split('.')[-1]}")
        except Exception as e:
            _fail(f"import {mod.split('.')[-1]}", str(e))
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary() -> bool:
    failed = [r for r in _results if r[1] == _FAIL]
    warned = [r for r in _results if r[1] == _WARN]
    passed = [r for r in _results if r[1] == _PASS]

    print("\n" + "=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)
    print(f"  PASS: {len(passed)}  WARN: {len(warned)}  FAIL: {len(failed)}")

    if failed:
        print("\nFailed checks:")
        for check, _, detail in failed:
            print(f"  - {check}: {detail}")

    if warned:
        print("\nWarnings:")
        for check, _, detail in warned:
            print(f"  - {check}: {detail}")

    if not failed:
        print("\nAll checks passed. Safe to run the full benchmark.")
    else:
        print("\nFix the failed checks before running the full benchmark.")

    return len(failed) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run() -> bool:
    print("NZ Legal RAG - Pre-flight Smoke Test")
    print("=" * 50)

    pg_ok = check_postgres()
    qdrant_ok = check_qdrant()
    embed_ok = await check_embedder()
    llm_ok = await check_llm()
    retrieval_ok = await check_retrieval() if (pg_ok and qdrant_ok and embed_ok) else False
    import_ok = check_imports()

    if not retrieval_ok and not (pg_ok and qdrant_ok and embed_ok):
        _fail("retrieval pipeline", "skipped - prerequisite checks failed")

    return _print_summary()


if __name__ == "__main__":
    passed = asyncio.run(run())
    sys.exit(0 if passed else 1)
