"""Performance and latency tests.

These tests verify that each pipeline stage completes within acceptable
time budgets. They catch regressions like an N+1 query bug or an
accidentally synchronous blocking call in an async path.

Thresholds are generous to avoid flakiness on loaded hardware.

Run:
    pytest tests/test_perf.py -v
"""

import asyncio
import time

import pytest

from db.filter import FilterParams, bm25_search, get_point_ids
from rag.embedder import Embedder
from rag.retriever import VectorStore


# Latency budgets (seconds)
BUDGET_SQL_FILTER = 0.5      # simple SQL pre-filter
BUDGET_BM25 = 1.0            # BM25 full-text search
BUDGET_EMBED = 3.0           # embedding a short query
BUDGET_QDRANT_SEARCH = 5.0   # Qdrant search (may need to load vectors)
BUDGET_HYBRID_FULL = 8.0     # SQL filter + Qdrant search_within combined
BUDGET_HEALTH_API = 0.5      # /health endpoint


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def embedder(event_loop):
    emb = Embedder()
    yield emb
    event_loop.run_until_complete(emb.close())


@pytest.fixture(scope="module")
def store():
    return VectorStore()


@pytest.fixture(scope="module")
def embed_fn(embedder, event_loop):
    def _embed(text: str):
        return event_loop.run_until_complete(embedder.embed(text))
    return _embed


# Pre-warm the embedder so first-call loading doesn't count
@pytest.fixture(scope="module", autouse=True)
def warmup(embed_fn):
    embed_fn("warmup")


# ---------------------------------------------------------------------------
# SQL filter latency
# ---------------------------------------------------------------------------

def test_sql_filter_no_params_latency():
    t = time.monotonic()
    ids = get_point_ids(FilterParams())
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_SQL_FILTER, (
        f"get_point_ids() took {elapsed:.3f}s (budget: {BUDGET_SQL_FILTER}s)"
    )
    assert len(ids) > 0


def test_sql_filter_court_latency():
    t = time.monotonic()
    ids = get_point_ids(FilterParams(courts=["NZCA"]))
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_SQL_FILTER, (
        f"get_point_ids(courts=NZCA) took {elapsed:.3f}s"
    )
    assert len(ids) > 0


def test_sql_filter_year_range_latency():
    t = time.monotonic()
    ids = get_point_ids(FilterParams(year_from=2020, year_to=2024))
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_SQL_FILTER, (
        f"get_point_ids(year_from=2020) took {elapsed:.3f}s"
    )


def test_sql_filter_combined_latency():
    t = time.monotonic()
    ids = get_point_ids(FilterParams(courts=["NZHC", "NZCA"], year_from=2022, year_to=2024))
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_SQL_FILTER, (
        f"get_point_ids(courts+year) took {elapsed:.3f}s"
    )


# ---------------------------------------------------------------------------
# BM25 latency
# ---------------------------------------------------------------------------

def test_bm25_latency_basic():
    t = time.monotonic()
    bm25_search("unjustified dismissal redundancy", limit=10)
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_BM25, f"bm25_search took {elapsed:.3f}s"


def test_bm25_latency_with_filter():
    t = time.monotonic()
    bm25_search("dismissal", courts=["NZERA"], limit=10)
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_BM25, f"bm25_search(courts=NZERA) took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Embedding latency
# ---------------------------------------------------------------------------

def test_embed_latency_short_query(embed_fn):
    t = time.monotonic()
    embed_fn("unjustified dismissal")
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_EMBED, f"embed() took {elapsed:.3f}s"


def test_embed_latency_long_query(embed_fn):
    long_query = (
        "What are the key factors courts consider when assessing "
        "whether a dismissal is unjustified under the Employment Relations Act 2000, "
        "particularly in relation to procedural fairness and substantive grounds?"
    )
    t = time.monotonic()
    embed_fn(long_query)
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_EMBED, f"embed(long query) took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Qdrant search latency
# ---------------------------------------------------------------------------

def test_qdrant_search_latency(embed_fn, store):
    vec = embed_fn("employment personal grievance")
    t = time.monotonic()
    store.search(vec, top_k=10)
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_QDRANT_SEARCH, f"VectorStore.search() took {elapsed:.3f}s"


def test_qdrant_search_with_filter_latency(embed_fn, store):
    vec = embed_fn("sentencing robbery")
    t = time.monotonic()
    store.search(vec, top_k=10, courts=["NZHC"])
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_QDRANT_SEARCH, f"VectorStore.search(court filter) took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Hybrid retrieval latency (SQL + Qdrant together)
# ---------------------------------------------------------------------------

def test_hybrid_latency_court_filter(embed_fn, store):
    t = time.monotonic()
    ids = get_point_ids(FilterParams(courts=["NZCA"]))
    vec = embed_fn("criminal appeal")
    store.search_within(vec, ids, top_k=10)
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_HYBRID_FULL, f"SQL+Qdrant hybrid took {elapsed:.3f}s"


def test_hybrid_latency_full_filter(embed_fn, store):
    t = time.monotonic()
    ids = get_point_ids(FilterParams(courts=["NZHC", "NZCA"], year_from=2020, year_to=2024))
    vec = embed_fn("sentencing starting point discount")
    store.search_within(vec, ids, top_k=10)
    elapsed = time.monotonic() - t
    assert elapsed < BUDGET_HYBRID_FULL, f"Full hybrid latency took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# API health endpoint latency
# ---------------------------------------------------------------------------

def test_health_endpoint_latency():
    from fastapi.testclient import TestClient
    from api.server import app
    with TestClient(app) as client:
        t = time.monotonic()
        r = client.get("/health")
        elapsed = time.monotonic() - t
    assert r.status_code == 200
    assert elapsed < BUDGET_HEALTH_API, f"/health took {elapsed:.3f}s"
