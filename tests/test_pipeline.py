"""Pipeline integration tests - no LLM required.

Tests the full retrieval path: embed -> SQL filter -> Qdrant search -> dedup.
Requires Qdrant + PostgreSQL populated.

Run:
    pytest tests/test_pipeline.py -v
"""

import asyncio
import re
import time

import pytest

from db.filter import FilterParams
from rag.embedder import Embedder
from rag.retriever import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

def test_embed_returns_vector(embed_fn):
    vec = embed_fn("unjustified dismissal employment")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)


def test_embed_different_texts_differ(embed_fn):
    v1 = embed_fn("unjustified dismissal")
    v2 = embed_fn("criminal sentencing robbery")
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a ** 2 for a in v1) ** 0.5
    norm2 = sum(b ** 2 for b in v2) ** 0.5
    cosine = dot / (norm1 * norm2)
    # Different domains should have low cosine similarity
    assert cosine < 0.95


def test_embed_similar_texts_close(embed_fn):
    v1 = embed_fn("unjustified dismissal personal grievance")
    v2 = embed_fn("unfair dismissal employment grievance")
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a ** 2 for a in v1) ** 0.5
    norm2 = sum(b ** 2 for b in v2) ** 0.5
    cosine = dot / (norm1 * norm2)
    # Same domain should have high cosine similarity
    assert cosine > 0.80


# ---------------------------------------------------------------------------
# VectorStore.search
# ---------------------------------------------------------------------------

def test_search_returns_results(embed_fn, store):
    vec = embed_fn("unjustified dismissal personal grievance")
    hits = store.search(vec, top_k=5)
    assert isinstance(hits, list)
    assert len(hits) > 0


def test_search_result_has_required_fields(embed_fn, store):
    vec = embed_fn("sentencing aggravated robbery starting point")
    hits = store.search(vec, top_k=3)
    for h in hits:
        assert h.case_id
        assert h.title
        assert h.court_name
        assert isinstance(h.score, float)
        assert 0.0 <= h.score <= 1.0
        assert h.text


def test_search_results_ordered_by_score(embed_fn, store):
    vec = embed_fn("employment personal grievance reinstatement")
    hits = store.search(vec, top_k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_court_filter(embed_fn, store):
    vec = embed_fn("employment personal grievance")
    hits = store.search(vec, top_k=5, courts=["NZERA"])
    for h in hits:
        assert "ERA" in h.court_name or "Employment" in h.court_name


def test_search_year_filter(embed_fn, store):
    vec = embed_fn("employment dismissal")
    hits = store.search(vec, top_k=5, year_from=2022, year_to=2024)
    for h in hits:
        m = re.search(r'\d{4}', h.date) if h.date else None
        year = int(m.group()) if m else None
        if year:
            assert 2022 <= year <= 2024


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_search_only_no_duplicate_case_ids(embed_fn, store):
    from rag.pipeline import _deduplicate
    vec = embed_fn("sentencing robbery")
    raw_hits = store.search(vec, top_k=20)
    deduped = _deduplicate(raw_hits, top_k=10)
    case_ids = [h.case_id for h in deduped]
    assert len(case_ids) == len(set(case_ids))


def test_dedup_keeps_best_score(embed_fn, store):
    from rag.pipeline import _deduplicate
    vec = embed_fn("sentencing")
    raw_hits = store.search(vec, top_k=30)
    deduped = _deduplicate(raw_hits, top_k=20)
    # For any case_id in deduped, no raw hit with same case_id has higher score
    deduped_map = {h.case_id: h.score for h in deduped}
    for rh in raw_hits:
        if rh.case_id in deduped_map:
            assert rh.score <= deduped_map[rh.case_id] + 1e-6


# ---------------------------------------------------------------------------
# SQL-first hybrid retrieval
# ---------------------------------------------------------------------------

def test_search_within_court_filter(embed_fn, store):
    from db.filter import get_point_ids
    params = FilterParams(courts=["NZCA"])
    point_ids = get_point_ids(params)
    assert len(point_ids) > 0
    vec = embed_fn("criminal appeal sentencing")
    hits = store.search_within(vec, point_ids, top_k=5)
    for h in hits:
        assert "Court of Appeal" in h.court_name or "NZCA" in h.court_name


def test_search_within_year_filter(embed_fn, store):
    from db.filter import get_point_ids
    params = FilterParams(year_from=2022, year_to=2024)
    point_ids = get_point_ids(params)
    vec = embed_fn("employment dismissal")
    hits = store.search_within(vec, point_ids, top_k=5)
    for h in hits:
        m = re.search(r'\d{4}', h.date) if h.date else None
        year = int(m.group()) if m else None
        if year:
            assert 2022 <= year <= 2024


def test_search_within_empty_ids_returns_empty(embed_fn, store):
    vec = embed_fn("employment law")
    hits = store.search_within(vec, [], top_k=5)
    assert hits == []


def test_search_within_scores_in_range(embed_fn, store):
    from db.filter import get_point_ids
    params = FilterParams(courts=["NZHC"])
    point_ids = get_point_ids(params)
    vec = embed_fn("criminal sentencing robbery")
    hits = store.search_within(vec, point_ids, top_k=5)
    for h in hits:
        assert 0.0 <= h.score <= 1.0


# ---------------------------------------------------------------------------
# SQL filter + Qdrant roundtrip (combined)
# ---------------------------------------------------------------------------

def test_hybrid_fewer_results_than_unrestricted(embed_fn, store):
    from db.filter import get_point_ids
    vec = embed_fn("employment dismissal personal grievance")
    unrestricted = store.search(vec, top_k=20)
    params = FilterParams(courts=["NZERA"])
    point_ids = get_point_ids(params)
    restricted = store.search_within(vec, point_ids, top_k=20)
    # Restricted set should not exceed unrestricted
    assert len(restricted) <= len(unrestricted)
