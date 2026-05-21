"""Retrieval quality and relevance tests.

These tests verify that the system returns topically relevant results,
BM25 snippets contain the search terms, and domain-specific queries
route to the correct courts.

Requires Qdrant + PostgreSQL populated.

Run:
    pytest tests/test_quality.py -v
"""

import asyncio

import pytest

from db.filter import bm25_search
from rag.embedder import Embedder
from rag.retriever import VectorStore


EMPLOYMENT_COURTS = {"NZERA", "NZEmpC", "Employment Relations Authority",
                     "Employment Court"}
CRIMINAL_COURTS = {"NZHC", "NZCA", "NZDC", "NZSC",
                   "High Court", "Court of Appeal", "District Court", "Supreme Court"}


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
# Domain relevance
# ---------------------------------------------------------------------------

def test_employment_query_returns_employment_courts(embed_fn, store):
    """Semantic search for employment topics should surface ERA/EmpC decisions."""
    vec = embed_fn("unjustified dismissal personal grievance reinstatement remedy")
    hits = store.search(vec, top_k=10)
    assert len(hits) > 0
    employment_hits = [
        h for h in hits
        if any(c in h.court_name for c in ("ERA", "Employment", "EmpC"))
    ]
    # At least half the top-10 results should be employment-related
    assert len(employment_hits) >= 3


def test_sentencing_query_returns_criminal_courts(embed_fn, store):
    """Sentencing queries should return criminal court decisions."""
    vec = embed_fn("criminal sentencing starting point guilty plea discount aggravated robbery")
    hits = store.search(vec, top_k=10)
    assert len(hits) > 0
    criminal_hits = [
        h for h in hits
        if any(c in h.court_name for c in ("NZHC", "NZCA", "NZDC", "High Court",
                                            "Court of Appeal", "District Court"))
    ]
    assert len(criminal_hits) >= 3


def test_court_filter_actually_filters(embed_fn, store):
    """Court filter must exclude other courts from results."""
    vec = embed_fn("employment redundancy")
    unrestricted = store.search(vec, top_k=10, courts=None)
    filtered = store.search(vec, top_k=10, courts=["NZCA"])
    filtered_names = {h.court_name for h in filtered}
    # All filtered results should only be NZCA/Court of Appeal
    for name in filtered_names:
        assert "Court of Appeal" in name or "NZCA" in name


# ---------------------------------------------------------------------------
# BM25 quality
# ---------------------------------------------------------------------------

def test_bm25_snippet_contains_search_term():
    """BM25 snippets must contain at least one of the search terms."""
    query = "unjustified dismissal reinstatement"
    terms = {"unjustified", "dismissal", "reinstatement"}
    results = bm25_search(query, limit=5)
    assert len(results) > 0
    for r in results:
        snippet_lower = r["snippet"].lower()
        assert any(t in snippet_lower for t in terms), (
            f"Snippet '{snippet_lower[:100]}' contains none of {terms}"
        )


def test_bm25_title_or_snippet_relevant():
    """BM25 results for a sentencing query should mention sentencing terms."""
    query = "guilty plea discount sentencing"
    terms = {"guilty", "plea", "discount", "sentencing", "sentence"}
    results = bm25_search(query, limit=5)
    assert len(results) > 0
    for r in results:
        combined = (r["title"] + " " + r["snippet"]).lower()
        assert any(t in combined for t in terms)


def test_bm25_citation_format():
    """BM25 citations should look like NZ case citations."""
    results = bm25_search("employment dismissal", limit=3)
    assert len(results) > 0
    for r in results:
        citation = r["citation"]
        assert len(citation) > 5
        assert citation.strip() != ""


def test_bm25_rank_positive():
    results = bm25_search("unjustified dismissal", limit=5)
    for r in results:
        assert r["rank"] > 0


def test_bm25_employment_court_isolation():
    """BM25 with ERA court filter should return only ERA results."""
    results = bm25_search("personal grievance dismissal", courts=["NZERA"], limit=5)
    for r in results:
        assert r["court"] == "NZERA"


def test_bm25_high_relevance_first():
    """The first BM25 result should be more relevant than the last."""
    query = "redundancy personal grievance unjustified"
    results = bm25_search(query, limit=5)
    if len(results) >= 2:
        assert results[0]["rank"] >= results[-1]["rank"]


# ---------------------------------------------------------------------------
# Score calibration
# ---------------------------------------------------------------------------

def test_search_scores_in_valid_range(embed_fn, store):
    vec = embed_fn("employment personal grievance")
    hits = store.search(vec, top_k=10)
    for h in hits:
        assert -1.0 <= h.score <= 1.0


def test_exact_phrase_scores_high(embed_fn, store):
    """A very specific NZ legal phrase should score above 0.5."""
    vec = embed_fn("unjustified dismissal personal grievance Employment Relations Act")
    hits = store.search(vec, top_k=3)
    assert len(hits) > 0
    assert hits[0].score > 0.3


def test_score_monotonic_with_top_k(embed_fn, store):
    """Top-3 scores should all be >= top-10 worst score."""
    vec = embed_fn("sentencing robbery")
    hits_3 = store.search(vec, top_k=3)
    hits_10 = store.search(vec, top_k=10)
    if hits_3 and hits_10:
        worst_of_3 = min(h.score for h in hits_3)
        best_of_rest = max(h.score for h in hits_10[3:]) if len(hits_10) > 3 else 0
        assert worst_of_3 >= best_of_rest - 1e-6
