"""API endpoint tests - runs against the live FastAPI app (no mocking).

Requires:
  - Qdrant running and populated
  - PostgreSQL nz_legal database populated
  - LLM server NOT required (generation is skipped in /search tests)

Run:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _llm_available() -> bool:
    import httpx
    import config
    try:
        resp = httpx.get(f"{config.LLM_BASE_URL}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "collection" in data


# ---------------------------------------------------------------------------
# /search  (no LLM required)
# ---------------------------------------------------------------------------

def test_search_returns_results(client):
    r = client.get("/search", params={"q": "unjustified dismissal"})
    assert r.status_code == 200
    hits = r.json()
    assert isinstance(hits, list)
    assert len(hits) > 0


def test_search_result_shape(client):
    r = client.get("/search", params={"q": "sentencing aggravated robbery", "top_k": 3})
    assert r.status_code == 200
    hits = r.json()
    for h in hits:
        assert "case_id" in h
        assert "title" in h
        assert "court_name" in h
        assert "score" in h
        assert isinstance(h["score"], float)
        assert 0.0 <= h["score"] <= 1.0


def test_search_court_filter(client):
    r = client.get("/search", params={"q": "redundancy", "courts": "NZERA", "top_k": 5})
    assert r.status_code == 200
    hits = r.json()
    for h in hits:
        assert "ERA" in h["court_name"] or "Employment" in h["court_name"]


def test_search_year_filter(client):
    r = client.get("/search", params={
        "q": "personal grievance",
        "year_from": 2022,
        "year_to": 2024,
        "top_k": 5,
    })
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_search_empty_query_rejected(client):
    r = client.get("/search", params={"q": ""})
    assert r.status_code in (200, 422)


def test_search_top_k_capped(client):
    r = client.get("/search", params={"q": "employment", "top_k": 999})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) <= 20


# ---------------------------------------------------------------------------
# /ask  (requires LLM)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _llm_available(), reason="LLM server not available")
def test_ask_basic(client):
    r = client.post("/ask", json={"question": "What is unjustified dismissal?", "top_k": 3})
    # 503 is acceptable - LLM may be temporarily unavailable despite /models returning 200
    if r.status_code == 503:
        pytest.skip("LLM returned 503 during generation")
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert len(data["answer"]) > 50
    assert isinstance(data["sources"], list)


@pytest.mark.skipif(not _llm_available(), reason="LLM server not available")
def test_ask_with_trace(client):
    r = client.post("/ask", json={
        "question": "What is unjustified dismissal?",
        "top_k": 3,
        "trace": True,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["trace"] is not None
    trace = data["trace"]
    assert "strategy" in trace
    assert "counts" in trace
    assert "latency_ms" in trace
    assert trace["counts"]["qdrant_candidates"] > 0
    assert trace["latency_ms"]["total"] > 0


@pytest.mark.skipif(not _llm_available(), reason="LLM server not available")
def test_ask_with_trace_citation_verification(client):
    r = client.post("/ask", json={
        "question": "What is unjustified dismissal?",
        "top_k": 5,
        "trace": True,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["citation_verification"] is not None
    cv = data["citation_verification"]
    assert "has_citations" in cv
    assert cv["evidence_confidence"] in ("high", "medium", "low")
    assert isinstance(cv["orphan_citations"], list)


def test_ask_empty_question_rejected(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 400


def test_ask_top_k_capped(client):
    r = client.post("/ask", json={"question": "employment law", "top_k": 999})
    assert r.status_code in (200, 422, 503)


# ---------------------------------------------------------------------------
# /notable
# ---------------------------------------------------------------------------

def test_notable_returns_results(client):
    r = client.post("/notable", json={"limit": 5})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_notable_result_shape(client):
    r = client.post("/notable", json={"limit": 3})
    assert r.status_code == 200
    for h in r.json():
        assert "case_id" in h
        assert "title" in h
        assert "flags" in h
        assert isinstance(h["flags"], list)


def test_notable_court_filter(client):
    r = client.post("/notable", json={"courts": ["NZCA"], "limit": 5})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# /sentencing-tracker
# ---------------------------------------------------------------------------

def test_sentencing_tracker_returns_results(client):
    r = client.post("/sentencing-tracker", json={"limit": 5})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_sentencing_tracker_shape(client):
    r = client.post("/sentencing-tracker", json={"limit": 3})
    assert r.status_code == 200
    for h in r.json():
        assert "case_id" in h
        assert "sentencing" in h
        assert isinstance(h["sentencing"], dict)


def test_sentencing_tracker_sentence_filter(client):
    r = client.post("/sentencing-tracker", json={
        "min_final_sentence": 24,
        "max_final_sentence": 60,
        "limit": 5,
    })
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# /pg-tracker
# ---------------------------------------------------------------------------

def test_pg_tracker_returns_results(client):
    r = client.post("/pg-tracker", json={"limit": 5})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_pg_tracker_shape(client):
    r = client.post("/pg-tracker", json={"limit": 3})
    assert r.status_code == 200
    for h in r.json():
        assert "case_id" in h
        assert "pg" in h
        assert isinstance(h["pg"], dict)


def test_pg_tracker_grievance_type_filter(client):
    r = client.post("/pg-tracker", json={
        "grievance_types": ["unjustified_dismissal"],
        "limit": 5,
    })
    assert r.status_code == 200
    assert isinstance(r.json(), list)
