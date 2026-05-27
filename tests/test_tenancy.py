"""Smoke tests for tenancy.localrun.ai stack.

Verifies the full chain: Qdrant (nztt_moj) -> embedder -> llama-server -> tenancy API.

Requires all services running:
  - Qdrant on port 6333
  - llama-server on port 8080
  - tenancy-api on port 8081  (tested via TestClient, not HTTP)

Run:
    pytest tests/test_tenancy.py -v
"""

import pytest
import httpx
from fastapi.testclient import TestClient

import config
from tenancy.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _llm_available() -> bool:
    try:
        resp = httpx.get(f"{config.LLM_BASE_URL}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _qdrant_available() -> bool:
    try:
        resp = httpx.get(f"{config.QDRANT_URL}/collections/{config.QDRANT_TENANCY_COLLECTION}", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Infrastructure checks
# ---------------------------------------------------------------------------

def test_qdrant_collection_exists():
    resp = httpx.get(f"{config.QDRANT_URL}/collections/{config.QDRANT_TENANCY_COLLECTION}", timeout=5)
    assert resp.status_code == 200, "nztt_moj collection not found in Qdrant"


def test_qdrant_collection_has_points():
    resp = httpx.get(f"{config.QDRANT_URL}/collections/{config.QDRANT_TENANCY_COLLECTION}", timeout=5)
    count = resp.json()["result"]["points_count"]
    assert count > 600_000, f"Expected 600k+ points in nztt_moj, got {count}"


def test_llm_server_reachable():
    try:
        resp = httpx.get(f"{config.LLM_BASE_URL}/models", timeout=5)
        assert resp.status_code == 200
    except Exception as e:
        pytest.fail(f"llama-server not reachable: {e}")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "active" in data
    assert "waiting" in data
    assert "estimated_wait_seconds" in data


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_ask_empty_question_rejected(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 400


def test_ask_whitespace_only_rejected(client):
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 400


def test_ask_too_long_rejected(client):
    r = client.post("/ask", json={"question": "x" * 2001})
    assert r.status_code == 400


def test_ask_exactly_at_limit_accepted(client):
    # 2000 chars is within limit - should not be rejected for length
    # (may fail with 503 if LLM unavailable, but not 400)
    r = client.post("/ask", json={"question": "x" * 2000})
    assert r.status_code != 400


# ---------------------------------------------------------------------------
# Ask endpoint - full chain (LLM required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _llm_available(), reason="llama-server not running")
def test_ask_returns_answer(client):
    r = client.post("/ask", json={"question": "Can my landlord keep my full bond for minor damage?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["answer"]) > 50, "Answer too short"


@pytest.mark.skipif(not _llm_available(), reason="llama-server not running")
def test_ask_returns_sources(client):
    r = client.post("/ask", json={"question": "What notice must a landlord give before entering my home?"})
    assert r.status_code == 200
    sources = r.json()["sources"]
    assert len(sources) > 0, "No sources returned"


@pytest.mark.skipif(not _llm_available(), reason="llama-server not running")
def test_ask_source_shape(client):
    r = client.post("/ask", json={"question": "What is fair wear and tear?"})
    assert r.status_code == 200
    for s in r.json()["sources"]:
        assert "case_id" in s
        assert "title" in s
        assert "court_name" in s
        assert "date" in s
        assert "url" in s


@pytest.mark.skipif(not _llm_available(), reason="llama-server not running")
def test_ask_sources_are_nztt(client):
    r = client.post("/ask", json={"question": "Can a landlord increase rent more than once per year?"})
    assert r.status_code == 200
    for s in r.json()["sources"]:
        assert s["case_id"].startswith("NZTT-MOJ-"), f"Unexpected case_id: {s['case_id']}"
        assert s["court_name"] == "Tenancy Tribunal", f"Wrong court: {s['court_name']}"


@pytest.mark.skipif(not _llm_available(), reason="llama-server not running")
def test_ask_source_urls_are_nzlii(client):
    r = client.post("/ask", json={"question": "What happens if a landlord does not lodge the bond?"})
    assert r.status_code == 200
    for s in r.json()["sources"]:
        url = s["url"]
        assert url.startswith("https://www.nzlii.org/"), (
            f"Source URL is not NZLII: {url}"
        )
        assert "forms.justice.govt.nz/search/TT" not in url, (
            f"Source URL still points to generic MoJ search page: {url}"
        )


@pytest.mark.skipif(not _llm_available(), reason="llama-server not running")
def test_ask_answer_contains_citation(client):
    r = client.post("/ask", json={"question": "My landlord is refusing to fix the heating. What can I do?"})
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert "[S" in answer, "Answer contains no [SN] citations"


@pytest.mark.skipif(not _llm_available(), reason="llama-server not running")
def test_ask_answer_contains_community_law_signpost(client):
    r = client.post("/ask", json={"question": "How much notice does a landlord need to give to end a tenancy?"})
    assert r.status_code == 200
    answer = r.json()["answer"].lower()
    assert "community law" in answer or "tenancy services" in answer, (
        "System prompt signpost missing from answer"
    )


# ---------------------------------------------------------------------------
# Feedback endpoint
# ---------------------------------------------------------------------------

def test_feedback_helpful(client):
    r = client.post("/feedback", json={
        "question": "Can my landlord keep my bond?",
        "rating": 1,
    })
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_feedback_rate_limited_on_second_call(client):
    # Same IP (TestClient default) hits the 30s cooldown after test_feedback_helpful
    r = client.post("/feedback", json={
        "question": "Can my landlord keep my bond?",
        "rating": -1,
        "comment": "Answer was too vague",
    })
    assert r.status_code == 429


def test_feedback_invalid_rating(client):
    r = client.post("/feedback", json={
        "question": "Can my landlord keep my bond?",
        "rating": 0,
    })
    assert r.status_code == 400


def test_feedback_invalid_rating_out_of_range(client):
    r = client.post("/feedback", json={
        "question": "test",
        "rating": 5,
    })
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert "content-security-policy" in r.headers
    assert "referrer-policy" in r.headers


def test_csp_blocks_inline_scripts(client):
    csp = client.get("/health").headers.get("content-security-policy", "")
    assert "script-src 'self'" in csp
    assert "'unsafe-inline'" not in csp


def test_no_api_docs_exposed(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
