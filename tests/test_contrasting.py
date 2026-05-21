"""Tests for POST /contrasting-cases and rag/contrasting.

Requires Qdrant + PostgreSQL populated with sentencing data.
Employment split tests may return empty groups due to sparse pg.reinstatement_ordered data.

Run:
    pytest tests/test_contrasting.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api.server import app
from rag.contrasting import get_split_config


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# get_split_config unit tests (no DB required)
# ---------------------------------------------------------------------------

def test_split_config_criminal_default():
    result = get_split_config("criminal", None)
    assert result is not None
    split_name, cfg_a, cfg_b = result
    assert split_name == "sentence_type"
    assert cfg_a[2] == "Imprisonment"
    assert cfg_b[2] == "Home detention"


def test_split_config_employment_default():
    result = get_split_config("employment", None)
    assert result is not None
    split_name, cfg_a, cfg_b = result
    assert split_name == "reinstatement"
    assert "ordered" in cfg_a[2].lower()
    assert "declined" in cfg_b[2].lower()


def test_split_config_criminal_guilty_plea():
    result = get_split_config("criminal", "guilty_plea")
    assert result is not None
    split_name, cfg_a, cfg_b = result
    assert split_name == "guilty_plea"
    assert cfg_a[2] == "Guilty plea"
    assert cfg_b[2] == "No guilty plea"


def test_split_config_unknown_domain_returns_none():
    assert get_split_config("maritime", None) is None


def test_split_config_unknown_split_returns_none():
    assert get_split_config("criminal", "moon_phase") is None


def test_split_config_all_groups_have_four_fields():
    for domain in ("criminal", "employment"):
        result = get_split_config(domain, None)
        assert result is not None
        _, cfg_a, cfg_b = result
        assert len(cfg_a) == 4  # key, value, label, description
        assert len(cfg_b) == 4


# ---------------------------------------------------------------------------
# API - basic shape
# ---------------------------------------------------------------------------

def test_contrasting_criminal_returns_200(client):
    r = client.post("/contrasting-cases", json={
        "query": "aggravated robbery weapon group offending",
        "domain": "criminal",
        "top_k": 3,
    })
    assert r.status_code == 200


def test_contrasting_response_shape(client):
    r = client.post("/contrasting-cases", json={
        "query": "robbery sentencing",
        "domain": "criminal",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["domain"] == "criminal"
    assert data["split_by"] == "sentence_type"
    assert data["group_a"]["label"] == "Imprisonment"
    assert data["group_b"]["label"] == "Home detention"
    assert isinstance(data["group_a"]["cases"], list)
    assert isinstance(data["group_b"]["cases"], list)
    assert data["explanation"] is None


def test_contrasting_case_item_shape(client):
    r = client.post("/contrasting-cases", json={
        "query": "aggravated robbery sentencing starting point",
        "domain": "criminal",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    for group in (data["group_a"], data["group_b"]):
        for case in group["cases"]:
            assert "case_id" in case
            assert "title" in case
            assert "court_name" in case
            assert "date" in case
            assert "url" in case
            assert "score" in case
            assert "structured" in case
            assert isinstance(case["score"], float)
            assert isinstance(case["structured"], dict)


def test_contrasting_top_k_respected(client):
    r = client.post("/contrasting-cases", json={
        "query": "robbery sentencing imprisonment",
        "domain": "criminal",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["group_a"]["cases"]) <= 3
    assert len(data["group_b"]["cases"]) <= 3


def test_contrasting_top_k_capped_at_10(client):
    r = client.post("/contrasting-cases", json={
        "query": "sentencing robbery",
        "domain": "criminal",
        "top_k": 999,
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["group_a"]["cases"]) <= 10
    assert len(data["group_b"]["cases"]) <= 10


# ---------------------------------------------------------------------------
# API - data quality
# ---------------------------------------------------------------------------

def test_contrasting_scores_in_valid_range(client):
    r = client.post("/contrasting-cases", json={
        "query": "aggravated robbery",
        "domain": "criminal",
        "top_k": 5,
    })
    assert r.status_code == 200
    data = r.json()
    for group in (data["group_a"], data["group_b"]):
        for case in group["cases"]:
            assert 0.0 <= case["score"] <= 1.0


def test_contrasting_no_duplicate_case_ids_per_group(client):
    r = client.post("/contrasting-cases", json={
        "query": "robbery sentencing starting point",
        "domain": "criminal",
        "top_k": 5,
    })
    assert r.status_code == 200
    data = r.json()
    for group in (data["group_a"], data["group_b"]):
        ids = [c["case_id"] for c in group["cases"]]
        assert len(ids) == len(set(ids))


def test_contrasting_structured_has_relevant_fields(client):
    r = client.post("/contrasting-cases", json={
        "query": "robbery imprisonment sentencing",
        "domain": "criminal",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    # Group A should be imprisonment - structured should have sentence_type
    for case in data["group_a"]["cases"]:
        if case["structured"]:
            # sentence_type or final_sentence should be present when data exists
            has_sentencing_field = any(
                k in case["structured"]
                for k in ("sentence_type", "final_sentence_months",
                          "starting_point_months", "has_guilty_plea")
            )
            assert has_sentencing_field


def test_contrasting_imprisonment_group_has_criminal_courts(client):
    r = client.post("/contrasting-cases", json={
        "query": "robbery aggravated weapon",
        "domain": "criminal",
        "top_k": 5,
    })
    assert r.status_code == 200
    data = r.json()
    criminal_courts = {"NZHC", "NZCA", "NZDC", "NZSC",
                       "High Court", "Court of Appeal", "District Court", "Supreme Court"}
    for group in (data["group_a"], data["group_b"]):
        for case in group["cases"]:
            assert any(c in case["court_name"] for c in criminal_courts), (
                f"Expected criminal court, got: {case['court_name']}"
            )


# ---------------------------------------------------------------------------
# API - filters and splits
# ---------------------------------------------------------------------------

def test_contrasting_guilty_plea_split(client):
    r = client.post("/contrasting-cases", json={
        "query": "robbery sentencing discount reduction",
        "domain": "criminal",
        "split_by": "guilty_plea",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["split_by"] == "guilty_plea"
    assert data["group_a"]["label"] == "Guilty plea"
    assert data["group_b"]["label"] == "No guilty plea"


def test_contrasting_court_filter(client):
    r = client.post("/contrasting-cases", json={
        "query": "robbery sentencing",
        "domain": "criminal",
        "courts": ["NZHC"],
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    for group in (data["group_a"], data["group_b"]):
        for case in group["cases"]:
            assert "High Court" in case["court_name"] or "NZHC" in case["court_name"]


def test_contrasting_year_filter(client):
    r = client.post("/contrasting-cases", json={
        "query": "robbery sentencing",
        "domain": "criminal",
        "year_from": 2022,
        "year_to": 2024,
        "top_k": 3,
    })
    assert r.status_code == 200
    assert isinstance(r.json()["group_a"]["cases"], list)


def test_contrasting_employment_structure(client):
    r = client.post("/contrasting-cases", json={
        "query": "unjustified dismissal personal grievance reinstatement",
        "domain": "employment",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["domain"] == "employment"
    assert data["split_by"] == "reinstatement"
    # Employment data is sparse - groups may be empty but structure must be correct
    assert "label" in data["group_a"]
    assert "label" in data["group_b"]
    assert data["group_a"]["label"] == "Reinstatement ordered"
    assert data["group_b"]["label"] == "Reinstatement declined"


# ---------------------------------------------------------------------------
# API - validation
# ---------------------------------------------------------------------------

def test_contrasting_empty_query_rejected(client):
    r = client.post("/contrasting-cases", json={"query": "", "domain": "criminal"})
    assert r.status_code == 400


def test_contrasting_invalid_domain_rejected(client):
    r = client.post("/contrasting-cases", json={"query": "test", "domain": "unknown"})
    assert r.status_code == 400


def test_contrasting_invalid_split_rejected(client):
    r = client.post("/contrasting-cases", json={
        "query": "test",
        "domain": "criminal",
        "split_by": "moon_phase",
    })
    assert r.status_code == 400
