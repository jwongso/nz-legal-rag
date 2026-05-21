"""Tests for db.filter - SQL pre-filter and BM25 search.

Requires PostgreSQL nz_legal database populated.

Run:
    pytest tests/test_filter.py -v
"""

import pytest
from db.filter import FilterParams, bm25_search, get_document_metadata, get_point_ids


# ---------------------------------------------------------------------------
# get_point_ids
# ---------------------------------------------------------------------------

def test_filter_no_params_returns_ids():
    p = FilterParams()
    ids = get_point_ids(p)
    assert isinstance(ids, list)
    assert len(ids) > 0


def test_filter_court():
    p = FilterParams(courts=["NZCA"])
    ids = get_point_ids(p)
    assert len(ids) > 0


def test_filter_multiple_courts():
    p = FilterParams(courts=["NZERA", "NZEmpC"])
    ids = get_point_ids(p)
    assert len(ids) > 0


def test_filter_year_range():
    p = FilterParams(year_from=2022, year_to=2024)
    ids = get_point_ids(p)
    assert len(ids) > 0


def test_filter_court_and_year():
    p = FilterParams(courts=["NZCA"], year_from=2020, year_to=2026)
    ids_filtered = get_point_ids(p)
    ids_all = get_point_ids(FilterParams(courts=["NZCA"]))
    # Narrower filter should return fewer or equal IDs
    assert len(ids_filtered) <= len(ids_all)


def test_filter_employment_grievance_type():
    p = FilterParams(grievance_type="unjustified_dismissal")
    ids = get_point_ids(p)
    assert isinstance(ids, list)


def test_filter_sentencing_final_sentence_range():
    p = FilterParams(min_final_sentence=24.0, max_final_sentence=60.0)
    ids = get_point_ids(p)
    assert isinstance(ids, list)


def test_filter_impossible_range_returns_empty():
    # Starting point > 9999 months is impossible
    p = FilterParams(min_starting_point=99999.0)
    ids = get_point_ids(p)
    assert ids == []


def test_filter_respects_max_ids():
    p = FilterParams(max_ids=10)
    ids = get_point_ids(p)
    assert len(ids) <= 10


def test_filter_ids_are_strings():
    p = FilterParams(courts=["NZCA"], max_ids=5)
    ids = get_point_ids(p)
    for id_ in ids:
        assert isinstance(id_, str)


# ---------------------------------------------------------------------------
# bm25_search
# ---------------------------------------------------------------------------

def test_bm25_basic():
    results = bm25_search("unjustified dismissal redundancy")
    assert isinstance(results, list)
    assert len(results) > 0


def test_bm25_result_shape():
    results = bm25_search("good faith employment", limit=3)
    for r in results:
        assert "citation" in r
        assert "title" in r
        assert "court" in r
        assert "rank" in r
        assert "snippet" in r
        assert r["rank"] > 0


def test_bm25_court_filter():
    results = bm25_search("dismissal", courts=["NZERA"], limit=5)
    for r in results:
        assert r["court"] == "NZERA"


def test_bm25_year_filter():
    results = bm25_search("sentencing", year_from=2022, year_to=2024, limit=5)
    assert isinstance(results, list)


def test_bm25_respects_limit():
    results = bm25_search("employment", limit=3)
    assert len(results) <= 3


def test_bm25_ranked_descending():
    results = bm25_search("unjustified dismissal reinstatement", limit=5)
    ranks = [r["rank"] for r in results]
    assert ranks == sorted(ranks, reverse=True)


def test_bm25_nonsense_query_returns_empty():
    results = bm25_search("xyzzy foobar qqqqqq")
    assert results == []


# ---------------------------------------------------------------------------
# get_document_metadata
# ---------------------------------------------------------------------------

def test_get_document_metadata_known_case():
    # Grab a known citation from the DB first
    import psycopg2
    conn = psycopg2.connect(dbname="nz_legal")
    cur = conn.cursor()
    cur.execute("SELECT citation FROM documents LIMIT 1")
    row = cur.fetchone()
    conn.close()

    if row:
        meta = get_document_metadata(row[0])
        assert meta is not None
        assert "citation" in meta
        assert "court" in meta


def test_get_document_metadata_unknown_returns_none():
    meta = get_document_metadata("DOES/NOT/EXIST/9999")
    assert meta is None
