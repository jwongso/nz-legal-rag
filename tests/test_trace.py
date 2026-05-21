"""Tests for rag.trace - RetrievalTrace and citation verification."""

import pytest
from rag.trace import CitationVerification, RetrievalTrace, verify_citations


# ---------------------------------------------------------------------------
# RetrievalTrace
# ---------------------------------------------------------------------------

def test_trace_defaults():
    t = RetrievalTrace()
    assert t.strategy == "pure_qdrant"
    assert t.sql_point_ids_count == 0
    assert t.qdrant_candidates == 0
    assert t.reranker_enabled is False


def test_trace_to_dict_shape():
    t = RetrievalTrace(
        strategy="sql_first_hybrid",
        sql_filters={"courts": ["NZCA"]},
        sql_point_ids_count=342,
        qdrant_candidates=50,
        after_dedup=20,
        after_rerank=8,
        latency_embed_ms=110.5,
        latency_sql_ms=8.2,
        latency_qdrant_ms=120.3,
        latency_rerank_ms=5500.0,
        latency_generate_ms=34000.0,
        latency_total_ms=39739.0,
        top_scores=[0.88, 0.85, 0.82],
        model_name="qwen3",
        embedding_model="nomic-embed-text",
        reranker_enabled=True,
    )
    d = t.to_dict()
    assert d["strategy"] == "sql_first_hybrid"
    assert d["sql_point_ids_count"] == 342
    assert d["counts"]["qdrant_candidates"] == 50
    assert d["counts"]["after_dedup"] == 20
    assert d["counts"]["after_rerank"] == 8
    assert d["latency_ms"]["embed"] == 110.5
    assert d["latency_ms"]["total"] == 39739.0
    assert len(d["top_scores"]) == 3
    assert d["models"]["reranker_enabled"] is True


def test_trace_to_dict_rounds_latency():
    t = RetrievalTrace(latency_embed_ms=110.12345)
    d = t.to_dict()
    assert d["latency_ms"]["embed"] == 110.1


# ---------------------------------------------------------------------------
# verify_citations
# ---------------------------------------------------------------------------

def _sources(n: int) -> list[dict]:
    return [{"case_id": f"CASE/{i}", "title": f"Case {i}"} for i in range(1, n + 1)]


def test_verify_all_cited_high_confidence():
    answer = "As noted in [1] and [2] and [3] and [4] and [5], the test is met."
    cv = verify_citations(answer, _sources(5))
    assert cv.has_citations is True
    assert cv.cited_count == 5
    assert cv.orphan_citations == []
    assert cv.uncited_sources == []
    assert cv.evidence_confidence == "high"
    assert cv.has_warning is False


def test_verify_no_citations_low_confidence():
    answer = "The employment relations act requires good faith."
    cv = verify_citations(answer, _sources(5))
    assert cv.has_citations is False
    assert cv.evidence_confidence == "low"
    assert cv.has_warning is True


def test_verify_orphan_citation():
    # Answer cites [9] but only 3 sources exist
    answer = "See [1] and [9] for details."
    cv = verify_citations(answer, _sources(3))
    assert 9 in cv.orphan_citations
    assert cv.has_warning is True


def test_verify_uncited_sources():
    # Only cites [1], ignores [2] and [3]
    answer = "As held in [1], the claimant succeeded."
    cv = verify_citations(answer, _sources(3))
    assert cv.cited_count == 1
    assert 2 in cv.uncited_sources
    assert 3 in cv.uncited_sources


def test_verify_years_not_treated_as_citations():
    # [1994], [2022] should NOT be treated as citation references
    answer = "In [1994] the Employment Relations Act was not yet in force. See [1]."
    cv = verify_citations(answer, _sources(3))
    assert cv.cited_count == 1
    assert 1994 not in cv.orphan_citations
    assert 2022 not in cv.orphan_citations


def test_verify_medium_confidence():
    # 3 sources, all cited, no orphans
    answer = "See [1], [2], and [3]."
    cv = verify_citations(answer, _sources(3))
    assert cv.evidence_confidence == "medium"


def test_verify_empty_answer():
    cv = verify_citations("", _sources(3))
    assert cv.has_citations is False
    assert cv.has_warning is True


def test_verify_empty_sources():
    answer = "See [1] for details."
    cv = verify_citations(answer, [])
    assert cv.orphan_citations == [1]
    assert cv.has_warning is True


def test_verify_to_dict():
    cv = CitationVerification(
        has_citations=True,
        cited_count=3,
        orphan_citations=[],
        uncited_sources=[4, 5],
        evidence_confidence="medium",
        has_warning=False,
    )
    d = cv.to_dict()
    assert d["cited_count"] == 3
    assert d["evidence_confidence"] == "medium"
    assert d["uncited_sources"] == [4, 5]
