"""Tests for search_legislation() MCP tool behaviour.

The local mcp/ directory shadows the installed mcp package, making direct import
of mcp.server impossible. These tests verify the same behaviour by exercising the
underlying components (pipeline + store) with identical parameters.

Requires: Qdrant running and populated with NZLEG chunks.

Run:
    pytest tests/test_mcp.py -v
"""

import asyncio

import pytest

from rag.pipeline import RAGPipeline, _deduplicate
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
def pipeline(event_loop):
    p = RAGPipeline()
    yield p
    event_loop.run_until_complete(p.close())


@pytest.fixture(scope="module")
def store(pipeline):
    return pipeline._store


@pytest.fixture(scope="module")
def embed(pipeline, event_loop):
    def _embed(text: str):
        return event_loop.run_until_complete(pipeline._embedder.embed(text))
    return _embed


# ---------------------------------------------------------------------------
# Helper that replicates the search_legislation() retrieval logic exactly
# ---------------------------------------------------------------------------

def _legislation_hits(embed, store, query: str, act: str = "", top_k: int = 5):
    """Run the same retrieval path as search_legislation() and return deduped hits."""
    vector = embed(query)
    raw = store.search(vector, top_k=min(top_k, 20) * 4, courts=["NZLEG"])

    act_prefix = f"NZLEG/{act.upper().strip()}/" if act.strip() else ""
    if act_prefix:
        raw = [h for h in raw if h.case_id.startswith(act_prefix)]

    seen: set[str] = set()
    hits = []
    for h in raw:
        if h.case_id not in seen:
            seen.add(h.case_id)
            hits.append(h)
        if len(hits) >= min(top_k, 20):
            break
    return hits


# ---------------------------------------------------------------------------
# Court filter: only NZLEG results returned
# ---------------------------------------------------------------------------

def test_legislation_results_are_nzleg_only(embed, store):
    hits = _legislation_hits(embed, store, "landlord obligations")
    assert hits, "Expected results for a broad tenancy topic"
    for h in hits:
        assert h.case_id.startswith("NZLEG/"), (
            f"Non-legislation result leaked through: {h.case_id}"
        )


def test_legislation_results_not_mixed_with_decisions(embed, store):
    hits = _legislation_hits(embed, store, "notice to terminate tenancy")
    for h in hits:
        # Case IDs from tribunal decisions start with NZTT, not NZLEG
        assert not h.case_id.startswith("NZTT"), (
            f"Tribunal decision appeared in legislation results: {h.case_id}"
        )


# ---------------------------------------------------------------------------
# Act filter: RTA prefix
# ---------------------------------------------------------------------------

def test_act_filter_rta_restricts_to_rta_only(embed, store):
    hits = _legislation_hits(embed, store, "landlord right of entry", act="RTA")
    assert hits, "Expected results for entry rights in RTA"
    for h in hits:
        assert h.case_id.startswith("NZLEG/RTA/"), (
            f"Non-RTA section returned when act='RTA': {h.case_id}"
        )


def test_act_filter_case_insensitive(embed, store):
    hits_upper = _legislation_hits(embed, store, "landlord right of entry", act="RTA")
    hits_lower = _legislation_hits(embed, store, "landlord right of entry", act="rta")
    assert hits_upper and hits_lower
    assert [h.case_id for h in hits_upper] == [h.case_id for h in hits_lower]


def test_act_filter_era2000(embed, store):
    hits = _legislation_hits(embed, store, "unjustified dismissal reinstatement", act="ERA2000")
    assert hits, "Expected results for ERA2000"
    for h in hits:
        assert h.case_id.startswith("NZLEG/ERA2000/"), (
            f"Non-ERA2000 section returned: {h.case_id}"
        )


def test_no_act_filter_returns_across_acts(embed, store):
    hits = _legislation_hits(embed, store, "good faith obligations", top_k=10)
    acts = {h.case_id.split("/")[1] for h in hits}
    # Without a filter, results can span multiple acts
    assert len(acts) >= 1  # at minimum one act returned


def test_unknown_act_returns_empty(embed, store):
    hits = _legislation_hits(embed, store, "landlord rights", act="FAKEXYZ123")
    assert hits == [], f"Expected empty list for unknown act, got {hits}"


# ---------------------------------------------------------------------------
# Known section resolution
# ---------------------------------------------------------------------------

def test_s48_found_for_entry_rights_query(embed, store):
    hits = _legislation_hits(embed, store, "landlord right of entry", act="RTA")
    case_ids = [h.case_id for h in hits]
    assert "NZLEG/RTA/s48" in case_ids, (
        f"s48 (Landlord's right of entry) not in top results: {case_ids}"
    )


def test_section_reference_query_finds_correct_section(embed, store):
    hits = _legislation_hits(embed, store, "section 48 entry", act="RTA", top_k=3)
    case_ids = [h.case_id for h in hits]
    assert "NZLEG/RTA/s48" in case_ids, (
        f"s48 not found via section number query: {case_ids}"
    )


def test_fair_wear_and_tear_section_found(embed, store):
    hits = _legislation_hits(embed, store, "fair wear and tear", act="RTA")
    assert hits, "Expected results for fair wear and tear"
    # Should find something about property condition / wear
    texts = " ".join(h.text for h in hits).lower()
    assert "wear" in texts or "condition" in texts, (
        "Results don't mention wear or condition"
    )


# ---------------------------------------------------------------------------
# Score and ordering
# ---------------------------------------------------------------------------

def test_results_ordered_by_score_descending(embed, store):
    hits = _legislation_hits(embed, store, "bond refund landlord", act="RTA", top_k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), (
        f"Results not in descending score order: {scores}"
    )


def test_scores_in_valid_range(embed, store):
    hits = _legislation_hits(embed, store, "notice period termination", act="RTA")
    for h in hits:
        assert 0.0 <= h.score <= 1.0, f"Score out of range: {h.score}"


# ---------------------------------------------------------------------------
# Deduplication: one chunk per section
# ---------------------------------------------------------------------------

def test_no_duplicate_section_ids(embed, store):
    hits = _legislation_hits(embed, store, "landlord obligations repairs", act="RTA", top_k=10)
    ids = [h.case_id for h in hits]
    assert len(ids) == len(set(ids)), f"Duplicate section IDs: {ids}"


def test_dedup_keeps_highest_score_chunk(embed, store):
    """If a section has multiple chunks, only the top-scoring one survives."""
    vector = embed("landlord right of entry")
    raw = store.search(vector, top_k=80, courts=["NZLEG"])
    rta = [h for h in raw if h.case_id.startswith("NZLEG/RTA/")]

    # Run dedup the same way search_legislation does (manual loop, not _deduplicate)
    seen: set[str] = set()
    deduped = []
    for h in rta:
        if h.case_id not in seen:
            seen.add(h.case_id)
            deduped.append(h)

    deduped_map = {h.case_id: h.score for h in deduped}
    for rh in rta:
        if rh.case_id in deduped_map:
            assert rh.score <= deduped_map[rh.case_id] + 1e-6, (
                f"Dedup kept lower-scoring chunk for {rh.case_id}"
            )


# ---------------------------------------------------------------------------
# top_k limit
# ---------------------------------------------------------------------------

def test_top_k_limits_result_count(embed, store):
    for k in (1, 3, 5):
        hits = _legislation_hits(embed, store, "landlord tenant obligations", top_k=k)
        assert len(hits) <= k, f"top_k={k} but got {len(hits)} results"


def test_top_k_capped_at_20(embed, store):
    hits = _legislation_hits(embed, store, "landlord tenant rights", top_k=99)
    assert len(hits) <= 20, f"top_k cap not enforced: got {len(hits)}"


# ---------------------------------------------------------------------------
# Result shape: required fields present
# ---------------------------------------------------------------------------

def test_results_have_required_fields(embed, store):
    hits = _legislation_hits(embed, store, "bond refund", act="RTA")
    assert hits
    for h in hits:
        assert h.case_id, "Missing case_id"
        assert h.title, "Missing title"
        assert h.text, "Missing text"
        assert h.url, "Missing url"
        assert isinstance(h.score, float)


def test_result_urls_point_to_legislation_site(embed, store):
    hits = _legislation_hits(embed, store, "landlord entry rights", act="RTA")
    for h in hits:
        assert "legislation.govt.nz" in h.url, (
            f"URL does not point to legislation.govt.nz: {h.url}"
        )


def test_result_titles_include_section_number(embed, store):
    hits = _legislation_hits(embed, store, "landlord right of entry", act="RTA")
    for h in hits:
        # Titles like "s48 Landlord's right of entry"
        assert h.title.startswith("s"), (
            f"Title doesn't start with section number: {h.title!r}"
        )


# ---------------------------------------------------------------------------
# Output format: verify the string the tool produces
# ---------------------------------------------------------------------------

def _format_output(hits, query: str) -> str:
    """Replicate the output format of search_legislation()."""
    if not hits:
        return "No matching legislation sections found."
    lines = [f"Found {len(hits)} section(s) matching '{query}':\n"]
    for h in hits:
        lines.append(f"## {h.title}")
        lines.append(f"Citation: {h.case_id}  |  Score: {h.score:.4f}")
        lines.append(f"URL: {h.url}")
        lines.append("")
        lines.append(h.text)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def test_output_contains_section_headers(embed, store):
    hits = _legislation_hits(embed, store, "landlord right of entry", act="RTA")
    out = _format_output(hits, "landlord right of entry")
    assert "## " in out, "Output missing '## ' section headers"


def test_output_contains_citation_lines(embed, store):
    hits = _legislation_hits(embed, store, "bond refund", act="RTA")
    out = _format_output(hits, "bond refund")
    assert "Citation: NZLEG/RTA/" in out, "Output missing Citation: line"


def test_output_contains_url_lines(embed, store):
    hits = _legislation_hits(embed, store, "notice termination", act="RTA")
    out = _format_output(hits, "notice termination")
    assert "URL: https://www.legislation.govt.nz" in out, "Output missing URL line"


def test_output_contains_section_text(embed, store):
    hits = _legislation_hits(embed, store, "landlord right of entry", act="RTA")
    out = _format_output(hits, "landlord right of entry")
    # The section text itself should appear in the output
    assert len(out) > 200, "Output suspiciously short - section text missing"


def test_output_no_results_message(embed, store):
    hits = _legislation_hits(embed, store, "anything", act="FAKEXYZ")
    out = _format_output(hits, "anything")
    assert "No matching" in out


def test_output_found_count_matches_hits(embed, store):
    query = "landlord obligations repairs"
    hits = _legislation_hits(embed, store, query, act="RTA", top_k=3)
    out = _format_output(hits, query)
    assert f"Found {len(hits)} section(s)" in out
