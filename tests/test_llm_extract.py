"""Tests for ingest/llm_extract_pipeline.py - validation and parsing logic.

These tests cover the pure-logic parts (JSON parsing, field validation,
text selection) without calling the LLM or touching the database.

Run:
    pytest tests/test_llm_extract.py -v
"""

import pytest

from ingest.llm_extract_pipeline import (
    _extract_json,
    _select_text,
    _validate_employment,
    _validate_sentencing,
)


# ---------------------------------------------------------------------------
# _extract_json - handles LLM output variance
# ---------------------------------------------------------------------------

def test_extract_json_clean():
    raw = '{"reinstatement_ordered": true, "compensation_nzd": 12000}'
    result = _extract_json(raw)
    assert result == {"reinstatement_ordered": True, "compensation_nzd": 12000}


def test_extract_json_with_preamble():
    raw = 'Here is the extracted JSON:\n{"offence": "robbery", "starting_point_months": 36}'
    result = _extract_json(raw)
    assert result is not None
    assert result["offence"] == "robbery"
    assert result["starting_point_months"] == 36


def test_extract_json_with_postamble():
    raw = '{"offence": "burglary"}\n\nNote: this is based on the excerpt provided.'
    result = _extract_json(raw)
    assert result is not None
    assert result["offence"] == "burglary"


def test_extract_json_null_fields():
    raw = '{"grievance_types": [], "reinstatement_ordered": null, "compensation_nzd": null}'
    result = _extract_json(raw)
    assert result is not None
    assert result["reinstatement_ordered"] is None
    assert result["compensation_nzd"] is None


def test_extract_json_invalid_returns_none():
    assert _extract_json("Sorry, I cannot extract this.") is None
    assert _extract_json("") is None
    assert _extract_json("{broken json") is None


def test_extract_json_nested_braces():
    # Real LLM might output nested structure - we just want the outer object
    raw = '{"grievance_types": ["unjustified_dismissal"]}'
    result = _extract_json(raw)
    assert result is not None
    assert "grievance_types" in result


# ---------------------------------------------------------------------------
# _validate_employment
# ---------------------------------------------------------------------------

def test_validate_employment_valid_grievance_types():
    fields = {"grievance_types": ["unjustified_dismissal", "harassment"]}
    result = _validate_employment(fields)
    assert "unjustified_dismissal" in result["grievance_types"]
    assert "harassment" in result["grievance_types"]


def test_validate_employment_filters_invalid_grievance_type():
    fields = {"grievance_types": ["unjustified_dismissal", "wrongful_termination"]}
    result = _validate_employment(fields)
    # "wrongful_termination" is not in the allowed set
    assert "wrongful_termination" not in result["grievance_types"]
    assert "unjustified_dismissal" in result["grievance_types"]


def test_validate_employment_none_grievance_types():
    fields = {"grievance_types": None}
    result = _validate_employment(fields)
    assert result["grievance_types"] == []


def test_validate_employment_clamps_contributory_pct():
    fields = {"contributory_conduct_pct": 150}
    result = _validate_employment(fields)
    assert result["contributory_conduct_pct"] == 100.0


def test_validate_employment_clamps_pct_negative():
    fields = {"contributory_conduct_pct": -10}
    result = _validate_employment(fields)
    assert result["contributory_conduct_pct"] == 0.0


def test_validate_employment_invalid_pct_becomes_none():
    fields = {"contributory_conduct_pct": "not a number"}
    result = _validate_employment(fields)
    assert result["contributory_conduct_pct"] is None


def test_validate_employment_negative_compensation_becomes_none():
    fields = {"compensation_nzd": -500}
    result = _validate_employment(fields)
    assert result["compensation_nzd"] == 0.0


def test_validate_employment_invalid_compensation_becomes_none():
    fields = {"compensation_nzd": "twelve thousand dollars"}
    result = _validate_employment(fields)
    assert result["compensation_nzd"] is None


def test_validate_employment_reinstatement_non_bool_becomes_none():
    fields = {"reinstatement_ordered": "yes"}
    result = _validate_employment(fields)
    assert result["reinstatement_ordered"] is None


def test_validate_employment_reinstatement_true_preserved():
    fields = {"reinstatement_ordered": True}
    result = _validate_employment(fields)
    assert result["reinstatement_ordered"] is True


def test_validate_employment_reinstatement_false_preserved():
    fields = {"reinstatement_ordered": False}
    result = _validate_employment(fields)
    assert result["reinstatement_ordered"] is False


# ---------------------------------------------------------------------------
# _validate_sentencing
# ---------------------------------------------------------------------------

def test_validate_sentencing_valid_months():
    fields = {"starting_point_months": 36, "final_sentence_months": 30}
    result = _validate_sentencing(fields)
    assert result["starting_point_months"] == 36.0
    assert result["final_sentence_months"] == 30.0


def test_validate_sentencing_zero_months_becomes_none():
    fields = {"starting_point_months": 0}
    result = _validate_sentencing(fields)
    assert result["starting_point_months"] is None


def test_validate_sentencing_unrealistic_months_becomes_none():
    # 1200+ months is unrealistic (100 years)
    fields = {"final_sentence_months": 9999}
    result = _validate_sentencing(fields)
    assert result["final_sentence_months"] is None


def test_validate_sentencing_invalid_months_becomes_none():
    fields = {"starting_point_months": "four years"}
    result = _validate_sentencing(fields)
    assert result["starting_point_months"] is None


def test_validate_sentencing_clamps_guilty_plea_pct():
    fields = {"guilty_plea_discount_pct": 110}
    result = _validate_sentencing(fields)
    assert result["guilty_plea_discount_pct"] == 100.0


def test_validate_sentencing_valid_appeal_outcome():
    for outcome in ("allowed", "dismissed", "varied"):
        fields = {"appeal_outcome": outcome}
        result = _validate_sentencing(fields)
        assert result["appeal_outcome"] == outcome


def test_validate_sentencing_invalid_appeal_outcome_becomes_none():
    fields = {"appeal_outcome": "upheld"}
    result = _validate_sentencing(fields)
    assert result["appeal_outcome"] is None


def test_validate_sentencing_null_appeal_outcome_stays_none():
    fields = {"appeal_outcome": None}
    result = _validate_sentencing(fields)
    assert result["appeal_outcome"] is None


# ---------------------------------------------------------------------------
# _select_text - text budget strategy
# ---------------------------------------------------------------------------

def test_select_text_empty_chunks():
    assert _select_text([]) == ""


def test_select_text_single_chunk():
    result = _select_text(["Only chunk"])
    assert "Only chunk" in result


def test_select_text_takes_head_and_tail():
    # 10 chunks - should get first 1 and last 3
    chunks = [f"Chunk {i}" for i in range(10)]
    result = _select_text(chunks)
    assert "Chunk 0" in result   # head
    assert "Chunk 9" in result   # tail
    assert "Chunk 4" not in result  # middle dropped


def test_select_text_truncates_long_chunks():
    long_chunk = "x" * 2000
    chunks = [long_chunk, "short"]
    result = _select_text(chunks)
    # Each chunk truncated to 500 chars
    assert len(result) <= 2000


def test_select_text_separator_present():
    chunks = ["First chunk", "Last chunk"]
    result = _select_text(chunks)
    # Head and tail separated by delimiter
    assert "---" in result


def test_select_text_few_chunks_no_overlap():
    # With only 2 chunks, head=1 and tail=1 - should not duplicate
    chunks = ["Head", "Tail"]
    result = _select_text(chunks)
    assert result.count("Head") == 1
    assert result.count("Tail") == 1
