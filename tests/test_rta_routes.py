"""Integration test: validate all forced_sections in ROUTES against Qdrant.

Run with: pytest tests/test_rta_routes.py -v
Requires: Qdrant running at QDRANT_URL (default http://localhost:6333)
"""

import pytest

try:
    from rag.retriever import VectorStore
    from tenancy.rta_routes import ROUTES
    _IMPORTS_OK = True
except Exception:
    _IMPORTS_OK = False

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason="rag package not importable")

# Per-section content expectations. Must stay in sync with ROUTES forced_sections.
# If a section ID resolves to regulation text or a penalty table row instead of
# the expected main-Act section, one of these checks will catch it.
SECTION_EXPECTATIONS: dict[str, dict] = {
    "NZLEG/RTA/s13A": {
        "must_contain_any": ["tenancy agreement", "agreement in writing", "written", "contents"],
        "must_not_contain_any": ["smoke alarm", "infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s13B": {
        "must_contain_any": ["agreement", "written", "copy", "landlord"],
        "must_not_contain_any": ["smoke alarm", "infringement fee"],
    },
    "NZLEG/RTA/s18": {
        "must_contain_any": ["bond", "receipt", "weeks", "landlord"],
        "must_not_contain_any": ["smoke alarm", "infringement fee", "19(2)"],
    },
    "NZLEG/RTA/s28": {
        "must_contain_any": ["rent", "increase", "notice"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s28A": {
        "must_contain_any": ["rent", "increase", "order"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s40": {
        "must_contain_any": ["tenant", "clean", "land", "garden", "responsibilities"],
        "must_not_contain_any": ["40(3A)(a)", "infringement fee", "1,800", "4,000"],
    },
    "NZLEG/RTA/s42A": {
        "must_contain_any": ["fixture", "improvement", "consent"],
        "must_not_contain_any": ["42A(7)", "infringement fee", "1,500"],
    },
    "NZLEG/RTA/s42B": {
        "must_contain_any": ["minor", "change"],
        "must_not_contain_any": ["42B(3)", "42B(6)", "infringement fee", "1,500"],
    },
    "NZLEG/RTA/s45": {
        "must_contain_any": ["repair", "maintain", "landlord"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s48": {
        "must_contain_any": ["entry", "enter", "landlord"],
        "must_not_contain_any": ["48(5)", "infringement fee"],
    },
    "NZLEG/RTA/s49A": {
        "must_contain_any": ["wear", "tear", "tenant"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s49B": {
        "must_contain_any": ["damage", "tenant", "landlord"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
    "NZLEG/RTA/s51": {
        "must_contain_any": ["terminate", "notice", "periodic"],
        "must_not_contain_any": ["infringement fee", "Schedule 1A"],
    },
}


def _all_forced_sections() -> set[str]:
    sections: set[str] = set()
    for route in ROUTES:
        sections.update(route.forced_sections)
    return sections


@pytest.fixture(scope="module")
def leg_store():
    try:
        return VectorStore()
    except Exception as exc:
        pytest.skip(f"Qdrant not available: {exc}")


def test_all_forced_sections_have_expectations():
    """Every forced_section must have a SECTION_EXPECTATIONS entry."""
    missing = _all_forced_sections() - set(SECTION_EXPECTATIONS)
    assert not missing, (
        f"Sections in ROUTES with no expectation entry: {sorted(missing)}. "
        "Add them to SECTION_EXPECTATIONS in tests/test_rta_routes.py."
    )


@pytest.mark.parametrize("section_id", sorted(_all_forced_sections()))
def test_forced_section_exists_in_qdrant(section_id, leg_store):
    results = leg_store.get_by_case_id(section_id)
    assert results, (
        f"{section_id} not found in Qdrant. "
        "The corpus may use a different ID - verify with the Qdrant scroll API."
    )


@pytest.mark.parametrize("section_id", sorted(_all_forced_sections()))
def test_forced_section_content(section_id, leg_store):
    exp = SECTION_EXPECTATIONS.get(section_id)
    if not exp:
        pytest.skip(f"No expectations defined for {section_id}")

    results = leg_store.get_by_case_id(section_id)
    if not results:
        pytest.fail(f"{section_id} not found in Qdrant")

    combined = " ".join(r.text.lower() for r in results)

    must_any = exp.get("must_contain_any", [])
    if must_any:
        assert any(term.lower() in combined for term in must_any), (
            f"{section_id}: text does not contain any of {must_any!r}. "
            f"First 300 chars: {combined[:300]!r}"
        )

    for term in exp.get("must_not_contain_any", []):
        assert term.lower() not in combined, (
            f"{section_id}: contains forbidden term {term!r}. "
            "This ID may resolve to regulation text or a penalty table row, not the main Act."
        )
