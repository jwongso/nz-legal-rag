"""Unit tests for _extract_rta_section() - the heading-aware RTA section extractor.

These tests are fully deterministic: no network calls, no Qdrant, no llama-server.
They use a synthetic RTA page text that mimics the structure of the real
legislation.govt.nz page, including both substantive sections and Schedule 1A
penalty table rows that triggered the anchor-extraction bug.

Run:
    pytest tests/test_rta_extractor.py -v
"""

import pytest
from tenancy.app import _extract_rta_section

# ---------------------------------------------------------------------------
# Synthetic RTA page fixture
#
# Mirrors the real legislation.govt.nz page structure:
#   - Substantive sections appear first, each with a heading and subsections.
#   - Schedule 1A appears later as a penalty/infringement-fee table.
#   - The same section numbers appear in BOTH places, which caused the bug:
#     "42A(7)" in the penalty table was matched before "42A  Consent..." heading.
# ---------------------------------------------------------------------------

_FAKE_RTA_PAGE = """
Residential Tenancies Act 1986

Public Act     1986 No 120
Date of assent 18 December 1986
Commencement   see section 1(2)


40  Responsibilities of tenant

(1)  Every tenant shall-

(a)  pay the rent in the manner and at the time specified in the tenancy
     agreement; and

(b)  use the premises principally for residential purposes; and

(c)  keep the premises in a reasonable state of cleanliness and tidiness,
     including any garden, lawn, or other outdoor area forming part of the
     premises; and

(d)  not damage, or permit any other person to damage, the premises.

(2)  A tenant who fails to comply with subsection (1) is liable to the
     landlord for any reasonable costs incurred as a result of that failure.

(3)  For the purposes of this section, premises includes the land on which
     any dwelling house is situated.

(3A)  In addition to the obligations in subsection (1), every tenant shall-

(a)  not use or permit the premises to be used for any unlawful purpose; and

(b)  not interfere with any means of escape from fire in or on the premises.


42A  Consent for tenant's fixtures, etc

(1)  The landlord must not unreasonably withhold consent to any fixture,
     renovation, alteration, addition, or improvement that a tenant proposes
     to make to the premises.

(2)  Any consent must be in writing.

(3)  Without limiting subsection (1), it is unreasonable for the landlord to
     withhold consent unless-

(a)  the proposed fixture, renovation, alteration, addition, or improvement
     is likely to cause serious damage to the premises; or

(b)  the landlord has reasonable grounds for believing that the proposed
     fixture, renovation, alteration, addition, or improvement would be
     unlawful.

(4)  A landlord must respond to a tenant's written request for consent within
     21 days of receiving the request.

(5)  If a landlord does not respond within 21 days, the landlord is treated
     as having given consent.

(6)  A tenant who makes a fixture, renovation, alteration, addition, or
     improvement with the landlord's consent must, on the termination of the
     tenancy, restore the premises to the condition they were in before the
     fixture, renovation, alteration, addition, or improvement was made,
     unless the landlord agrees otherwise.

(7)  See Schedule 1A for the infringement offence relating to this section.


42B  Minor changes

(1)  Without limiting section 42A(1), it is unreasonable for a landlord to
     withhold consent to a minor change to the premises.

(2)  A minor change is a change that-

(a)  does not affect the structure of the premises; and

(b)  is reasonably capable of being reversed when the tenancy ends; and

(c)  does not require a building consent.

(3)  The landlord must respond to a request for consent to a minor change
     within 21 days of receiving the request.

(4)  If the landlord does not respond within 21 days, the landlord is treated
     as having given consent.

(5)  A tenant who makes a minor change with the landlord's consent must, at
     the end of the tenancy, restore the premises unless the landlord agrees
     otherwise.

(6)  See Schedule 1A for infringement offences.


48  Landlord's right of entry

(1)  A landlord may enter premises that are subject to a tenancy-

(a)  with the consent of the tenant; or

(b)  where the tenancy agreement so provides, for the purposes and subject
     to the conditions specified in the agreement; or

(c)  in any case, after giving the tenant not less than 24 hours notice in
     writing of the intention to enter and the reason for entry.

(2)  A landlord shall not exercise the right of entry between 10 pm and 8 am.

(3)  A tenant may refuse entry to the landlord if the landlord fails to
     comply with subsection (1).


---

Schedule 1A
Unlawful acts and penalties - Infringement fees

Infringement notices may be issued under this Schedule. Maximum amount
for each infringement offence is as set out below.

Section     Description                                                       Amount

16A(6)      Landlord failing to appoint agent when outside New Zealand        1,500
            for longer than 21 consecutive days

17(3)       Requiring key money                                                1,500

17A(3)      Requiring letting fee                                              1,500

18(4)(a)    Landlord requiring general bond greater than amount permitted      1,500

19(2)       Breaching duties on receipt of bond                                1,500

22F(3)(a)   Landlord failing to state amount of rent when offering tenancy     1,500

23(4)(a)    Landlord requiring rent more than 2 weeks in advance               1,500

38(3)       Interference with privacy of tenant                                3,000

40(3A)(a)   Tenant failing, without reasonable excuse, to quit premises        1,500
            upon termination

40(3A)(b)   Tenant's interference with means of escape from fire               4,000

40(3A)(c)   Tenant using or permitting premises to be used for unlawful        1,800
            purpose

42A(7)      Landlord failing to respond to written request seeking consent     1,500
            for fixtures, etc

42B(3)      Landlord failing to consent to request for minor change            1,500

42B(6)      Tenant failing to reinstate premises at end of tenancy following   1,500
            minor change

48(5)       Landlord entering premises without required notice                 1,500

"""


# ---------------------------------------------------------------------------
# Positive tests: must return substantive section text
# ---------------------------------------------------------------------------

class TestExtractReturnsSubstantiveText:

    def test_s40_returns_tenant_responsibilities(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "40")
        assert result is not None, "s40 extraction returned None"
        assert "tenant" in result.lower()
        assert "responsibilities" in result.lower() or "clean" in result.lower()

    def test_s40_contains_premises_includes_land(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "40")
        assert result is not None
        assert "land" in result.lower() or "garden" in result.lower()

    def test_s42a_returns_fixtures_and_consent(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "42A")
        assert result is not None, "s42A extraction returned None"
        text = result.lower()
        assert "consent" in text
        assert "fixture" in text or "improvement" in text or "renovation" in text

    def test_s42a_contains_unreasonably_withhold(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "42A")
        assert result is not None
        assert "unreasonably" in result.lower()

    def test_s42b_returns_minor_changes(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "42B")
        assert result is not None, "s42B extraction returned None"
        assert "minor" in result.lower()
        assert "change" in result.lower()

    def test_s48_returns_landlord_entry(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "48")
        assert result is not None, "s48 extraction returned None"
        text = result.lower()
        assert "entry" in text or "enter" in text
        assert "landlord" in text

    def test_s48_contains_notice_requirement(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "48")
        assert result is not None
        assert "24" in result or "notice" in result.lower()

    def test_result_contains_subsection_markers(self):
        for num in ("40", "42A", "42B", "48"):
            result = _extract_rta_section(_FAKE_RTA_PAGE, num)
            assert result is not None, f"s{num} returned None"
            assert "(1)" in result, f"s{num} result has no subsection markers"


# ---------------------------------------------------------------------------
# Negative tests: must NOT return penalty-table content
# ---------------------------------------------------------------------------

class TestExtractRejectsScheduleContent:

    def test_s40_excludes_penalty_table_row(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "40")
        assert result is not None
        assert "40(3A)(a)" not in result
        assert "40(3A)(b)" not in result
        assert "40(3A)(c)" not in result

    def test_s40_excludes_infringement_fee_amount(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "40")
        assert result is not None
        # Penalty table amounts should not appear in extracted section text
        assert "1,800" not in result
        assert "4,000" not in result

    def test_s42a_excludes_penalty_row_42a7(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "42A")
        assert result is not None
        assert "42A(7)" not in result

    def test_s42a_excludes_infringement_fee(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "42A")
        assert result is not None
        assert "infringement fee" not in result.lower()
        assert "infringement" not in result.lower() or "notice" not in result.lower()

    def test_s42a_excludes_penalty_table_amounts(self):
        # s42A(7) legitimately cross-references Schedule 1A inline, so "Schedule 1A"
        # may appear in the result. What must NOT appear is the actual penalty table
        # row format: "42A(7)   description   1,500".
        result = _extract_rta_section(_FAKE_RTA_PAGE, "42A")
        assert result is not None
        assert "1,500" not in result
        assert "42A(7)      Landlord" not in result

    def test_s42b_excludes_penalty_rows(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "42B")
        assert result is not None
        assert "42B(3)" not in result
        assert "42B(6)" not in result

    def test_s42b_excludes_penalty_amount(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "42B")
        assert result is not None
        assert "1,500" not in result

    def test_s48_excludes_penalty_row_48_5(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "48")
        assert result is not None
        assert "48(5)" not in result

    def test_s48_excludes_infringement_fee(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "48")
        assert result is not None
        assert "infringement fee" not in result.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestExtractEdgeCases:

    def test_nonexistent_section_returns_none(self):
        result = _extract_rta_section(_FAKE_RTA_PAGE, "999")
        assert result is None

    def test_empty_text_returns_none(self):
        result = _extract_rta_section("", "40")
        assert result is None

    def test_text_with_only_penalty_table_returns_none(self):
        penalty_only = """
Schedule 1A
Infringement fees

42A(7)      Landlord failing to respond to written request seeking consent     1,500
            for fixtures, etc

42B(3)      Landlord failing to consent to request for minor change            1,500
"""
        assert _extract_rta_section(penalty_only, "42A") is None
        assert _extract_rta_section(penalty_only, "42B") is None

    def test_section_without_subsections_returns_none(self):
        no_subsections = """
40  Responsibilities of tenant

General heading with no subsection markers.
"""
        result = _extract_rta_section(no_subsections, "40")
        assert result is None, "Should reject headings with no (1), (2) subsection markers"

    def test_result_length_is_reasonable(self):
        for num in ("40", "42A", "42B", "48"):
            result = _extract_rta_section(_FAKE_RTA_PAGE, num)
            assert result is not None
            assert len(result) > 100, f"s{num} result suspiciously short: {len(result)} chars"
            assert len(result) <= 1800, f"s{num} result exceeds 1800-char cap: {len(result)} chars"
