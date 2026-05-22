"""Heuristic court planner for NZ legal queries.

Maps query text to NZ court codes for SQL pre-filtering. Used in production when
the caller does not provide an explicit court filter, and in the planner benchmark
to measure how much quality is lost vs oracle (gold expected_courts).

Returns None when no domain signal is detected -- caller falls back to full corpus.

Key design decisions (benchmarked in benchmarks/runners/run_retrieval.py):
- Employment signals -> NZERA only by default. NZEmpC added only when "Employment
  Court" is explicitly mentioned. This avoids diluting NZERA-specific queries.
- NZLEG triggered only by explicit section references or ERA-authority mentions,
  not by named-Act mentions alone. Adding NZLEG for every Privacy Act / ERA Act
  mention increased the candidate pool and caused coverage regressions.
"""

import re
from dataclasses import dataclass

# Matches years only in temporal phrases: "in 2023", "from 2024", "2023 decisions".
# Deliberately does NOT match "Privacy Act 2020" or "Act 2021" (Act name years).
_YEAR_RE = re.compile(
    r'\b(?:in|from|during|of)\s+(20[2-3]\d)\b'
    r'|\b(20[2-3]\d)\s+(?:decisions?|cases?|judgments?)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Each list is (pattern, signal_name) pairs; pattern is a regex string.
# Patterns are checked case-insensitively against the lowercased query.
# `check()` stops at the first match per group (one signal per domain is enough).

_ACC = [
    (r"\bacc\b(?!\w)", "acc_keyword"),
    (r"accident compensation", "accident_compensation"),
    (r"acc[- ]covered", "acc_covered"),
    (r"acc claim", "acc_claim"),
    (r"earners.{0,5}levy", "earners_levy"),
]

_CRIMINAL = [
    (r"\bsentenc", "sentencing"),
    (r"\bguilty plea\b", "guilty_plea"),
    (r"\bhome detention\b", "home_detention"),
    (r"\bimprisonment\b", "imprisonment"),
    (r"\bstarting point\b", "starting_point"),
    (r"\baggravated\b", "aggravated"),
    (r"\bmanslaughter\b", "manslaughter"),
    (r"\bmurder\b", "murder"),
    (r"\brobbery\b", "robbery"),
    (r"\bcriminal\b", "criminal"),
    (r"\boffending\b", "offending"),
    (r"\bappeal against sentence\b", "appeal_against_sentence"),
    (r"\bprevious convictions\b", "previous_convictions"),
    (r"\bcourt of appeal\b", "court_of_appeal"),
]

# Employment signals map to NZERA only. NZEmpC is added separately when
# "Employment Court" is explicitly named. This prevents diluting NZERA-specific
# queries (e.g., "Find ERA decisions about redundancy") with 300 NZEmpC docs.
_EMPLOYMENT_ERA = [
    (r"\bemployers?\b", "employer"),
    (r"\bemployees?\b", "employee"),
    (r"\bemployment\b", "employment"),
    (r"\bdismissal\b", "dismissal"),
    (r"\bredundancy\b", "redundancy"),
    (r"\breinstatement\b", "reinstatement"),
    (r"\bpersonal grievance\b", "personal_grievance"),
    (r"\bsick leave\b", "sick_leave"),
    (r"\bunjustified\b", "unjustified"),
    (r"\bconstructive dismissal\b", "constructive_dismissal"),
    (r"\bworkplace harassment\b", "workplace_harassment"),
    (r"\bemployment relations act\b", "employment_relations_act"),
    (r"\bcontributory conduct\b", "contributory_conduct"),
    (r"\bunjustified disadvantage\b", "unjustified_disadvantage"),
    (r"\bmedical certificate\b", "medical_certificate"),
    (r"\bera decisions?\b", "era_decisions"),
    (r"\bera determination\b", "era_determination"),
]

# NZEmpC-specific: triggered only when Employment Court is explicitly named.
_EMPLOYMENT_COURT = [
    (r"\bemployment court\b", "employment_court"),
    (r"\bnzempc\b", "nzempc"),
]

_TENANCY = [
    (r"\blandlord\b", "landlord"),
    (r"\btenant\b", "tenant"),
    (r"\btenancy\b", "tenancy"),
    (r"\btenancies\b", "tenancies"),
    (r"\brental property\b", "rental_property"),
    (r"\brent increase\b", "rent_increase"),
    (r"\bperiodic tenancy\b", "periodic_tenancy"),
    (r"\bresidential tenancies act\b", "residential_tenancies_act"),
    (r"\bbond refund\b", "bond_refund"),
    (r"\btenancy tribunal\b", "tenancy_tribunal"),
]

_HUMAN_RIGHTS = [
    (r"\bhuman rights act\b", "human_rights_act"),
    (r"\bprivacy act\b", "privacy_act"),
    (r"\bdiscrimination\b", "discrimination"),
    (r"\bequal opportunities\b", "equal_opportunities"),
    (r"\bnzhrrt\b", "nzhrrt"),
    (r"\bhuman rights review tribunal\b", "hrrt"),
]

# NZLEG triggered by explicit section references, named NZ Acts, or ERA-authority
# mentions. Named-Act triggers are safe because the dilution regressions in V1 of
# the planner were caused by NZEmpC co-triggering with employment signals, not by
# NZLEG from named Acts. With NZERA-only default for employment, NZLEG from named
# Acts is no longer a regression risk.
#
# Section ref patterns use [A-Za-z]* (not [A-Z]*) because the query is lowercased
# before matching: "section 103A" becomes "section 103a".
_LEGISLATION = [
    (r"\bsection\s+\d+[A-Za-z]*\b", "section_ref"),
    (r"\bs\s*\d{2,}[A-Za-z]*\b", "section_abbrev"),
    (r"\bemployment relations act\b", "era_act"),
    (r"\bresidential tenancies act\b", "rta_act"),
    (r"\bholidays act\b", "holidays_act"),
    (r"\bcrimes act\b", "crimes_act"),
    # Privacy Act and Human Rights Act are NOT included here -- they are NZHRRT
    # signals already handled by _HUMAN_RIGHTS. Adding them to NZLEG inflates
    # the candidate pool with legislation chunks and causes ranking regressions
    # on queries that seek court decisions, not statute text.
    (r"\bhealth and safety at work act\b", "hswa_act"),
    # ERA as a statutory body implies the Employment Relations Act text is relevant
    (r"\bemployment relations authority\b", "era_authority"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CourtPlan:
    courts: list[str] | None  # None = no filter (full corpus)
    years: list[int] | None   # year filter extracted from query, or None
    signals: list[str]         # triggered signal names, for tracing
    confidence: str            # "high" | "medium" | "low"


def plan_courts(query: str) -> CourtPlan:
    """Detect likely NZ courts from query text using heuristic signals.

    When uncertain between narrowing and broadening, prefer to broaden (include
    more courts) rather than risk excluding the relevant court. The exception is
    NZERA vs NZEmpC: NZEmpC is only added when explicitly named to avoid diluting
    NZERA-specific queries with Employment Court documents.

    Returns CourtPlan(courts=None) when no domain signal is detected.
    """
    q = query.lower()
    courts: set[str] = set()
    signals: list[str] = []

    def check(patterns, court_set):
        for pat, name in patterns:
            if re.search(pat, q):
                courts.update(court_set)
                signals.append(name)
                return  # first match per domain group is enough

    check(_ACC, {"NZACC"})
    check(_CRIMINAL, {"NZCA", "NZHC"})
    check(_EMPLOYMENT_ERA, {"NZERA"})
    check(_EMPLOYMENT_COURT, {"NZEmpC"})
    check(_TENANCY, {"NZTT"})
    check(_HUMAN_RIGHTS, {"NZHRRT"})
    check(_LEGISLATION, {"NZLEG"})

    # Year extraction: years in temporal phrases only (e.g., "in 2023", "2024 decisions").
    # The two-group regex has group(1) for "in/from YEAR" and group(2) for "YEAR decisions".
    year_matches = [
        int(m.group(1) or m.group(2)) for m in _YEAR_RE.finditer(query)
    ]
    years = sorted(set(year_matches)) if year_matches else None

    if not courts:
        return CourtPlan(courts=None, years=years, signals=[], confidence="low")

    # Confidence: how many distinct non-legislation domains matched
    n_domains = sum([
        any(re.search(p, q) for p, _ in _ACC),
        any(re.search(p, q) for p, _ in _CRIMINAL),
        any(re.search(p, q) for p, _ in _EMPLOYMENT_ERA),
        any(re.search(p, q) for p, _ in _TENANCY),
        any(re.search(p, q) for p, _ in _HUMAN_RIGHTS),
    ])
    confidence = "high" if n_domains == 1 else "medium" if n_domains == 2 else "low"

    return CourtPlan(courts=sorted(courts), years=years, signals=signals, confidence=confidence)
