"""Intent-sensitive legal authority ranker applied after Qdrant dedup, before cross-encoder.

Four ranker profiles, each tuned for a different query intent:

  authority_mode  - legal principle / doctrine queries ("what is the test for X")
                    boosts court hierarchy, citation density, early chunks
  example_mode    - search-for-cases queries ("find ERA decisions about redundancy in 2023")
                    respects court-level filter, boosts recency, avoids penalising tribunals
  tracker_mode    - quantitative / structured queries ("typical guilty plea discount")
                    low authority weight, boosts chunks that carry structured payload fields
                    (sentencing, employment outcomes, compensation amounts)
  statute_mode    - specific section queries ("what does s103A require")
                    maximum boost for NZLEG legislation chunks, then cases citing the section

The planner should set RankerMode explicitly. If it cannot, QueryContext.from_query()
infers the mode from the query text using heuristics.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Court authority weights (shared across all profiles, scaled by authority_weight)
# ---------------------------------------------------------------------------

_COURT_AUTHORITY: dict[str, float] = {
    "NZSC":    1.00,
    "NZLEG":   0.95,
    "NZCA":    0.85,
    "NZHC":    0.70,
    "NZEmpC":  0.65,
    "NZEnvC":  0.60,
    "NZFC":    0.55,
    "NZERA":   0.50,
    "NZACC":   0.50,
    "NZHRRT":  0.50,
    "NZTT":    0.45,
    "NZCorC":  0.45,
    "NZLCDT":  0.40,
    "NZREADT": 0.40,
}
_DEFAULT_AUTHORITY = 0.45

_TRACKER_LEGAL_AREAS = {"sentencing", "employment", "personal_grievance", "accident_compensation"}


# ---------------------------------------------------------------------------
# Ranker modes and profiles
# ---------------------------------------------------------------------------

class RankerMode(str, Enum):
    AUTHORITY = "authority"  # legal principle / doctrine
    EXAMPLE   = "example"   # find cases, recent decisions
    TRACKER   = "tracker"   # sentencing / PG structured data queries
    STATUTE   = "statute"   # specific section lookup


@dataclass
class RankerProfile:
    mode: RankerMode
    authority_weight: float     # fraction of final score contributed by court hierarchy
    early_chunk_weight: float   # max boost for chunk_index=0 (decays by 0.005 per chunk)
    citation_rich_weight: float # max boost for citation-dense chunks (scales with count)
    statute_boost: float        # extra boost for legislation doc_type (statute/authority modes)
    recency_boost: float        # boost for docs newer than recency_threshold_year
    recency_threshold_year: int
    tracker_boost: float        # boost for chunks with structured sentencing/employment payload


_PROFILES: dict[RankerMode, RankerProfile] = {
    RankerMode.AUTHORITY: RankerProfile(
        mode=RankerMode.AUTHORITY,
        authority_weight=0.18,
        early_chunk_weight=0.015,
        citation_rich_weight=0.020,
        statute_boost=0.04,
        recency_boost=0.010,
        recency_threshold_year=2022,
        tracker_boost=0.0,
    ),
    RankerMode.EXAMPLE: RankerProfile(
        mode=RankerMode.EXAMPLE,
        authority_weight=0.04,    # low - do not penalise tribunal courts
        early_chunk_weight=0.008,
        citation_rich_weight=0.008,
        statute_boost=0.0,
        recency_boost=0.050,      # high - examples should be recent
        recency_threshold_year=2021,
        tracker_boost=0.0,
    ),
    RankerMode.TRACKER: RankerProfile(
        mode=RankerMode.TRACKER,
        authority_weight=0.05,    # low - exact match matters more than court prestige
        early_chunk_weight=0.010,
        citation_rich_weight=0.008,
        statute_boost=0.0,
        recency_boost=0.025,
        recency_threshold_year=2021,
        tracker_boost=0.040,      # boost chunks carrying structured sentencing / PG fields
    ),
    RankerMode.STATUTE: RankerProfile(
        mode=RankerMode.STATUTE,
        authority_weight=0.22,    # high - NZLEG and binding courts matter most
        early_chunk_weight=0.015,
        citation_rich_weight=0.020,
        statute_boost=0.15,       # legislation chunks win clearly over case law
        recency_boost=0.005,      # statutes change slowly; recency matters less
        recency_threshold_year=2022,
        tracker_boost=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Intent detection regexes
# ---------------------------------------------------------------------------

# Section reference: "s 103A", "section 103", "s127" etc.
_SECTION_RE = re.compile(
    r"\b(s\s*\d+[A-Z]*|section\s+\d+[A-Z]*)\b", re.IGNORECASE
)

# Structured / quantitative signals for tracker queries.
# These are query-TYPE signals (quantitative, range, structured outcome), NOT topic words.
# Topic words like "unjustified dismissal", "redundancy", "manslaughter" are intentionally
# excluded - they appear in example queries too. Only include signals that indicate the
# user wants aggregate/structured data rather than specific case examples.
_TRACKER_RE = re.compile(
    r"\b(starting\s+points?|guilty\s+plea|home\s+detention|"
    r"sentencing\s+ranges?|sentence\s+ranges?|"
    r"contributory\s+conduct|compensation\s+award|"
    r"typical\s+sentence|usual\s+sentence|average\s+sentence|"
    r"typical\s+discount|usual\s+discount|"
    r"how\s+much|what\s+range|what\s+discount|"
    r"sentence\s+reduc|appeal\s+sentence|appeal\s+varied)\b",
    re.IGNORECASE,
)

# Search-for-examples / recent-case signals
_EXAMPLE_RE = re.compile(
    r"\b(find|show|examples?\s+of|cases?\s+where|decisions?\s+where|"
    r"decisions?\s+about|decisions?\s+on|recent|"
    r"ERA\s+decisions?|NZTT\s+decisions?|tribunal\s+decisions?|"
    r"employment\s+court\s+decisions?)\b",
    re.IGNORECASE,
)

# Recency signals
_RECENCY_RE = re.compile(
    r"\b(latest|recent|current|2022|2023|2024|2025|2026)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# QueryContext
# ---------------------------------------------------------------------------

@dataclass
class QueryContext:
    mode: RankerMode = RankerMode.AUTHORITY
    wants_recent: bool = False

    @classmethod
    def from_query(cls, question: str) -> "QueryContext":
        """Infer ranker mode from query text.

        Priority: statute > tracker > example > authority (default).
        The planner can bypass this by constructing QueryContext directly with an explicit mode.
        """
        wants_recent = bool(_RECENCY_RE.search(question))

        if _SECTION_RE.search(question):
            mode = RankerMode.STATUTE
        elif _TRACKER_RE.search(question):
            mode = RankerMode.TRACKER
        elif _EXAMPLE_RE.search(question) or wants_recent:
            mode = RankerMode.EXAMPLE
        else:
            mode = RankerMode.AUTHORITY

        return cls(mode=mode, wants_recent=wants_recent)

    @property
    def profile(self) -> RankerProfile:
        return _PROFILES[self.mode]


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

def rerank(hits: list, ctx: QueryContext, top_k: int | None = None) -> list:
    """Re-score and sort hits using the profile selected by ctx.mode.

    Original .score on each SearchResult is not mutated.
    Returns a new list sorted descending by legal-blended score.
    """
    profile = ctx.profile
    scored: list[tuple[float, object]] = []

    for h in hits:
        p = h.payload

        # Court authority
        court = p.get("court", "")
        authority = _COURT_AUTHORITY.get(court, _DEFAULT_AUTHORITY)

        # Early chunk boost (decays linearly per chunk position)
        chunk_index = p.get("chunk_index", 99)
        early_boost = max(0.0, profile.early_chunk_weight - chunk_index * 0.005)

        # Citation density boost
        citations = p.get("citations") or []
        citation_boost = min(len(citations), 5) / 5.0 * profile.citation_rich_weight

        # Statute boost: legislation chunks on statute-intent queries
        statute_boost = 0.0
        if profile.statute_boost > 0 and p.get("document_type", "") == "legislation":
            statute_boost = profile.statute_boost

        # Recency boost
        recency_boost = 0.0
        if profile.recency_boost > 0 and ctx.wants_recent:
            year = p.get("year") or 0
            if year >= profile.recency_threshold_year:
                recency_boost = profile.recency_boost

        # Tracker boost: chunks carrying structured sentencing / employment payload
        tracker_boost = 0.0
        if profile.tracker_boost > 0:
            has_structured = (
                p.get("sentencing") is not None
                or p.get("legal_area", "") in _TRACKER_LEGAL_AREAS
            )
            if has_structured:
                tracker_boost = profile.tracker_boost

        legal_score = (
            h.score * (1.0 - profile.authority_weight)
            + authority * profile.authority_weight
            + early_boost
            + citation_boost
            + statute_boost
            + recency_boost
            + tracker_boost
        )
        scored.append((legal_score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [h for _, h in scored]
    return result[:top_k] if top_k is not None else result
