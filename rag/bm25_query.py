"""Conditional BM25 query builder for legal retrieval.

BM25 is activated only for queries where exact keyword matching adds signal
over vector-only search. For broad natural language questions it is suppressed,
because AND-matching via websearch_to_tsquery requires all stems to co-occur
in the same chunk and systematically excludes relevant documents.

Activation rules (any one is sufficient):
  1. Section reference   s103A, section 103A, s 127
  2. Case citation        NZCA/2024/50, [2024] NZCA 50
  3. Quoted phrase        "interim reinstatement"
  4. Short keyword query  <= 6 tokens, no question words

When BM25 is activated, query_terms is an OR-joined set of specific anchors
extracted from the question. Broad terms from a general question are never
passed into BM25; only the anchors are. This avoids the AND-strictness problem.

Examples:
  "s103A"                                    -> activate, terms="\"s103A\""
  "section 127 Employment Relations Act"     -> activate, terms="\"section 127\" OR ..."
  "interim reinstatement"                    -> activate (short keyword), terms=question
  "guilty plea discount"                     -> activate (short keyword), terms=question
  "What does a fair dismissal process require?" -> suppress (broad question)
  "What obligations does the Privacy Act place on employers?" -> suppress (broad)
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"\b(s\s*\d+[A-Z]*|section\s+\d+[A-Z]*)\b", re.IGNORECASE
)

_ACT_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Za-z]+)+\s+Act(?:\s+\d{4})?)\b"
)

_CITATION_RE = re.compile(
    r"\b[A-Z]+/\d{4}/\d+\b|\[\d{4}\]\s+NZ[A-Z]+\s+\d+"
)

_QUOTED_RE = re.compile(r'"([^"]{2,})"')

_QUESTION_WORDS = frozenset({
    "what", "how", "when", "where", "why", "which", "who",
    "does", "do", "can", "is", "are", "was", "were", "will",
})

# Legal phrases added to the BM25 query when they appear in an activated query.
# Sorted longest-first so longer phrases match before shorter substrings.
_LEGAL_PHRASES: list[str] = sorted([
    "guilty plea discount",
    "guilty plea",
    "starting point",
    "interim reinstatement",
    "constructive dismissal",
    "unjustified dismissal",
    "personal grievance",
    "home detention",
    "minimum period of imprisonment",
    "minimum non-parole period",
    "contributory conduct",
    "aggravated robbery",
    "grievous bodily harm",
    "sentencing range",
    "sentence range",
    "appeal against sentence",
    "sentence varied",
    "previous convictions",
    "good faith",
    "natural justice",
    "unjustified disadvantage",
], key=len, reverse=True)

_LEGAL_PHRASE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _LEGAL_PHRASES) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class BM25Query:
    should_use: bool
    query_terms: str | None   # websearch_to_tsquery-compatible OR-joined string
    reason: str               # for tracing and logging


def build_bm25_query(question: str) -> BM25Query:
    """Decide whether to activate BM25 and build an OR-anchored key-term string.

    Returns BM25Query(should_use=False) for broad natural language questions.
    When should_use=True, query_terms is safe to pass directly to
    websearch_to_tsquery('english', ...) which supports OR and quoted phrases.

    When BM25 is activated, extracted terms are OR-joined so the FTS query
    becomes: chunk must match AT LEAST ONE anchor, not all of them.
    """
    words = question.strip().split()
    first_word = words[0].lower() if words else ""
    is_short_keyword = (
        len(words) <= 6
        and first_word not in _QUESTION_WORDS
        and not question.strip().endswith("?")
    )

    # Step 1: determine whether to activate
    has_section = bool(_SECTION_RE.search(question))
    has_citation = bool(_CITATION_RE.search(question))
    has_quoted = bool(_QUOTED_RE.search(question))
    should_use = has_section or has_citation or has_quoted or is_short_keyword

    if not should_use:
        return BM25Query(
            should_use=False,
            query_terms=None,
            reason="broad_question:no_keyword_anchors",
        )

    # Step 2: extract terms for the BM25 query string
    terms: list[str] = []
    reasons: list[str] = []

    for m in _SECTION_RE.finditer(question):
        raw = m.group(0).strip()
        terms.append(f'"{raw}"')
        reasons.append(f"section:{raw}")

    for m in _ACT_RE.finditer(question):
        name = m.group(0).strip()
        terms.append(f'"{name}"')
        reasons.append(f"act:{name}")

    for m in _CITATION_RE.finditer(question):
        cit = m.group(0)
        terms.append(f'"{cit}"')
        reasons.append(f"citation:{cit}")

    for m in _QUOTED_RE.finditer(question):
        phrase = m.group(1).strip()
        terms.append(f'"{phrase}"')
        reasons.append(f"quoted:{phrase}")

    seen_phrases: set[str] = set()
    for m in _LEGAL_PHRASE_RE.finditer(question):
        phrase = m.group(0).lower()
        if phrase not in seen_phrases:
            seen_phrases.add(phrase)
            terms.append(f'"{m.group(0)}"')
            reasons.append(f"phrase:{phrase}")

    # For a short keyword query with no extracted anchors, use the raw text
    if not terms and is_short_keyword:
        terms.append(question.strip())
        reasons.append("raw_keyword_query")

    if not terms:
        # Activation criteria met but nothing to extract - should not happen,
        # but fall back to suppress rather than an empty query.
        return BM25Query(
            should_use=False,
            query_terms=None,
            reason="activated_but_no_terms_extracted",
        )

    return BM25Query(
        should_use=True,
        query_terms=" OR ".join(terms),
        reason=", ".join(reasons),
    )
