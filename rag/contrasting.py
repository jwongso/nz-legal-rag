"""Similar Cases With Opposite Outcomes - contrastive retrieval (Phase 4, #6).

Pipeline:
  1. Embed the query (caller's responsibility)
  2. Two Qdrant semantic searches, one per outcome group, each with a filter
     restricting results to chunks that carry the target structured outcome
  3. Deduplicate within each group (one case per case_id, best score wins)
  4. Return both groups - caller may optionally pass to the LLM for explanation
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range


# ---------------------------------------------------------------------------
# Split configurations
# ---------------------------------------------------------------------------

# Each split entry: (filter_key, filter_value, label, description)
_SPLITS: dict[str, dict[str, tuple]] = {
    "criminal": {
        "sentence_type": (
            ("sentencing.sentence_type", "imprisonment",
             "Imprisonment", "Cases where imprisonment was imposed"),
            ("sentencing.sentence_type", "home_detention",
             "Home detention", "Cases where home detention was imposed"),
        ),
        "guilty_plea": (
            ("sentencing.has_guilty_plea", True,
             "Guilty plea", "Cases where a guilty plea was entered"),
            ("sentencing.has_guilty_plea", False,
             "No guilty plea", "Cases without a guilty plea"),
        ),
    },
    "employment": {
        "reinstatement": (
            ("pg.reinstatement_ordered", True,
             "Reinstatement ordered", "Cases where reinstatement was ordered"),
            ("pg.reinstatement_ordered", False,
             "Reinstatement declined", "Cases where reinstatement was declined"),
        ),
    },
}

_DEFAULT_SPLIT = {"criminal": "sentence_type", "employment": "reinstatement"}
_HAS_DATA_KEY  = {"criminal": "sentencing.has_data", "employment": "pg.has_data"}
_STRUCT_KEY    = {"criminal": "sentencing",           "employment": "pg"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContrastingCase:
    case_id: str
    title: str
    court_name: str
    date: str
    url: str
    score: float
    structured: dict  # sentencing.* or pg.* fields, nulls removed

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "court_name": self.court_name,
            "date": self.date,
            "url": self.url,
            "score": round(self.score, 4),
            "structured": self.structured,
        }


@dataclass
class ContrastingGroup:
    label: str
    description: str
    cases: list[ContrastingCase] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "description": self.description,
            "cases": [c.to_dict() for c in self.cases],
        }


@dataclass
class ContrastingResult:
    query: str
    domain: str
    split_by: str
    group_a: ContrastingGroup
    group_b: ContrastingGroup
    explanation: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "domain": self.domain,
            "split_by": self.split_by,
            "group_a": self.group_a.to_dict(),
            "group_b": self.group_b.to_dict(),
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_split_config(
    domain: str,
    split_by: str | None,
) -> tuple[str, tuple, tuple] | None:
    """Return (split_name, group_a_cfg, group_b_cfg) or None if invalid."""
    domain_splits = _SPLITS.get(domain)
    if not domain_splits:
        return None
    split_name = split_by or _DEFAULT_SPLIT.get(domain)
    split = domain_splits.get(split_name)
    if not split:
        return None
    return split_name, split[0], split[1]


def _build_filter(
    outcome_key: str,
    outcome_value,
    has_data_key: str,
    courts: list[str] | None,
    year_from: int | None,
    year_to: int | None,
) -> Filter:
    must = [
        FieldCondition(key=has_data_key, match=MatchValue(value=True)),
        FieldCondition(key=outcome_key,  match=MatchValue(value=outcome_value)),
    ]
    if courts:
        must.append(FieldCondition(key="court", match=MatchAny(any=courts)))
    if year_from is not None or year_to is not None:
        must.append(FieldCondition(
            key="year",
            range=Range(
                gte=year_from if year_from is not None else 1900,
                lte=year_to   if year_to   is not None else 2100,
            ),
        ))
    return Filter(must=must)


def _hits_to_cases(hits, struct_key: str, top_k: int) -> list[ContrastingCase]:
    """Deduplicate by case_id (best score wins), then return top_k."""
    seen: dict[str, object] = {}
    for h in hits:
        cid = h.case_id
        if cid not in seen or h.score > seen[cid].score:
            seen[cid] = h

    ordered = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]

    cases = []
    for h in ordered:
        raw = h.payload.get(struct_key, {})
        structured = {k: v for k, v in raw.items() if k != "has_data" and v is not None}
        cases.append(ContrastingCase(
            case_id=h.case_id,
            title=h.title,
            court_name=h.court_name,
            date=h.date,
            url=h.url,
            score=h.score,
            structured=structured,
        ))
    return cases


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_contrasting_cases(
    query: str,
    domain: str,
    query_vector: list[float],
    store,            # VectorStore - passed in to avoid circular imports
    split_by: str | None = None,
    courts: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    top_k: int = 5,
) -> ContrastingResult:
    """Return two groups of semantically similar cases with opposite structured outcomes.

    Each group is retrieved as a separate Qdrant semantic search, restricted to
    chunks that carry the target outcome value (e.g. sentence_type=imprisonment).
    The caller is responsible for embedding the query before calling this function.
    """
    cfg = get_split_config(domain, split_by)
    if cfg is None:
        raise ValueError(f"Unknown domain/split: {domain!r}/{split_by!r}")

    split_name, (key_a, val_a, label_a, desc_a), (key_b, val_b, label_b, desc_b) = cfg
    has_data_key = _HAS_DATA_KEY[domain]
    struct_key   = _STRUCT_KEY[domain]

    filter_a = _build_filter(key_a, val_a, has_data_key, courts, year_from, year_to)
    filter_b = _build_filter(key_b, val_b, has_data_key, courts, year_from, year_to)

    hits_a = store.search_filtered(query_vector, filter_a, top_k=top_k * 4)
    hits_b = store.search_filtered(query_vector, filter_b, top_k=top_k * 4)

    return ContrastingResult(
        query=query,
        domain=domain,
        split_by=split_name,
        group_a=ContrastingGroup(label_a, desc_a, _hits_to_cases(hits_a, struct_key, top_k)),
        group_b=ContrastingGroup(label_b, desc_b, _hits_to_cases(hits_b, struct_key, top_k)),
    )
