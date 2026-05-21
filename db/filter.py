"""SQL pre-filter: translate structured criteria into qdrant_point_ids.

Strategy: SQL-first hybrid retrieval.
  1. Call get_point_ids() with structured filters to get candidate chunk IDs from PostgreSQL.
  2. Pass those IDs to VectorStore.search_within() for semantic ranking inside the filtered scope.
  3. Result: semantically ranked results grounded in structured metadata.

Also provides bm25_search() as a pure-SQL keyword retrieval alternative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg2
import psycopg2.extras


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(dbname="nz_legal")


@dataclass
class FilterParams:
    """Structured filter criteria that map to PostgreSQL columns.

    All fields are optional - only non-None fields are added to the WHERE clause.
    Combining sentencing and employment filters in one query returns no results
    (a document cannot be both) - pass one domain at a time.
    """

    # Document-level filters
    courts:     list[str] | None = None
    year_from:  int | None = None
    year_to:    int | None = None
    legal_area: str | None = None   # 'criminal', 'employment', 'family', 'civil', 'environment'

    # Sentencing filters (joined to sentencing_cases)
    offence:                   str | None = None   # ILIKE match, e.g. 'robbery'
    min_starting_point:        float | None = None  # months
    max_starting_point:        float | None = None
    min_final_sentence:        float | None = None
    max_final_sentence:        float | None = None
    min_guilty_plea_discount:  float | None = None  # percentage
    max_guilty_plea_discount:  float | None = None
    flag_self_defence:         bool | None = None
    flag_provocation:          bool | None = None
    flag_mental_health:        bool | None = None
    flag_intoxication:         bool | None = None
    flag_youth:                bool | None = None
    flag_tikanga_maori:        bool | None = None
    flag_cultural_factors:     bool | None = None
    flag_previous_convictions: bool | None = None

    # Employment filters (joined to employment_cases)
    grievance_type:    str | None = None   # exact match, e.g. 'unjustified_dismissal'
    reinstatement:     bool | None = None
    min_contributory:  float | None = None  # percentage
    max_contributory:  float | None = None
    min_compensation:  float | None = None  # NZD
    max_compensation:  float | None = None

    # Safety cap on returned point IDs (Qdrant HasIdCondition performance)
    max_ids: int = 5000


_SENTENCING_FLAGS = [
    "flag_self_defence", "flag_provocation", "flag_mental_health",
    "flag_intoxication", "flag_youth", "flag_tikanga_maori",
    "flag_cultural_factors", "flag_previous_convictions",
]

_SENTENCING_RANGES = {
    "offence":                  ("sc.offence ILIKE %s",             lambda v: f"%{v}%"),
    "min_starting_point":       ("sc.starting_point >= %s",         None),
    "max_starting_point":       ("sc.starting_point <= %s",         None),
    "min_final_sentence":       ("sc.final_sentence >= %s",         None),
    "max_final_sentence":       ("sc.final_sentence <= %s",         None),
    "min_guilty_plea_discount": ("sc.guilty_plea_discount >= %s",   None),
    "max_guilty_plea_discount": ("sc.guilty_plea_discount <= %s",   None),
}

_EMPLOYMENT_RANGES = {
    "grievance_type":   ("ec.grievance_type = %s",               None),
    "reinstatement":    ("ec.reinstatement = %s",                None),
    "min_contributory": ("ec.contributory_conduct_pct >= %s",    None),
    "max_contributory": ("ec.contributory_conduct_pct <= %s",    None),
    "min_compensation": ("ec.compensation >= %s",                None),
    "max_compensation": ("ec.compensation <= %s",                None),
}


def _has_sentencing_filters(p: FilterParams) -> bool:
    return any([
        p.offence,
        p.min_starting_point, p.max_starting_point,
        p.min_final_sentence, p.max_final_sentence,
        p.min_guilty_plea_discount, p.max_guilty_plea_discount,
        *(getattr(p, f) is not None for f in _SENTENCING_FLAGS),
    ])


def _has_employment_filters(p: FilterParams) -> bool:
    return any([
        p.grievance_type,
        p.reinstatement is not None,
        p.min_contributory, p.max_contributory,
        p.min_compensation, p.max_compensation,
    ])


def get_point_ids(params: FilterParams) -> list[str]:
    """Return qdrant_point_ids matching the structured filter.

    Pass the result to VectorStore.search_within() for semantic ranking.
    Returns an empty list if no matching chunks found.
    """
    use_sentencing = _has_sentencing_filters(params)
    use_employment = _has_employment_filters(params)

    conditions: list[str] = ["ch.qdrant_point_id IS NOT NULL"]
    args: list = []

    # Document-level
    if params.courts:
        conditions.append("d.court = ANY(%s)")
        args.append(params.courts)

    if params.year_from is not None:
        conditions.append("EXTRACT(YEAR FROM d.decision_date) >= %s")
        args.append(params.year_from)

    if params.year_to is not None:
        conditions.append("EXTRACT(YEAR FROM d.decision_date) <= %s")
        args.append(params.year_to)

    if params.legal_area:
        # legal_area is not stored in documents; derive via court jurisdiction
        conditions.append("ct.jurisdiction = %s")
        args.append(params.legal_area)

    # Sentencing filters
    if use_sentencing:
        for attr, (sql_expr, transform) in _SENTENCING_RANGES.items():
            val = getattr(params, attr)
            if val is not None:
                conditions.append(sql_expr)
                args.append(transform(val) if transform else val)

        for flag in _SENTENCING_FLAGS:
            val = getattr(params, flag)
            if val is not None:
                conditions.append(f"sc.{flag} = %s")
                args.append(val)

    # Employment filters
    if use_employment:
        for attr, (sql_expr, _) in _EMPLOYMENT_RANGES.items():
            val = getattr(params, attr)
            if val is not None:
                conditions.append(sql_expr)
                args.append(val)

    # Build FROM / JOIN clause
    joins = ["JOIN documents d ON ch.document_id = d.id",
             "JOIN courts ct ON d.court = ct.code"]

    if use_sentencing:
        joins.append("JOIN sentencing_cases sc ON sc.document_id = d.id")
    if use_employment:
        joins.append("JOIN employment_cases ec ON ec.document_id = d.id")

    where = " AND ".join(conditions)
    sql = f"""
        SELECT ch.qdrant_point_id
        FROM chunks ch
        {chr(10).join(joins)}
        WHERE {where}
        LIMIT %s
    """
    args.append(params.max_ids)

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, args)
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def bm25_search(
    query: str,
    courts: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    legal_area: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Full-text BM25 search using PostgreSQL tsvector on chunk text.

    Returns ranked results with snippet highlights.
    Requires: CREATE INDEX idx_chunks_fts ON chunks USING GIN (to_tsvector('english', COALESCE(text, '')));
    """
    conditions: list[str] = ["to_tsvector('english', COALESCE(ch.text, '')) @@ query"]
    args: list = []

    if courts:
        conditions.append("d.court = ANY(%s)")
        args.append(courts)

    if year_from is not None:
        conditions.append("EXTRACT(YEAR FROM d.decision_date) >= %s")
        args.append(year_from)

    if year_to is not None:
        conditions.append("EXTRACT(YEAR FROM d.decision_date) <= %s")
        args.append(year_to)

    if legal_area:
        conditions.append("ct.jurisdiction = %s")
        args.append(legal_area)

    where = " AND ".join(conditions)
    args.append(limit)

    sql = f"""
        SELECT
            d.citation,
            d.title,
            d.court,
            d.decision_date,
            d.source_url,
            ch.qdrant_point_id,
            ch.section_title,
            ts_rank(to_tsvector('english', COALESCE(ch.text, '')), query) AS rank,
            ts_headline(
                'english', COALESCE(ch.text, ''), query,
                'MaxWords=60, MinWords=25, StartSel=**, StopSel=**'
            ) AS snippet
        FROM chunks ch
        JOIN documents d  ON ch.document_id = d.id
        JOIN courts ct    ON d.court = ct.code,
             plainto_tsquery('english', %s) AS query
        WHERE {where}
        ORDER BY rank DESC
        LIMIT %s
    """

    # plainto_tsquery arg goes first in the FROM clause - prepend it
    conn = _connect()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, [query] + args)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_document_metadata(citation: str) -> dict | None:
    """Fetch full structured metadata for a case by citation (case_id)."""
    sql = """
        SELECT
            d.id, d.citation, d.title, d.court, d.decision_date, d.source_url,
            d.document_type, d.ingestion_status,
            sc.offence, sc.offences, sc.starting_point, sc.final_sentence,
            sc.home_detention_months, sc.community_work_hours,
            sc.guilty_plea_discount, sc.appeal_outcome,
            sc.flag_self_defence, sc.flag_provocation, sc.flag_mental_health,
            sc.flag_intoxication, sc.flag_youth, sc.flag_tikanga_maori,
            sc.flag_cultural_factors, sc.flag_previous_convictions,
            ec.grievance_type, ec.grievance_types, ec.outcome,
            ec.reinstatement, ec.compensation, ec.contributory_conduct_pct
        FROM documents d
        LEFT JOIN sentencing_cases sc ON sc.document_id = d.id
        LEFT JOIN employment_cases ec ON ec.document_id = d.id
        WHERE d.citation = %s
    """
    conn = _connect()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, (citation,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
