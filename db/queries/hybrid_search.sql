-- Hybrid search patterns: SQL pre-filter + Qdrant semantic search
--
-- These are template queries showing how to combine SQL structured filters
-- with Qdrant vector search. In practice the application code:
--   1. Runs the SQL pre-filter to get a list of case_ids / qdrant_point_ids
--   2. Passes those IDs as a filter to qdrant.search()
--   3. Re-ranks or merges results as needed
--
-- Approach A: SQL-first (structured filter -> semantic search)
--   Good for: sentencing range, court, date range, specific offence queries
--   SQL narrows candidate set, Qdrant ranks by semantic relevance.
--
-- Approach B: Qdrant-first (semantic search -> SQL enrichment)
--   Good for: open-ended legal questions where the semantic match is primary
--   Qdrant returns top-N chunks, SQL adds structured metadata / related cases.
--
-- Approach C: Parallel + merge
--   Run SQL filter AND Qdrant search independently, RRF-merge the result lists.
--   Best recall; more latency.


-- Example A1: Pre-filter for sentencing search
-- "NZCA robbery cases 2022-2024 with starting point < 5 years"
-- Returns qdrant_point_ids to pass as Qdrant filter.
SELECT DISTINCT ch.qdrant_point_id
FROM chunks ch
JOIN documents d  ON ch.document_id = d.id
JOIN sentencing_cases sc ON sc.document_id = d.id
WHERE d.court = 'NZCA'
  AND d.decision_date BETWEEN '2022-01-01' AND '2024-12-31'
  AND sc.starting_point < 60   -- months
  AND (sc.offence ILIKE '%robbery%' OR 'robbery' = ANY(
        SELECT unnest(sc2.offences) FROM sentencing_cases sc2 WHERE sc2.document_id = d.id
      ))
  AND ch.qdrant_point_id IS NOT NULL;


-- Example A2: Pre-filter for employment search
-- "ERA unjustified dismissal cases with reinstatement 2023"
SELECT DISTINCT ch.qdrant_point_id
FROM chunks ch
JOIN documents d  ON ch.document_id = d.id
JOIN employment_cases ec ON ec.document_id = d.id
WHERE d.court IN ('NZERA', 'NZEmpC')
  AND d.decision_date BETWEEN '2023-01-01' AND '2023-12-31'
  AND ec.grievance_type = 'unjustified_dismissal'
  AND ec.reinstatement = TRUE
  AND ch.qdrant_point_id IS NOT NULL;


-- Example B1: Post-search enrichment
-- After Qdrant returns a list of qdrant_point_ids, fetch full metadata.
-- Replace the IN list with the actual IDs from Qdrant results.
SELECT
    d.citation,
    d.title,
    d.court,
    d.decision_date,
    d.source_url,
    ch.chunk_index,
    ch.section_title,
    sc.offence,
    sc.starting_point,
    sc.final_sentence,
    ec.grievance_type,
    ec.outcome
FROM chunks ch
JOIN documents d         ON ch.document_id = d.id
LEFT JOIN sentencing_cases sc ON sc.document_id = d.id
LEFT JOIN employment_cases ec ON ec.document_id = d.id
WHERE ch.qdrant_point_id = ANY(ARRAY[
    -- paste qdrant point ids here
    '00000000-0000-0000-0000-000000000000'
]);


-- Example C: BM25 full-text search (PostgreSQL tsvector)
-- Runs entirely in SQL without Qdrant. Add GIN index first:
--   CREATE INDEX idx_chunks_fts ON chunks USING GIN (to_tsvector('english', text));
SELECT
    d.citation,
    d.title,
    d.court,
    d.decision_date,
    ch.section_title,
    ts_rank(to_tsvector('english', ch.text), query) AS rank,
    ts_headline('english', ch.text, query, 'MaxWords=50, MinWords=20') AS snippet
FROM chunks ch
JOIN documents d ON ch.document_id = d.id,
     to_tsquery('english', 'reinstatement & unjustified & dismissal') AS query
WHERE to_tsvector('english', ch.text) @@ query
  AND d.court IN ('NZERA', 'NZEmpC')
ORDER BY rank DESC
LIMIT 20;
