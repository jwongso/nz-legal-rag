-- Corpus coverage: decisions and chunks per court per year
-- Use this to find gaps and plan ingestion runs.

SELECT
    d.court,
    ct.name                                             AS court_name,
    EXTRACT(YEAR FROM d.decision_date)::INTEGER         AS year,
    COUNT(DISTINCT d.id)                                AS decisions,
    COALESCE(SUM(c.chunk_count), 0)                     AS chunks,
    ROUND(AVG(c.chunk_count), 1)                        AS avg_chunks_per_doc
FROM documents d
JOIN courts ct ON d.court = ct.code
LEFT JOIN (
    SELECT document_id, COUNT(*) AS chunk_count FROM chunks GROUP BY document_id
) c ON d.id = c.document_id
WHERE d.decision_date IS NOT NULL
GROUP BY d.court, ct.name, EXTRACT(YEAR FROM d.decision_date)
ORDER BY d.court, year;


-- Quick totals per court
SELECT
    d.court,
    ct.name                     AS court_name,
    COUNT(DISTINCT d.id)        AS total_decisions,
    COUNT(ch.id)                AS total_chunks,
    MIN(d.decision_date)        AS earliest,
    MAX(d.decision_date)        AS latest
FROM documents d
JOIN courts ct ON d.court = ct.code
LEFT JOIN chunks ch ON ch.document_id = d.id
GROUP BY d.court, ct.name
ORDER BY total_decisions DESC;


-- Missing years (gaps in coverage) for a given court
-- Replace 'NZCA' and the year range as needed.
SELECT gs.year
FROM generate_series(1985, 2025) AS gs(year)
WHERE NOT EXISTS (
    SELECT 1
    FROM documents d
    WHERE d.court = 'NZCA'
      AND EXTRACT(YEAR FROM d.decision_date) = gs.year
);
