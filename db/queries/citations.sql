-- Citation graph analytics

-- 1. Most cited cases in the corpus
SELECT
    td.citation                 AS cited_case,
    td.title                    AS title,
    td.court                    AS court,
    td.decision_date            AS decision_date,
    COUNT(*)                    AS times_cited
FROM citations ci
JOIN documents td ON ci.to_document_id = td.id
GROUP BY td.id, td.citation, td.title, td.court, td.decision_date
ORDER BY times_cited DESC
LIMIT 30;


-- 2. Cases that cite the most other cases (most research-dense)
SELECT
    fd.citation                 AS from_case,
    fd.title                    AS title,
    fd.court                    AS court,
    fd.decision_date            AS decision_date,
    COUNT(*)                    AS citations_made
FROM citations ci
JOIN documents fd ON ci.from_document_id = fd.id
GROUP BY fd.id, fd.citation, fd.title, fd.court, fd.decision_date
ORDER BY citations_made DESC
LIMIT 30;


-- 3. Citation network: cases citing a specific case
-- Replace the citation value as needed.
SELECT
    fd.citation                 AS citing_case,
    fd.title,
    fd.court,
    fd.decision_date,
    ci.citation_type
FROM citations ci
JOIN documents fd ON ci.from_document_id = fd.id
JOIN documents td ON ci.to_document_id   = td.id
WHERE td.citation = 'NZCA/2008/1'
ORDER BY fd.decision_date DESC;


-- 4. Unresolved citations (cited text not matching any document in corpus)
SELECT
    fd.citation                 AS from_case,
    fd.court,
    ci.cited_text,
    fd.decision_date
FROM citations ci
JOIN documents fd ON ci.from_document_id = fd.id
WHERE ci.to_document_id IS NULL
ORDER BY fd.decision_date DESC
LIMIT 50;


-- 5. Citation resolution stats
SELECT
    COUNT(*)                                    AS total_citations,
    COUNT(*) FILTER (WHERE to_document_id IS NOT NULL) AS resolved,
    COUNT(*) FILTER (WHERE to_document_id IS NULL)     AS unresolved,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE to_document_id IS NOT NULL) / COUNT(*), 1
    )                                           AS resolution_pct
FROM citations;


-- 6. Resolve unresolved citations (run after adding more documents)
-- Matches cited_text against documents.citation using ILIKE for flexibility.
-- Run this periodically to improve the citation graph.
UPDATE citations c
SET to_document_id = d.id
FROM documents d
WHERE c.to_document_id IS NULL
  AND d.citation ILIKE '%' || split_part(c.cited_text, '/', 2) || '%';
