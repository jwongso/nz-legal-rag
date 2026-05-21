-- BM25 / full-text search index on chunk text.
-- Run AFTER migration is complete (chunks table must be populated first).
-- This can take a few minutes on large corpora.

-- Create GIN index for tsvector full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks
    USING GIN (to_tsvector('english', COALESCE(text, '')));

-- Verify index was created
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'chunks'
  AND indexname = 'idx_chunks_fts';

-- Example BM25 query (after index is built):
-- SELECT
--     d.citation,
--     d.title,
--     d.court,
--     ts_rank(to_tsvector('english', ch.text), query) AS rank,
--     ts_headline('english', ch.text, query, 'MaxWords=60, MinWords=25') AS snippet
-- FROM chunks ch
-- JOIN documents d ON ch.document_id = d.id,
--      plainto_tsquery('english', 'personal grievance unjustified dismissal reinstatement') AS query
-- WHERE to_tsvector('english', ch.text) @@ query
-- ORDER BY rank DESC
-- LIMIT 10;
