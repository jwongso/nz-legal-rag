-- Reset all migrated data (documents, chunks, citations, structured tables).
-- Safe to run before re-migrating from Qdrant.
-- Does NOT drop the schema - just truncates data tables in dependency order.
-- Courts reference data is preserved.

TRUNCATE TABLE
    evaluation_results,
    legislation_references,
    document_judges,
    citations,
    sentencing_cases,
    employment_cases,
    chunks,
    ingest_runs,
    documents
RESTART IDENTITY CASCADE;
