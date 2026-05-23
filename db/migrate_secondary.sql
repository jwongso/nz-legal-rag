-- Secondary source ingestion schema (Phase 1)
-- Run once: psql -d nz_legal -f db/migrate_secondary.sql

CREATE TABLE IF NOT EXISTS secondary_documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type     text NOT NULL,           -- journal_article, law_review, legal_memo, commentary, user_note
    title           text,
    authors         text[],
    publication     text,
    publication_year int,
    file_path       text NOT NULL,
    file_hash       text NOT NULL UNIQUE,    -- SHA-256 of original file, used for dedup
    parse_status    text NOT NULL DEFAULT 'pending',  -- pending, parsed, chunked, embedded, failed
    parse_method    text,                    -- pymupdf, python-docx, plaintext
    parse_error     text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sec_docs_status ON secondary_documents(parse_status);
CREATE INDEX IF NOT EXISTS idx_sec_docs_type   ON secondary_documents(source_type);

CREATE TABLE IF NOT EXISTS secondary_chunks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     uuid NOT NULL REFERENCES secondary_documents(id) ON DELETE CASCADE,
    chunk_index     int  NOT NULL,
    section_title   text,
    chunk_type      text,                    -- abstract, argument, case_discussion, footnote, conclusion, body
    text            text NOT NULL,
    token_count     int,
    qdrant_point_id text UNIQUE,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_sec_chunks_doc    ON secondary_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_sec_chunks_qdrant ON secondary_chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_sec_chunks_fts
    ON secondary_chunks USING gin(to_tsvector('english', coalesce(text, '')));

-- secondary_citations populated in Phase 2
-- secondary_concepts populated in Phase 2

-- Phase 2: citation linking (run after Phase 1 tables exist)
CREATE TABLE IF NOT EXISTS secondary_citations (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    secondary_document_id uuid NOT NULL REFERENCES secondary_documents(id) ON DELETE CASCADE,
    secondary_chunk_id    uuid NOT NULL REFERENCES secondary_chunks(id) ON DELETE CASCADE,
    raw_citation          text NOT NULL,
    normalised_citation   text NOT NULL,       -- COURT/YEAR/NUM or NZLEG/ABBREV/sN
    citation_type         text NOT NULL,       -- case | legislation
    target_document_id    integer REFERENCES documents(id),   -- NULL if not found in corpus
    confidence            real NOT NULL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_sec_cit_doc    ON secondary_citations(secondary_document_id);
CREATE INDEX IF NOT EXISTS idx_sec_cit_target ON secondary_citations(target_document_id);
CREATE INDEX IF NOT EXISTS idx_sec_cit_norm   ON secondary_citations(normalised_citation);
