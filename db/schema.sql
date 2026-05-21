-- NZ Legal RAG - PostgreSQL schema
-- Complements Qdrant (semantic/vector search) with structured metadata,
-- citation graph, tracker analytics, and ingest pipeline state.
--
-- Design principles:
--   - documents.citation is the shared key between Postgres and Qdrant payload
--   - Qdrant stores embeddings for semantic search
--   - Postgres stores structure, relationships, analytics, and pipeline state
--   - Views expose tracker-friendly flat tables for the API and direct SQL queries

CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy title/name search
CREATE EXTENSION IF NOT EXISTS unaccent;  -- normalise accented characters


-- -----------------------------------------------------------------------
-- Reference tables
-- -----------------------------------------------------------------------

CREATE TABLE courts (
    code         VARCHAR(20) PRIMARY KEY,  -- 'NZCA', 'NZHC', 'ERA', etc.
    name         TEXT        NOT NULL,
    jurisdiction VARCHAR(20) NOT NULL,     -- 'criminal', 'civil', 'employment', 'family'
    level        SMALLINT    NOT NULL,     -- 1=Supreme 2=Appeal 3=High 4=Specialist
    nzlii_path   TEXT                      -- '/nz/cases/NZCA'
);

INSERT INTO courts (code, name, jurisdiction, level, nzlii_path) VALUES
    ('NZSC',    'Supreme Court of New Zealand',                   'civil',       1, '/nz/cases/NZSC'),
    ('NZCA',    'Court of Appeal of New Zealand',                 'civil',       2, '/nz/cases/NZCA'),
    ('NZHC',    'High Court of New Zealand',                      'civil',       3, '/nz/cases/NZHC'),
    ('NZDC',    'District Court of New Zealand',                  'civil',       4, '/nz/cases/NZDC'),
    ('NZEmpC',  'Employment Court of New Zealand',                'employment',  3, '/nz/cases/NZEmpC'),
    ('NZERA',   'Employment Relations Authority',                  'employment',  4, '/nz/cases/NZERA'),
    ('NZFC',    'Family Court of New Zealand',                    'family',      4, '/nz/cases/NZFC'),
    ('NZEnvC',  'Environment Court of New Zealand',               'environment', 3, '/nz/cases/NZEnvC'),
    ('NZACC',   'ACC Appeals',                                    'civil',       4, '/nz/cases/NZACC'),
    ('NZCorC',  'Coroners Court of New Zealand',                  'civil',       4, '/nz/cases/NZCorC'),
    ('NZLCDT',  'Lawyers and Conveyancers Disciplinary Tribunal', 'civil',       4, '/nz/cases/NZLCDT'),
    ('NZHRRT',  'Human Rights Review Tribunal',                   'civil',       4, '/nz/cases/NZHRRT'),
    ('NZREADT', 'Real Estate Agents Disciplinary Tribunal',       'civil',       4, '/nz/cases/NZREADT'),
    ('NZTT',    'Tenancy Tribunal',                               'civil',       4, '/nz/cases/NZTT');


CREATE TABLE judges (
    id    SERIAL PRIMARY KEY,
    name  TEXT   NOT NULL UNIQUE,  -- 'Winkelmann CJ', 'Collins J'
    title TEXT                     -- 'CJ', 'P', 'J', 'JA', 'DCJ'
);


-- -----------------------------------------------------------------------
-- Core document table
-- -----------------------------------------------------------------------

CREATE TABLE documents (
    id               SERIAL      PRIMARY KEY,
    title            TEXT,
    citation         TEXT        NOT NULL UNIQUE,  -- 'NZCA/2023/123' - matches Qdrant case_id
    court            VARCHAR(20) NOT NULL REFERENCES courts(code),
    decision_date    DATE,
    source_url       TEXT,
    document_type    VARCHAR(20) DEFAULT 'decision',   -- 'decision', 'legislation'
    jurisdiction     VARCHAR(20),                      -- inherited from court, stored for convenience
    suppressed       BOOLEAN     DEFAULT FALSE,        -- name suppression order
    checksum         TEXT,                             -- SHA256 of raw source content
    ingestion_status VARCHAR(20) DEFAULT 'pending',    -- 'pending', 'in_progress', 'completed', 'failed'
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_documents_court        ON documents (court);
CREATE INDEX idx_documents_date         ON documents (decision_date);
CREATE INDEX idx_documents_status       ON documents (ingestion_status);
CREATE INDEX idx_documents_title_trgm   ON documents USING GIN (title gin_trgm_ops);


-- Document-judge relationship (future: parse judge names from decisions)
CREATE TABLE document_judges (
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    judge_id     INTEGER NOT NULL REFERENCES judges(id),
    role         VARCHAR(20) DEFAULT 'panel',  -- 'presiding', 'majority', 'dissent'
    PRIMARY KEY (document_id, judge_id)
);


-- -----------------------------------------------------------------------
-- Chunks (mirrors Qdrant points)
-- -----------------------------------------------------------------------

CREATE TABLE chunks (
    id              SERIAL   PRIMARY KEY,
    document_id     INTEGER  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     SMALLINT NOT NULL,
    section_title   TEXT,
    text            TEXT,
    token_count     INTEGER,
    qdrant_point_id TEXT     UNIQUE,     -- UUID5 matching Qdrant point ID
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (document_id, chunk_index)
);

CREATE INDEX idx_chunks_document ON chunks (document_id);
CREATE INDEX idx_chunks_qdrant   ON chunks (qdrant_point_id);


-- -----------------------------------------------------------------------
-- Citations (case-to-case graph)
-- -----------------------------------------------------------------------

CREATE TABLE citations (
    id               SERIAL  PRIMARY KEY,
    from_document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    to_document_id   INTEGER REFERENCES documents(id),   -- NULL if not in corpus
    cited_text       TEXT    NOT NULL,                   -- '[2020] NZCA 47'
    citation_type    VARCHAR(20) DEFAULT 'general',      -- 'general', 'applied', 'distinguished', 'overruled', 'approved'
    UNIQUE (from_document_id, cited_text)
);

CREATE INDEX idx_citations_from ON citations (from_document_id);
CREATE INDEX idx_citations_to   ON citations (to_document_id);


-- -----------------------------------------------------------------------
-- Legislation references
-- -----------------------------------------------------------------------

CREATE TABLE legislation_references (
    id           SERIAL  PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    act_name     TEXT    NOT NULL,   -- 'Employment Relations Act 2000'
    section      TEXT,               -- 's 103A'
    act_year     SMALLINT
);

CREATE INDEX idx_legref_document ON legislation_references (document_id);
CREATE INDEX idx_legref_act      ON legislation_references (act_name);


-- -----------------------------------------------------------------------
-- Sentencing cases (criminal courts: NZHC, NZCA, NZSC)
-- -----------------------------------------------------------------------

CREATE TABLE sentencing_cases (
    id                       SERIAL  PRIMARY KEY,
    document_id              INTEGER NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    offence                  TEXT,               -- primary offence, e.g. 'aggravated robbery'
    offences                 TEXT[]  DEFAULT '{}', -- all offences (array for multi-count cases)
    starting_point           NUMERIC(6,1),        -- months
    final_sentence           NUMERIC(6,1),        -- months
    home_detention_months    NUMERIC(6,1),
    community_work_hours     SMALLINT,
    guilty_plea_discount     NUMERIC(5,2),        -- percentage
    appeal_outcome           VARCHAR(20),          -- 'allowed', 'dismissed', 'varied', NULL
    -- aggravating and mitigating factors (text for flexibility, flags for filtering)
    aggravating_factors      TEXT,
    mitigating_factors       TEXT,
    flag_self_defence        BOOLEAN DEFAULT FALSE,
    flag_provocation         BOOLEAN DEFAULT FALSE,
    flag_mental_health       BOOLEAN DEFAULT FALSE,
    flag_intoxication        BOOLEAN DEFAULT FALSE,
    flag_youth               BOOLEAN DEFAULT FALSE,
    flag_tikanga_maori       BOOLEAN DEFAULT FALSE,
    flag_cultural_factors    BOOLEAN DEFAULT FALSE,
    flag_previous_convictions BOOLEAN DEFAULT FALSE,
    extracted_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sent_offence       ON sentencing_cases (offence);
CREATE INDEX idx_sent_offences      ON sentencing_cases USING GIN (offences);
CREATE INDEX idx_sent_sentence_type ON sentencing_cases (appeal_outcome);


-- -----------------------------------------------------------------------
-- Employment cases (ERA, NZEmpC)
-- -----------------------------------------------------------------------

CREATE TABLE employment_cases (
    id                      SERIAL  PRIMARY KEY,
    document_id             INTEGER NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    grievance_type          TEXT,                -- primary type: 'unjustified_dismissal', 'disadvantage', etc.
    grievance_types         TEXT[]  DEFAULT '{}', -- all types (array for multi-claim cases)
    outcome                 VARCHAR(20),          -- 'upheld', 'dismissed', 'settled', 'partial'
    remedy_amount           NUMERIC(12,2),        -- total monetary remedy
    reinstatement           BOOLEAN,
    compensation            NUMERIC(12,2),        -- compensation component
    contributory_conduct_pct NUMERIC(5,2),        -- percentage reduction for contributory conduct
    extracted_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_emp_grievance_type  ON employment_cases (grievance_type);
CREATE INDEX idx_emp_grievance_types ON employment_cases USING GIN (grievance_types);
CREATE INDEX idx_emp_outcome         ON employment_cases (outcome);


-- -----------------------------------------------------------------------
-- Evaluation results (RAGAS or manual eval)
-- -----------------------------------------------------------------------

CREATE TABLE evaluation_results (
    id                  SERIAL  PRIMARY KEY,
    query               TEXT    NOT NULL,
    answer              TEXT,
    retrieved_chunk_ids TEXT[]  DEFAULT '{}',
    faithfulness        NUMERIC(4,3),
    context_precision   NUMERIC(4,3),
    answer_relevance    NUMERIC(4,3),
    citation_accuracy   NUMERIC(4,3),
    hallucination_flag  BOOLEAN DEFAULT FALSE,
    retrieval_strategy  TEXT,
    model_name          TEXT,
    embedding_model     TEXT,
    evaluated_at        TIMESTAMPTZ DEFAULT NOW()
);


-- -----------------------------------------------------------------------
-- Views
-- -----------------------------------------------------------------------

-- Your example query works exactly as written:
--   SELECT * FROM sentencing_view
--   WHERE court = 'NZCA'
--     AND offence = 'aggravated robbery'
--     AND decision_date BETWEEN '2023-01-01' AND '2024-12-31';
CREATE VIEW sentencing_view AS
SELECT
    d.citation,
    d.court,
    ct.name                     AS court_name,
    d.title,
    d.decision_date,
    d.source_url,
    sc.offence,
    sc.offences,
    sc.starting_point,
    sc.final_sentence,
    sc.home_detention_months,
    sc.community_work_hours,
    sc.guilty_plea_discount,
    sc.appeal_outcome,
    sc.aggravating_factors,
    sc.mitigating_factors,
    sc.flag_self_defence,
    sc.flag_mental_health,
    sc.flag_youth,
    sc.flag_tikanga_maori,
    sc.flag_cultural_factors
FROM sentencing_cases sc
JOIN documents d    ON sc.document_id = d.id
JOIN courts ct      ON d.court = ct.code;


-- Employment grievance view
CREATE VIEW employment_view AS
SELECT
    d.citation,
    d.court,
    ct.name                     AS court_name,
    d.title,
    d.decision_date,
    d.source_url,
    ec.grievance_type,
    ec.grievance_types,
    ec.outcome,
    ec.reinstatement,
    ec.compensation,
    ec.remedy_amount,
    ec.contributory_conduct_pct
FROM employment_cases ec
JOIN documents d    ON ec.document_id = d.id
JOIN courts ct      ON d.court = ct.code;


-- Citation graph view
CREATE VIEW citation_graph AS
SELECT
    fd.citation             AS from_citation,
    fd.title                AS from_title,
    fd.court                AS from_court,
    fd.decision_date        AS from_date,
    ci.cited_text,
    ci.citation_type,
    td.citation             AS to_citation,
    td.title                AS to_title,
    td.court                AS to_court,
    td.decision_date        AS to_date
FROM citations ci
JOIN documents fd ON ci.from_document_id = fd.id
LEFT JOIN documents td ON ci.to_document_id  = td.id;


-- Ingest coverage view
CREATE VIEW ingest_coverage AS
SELECT
    court,
    EXTRACT(YEAR FROM decision_date)::INTEGER   AS year,
    ingestion_status,
    COUNT(*)                                    AS documents,
    SUM(c.chunk_count)                          AS total_chunks
FROM documents d
LEFT JOIN (
    SELECT document_id, COUNT(*) AS chunk_count FROM chunks GROUP BY document_id
) c ON d.id = c.document_id
WHERE decision_date IS NOT NULL
GROUP BY court, EXTRACT(YEAR FROM decision_date), ingestion_status
ORDER BY court, year;


-- Judge statistics view
CREATE VIEW judge_stats AS
SELECT
    j.name                              AS judge,
    j.title,
    d.court,
    COUNT(DISTINCT d.id)                AS total_decisions,
    MIN(d.decision_date)                AS first_decision,
    MAX(d.decision_date)                AS last_decision
FROM document_judges dj
JOIN judges j    ON dj.judge_id = j.id
JOIN documents d ON dj.document_id = d.id
GROUP BY j.id, j.name, j.title, d.court
ORDER BY total_decisions DESC;


-- -----------------------------------------------------------------------
-- Example analytics queries (for reference)
-- -----------------------------------------------------------------------

-- Your example - works exactly as written on the view:
--
-- SELECT * FROM sentencing_view
-- WHERE court = 'NZCA'
--   AND offence = 'aggravated robbery'
--   AND decision_date BETWEEN '2023-01-01' AND '2024-12-31';
--
-- Median guilty plea discount by offence:
--
-- SELECT offence, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY guilty_plea_discount) AS median_discount
-- FROM sentencing_view
-- WHERE guilty_plea_discount IS NOT NULL
-- GROUP BY offence ORDER BY median_discount DESC;
--
-- Most cited cases in the corpus:
--
-- SELECT to_citation, to_title, to_court, COUNT(*) AS times_cited
-- FROM citation_graph
-- WHERE to_citation IS NOT NULL
-- GROUP BY to_citation, to_title, to_court
-- ORDER BY times_cited DESC LIMIT 20;
--
-- Employment outcomes by grievance type and year:
--
-- SELECT grievance_type, EXTRACT(YEAR FROM decision_date) AS year,
--        COUNT(*) AS cases,
--        COUNT(*) FILTER (WHERE reinstatement) AS reinstated,
--        AVG(compensation) AS avg_compensation
-- FROM employment_view
-- GROUP BY grievance_type, year ORDER BY year DESC, cases DESC;
