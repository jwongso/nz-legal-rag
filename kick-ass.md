## Future Feature Ideas

This section collects ambitious but doable ideas for extending `nz-legal-rag` using PostgreSQL, Qdrant, RAG, MCP, local LLMs, rerankers, and structured legal metadata.

### Framing Principles (apply to all features)

Every user-facing output must be clearly framed as a summary of publicly available court decisions. The consistent language to use throughout:

```text
This information is derived from publicly available New Zealand court decisions.
It does not constitute legal advice, a legal opinion, or a prediction of any outcome.
All outputs should be verified against the original source decisions.
For advice about a specific situation, consult a qualified New Zealand lawyer.
```

Shorter inline variant for tooltips, footers, and UI labels:

```text
Summarised from public court decisions. Not legal advice. Verify against original sources.
```

Features that present outcomes, ranges, or patterns are especially sensitive and must carry this framing prominently - not buried in a footer.

### 1. Hybrid PostgreSQL + Qdrant Retrieval

Use PostgreSQL as the structured source of truth and Qdrant as the semantic retrieval index.

**PostgreSQL handles:**

- Courts, dates, citations, source URLs, document types
- Offence flags and sentencing fields
- Personal grievance outcomes and remedies
- Ingestion state, document hashes, chunk metadata
- Citation relationships and evaluation results

**Qdrant handles:**

- Chunk embeddings
- Semantic similarity search
- Metadata-filtered vector retrieval

Example flow:

```text
User query
-> extract structured filters
-> PostgreSQL narrows candidate documents/chunks
-> Qdrant searches semantically inside filtered scope
-> reranker selects best passages
-> LLM generates citation-grounded answer
````

### 2. Retrieval Planner

Add a query planner that chooses the best retrieval strategy depending on the legal question.

Possible strategies:

```text
exact_citation_lookup
statute_first
sql_filter_then_vector
vector_then_filter
hybrid_keyword_vector
tracker_then_rag
citation_graph_then_rag
comparative_multi_search
broad_exploratory
```

Example:

```text
Question:
"Compare ERA and Employment Court redundancy dismissal outcomes"

Detected:
- domain: employment
- intent: compare_outcomes
- issue: redundancy
- courts: ERA, Employment Court
- strategy: comparative_multi_search

Plan:
1. SQL filter ERA redundancy cases
2. SQL filter Employment Court redundancy cases
3. Vector search within each group
4. Rerank separately
5. Generate comparison with cited examples
```

### 3. Developer Retrieval Trace

Add a developer/debug mode that shows how the answer was produced.

Example trace:

```text
Detected intent: sentencing_comparison
Strategy: sql_filter_then_vector
SQL filters:
- court = NZCA
- year = 2023
- offence = aggravated robbery

Qdrant:
- collection = nz_legal_chunks
- top_k = 50

Reranker:
- model = bge-reranker-v2-m3
- kept = 8

Generator:
- model = Qwen2.5-14B-Instruct Q5_K_M
- latency = 4.2s
```

This improves debugging, demo quality, and interview value.

### 4. "Why This Result?" Explanation

For each retrieved case or passage, explain why it appeared.

Example:

```text
Matched because:
- court = Employment Court
- issue = unjustified dismissal
- vector similarity = 0.82
- reranker score = 0.91
- contains terms: redundancy, consultation, good faith
```

### 5. Case Similarity Explorer

Allow users to open one case and find similar cases.

Possible modes:

```text
similar facts
similar legal issue
similar outcome
similar sentence
similar compensation amount
different result despite similar facts
```

This combines Qdrant semantic similarity with PostgreSQL outcome fields.

Important framing:

```text
Similarity is based on indexed text patterns and structured fields from public decisions.
It does not mean the cases are legally equivalent or that the same outcome will apply elsewhere.
```

### 6. Similar Cases With Opposite Outcomes

Find legally or factually similar cases where the result was different.

Examples:

```text
Show redundancy dismissal cases where the employer won and similar cases where the employee won.

Show aggravated robbery cases with similar facts where one resulted in imprisonment and another resulted in home detention.
```

Pipeline:

```text
Qdrant -> find similar fact patterns
PostgreSQL -> compare structured outcomes
LLM -> explain the key distinguishing factors
```

This is one of the strongest differentiating features because normal keyword search cannot do it well.

Important framing:

```text
Contrasting outcomes are drawn from public court decisions and reflect what courts decided in those
specific cases. They do not predict what a court would decide in any other situation.
Summarised from public court decisions. Not legal advice. Verify against original sources.
```

### 7. Sentencing Range Research Tool

Summarise comparable sentencing outcomes from public cases.

Example input:

```text
Offence: aggravated robbery
Aggravating factors: weapon, group offending
Mitigating factors: youth, guilty plea
Court: NZHC
```

Example output:

```text
Comparable cases
Starting-point range
Final-sentence range
Guilty plea discounts
Cases with similar factors
Outliers
```

Important wording:

```text
This is not a prediction or legal advice. It summarizes comparable public cases.
```

### 8. Personal Grievance Outcome Explorer

Analyse employment-law outcomes using structured PG tracker data.

Example queries:

```text
Show typical compensation awards for unjustified dismissal where contributory conduct was 20-50%.

Compare reinstatement outcomes between ERA and Employment Court.

Find cases where compensation was reduced because of employee conduct.
```

Possible output:

```text
median compensation
range
representative cases
factors increasing award
factors reducing award
cited examples
```

Important framing:

```text
These figures summarise outcomes from publicly available ERA and Employment Court decisions.
They reflect what tribunals awarded in those specific cases, not what any future award would be.
Summarised from public court decisions. Not legal advice. Verify against original sources.
```

### 9. Counsel Intelligence

Extend counsel search into a public-case appearance research tool.

Possible features:

```text
search cases by counsel
show counsel appearance history in public decisions
show courts and legal areas counsel appears in
show outcomes where counsel appeared
detect prosecution / defence / civil role where extractable
```

Important framing:

```text
This is public case appearance research, not a ranking or performance score for lawyers.
```

### 10. Citation Graph

Build a graph of legal authorities.

Data model:

```text
case A cites case B
case B is cited by case C
case D follows, applies, distinguishes, or criticises case B
```

Possible features:

```text
most cited cases
later treatment
cases relying on this authority
cases distinguishing this authority
authority chain
```

Pipeline:

```text
PostgreSQL -> citation edges
Qdrant -> semantic search inside citing passages
LLM -> classify treatment as followed, applied, distinguished, criticised, or neutral
```

Important framing:

```text
Citation relationships are extracted from publicly indexed decisions only.
This is not a complete picture of how a case has been treated across all NZ courts.
Treatment classification is generated automatically and may be incorrect - verify against the original decisions.
```

### 11. Authority Strength Signal

Estimate the research strength of a case based on public metadata.

Possible factors:

```text
court level
recency
number of later citations
whether later cases follow or distinguish it
whether it comes from NZSC, NZCA, NZHC, ERA, etc.
```

Example output:

```text
Authority strength: high

Reasons:
- Court of Appeal decision
- cited by 18 later cases
- followed in 11 indexed cases
- not clearly overruled or criticised in the indexed corpus
```

Important framing:

```text
Citation frequency is measured within the indexed corpus only and does not reflect the full body
of NZ case law. This is a research signal based on public decisions, not a legal opinion on
whether a case remains good law. Always verify currency against an authoritative source.
```

### 12. Legal Issue Map

Cluster and label common legal issues automatically.

Example employment-law clusters:

```text
unjustified dismissal
redundancy consultation
constructive dismissal
good faith breach
harassment
disadvantage grievance
remedies and compensation
```

For each cluster, show:

```text
top cases
common phrases
common outcomes
representative passages
trend over time
```

Tech:

```text
embeddings
clustering
PostgreSQL metadata
LLM-generated labels
```

Important framing:

```text
Clusters and labels are generated automatically from indexed public decisions.
They are a research starting point, not a definitive classification of NZ law.
Summarised from public court decisions. Not legal advice. Verify against original sources.
```

### 13. Natural Language to SQL + Vector Search

Convert user questions into a structured retrieval plan.

Example:

```text
Find 2023 High Court cases involving self-defence where imprisonment was less than 3 years.
```

Generated plan:

```text
SQL filters:
- court = NZHC
- year = 2023
- self_defence = true
- final_sentence_months < 36

Vector search:
- semantic query = self-defence reasoning, mitigating factors, sentencing
```

The system can show the plan before running it. The generated SQL filter and search strategy are shown to the user before execution so they can verify the query is correctly interpreted.

Important framing:

```text
The structured query is generated automatically and may misinterpret the legal question.
Review the generated plan before relying on the results.
Summarised from public court decisions. Not legal advice. Verify against original sources.
```

### 14. Result Set Summarizer

After a search returns many cases, summarize the result set.

Example:

```text
Summarize patterns in these 30 redundancy dismissal cases.
```

Output:

```text
common legal issues
common outcomes
outlier cases
median award or sentence
cases worth reading first
key citations
```

Important framing:

```text
This summary is generated automatically from a limited set of indexed public decisions.
It may not reflect all relevant cases or legal principles. Not legal advice.
Verify all cited cases against the original source decisions.
```

### 15. Outlier Detector

Find unusual cases compared with similar cases.

Examples:

```text
Show unusually high compensation awards in unjustified dismissal cases.

Sho1w unusually low sentences for aggravated robbery compared with similar cases.
```

Tech:

```text
PostgreSQL statistics
percentiles or z-scores
Qdrant similarity grouping
LLM explanation with citations
```

Important framing:

```text
Outlier status is calculated from indexed cases only and reflects statistical patterns in public decisions.
An unusual outcome does not mean an incorrect or unlawful one.
Summarised from public court decisions. Not legal advice. Verify against original sources.
```

### 16. Timeline Mode

Show how a legal issue develops over time.

Examples:

```text
Show how unjustified dismissal reasoning changed from 2022 to 2024.

Show sentencing trends for aggravated robbery cases over time.
```

Pipeline:

```text
PostgreSQL date grouping
Qdrant topic retrieval
LLM trend summary
cited representative cases
```

Important framing:

```text
Trend summaries are generated from indexed public decisions over time and reflect patterns in
the indexed corpus, not the full body of NZ law. Trends may be affected by indexing gaps.
Summarised from public court decisions. Not legal advice. Verify against original sources.
```

### 17. Legal Update Monitor

Run scheduled ingestion and detect important new public decisions.

Possible outputs:

```text
new cases this week
important new authority
new sentencing patterns
new employment compensation awards
cases that cite older important cases
new cases relevant to saved topics
```

This could later power email, RSS, or dashboard notifications.

Important framing:

```text
Alerts are triggered by new publicly indexed decisions matching saved criteria.
They are not legal alerts and do not constitute advice about how new decisions affect any specific situation.
```

### 18. MCP Legal Research Server

Expose nz-legal-rag capabilities as MCP tools.

Possible tools:

```text
search_cases
search_legislation
get_case_by_citation
find_similar_cases
get_sentencing_comparables
get_pg_outcomes
trace_citations
summarize_result_set
build_research_bundle
```

This allows local AI clients to use nz-legal-rag as a legal research backend.

Important framing:

```text
All MCP tool responses are summaries of publicly available court decisions.
Consuming applications must carry forward the same disclaimer: not legal advice, verify against original sources.
```

### 19. Research Bundle Builder

Generate a structured research bundle for a topic.

Example:

```text
Create a research bundle for unjustified dismissal due to redundancy.
```

Output:

```text
10 key cases
3 leading principles
recurring employer arguments in public decisions
recurring employee arguments in public decisions
compensation range from indexed cases
contrasting outcomes
citations
downloadable markdown or PDF
```

Important framing:

```text
This bundle is generated from publicly available court decisions and is a research starting point only.
It does not constitute legal advice, a legal strategy, or a complete picture of the law.
Verify all cited cases and principles against the original source decisions.
```

### 20. Legal Argument Map

Build a research map showing arguments on each side.

Example output:

```text
Arguments appearing on the applicant side in indexed decisions
Arguments appearing on the respondent side in indexed decisions
How courts reasoned in comparable cases
Cases cited on each side
Distinguishing factors noted by courts
Key passages from the decisions
```

Important framing:

```text
This research map is generated from publicly available court decisions.
It reflects arguments and reasoning that appeared in those specific cases, not advice on how to argue any other case.
Not legal advice. Verify against original sources. Consult a qualified NZ lawyer for strategic advice.
```

### 21. Citation Verifier

Before showing the final answer, verify that important claims are supported by retrieved sources.

Pipeline:

```text
draft answer
-> extract claims
-> match claims to retrieved passages
-> flag unsupported claims
-> regenerate or show warning
```

This improves faithfulness and reduces hallucination risk.

### 22. Evidence Confidence Score

Show confidence based on retrieved evidence, not model confidence.

Example high confidence:

```text
Evidence confidence: high

Reasons:
- 8 relevant cases found
- 5 from higher courts
- citations directly support the key points
- retrieved sources are consistent
```

Example low confidence:

```text
Evidence confidence: low

Reasons:
- only 2 relevant cases found
- no higher-court authority found
- retrieved passages are indirect
- answer should be verified against original sources before relying on it
```

Important framing:

```text
Evidence confidence reflects the strength of retrieved indexed sources, not the correctness of the answer.
Even a high-confidence answer must be verified against the original decisions.
This score is not a legal opinion. Not legal advice.
```

### 23. Suppression and Sensitivity Scanner

Scan indexed/displayed material for sensitivity risk.

Possible flags:

```text
name suppression
suppressed identity
minor or youth
sexual offending
family court sensitivity
confidential details
```

Possible actions:

```text
hide snippets
show warning
link only to original source
exclude from public demo mode
require manual review
```

### 24. Benchmark Dashboard

Add an evaluation dashboard for retrieval and generation quality.

Compare:

```text
vector only
keyword only
SQL + vector
SQL + vector + reranker
SQL + keyword + vector + reranker
```

Metrics:

```text
top-5 hit rate
top-10 hit rate
reranker improvement
average latency
time to first token
tokens per second
citation correctness
faithfulness
```

### 25. Multi-Model Answer Comparison

In developer mode, compare outputs from different models.

Example:

```text
Qwen2.5-14B answer
Qwen2.5-32B answer
Qwen3-30B-A3B answer
optional external model answer
```

Compare:

```text
citation correctness
hallucination risk
latency
cost
answer completeness
```

### 26. Query Replay and Regression Testing

Store important queries as regression tests.

Store:

```text
query
expected citations
expected answer points
retrieval results
model version
embedding version
reranker version
chunking version
```

After changing chunking, embeddings, models, or retrieval logic, replay the test suite and detect quality regressions.

### 27. Legal Research Autopilot

Allow a broad topic and run staged research automatically.

Example topic:

```text
redundancy unjustified dismissal
```

Autopilot plan:

```text
1. find leading cases
2. find recent cases
3. find similar fact patterns
4. find contrary outcomes
5. summarize principles from indexed decisions
6. identify outliers in the indexed corpus
7. build research bundle
```

Important framing:

```text
Autopilot output is a structured summary of publicly available court decisions.
It is a research starting point and does not substitute for a lawyer's analysis.
Not legal advice. Verify all outputs against the original source decisions.
```

### 28. Saved Research Topics

Allow saved topics that can be re-run after new ingestion.

Example saved topics:

```text
aggravated robbery sentencing
unjustified dismissal redundancy
personal grievance compensation
name suppression decisions
```

Each topic can show:

```text
new matching cases in the indexed corpus
updated pattern summary from indexed decisions
new citations appearing in indexed cases
new statistical outliers in the indexed corpus
```

Important framing:

```text
Updates reflect newly indexed public court decisions only.
They do not constitute legal alerts or advice about how new decisions affect any specific situation.
Consult a qualified NZ lawyer to assess the impact of new decisions on your matter.
```

### 29. Source Freshness and Index Health

Show operational health of the corpus.

Possible fields:

```text
last ingestion time
source documents fetched
documents changed
documents removed
chunks indexed
Qdrant point count
embedding model version
failed ingestion jobs
stale documents
```

### 30. Public Demo Safety Mode

Add a public-demo mode with stricter limits.

Possible restrictions:

```text
no confidential facts
no legal strategy advice
no drafting court documents
no "will I win" predictions
no personal data entry
short excerpts only
citation required for every legal claim
```

Example refusal style:

```text
I can summarize public legal sources and comparable cases, but I cannot advise what you should do in a specific dispute. Please consult a qualified New Zealand lawyer for advice about your situation.
```

### 31. Counterfactual Sentencing ("What If...")

Let users adjust sentencing factors interactively and see how the comparable case range shifts.

Example:

```text
Offence: aggravated robbery
Court: NZHC

Adjust factors:
- Previous convictions: none / minor / serious
- Guilty plea: yes / no
- Youth: yes / no
- Mental health: yes / no

Output: starting point range and final sentence range from comparable cases
```

This is the sentencing research tool (#7) made interactive. Pure PostgreSQL + structured extraction - no LLM required for the core logic. The ranges update as the user changes factors, showing the statistical impact of each mitigating or aggravating element.

Important framing:

```text
These ranges are derived from comparable public cases. They are not a prediction and do not constitute legal advice.
```


### 32. Legislation-to-Case Bridge

User enters an Act and section and gets every indexed case that interpreted it.

Example:

```text
Input: s 103A Employment Relations Act 2000

Output:
- 47 indexed cases reference this section
- Leading cases: [list]
- Common interpretations: [summary]
- Divergent interpretations: [list]
- Semantic search within this filtered set
```

The `legislation_references` table is already in the schema. Ingest-time regex extracts Act references. Westlaw charges a lot for exactly this feature.

Important framing:

```text
Cases are matched to legislation sections based on text extracted from publicly available decisions.
This is not a complete or authoritative index of how a section has been interpreted.
Verify against the original decisions and current legislation. Not legal advice.
```


### 33. Procedural History Reconstruction

For cases that travelled through multiple levels, reconstruct the full chain and show how the outcome changed at each level.

Example:

```text
ERA (2021) -> dismissed
Employment Court (2022) -> upheld on appeal
Court of Appeal (2023) -> varied on remedy
```

The citation graph already captures these links. A traversal query over `citations` can follow the chain up or down. A legal researcher reading a Supreme Court decision often needs to understand what happened below to evaluate the reasoning.

Important framing:

```text
Procedural history is reconstructed from citation relationships in indexed public decisions.
It may be incomplete if earlier decisions are not in the indexed corpus.
Verify the full procedural history against the original decisions. Not legal advice.
```


### 34. IRAC Memo Generator

After retrieval, structure the answer as a formal legal research memo.

Standard format:

```text
Issue:
What is the legal question?

Rule:
Legal principles that have appeared in comparable indexed cases, with citations.

Application:
How courts have applied those principles in comparable public decisions.

Conclusion:
How comparable cases were decided in the indexed corpus, with key cited authority.

Important:
This memo is generated from publicly available court decisions and is not legal advice.
It does not constitute a legal opinion and should not be relied on without independent verification.
Consult a qualified New Zealand lawyer for advice about your specific situation.
```

This turns the system from a search engine into something a lawyer can paste into a document or send to a supervising partner. It is also a strong portfolio differentiator - "generates structured legal research memos grounded in cited NZ authority" is a concrete, professional claim.


### 35. Always-Fresh Ingestion (Automated Crawl)

Run scheduled ingestion of new NZLII decisions so the corpus stays current.

Pipeline:

```text
nightly or weekly cron job
-> check NZLII for new decisions since last run
-> ingest new documents
-> embed and index in Qdrant
-> update PostgreSQL
-> trigger Legal Update Monitor (#17) for saved topics
```

The batch pipeline already exists. This adds a "what is new since last run" check using the `ingest_runs` table and a scheduler. Without this the system is a snapshot. With it, the system is a live legal intelligence feed.

This is what separates a project from a product.


## Combo Features

Some of the most powerful experiences come from combining features:

### The Research Memo Combo

```text
#6  Similar Cases With Opposite Outcomes
+
#21 Citation Verifier
+
#34 IRAC Memo Generator

= "Here are two cases with similar facts that went different ways.
   Here is why the court distinguished them.
   Here is a structured memo you can take to your supervising partner."
```

This is a complete workflow that no NZ legal tool currently offers.

### The Sentencing Intelligence Combo

```text
#7  Sentencing Range Research Tool
+
#31 Counterfactual Sentencing
+
#15 Outlier Detector
+
#11 Authority Strength Signal

= Interactive sentencing research with comparable ranges, factor adjustment,
  outlier flagging, and citation strength - all grounded in public NZCA/NZHC/NZDC decisions.
```

### The Living Corpus Combo

```text
#35 Always-Fresh Ingestion
+
#17 Legal Update Monitor
+
#28 Saved Research Topics
+
#16 Timeline Mode

= A legal research assistant that watches the courts for you and
  surfaces new decisions relevant to topics you care about.
```


## Suggested Build Order

```text
Phase 1:
PostgreSQL schema + sync with Qdrant point IDs  [DONE]

Phase 2:
SQL-filter-then-vector retrieval  [DONE]
Legislation-to-case bridge (#32)

Phase 3:
Developer retrieval trace (#3)    [DONE]
Citation verifier (#21)           [DONE]

Phase 4:
Similar cases with opposite outcomes (#6)  [DONE]
LLM extraction backfill pipeline           [DONE]
Procedural history reconstruction (#33)

Phase 5:
Sentencing range research tool (#7)
Counterfactual sentencing (#31)
Outlier detector (#15)

Phase 6:
Citation graph (#10)
Authority strength signal (#11)

Phase 7:
IRAC memo generator (#34)
Research bundle builder (#19)

Phase 8:
Always-fresh ingestion (#35)
Legal update monitor (#17)
Saved research topics (#28)

Phase 9:
MCP tool expansion (#18)
Legal research autopilot (#27)

Phase 4.5 (benchmark suite - NEW):
Speed benchmarks (bench_generator, bench_embedder, bench_reranker)  [DONE]
30-record gold dataset (retrieval_gold.jsonl)                        [DONE]
Retrieval A/B runner (run_retrieval.py - oracle filter)              [DONE]
Citation correctness benchmark (legal-specific, stages 1-3)
Structured extraction accuracy benchmark
Answer quality (extend ragas_eval with full gold dataset)
Safety benchmark (prompt injection, legal advice refusal)

Phase 10:
Benchmark dashboard (#24)
Query replay and regression testing (#26)
```

## README Positioning Line

```text
nz-legal-rag combines PostgreSQL metadata filtering, Qdrant semantic retrieval, reranking, local LLM generation, and MCP tooling to support citation-grounded legal research over public New Zealand legal sources.
```
