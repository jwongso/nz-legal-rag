"""
MCP server exposing NZ legal RAG as tools for Claude Code / Claude Desktop.

Add to your MCP config:
  {
    "nz-legal": {
      "command": "python",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/nz-legal-rag"
    }
  }
"""

from mcp.server.fastmcp import FastMCP

import config
from rag.pipeline import RAGPipeline

mcp = FastMCP("nz-legal-rag", description="Search NZ court decisions and legislation")

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@mcp.tool()
async def search_nz_law(
    query: str,
    courts: str = "",
    year_from: int = 0,
    year_to: int = 0,
    top_k: int = 5,
) -> str:
    """
    Search NZ court decisions and legislation using semantic search.

    Args:
        query: The legal question or topic to search for.
        courts: Comma-separated court codes to filter by. Options: NZTT, NZHC, NZCA, NZSC, NZEmpC, NZERA.
                Leave empty to search all courts.
        year_from: Earliest year to include (e.g. 2018). 0 means no lower bound.
        year_to: Latest year to include (e.g. 2024). 0 means no upper bound.
        top_k: Number of results to return (default 5, max 20).
    """
    pipeline = _get_pipeline()

    court_list = [c.strip() for c in courts.split(",") if c.strip()] or None
    yf = year_from if year_from > 0 else None
    yt = year_to if year_to > 0 else None

    response = await pipeline.ask(
        question=query,
        top_k=min(top_k, 20),
        courts=court_list,
        year_from=yf,
        year_to=yt,
    )

    # Return answer + structured source list so the MCP client can verify citations
    lines = [response.answer, "", "--- Sources ---"]
    for i, s in enumerate(response.sources):
        lines.append(
            f"[{i + 1}] {s.get('title', 'Unknown')} | {s.get('court_name', '')} | "
            f"{s.get('date', '')} | {s.get('url', '')}"
        )
    return "\n".join(lines)


@mcp.tool()
async def get_case(case_id: str) -> str:
    """
    Retrieve all indexed chunks for a specific NZ case.

    Args:
        case_id: Case identifier in format COURT/YEAR/NUMBER, e.g. NZTT/2023/42
    """
    # Reuse the pipeline's store rather than creating a new Qdrant connection
    store = _get_pipeline()._store
    results = store.get_by_case_id(case_id)

    if not results:
        return f"Case '{case_id}' not found in the index."

    chunks = sorted(results, key=lambda r: r.payload.get("chunk_index", 0))
    header = chunks[0]
    output = [
        f"Case: {header.title}",
        f"Court: {header.court_name}",
        f"Date: {header.date}",
        f"URL: {header.url}",
        "",
        "--- Decision text ---",
        "",
    ]
    for chunk in chunks:
        output.append(chunk.text)
        output.append("")

    return "\n".join(output)


@mcp.tool()
async def search_legislation(
    query: str,
    act: str = "",
    top_k: int = 5,
) -> str:
    """
    Search NZ legislation sections by topic or section reference.

    Returns the raw section text from the Act - no LLM generation, just the actual law.
    Use this to look up what a specific section says, verify a section number, or find
    which sections cover a topic.

    Args:
        query: Topic or section to find, e.g. "landlord right of entry" or "section 48 notice to terminate".
        act: Optional act code to restrict to one Act. Options:
               RTA      - Residential Tenancies Act 1986
               ERA2000  - Employment Relations Act 2000
               PA2020   - Privacy Act 2020
               CCLA2017 - Contract and Commercial Law Act 2017
               CA1993   - Companies Act 1993
               CRA1961  - Crimes Act 1961
             Leave empty to search all indexed legislation.
        top_k: Number of sections to return (default 5, max 20).
    """
    pipeline = _get_pipeline()
    query_vector = await pipeline._embedder.embed(query)

    raw_hits = pipeline._store.search(
        query_vector,
        top_k=min(top_k, 20) * 4,
        courts=["NZLEG"],
    )

    act_prefix = f"NZLEG/{act.upper().strip()}/" if act.strip() else ""
    if act_prefix:
        raw_hits = [h for h in raw_hits if h.case_id.startswith(act_prefix)]

    # One chunk per section (each section is its own case_id in NZLEG)
    seen: set[str] = set()
    hits = []
    for h in raw_hits:
        if h.case_id not in seen:
            seen.add(h.case_id)
            hits.append(h)
        if len(hits) >= min(top_k, 20):
            break

    if not hits:
        act_label = f" in {act.upper()}" if act.strip() else ""
        return f"No matching legislation sections found{act_label}."

    lines: list[str] = [f"Found {len(hits)} section(s) matching '{query}':\n"]
    for h in hits:
        lines.append(f"## {h.title}")
        lines.append(f"Citation: {h.case_id}  |  Score: {h.score:.4f}")
        lines.append(f"URL: {h.url}")
        lines.append("")
        lines.append(h.text)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def list_courts() -> str:
    """List all courts in the index with their decision counts."""
    store = _get_pipeline()._store
    try:
        stats = store.collection_stats()
        courts_info = "\n".join(f"  {code}: {name}" for code, name in config.COURTS.items())
        return (
            f"Collection: {config.QDRANT_COLLECTION}\n"
            f"Total chunks indexed: {stats['points_count']}\n"
            f"Status: {stats['status']}\n\n"
            f"Available courts:\n{courts_info}\n\n"
            f"Use court codes in search_nz_law() to filter by court."
        )
    except Exception as e:
        return f"Could not reach Qdrant: {e}"


if __name__ == "__main__":
    mcp.run()
