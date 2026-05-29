"""MCP tool: search court decisions via vector RAG."""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
from rag.pipeline import RAGPipeline
from mcp.domains.base import DomainConfig


def register(mcp: FastMCP, pipeline: RAGPipeline, domain: DomainConfig) -> None:
    source_list = ", ".join(domain.source_labels.keys()) or "all"

    @mcp.tool()
    async def search_cases(
        query: str,
        sources: str = "",
        year_from: int = 0,
        year_to: int = 0,
        top_k: int = 5,
    ) -> str:
        f"""
        Search decisions and documents in the {domain.name} knowledge base.

        Args:
            query: The question or topic to search for.
            sources: Comma-separated source/court codes to filter by.
                     Available: {source_list}. Leave empty to search all.
            year_from: Earliest year to include (0 = no limit).
            year_to: Latest year to include (0 = no limit).
            top_k: Number of results (default 5, max 20).
        """
        court_list = [c.strip() for c in sources.split(",") if c.strip()] or None
        yf = year_from if year_from > 0 else None
        yt = year_to if year_to > 0 else None

        response = await pipeline.ask(
            question=query,
            top_k=min(top_k, 20),
            courts=court_list,
            year_from=yf,
            year_to=yt,
        )

        lines = [response.answer, "", "--- Sources ---"]
        for i, s in enumerate(response.sources):
            lines.append(
                f"[{i + 1}] {s.get('title', 'Unknown')} | "
                f"{s.get('court_name', '')} | {s.get('date', '')} | {s.get('url', '')}"
            )
        return "\n".join(lines)
