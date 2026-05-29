"""MCP tool: list available data sources and index stats."""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
from rag.pipeline import RAGPipeline
from mcp.domains.base import DomainConfig


def register(mcp: FastMCP, pipeline: RAGPipeline, domain: DomainConfig) -> None:

    @mcp.tool()
    async def list_sources() -> str:
        """List all data sources available in this knowledge base with index stats."""
        store = pipeline._store
        try:
            stats = store.collection_stats()
            source_lines = "\n".join(
                f"  {code}: {label}"
                for code, label in domain.source_labels.items()
            )
            return (
                f"Domain: {domain.name}\n"
                f"Collection: {domain.qdrant_collection}\n"
                f"Total chunks indexed: {stats['points_count']}\n"
                f"Status: {stats['status']}\n\n"
                f"Available sources:\n{source_lines}\n\n"
                f"Use source codes in search_cases() to filter by source."
            )
        except Exception as e:
            return f"Could not reach vector store: {e}"
