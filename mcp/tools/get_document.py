"""MCP tool: retrieve all indexed chunks for a specific document by ID."""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
from rag.pipeline import RAGPipeline
from mcp.domains.base import DomainConfig


def register(mcp: FastMCP, pipeline: RAGPipeline, domain: DomainConfig) -> None:

    @mcp.tool()
    async def get_document(document_id: str) -> str:
        """
        Retrieve all indexed text chunks for a specific document.

        Args:
            document_id: Document identifier as stored in the index,
                         e.g. "NZTT-MOJ-12345" or "NZLEG/RTA/s48".
        """
        store = pipeline._store
        results = store.get_by_case_id(document_id)

        if not results:
            return f"Document '{document_id}' not found in the index."

        chunks = sorted(results, key=lambda r: r.payload.get("chunk_index", 0))
        header = chunks[0]
        lines = [
            f"Document: {header.title}",
            f"Source: {header.court_name}",
            f"Date: {header.date}",
            f"URL: {header.url}",
            "",
            "--- Content ---",
            "",
        ]
        for chunk in chunks:
            lines.append(chunk.text)
            lines.append("")

        return "\n".join(lines)
