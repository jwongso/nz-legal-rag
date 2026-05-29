"""
MCP tool: live web verification via headless Firefox.

Browses an official source URL and returns the page text so the MCP client
(Claude) can check whether a claim is consistent with the current live content.
Simulates real human browsing - NZ locale, realistic user-agent, JS rendered.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
from rag.pipeline import RAGPipeline
from rag.live_verify.browser import BrowserSession
from mcp.domains.base import DomainConfig


def register(mcp: FastMCP, pipeline: RAGPipeline, domain: DomainConfig) -> None:
    verify_sources: list[dict] = domain.tool_kwargs.get("verify_sources", [])
    source_help = (
        ", ".join(f"'{s['id']}' ({s['label']})" for s in verify_sources)
        if verify_sources else "none configured"
    )
    source_map = {s["id"]: s for s in verify_sources}

    @mcp.tool()
    async def verify_live(
        url: str,
        source_id: str = "",
        query: str = "",
    ) -> str:
        f"""
        Browse a live official source and return its current text content.

        Use this to verify whether a legal or regulatory claim is consistent
        with the current published text of a law, guideline, or official page.
        Returns raw page text - you assess whether the claim holds.

        Args:
            url: Full URL to fetch, e.g.
                 "https://www.legislation.govt.nz/act/public/1986/120/en/latest/"
            source_id: Optional hint for known trusted sources.
                       Available: {source_help}
            query: Optional search term to locate within the page (returned
                   with surrounding context if found).
        """
        if not url.startswith("http"):
            return "Error: url must be a fully qualified https:// URL."

        # Validate against trusted sources if source_id is given
        if source_id and source_id in source_map:
            base = source_map[source_id]["base_url"]
            if not url.startswith(base):
                return (
                    f"Error: url does not match expected base for '{source_id}' "
                    f"({base}). Provide the full URL manually."
                )

        try:
            async with BrowserSession() as session:
                text = await session.fetch_text(url)
        except Exception as e:
            return f"Error fetching {url}: {e}"

        if not query:
            # Return a trimmed version - first 4000 chars is usually enough
            return f"[Fetched: {url}]\n\n{text[:4000]}"

        # Locate the query within the page and return surrounding context
        idx = text.lower().find(query.lower())
        if idx == -1:
            return (
                f"[Fetched: {url}]\n\n"
                f"Query '{query}' not found in page text.\n\n"
                f"Page excerpt (first 2000 chars):\n{text[:2000]}"
            )

        start = max(0, idx - 300)
        end = min(len(text), idx + 1500)
        excerpt = text[start:end]
        return (
            f"[Fetched: {url}]\n"
            f"[Query '{query}' found at position {idx}]\n\n"
            f"...{excerpt}..."
        )
