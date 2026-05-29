"""MCP tool: search indexed legislation sections."""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
from rag.pipeline import RAGPipeline
from mcp.domains.base import DomainConfig


def register(mcp: FastMCP, pipeline: RAGPipeline, domain: DomainConfig) -> None:
    act_prefixes: dict[str, str] = domain.tool_kwargs.get("act_prefixes", {})
    acts_help = (
        ", ".join(f"{k} ({v})" for k, v in act_prefixes.items())
        if act_prefixes else "none configured"
    )

    @mcp.tool()
    async def search_legislation(
        query: str,
        act: str = "",
        top_k: int = 5,
    ) -> str:
        f"""
        Search indexed legislation sections by topic or section reference.

        Returns raw section text - no LLM generation, actual law only.

        Args:
            query: Topic or section to find, e.g. "right of entry notice period".
            act: Optional act code to restrict search. Available: {acts_help}.
                 Leave empty to search all indexed legislation.
            top_k: Number of sections to return (default 5, max 20).
        """
        query_vector = await pipeline._embedder.embed(query)

        raw_hits = pipeline._store.search(
            query_vector,
            top_k=min(top_k, 20) * 4,
            courts=["NZLEG"],
        )

        act_prefix = f"NZLEG/{act.upper().strip()}/" if act.strip() else ""
        if act_prefix:
            raw_hits = [h for h in raw_hits if h.case_id.startswith(act_prefix)]

        seen: set[str] = set()
        hits = []
        for h in raw_hits:
            if h.case_id not in seen:
                seen.add(h.case_id)
                hits.append(h)
            if len(hits) >= min(top_k, 20):
                break

        if not hits:
            label = f" in {act.upper()}" if act.strip() else ""
            return f"No matching legislation sections found{label}."

        lines = [f"Found {len(hits)} section(s) matching '{query}':\n"]
        for h in hits:
            lines += [
                f"## {h.title}",
                f"Citation: {h.case_id}  |  Score: {h.score:.4f}",
                f"URL: {h.url}",
                "",
                h.text,
                "",
                "---",
                "",
            ]
        return "\n".join(lines)
