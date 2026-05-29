"""
Base interface for MCP tools.
Each tool module must expose a `register(mcp, pipeline, domain)` function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from rag.pipeline import RAGPipeline
    from mcp.domains.base import DomainConfig


class ToolModule(Protocol):
    """
    Protocol that every tool module must satisfy.

    A tool module is a Python module (e.g. mcp/tools/search_cases.py) that
    exposes a single top-level function:

        def register(mcp: FastMCP, pipeline: RAGPipeline, domain: DomainConfig) -> None

    register() calls mcp.tool() to attach one or more tools to the server,
    closing over `pipeline` and `domain` for runtime access.
    """

    def register(
        self,
        mcp: "FastMCP",
        pipeline: "RAGPipeline",
        domain: "DomainConfig",
    ) -> None: ...
