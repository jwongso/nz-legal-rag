"""
BabaYaga MCP Server - modular RAG + live-verification SDK.

Swap the domain config to serve any vertical (NZ law, healthcare, accounting...).
Each enabled tool is registered automatically from mcp/tools/.

Add to your MCP config:
  {
    "nz-legal": {
      "command": "python",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/nz-legal-rag"
    }
  }

To deploy a different domain:
  from mcp.domains import MY_DOMAIN
  server = build_server(MY_DOMAIN)
  server.run()
"""

from __future__ import annotations

import importlib

import importlib.util, sys

# The local mcp/ directory shadows the installed mcp package.
# Load FastMCP directly from site-packages to avoid the conflict.
_site = next(p for p in sys.path if "site-packages" in p and "mcp" not in p)
sys.path.insert(0, _site)
from mcp.server.fastmcp import FastMCP  # installed package, not local mcp/
sys.path.pop(0)

from mcp.domains.base import DomainConfig
from mcp.domains.nz_legal import NZ_LEGAL
from rag.pipeline import RAGPipeline


def build_server(domain: DomainConfig) -> FastMCP:
    """
    Construct an MCP server for the given domain.

    Loads the RAGPipeline once, then registers each tool listed in
    domain.enabled_tools by importing mcp.tools.<tool_name>.register().
    """
    mcp = FastMCP(domain.name, description=domain.description)
    pipeline = RAGPipeline()

    for tool_name in domain.enabled_tools:
        module = importlib.import_module(f"mcp.tools.{tool_name}")
        module.register(mcp, pipeline, domain)

    return mcp


if __name__ == "__main__":
    server = build_server(NZ_LEGAL)
    server.run()
