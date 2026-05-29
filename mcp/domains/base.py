"""
Abstract domain configuration.
Subclass this to define a new BabaYaga vertical (NZ law, healthcare, accounting, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainConfig:
    """
    Describes one deployable domain for the MCP server.

    Each domain maps to a specific RAG collection and exposes a curated set
    of MCP tools. Swap the domain config to serve a completely different
    business vertical without changing any tool code.
    """

    # Human-readable name shown in MCP server description
    name: str

    # Short description shown to the MCP client
    description: str

    # Qdrant collection to search
    qdrant_collection: str

    # Mapping of court/source codes to human-readable labels
    # e.g. {"NZTT": "Tenancy Tribunal", "NZLEG": "NZ Legislation"}
    source_labels: dict[str, str] = field(default_factory=dict)

    # Which tool modules to register. Each entry must be an importable path
    # relative to mcp.tools, e.g. "search_cases", "search_legislation".
    enabled_tools: list[str] = field(default_factory=list)

    # Extra domain-specific kwargs forwarded to each tool on registration.
    # Tools that do not recognise a key must ignore it gracefully.
    tool_kwargs: dict[str, Any] = field(default_factory=dict)
