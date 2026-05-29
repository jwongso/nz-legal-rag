"""Abstract base for domain-specific legislation source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag.live_verify.browser import BrowserSession


@dataclass
class FetchResult:
    url: str
    source_id: str
    title: str
    text: str
    query_excerpt: str | None  # surrounding context if a query was given


class LegislationSource(Protocol):
    """
    Protocol for a domain-specific legislation source.
    Implement this to add support for a new official website.
    """

    source_id: str
    label: str
    base_url: str

    async def fetch_section(
        self,
        session: BrowserSession,
        reference: str,
    ) -> FetchResult: ...

    async def search(
        self,
        session: BrowserSession,
        query: str,
    ) -> FetchResult: ...
