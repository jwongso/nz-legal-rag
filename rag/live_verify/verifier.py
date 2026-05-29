"""
LiveVerifier: high-level API combining BrowserSession + source adapters.
Use this from application code or benchmarks.
"""

from __future__ import annotations

from .browser import BrowserSession
from .sources.nz_legislation import NZLegislationSource
from .sources.nz_tenancy_services import NZTenancyServicesSource
from .sources.base import FetchResult


class LiveVerifier:
    """
    High-level facade for live web verification.

    Usage:
        async with LiveVerifier() as v:
            result = await v.fetch_legislation_section("s51")
            result = await v.fetch_tenancy_topic("entry")
            result = await v.fetch_url("https://...")
    """

    def __init__(self) -> None:
        self._session = BrowserSession()
        self._nz_leg = NZLegislationSource()
        self._nz_ts = NZTenancyServicesSource()

    async def __aenter__(self) -> "LiveVerifier":
        await self._session.open()
        return self

    async def __aexit__(self, *_) -> None:
        await self._session.close()

    async def fetch_legislation_section(
        self, reference: str, act: str = "RTA"
    ) -> FetchResult:
        """Fetch a specific section from an NZ Act."""
        return await self._nz_leg.fetch_section(self._session, reference, act)

    async def fetch_legislation_search(
        self, query: str, act: str = "RTA"
    ) -> FetchResult:
        """Search an NZ Act page for a query term."""
        return await self._nz_leg.search(self._session, query, act)

    async def fetch_tenancy_topic(self, topic: str) -> FetchResult:
        """Fetch a Tenancy Services guidance page by topic slug."""
        return await self._nz_ts.fetch_topic(self._session, topic)

    async def fetch_tenancy_search(self, query: str) -> FetchResult:
        """Search tenancy.govt.nz for a query."""
        return await self._nz_ts.search(self._session, query)

    async def fetch_url(self, url: str, query: str = "") -> FetchResult:
        """Fetch any URL and optionally locate a query within the page."""
        text = await self._session.fetch_text(url)
        excerpt: str | None = None
        if query:
            idx = text.lower().find(query.lower())
            if idx != -1:
                excerpt = text[max(0, idx - 200): idx + 1000]
        return FetchResult(
            url=url,
            source_id="custom",
            title=url,
            text=text,
            query_excerpt=excerpt,
        )
