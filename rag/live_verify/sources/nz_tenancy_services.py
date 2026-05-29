"""
Source adapter for tenancy.govt.nz.
Fetches plain-language Tenancy Services guidance via headless Firefox.
"""

from __future__ import annotations

from rag.live_verify.browser import BrowserSession
from .base import FetchResult

_BASE = "https://www.tenancy.govt.nz"

# Well-known topic slugs on tenancy.govt.nz
TOPIC_URLS: dict[str, str] = {
    "bond":           f"{_BASE}/rent-bond-and-bills/bonds/",
    "entry":          f"{_BASE}/maintenance-and-inspections/entry-to-the-property/",
    "notice":         f"{_BASE}/ending-a-tenancy/notice-to-end-tenancy/",
    "rent":           f"{_BASE}/rent-bond-and-bills/rent/",
    "repairs":        f"{_BASE}/maintenance-and-inspections/repairs-and-maintenance/",
    "healthy_homes":  f"{_BASE}/healthy-homes/",
    "meth":           f"{_BASE}/disputes/specific-issues/methamphetamine/",
    "anti_social":    f"{_BASE}/ending-a-tenancy/anti-social-behaviour/",
}


class NZTenancyServicesSource:
    source_id = "nz_tenancy_services"
    label = "Tenancy Services (tenancy.govt.nz)"
    base_url = _BASE

    async def fetch_topic(
        self,
        session: BrowserSession,
        topic: str,
    ) -> FetchResult:
        """
        Fetch a known topic page from tenancy.govt.nz.

        Args:
            session: Open BrowserSession.
            topic: Topic slug from TOPIC_URLS keys, or a full URL.
        """
        url = TOPIC_URLS.get(topic.lower().strip(), topic if topic.startswith("http") else f"{_BASE}/{topic}/")
        text = await session.fetch_text(url, wait="networkidle")
        return FetchResult(
            url=url,
            source_id=self.source_id,
            title=f"Tenancy Services: {topic}",
            text=text,
            query_excerpt=None,
        )

    async def search(
        self,
        session: BrowserSession,
        query: str,
    ) -> FetchResult:
        """
        Fetch the tenancy.govt.nz search results page for a query.
        """
        import urllib.parse
        search_url = f"{_BASE}/search/?query={urllib.parse.quote(query)}"
        text = await session.fetch_text(search_url, wait="networkidle")

        idx = text.lower().find(query.lower())
        excerpt: str | None = None
        if idx != -1:
            excerpt = text[max(0, idx - 100): idx + 800]

        return FetchResult(
            url=search_url,
            source_id=self.source_id,
            title=f"Tenancy Services search: {query}",
            text=text,
            query_excerpt=excerpt,
        )
