"""
Source adapter for legislation.govt.nz.
Fetches current NZ Act text via headless Firefox.
"""

from __future__ import annotations

import re

from rag.live_verify.browser import BrowserSession
from .base import FetchResult

# RTA 1986 base URL - always the latest version
_RTA_URL = "https://www.legislation.govt.nz/act/public/1986/120/en/latest/"

# Map well-known act identifiers to their latest-version URLs
ACT_URLS: dict[str, str] = {
    "RTA":      _RTA_URL,
    "ERA2000":  "https://www.legislation.govt.nz/act/public/2000/24/en/latest/",
    "PA2020":   "https://www.legislation.govt.nz/act/public/2020/31/en/latest/",
    "CA1993":   "https://www.legislation.govt.nz/act/public/1993/105/en/latest/",
    "CRA1961":  "https://www.legislation.govt.nz/act/public/1961/43/en/latest/",
}


class NZLegislationSource:
    source_id = "nz_legislation"
    label = "NZ Legislation (legislation.govt.nz)"
    base_url = "https://www.legislation.govt.nz"

    async def fetch_section(
        self,
        session: BrowserSession,
        reference: str,
        act: str = "RTA",
    ) -> FetchResult:
        """
        Fetch a specific section from an NZ Act.

        Args:
            session: Open BrowserSession.
            reference: Section reference, e.g. "s51", "51", "s54A".
            act: Act code from ACT_URLS keys (default: RTA).
        """
        url = ACT_URLS.get(act.upper(), _RTA_URL)
        full_text = await session.fetch_text(url, wait="networkidle")

        # Normalise reference: strip leading 's' or 'S'
        sec = re.sub(r"^[sS]", "", reference).strip()

        # The page renders the full Act - find the section by number
        pattern = rf"\n{re.escape(sec)}\b"
        matches = list(re.finditer(pattern, full_text))

        # Usually appears twice: once in ToC, once in body. Take last two.
        excerpt: str | None = None
        for m in matches[-2:]:
            candidate = full_text[m.start(): m.start() + 1200]
            # Confirm it is a section body (contains a numbered sub-clause or paragraph)
            if re.search(r"\(\d+\)", candidate):
                excerpt = candidate
                break

        title_match = re.search(r"title[^|]+", full_text[:200])
        title = title_match.group() if title_match else f"NZ Act - {act}"

        return FetchResult(
            url=url,
            source_id=self.source_id,
            title=title,
            text=full_text,
            query_excerpt=excerpt,
        )

    async def search(
        self,
        session: BrowserSession,
        query: str,
        act: str = "RTA",
    ) -> FetchResult:
        """
        Fetch an act and return page text + excerpt around the query term.
        """
        url = ACT_URLS.get(act.upper(), _RTA_URL)
        full_text = await session.fetch_text(url, wait="networkidle")

        idx = full_text.lower().find(query.lower())
        excerpt: str | None = None
        if idx != -1:
            start = max(0, idx - 200)
            excerpt = full_text[start: start + 1000]

        return FetchResult(
            url=url,
            source_id=self.source_id,
            title=f"NZ Legislation search: {query}",
            text=full_text,
            query_excerpt=excerpt,
        )
