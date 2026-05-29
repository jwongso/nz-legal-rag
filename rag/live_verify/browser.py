"""
Shared async browser context using Playwright Firefox.
Human-like fingerprint: real user-agent, viewport, locale, timezone.
Reuse a single context across multiple fetches within a session.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) "
    "Gecko/20100101 Firefox/136.0"
)
_VIEWPORT = {"width": 1366, "height": 768}
_LOCALE = "en-NZ"
_TIMEZONE = "Pacific/Auckland"


class BrowserSession:
    """
    Manages a single headless Firefox browser instance for the session.
    Use as an async context manager or call open()/close() explicitly.

    Usage:
        async with BrowserSession() as session:
            text = await session.fetch_text("https://example.com")
    """

    def __init__(
        self,
        headless: bool = True,
        user_agent: str = _USER_AGENT,
        viewport: dict = _VIEWPORT,
        locale: str = _LOCALE,
        timezone: str = _TIMEZONE,
        timeout_ms: int = 30_000,
    ) -> None:
        self._headless = headless
        self._ua = user_agent
        self._viewport = viewport
        self._locale = locale
        self._timezone = timezone
        self._timeout_ms = timeout_ms
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def open(self) -> None:
        self._pw = await async_playwright().start()
        self._browser = await self._pw.firefox.launch(headless=self._headless)
        self._context = await self._browser.new_context(
            user_agent=self._ua,
            viewport=self._viewport,
            locale=self._locale,
            timezone_id=self._timezone,
        )
        self._context.set_default_timeout(self._timeout_ms)

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def __aenter__(self) -> "BrowserSession":
        await self.open()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def new_page(self) -> Page:
        if not self._context:
            raise RuntimeError("BrowserSession is not open. Call open() first.")
        return await self._context.new_page()

    async def fetch_text(self, url: str, wait: str = "networkidle") -> str:
        """Navigate to url and return the full visible text of the page body."""
        page = await self.new_page()
        try:
            await page.goto(url, wait_until=wait, timeout=self._timeout_ms)
            return await page.inner_text("body")
        finally:
            await page.close()

    async def fetch_html(self, url: str, wait: str = "networkidle") -> str:
        """Return raw HTML of the page."""
        page = await self.new_page()
        try:
            await page.goto(url, wait_until=wait, timeout=self._timeout_ms)
            return await page.content()
        finally:
            await page.close()

    async def search_ddg(self, query: str, max_results: int = 5) -> list[dict]:
        """Search DuckDuckGo (HTML endpoint) and return results as {title, url, body} dicts."""
        from urllib.parse import quote_plus, unquote, urlparse, parse_qs
        from bs4 import BeautifulSoup

        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        html = await self.fetch_html(search_url, wait="networkidle")
        soup = BeautifulSoup(html, "html.parser")

        results = []
        for div in soup.select(".result"):
            title_el = div.select_one(".result__a")
            snippet_el = div.select_one(".result__snippet")
            if not title_el:
                continue
            title = " ".join(title_el.get_text(separator=" ", strip=True).split())
            href = title_el.get("href", "")
            # DDG wraps redirect URLs - extract the real destination
            if "uddg=" in href:
                uddg = parse_qs(urlparse(href).query).get("uddg", [""])[0]
                if uddg:
                    href = unquote(uddg)
            body = " ".join(snippet_el.get_text(separator=" ", strip=True).split()) if snippet_el else ""
            # Skip DuckDuckGo ad redirects (y.js links)
            if "duckduckgo.com/y.js" in href:
                continue
            results.append({"title": title, "url": href, "body": body})
            if len(results) >= max_results:
                break

        return results
