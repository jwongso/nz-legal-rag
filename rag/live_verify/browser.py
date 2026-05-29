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
